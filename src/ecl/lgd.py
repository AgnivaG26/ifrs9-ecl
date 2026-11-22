"""LGD: loss given default, from Freddie Mac's actual-loss records.

Realised LGD = actual loss / balance removed at liquidation, for loans that
ended in a credit event (third-party sale, short sale, REO disposition, whole
loan sale). Modelled with a fractional logit: a logistic link on a 0-1 outcome,
which keeps predictions inside [0, 1] without forcing them to 0 or 1.

Drivers: loan-to-value at liquidation (the collateral covers less of the debt
when it is high) and mortgage insurance (cuts the loss). Because LTV is
projected along each scenario's house-price path, a scenario with falling
prices raises LGD as well as PD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import CREDIT_EVENT_ZB_CODES
from .io import yyyymm_to_index


def lgd_dataset(orig: pd.DataFrame, perf: pd.DataFrame, macro_m: pd.DataFrame) -> pd.DataFrame:
    """Liquidated loans with realised LGD and its drivers.

    LTV at liquidation: Freddie Mac's estimated LTV (ELTV, from local house
    prices) from the last month it was reported, where available; otherwise
    marked to market with the national house price index. On the real data
    ELTV exists for only ~12% of liquidations, but where it does, LGD rises
    steadily with it, while national-index LTV is a much blunter measure.
    """
    prev_eltv = perf.groupby("loan_id", sort=False)["eltv"].shift(1)
    liq_mask = (perf.zero_balance_code.isin(CREDIT_EVENT_ZB_CODES)
                & perf.actual_loss.notna() & (perf.zero_balance_removal_upb > 0))
    liq = perf[liq_mask].copy()
    liq["eltv_best"] = liq["eltv"].fillna(prev_eltv[liq_mask])
    o = orig.set_index("loan_id")[["orig_upb", "orig_ltv", "orig_cltv", "mi_pct", "first_payment_date"]]
    liq = liq.join(o, on="loan_id")
    m = macro_m.set_index("t")
    fp_t = yyyymm_to_index(liq["first_payment_date"])
    ltv0 = liq.orig_ltv.fillna(liq.orig_cltv)
    value_now = (liq.orig_upb / (ltv0 / 100)) * m["hpi"].reindex(liq.t).to_numpy() \
        / m["hpi"].reindex(fp_t - 1).to_numpy()
    liq["ltv_national"] = 100 * liq.zero_balance_removal_upb / value_now
    liq["ltv_liq"] = liq["eltv_best"].fillna(liq["ltv_national"])
    liq["has_mi"] = (liq.mi_pct.fillna(0) > 0).astype(int)
    liq["hpi_growth_12m"] = m["hpi_growth_12m"].reindex(liq.t).to_numpy()
    liq["lgd_raw"] = liq.actual_loss / liq.zero_balance_removal_upb
    liq["lgd"] = liq["lgd_raw"].clip(0, 1)
    return liq[["loan_id", "period", "ltv_liq", "eltv_best", "has_mi", "hpi_growth_12m",
                "lgd_raw", "lgd"]].dropna(subset=["ltv_liq", "lgd", "hpi_growth_12m"])


def _design(ltv, has_mi, hpi_growth=None):
    # 12-month house-price growth at liquidation was tested and dropped: on the
    # real data it took a positive (wrong) sign and was not significant (p = 0.07).
    # LGD still responds to scenarios through LTV, which moves with house prices.
    ltv = np.asarray(ltv, float)
    return sm.add_constant(pd.DataFrame({
        "ltv": ltv / 100, "ltv_above_100": np.maximum(ltv - 100, 0) / 100,
        "has_mi": np.asarray(has_mi, float)}), has_constant="add")


def fit_lgd(d: pd.DataFrame):
    return sm.GLM(d["lgd"].to_numpy(), _design(d.ltv_liq, d.has_mi, d.hpi_growth_12m),
                  family=sm.families.Binomial()).fit()


def predict_lgd(model, ltv, has_mi, hpi_growth) -> np.ndarray:
    return np.asarray(model.predict(_design(ltv, has_mi, hpi_growth)))
