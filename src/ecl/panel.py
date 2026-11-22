"""Build the 12-month PD development dataset from loan-month performance data.

One row = one performing loan at one quarter-end "snapshot". Features are what
the bank knew at that date; the target is whether the loan defaults in the
next 12 months. This is the standard layout for behavioural PD models.

Eligibility at a snapshot: the loan is active, not 90+ days past due, and has
never defaulted before (we model first default).
Censoring: a snapshot is dropped when the data ends (performance cutoff)
before the 12-month window closes and the loan has neither defaulted nor
paid off. A loan that prepays inside the window counts as a non-default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CREDIT_EVENT_ZB_CODES, DEFAULT_DPD, PD_HORIZON,
                     SNAPSHOT_MONTHS)
from .io import yyyymm_to_index


def default_flag(perf: pd.DataFrame) -> pd.Series:
    """90+ days past due outside forbearance, or a credit-event zero balance."""
    fb = perf["forbearance"] if "forbearance" in perf else False
    return (((perf["dpd"] >= DEFAULT_DPD) & ~fb)
            | perf["zero_balance_code"].isin(CREDIT_EVENT_ZB_CODES))


def loan_events(perf: pd.DataFrame) -> pd.DataFrame:
    """Per loan: first default month, termination month and code, last month seen."""
    is_def = default_flag(perf)
    g = perf.groupby("loan_id", sort=False)
    ev = pd.DataFrame({"last_t": g["t"].max()})
    ev["first_default_t"] = perf.loc[is_def].groupby("loan_id")["t"].min()
    term = perf.loc[perf["zero_balance_code"] != ""].groupby("loan_id").agg(
        term_t=("t", "min"), term_code=("zero_balance_code", "first"))
    return ev.join(term)


def rolling_max_dpd(perf: pd.DataFrame, window: int = 12) -> np.ndarray:
    """Max delinquency over the last `window` months, within each loan.

    Uses shifted copies of the array instead of a groupby-rolling, which is far
    faster on tens of millions of rows. Assumes rows are sorted by loan then month.
    """
    dpd = perf["dpd"].fillna(0).to_numpy()
    ids = perf["loan_id"].to_numpy()
    out = dpd.copy()
    for k in range(1, window):
        shifted = np.empty_like(dpd)
        shifted[:k] = 0
        shifted[k:] = dpd[:-k]
        same = np.zeros(len(ids), bool)
        same[k:] = ids[k:] == ids[:-k]
        out = np.maximum(out, np.where(same, shifted, 0))
    return out


def build_pd_dataset(orig: pd.DataFrame, perf: pd.DataFrame, macro: pd.DataFrame,
                     horizon: int = PD_HORIZON, max_rows: int | None = 3_000_000,
                     seed: int = 0) -> pd.DataFrame:
    # Adds a column to the caller's table instead of copying it: on the real data
    # a copy of ~15 million rows would double memory use.
    if "max_dpd_12m" not in perf:
        perf["max_dpd_12m"] = rolling_max_dpd(perf)
    ev = loan_events(perf)

    month = perf["period"] % 100
    snap = perf[month.isin(SNAPSHOT_MONTHS) & (perf["dpd"] < DEFAULT_DPD)
                & (perf["zero_balance_code"] == "")].copy()
    snap = snap.join(ev, on="loan_id")
    snap = snap[snap["first_default_t"].isna() | (snap["first_default_t"] > snap["t"])]

    # The whole 12-month window must lie inside the data. Dropping only the
    # survivors near the cutoff (while keeping early defaulters) would inflate
    # the default rate of the last year of snapshots.
    snap = snap[snap["t"] + horizon <= perf["t"].max()]
    t, fd = snap["t"], snap["first_default_t"]
    snap["default_12m"] = ((fd > t) & (fd <= t + horizon)).astype("int8")
    resolved = snap["term_t"].notna() & (snap["term_t"] <= t + horizon)
    observed = snap["last_t"] >= t + horizon
    snap = snap[(snap["default_12m"] == 1) | resolved | observed]

    snap = add_features(snap, orig, macro)

    cols = ["loan_id", "period", "t", "vintage", "default_12m", "current_upb", "current_rate",
            "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_rate",
            "log_orig_upb", "loan_purpose", "occupancy", "channel",
            "first_time_homebuyer", "multi_borrower", "property_type", "property_state",
            "has_mi", "short_term", "loan_age", "dpd_now", "max_dpd_12m", "mtm_ltv",
            "rate_incentive", "paid_down_pct", "unemployment", "unemp_chg_12m",
            "hpi_growth_12m"]
    out = snap[cols].reset_index(drop=True)
    if max_rows and len(out) > max_rows:
        # Keep every default, sample non-defaults, and carry a weight so rates
        # stay unbiased. (Needed only for the full real data.)
        rng = np.random.default_rng(seed)
        nd = out.index[out["default_12m"] == 0]
        keep_n = max_rows - int(out["default_12m"].sum())
        pick = rng.choice(nd, size=keep_n, replace=False)
        w = len(nd) / keep_n
        out = pd.concat([out[out["default_12m"] == 1].assign(weight=1.0),
                         out.loc[pick].assign(weight=w)]).sort_values(["loan_id", "t"])
    else:
        out["weight"] = 1.0
    return out.reset_index(drop=True)


def add_features(snap: pd.DataFrame, orig: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Loan features as known at each row's month: origination attributes,
    macro conditions, and dynamic measures (marked-to-market LTV, refinancing
    incentive, pay-down). Used for PD development and for the ECL portfolio."""
    # --- static features from origination
    o = orig.set_index("loan_id")
    keep = ["vintage", "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_rate",
            "orig_upb", "loan_purpose", "occupancy", "channel", "first_time_homebuyer",
            "num_borrowers", "property_type", "property_state", "mi_pct", "orig_term",
            "first_payment_date"]
    snap = snap.join(o[keep], on="loan_id")

    # --- macro at the snapshot, and house prices at origination
    m = macro.set_index("t")
    snap = snap.join(m[["unemployment", "unemp_chg_12m", "hpi", "hpi_growth_12m",
                        "mortgage_rate"]], on="t")
    fp_t = yyyymm_to_index(snap["first_payment_date"].fillna(190001))
    snap["hpi_orig"] = m["hpi"].reindex(fp_t - 1).to_numpy()

    # --- dynamic features
    value0 = snap["orig_upb"] / (snap["orig_ltv"] / 100)
    snap["mtm_ltv"] = 100 * snap["current_upb"] / (value0 * snap["hpi"] / snap["hpi_orig"])
    snap["rate_incentive"] = snap["current_rate"] - snap["mortgage_rate"]
    snap["paid_down_pct"] = 1 - snap["current_upb"] / snap["orig_upb"]
    snap["dpd_now"] = snap["dpd"].fillna(0).clip(upper=2)
    snap["has_mi"] = (snap["mi_pct"].fillna(0) > 0).astype("int8")
    snap["multi_borrower"] = (snap["num_borrowers"].fillna(1) > 1).astype("int8")
    snap["short_term"] = (snap["orig_term"] <= 180).astype("int8")
    snap["log_orig_upb"] = np.log(snap["orig_upb"])

    return snap
