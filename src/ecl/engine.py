"""IFRS 9 engine: scenarios, staging, EAD, and expected credit loss.

ECL for one loan under one scenario:

    ECL = sum over quarters h of  marginal PD(h) x LGD(h) x EAD(h) x discount(h)

summed over 4 quarters for Stage 1 (12-month ECL) and over the remaining life
for Stage 2 (lifetime ECL). Stage 3 loans have already defaulted: ECL = LGD x EAD.
The reported ECL is the probability-weighted average across scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lgd as lgd_mod
from . import lifetime as lt

H_MAX = 40  # quarters (10 years): lifetime ECL horizon cap, a stated simplification


@dataclass
class Scenario:
    name: str
    weight: float
    unemp_target: float      # unemployment level reached ...
    unemp_peak_q: int        # ... by this quarter
    unemp_long_run: float    # level it settles at by quarter 20
    hpi_shock: float         # total house-price change over the first `hpi_shock_q` quarters
    hpi_shock_q: int
    hpi_growth_after: float  # annual growth afterwards
    mortgage_rate: float     # level reached by quarter 8


SCENARIOS = [
    Scenario("Upside", 0.15, 3.8, 6, 4.2, 0.10, 8, 0.04, 5.5),
    Scenario("Base", 0.55, 4.6, 4, 4.5, 0.06, 8, 0.03, 6.0),
    Scenario("Downside", 0.25, 7.0, 8, 5.0, -0.08, 8, 0.03, 6.5),
    Scenario("Severe", 0.05, 10.0, 8, 6.0, -0.25, 12, 0.025, 5.0),
]


def scenario_path(sc: Scenario, start: dict, H: int = H_MAX) -> pd.DataFrame:
    """Quarterly macro path for quarters 1..H after the reporting date."""
    q = np.arange(1, H + 1)
    u0 = start["unemployment"]
    u = np.where(q <= sc.unemp_peak_q,
                 u0 + (sc.unemp_target - u0) * q / sc.unemp_peak_q,
                 np.interp(q, [sc.unemp_peak_q, max(20, sc.unemp_peak_q + 1)],
                           [sc.unemp_target, sc.unemp_long_run]))
    # Series from quarter -3 to H, so quarter q can look back 4 quarters (q-4).
    u_all = np.concatenate([[start["unemp_q-3"], start["unemp_q-2"], start["unemp_q-1"], u0], u])
    unemp_chg = u - u_all[: len(u)]
    g_q_shock = (1 + sc.hpi_shock) ** (1 / sc.hpi_shock_q) - 1
    g_q_after = (1 + sc.hpi_growth_after) ** 0.25 - 1
    g = np.where(q <= sc.hpi_shock_q, g_q_shock, g_q_after)
    hpi_cum = np.log(np.cumprod(1 + g))                    # log HPI vs reporting date
    step0 = np.log(1 + start["hpi_growth_12m"]) / 4        # recent history, per quarter
    hist = np.array([-3, -2, -1, 0]) * step0               # log HPI at quarters -3..0
    h_all = np.concatenate([hist, hpi_cum])
    hpi_growth_12m = np.exp(hpi_cum - h_all[: len(q)]) - 1
    r0 = start["mortgage_rate"]
    rate = np.where(q <= 8, r0 + (sc.mortgage_rate - r0) * q / 8, sc.mortgage_rate)
    return pd.DataFrame({"h": q, "unemployment": u, "unemp_chg_12m": unemp_chg,
                         "hpi_cum": hpi_cum, "hpi_growth_12m": hpi_growth_12m,
                         "mortgage_rate": rate})


def weighted_average_scenario(start: dict) -> pd.DataFrame:
    paths = [scenario_path(s, start) for s in SCENARIOS]
    w = np.array([s.weight for s in SCENARIOS])
    return sum(p * wi for p, wi in zip(paths, w)).assign(h=paths[0]["h"])


def amortized_balance(upb, rate, remaining_months, months_ahead):
    """Scheduled balance after `months_ahead` payments (annuity formula)."""
    r = np.asarray(rate) / 1200
    n = np.maximum(np.asarray(remaining_months), 1)
    k = np.minimum(months_ahead, n)
    growth = (1 + r) ** k
    pmt = upb * r / (1 - (1 + r) ** (-n))
    return np.maximum(upb * growth - pmt * (growth - 1) / r, 0)


def assign_stage(port: pd.DataFrame, rel: float = 3.0, abs_: float = 0.005) -> np.ndarray:
    """Stage 3: in default (90+ days past due).
    Stage 2: 30+ days past due (the IFRS 9 backstop), in forbearance, or a significant increase
    in credit risk: 12-month PD now >= `rel` x PD at origination AND at least
    `abs_` higher in absolute terms.
    Stage 1: everything else."""
    sicr = (port.pd12_now >= rel * port.pd12_orig) & (port.pd12_now - port.pd12_orig >= abs_)
    fb = port["forbearance"].astype(bool) if "forbearance" in port else False
    return np.select([(port.dpd >= 3) & ~fb, (port.dpd >= 1) | sicr | fb], [3, 2], 1)


def run_scenario(port: pd.DataFrame, haz: dict, lgd_model, path: pd.DataFrame,
                 H: int = H_MAX) -> dict:
    """Per-loan 12-month and lifetime ECL, plus the cumulative PD curve."""
    n = len(port)
    rem_q = np.clip(np.ceil(port.remaining_months.to_numpy() / 3), 1, H).astype(int)
    surv = np.ones(n)
    ecl12, ecl_life = np.zeros(n), np.zeros(n)
    cum_pd = np.zeros((n, H))
    has_mi = port.has_mi.to_numpy()
    value_now = port.value_now.to_numpy()
    for i, row in path.iterrows():
        h = int(row.h)
        d = pd.DataFrame({"h": h, "score_lo": port.score_lo.to_numpy(),
                          "unemployment": row.unemployment, "unemp_chg_12m": row.unemp_chg_12m,
                          "hpi_cum": row.hpi_cum, "hpi_growth_12m": row.hpi_growth_12m,
                          "rate_gap": port.current_rate.to_numpy() - row.mortgage_rate})
        lam_d = lt.predict_hazard(haz["default"], d, "default")
        lam_p = lt.predict_hazard(haz["prepay"], d, "prepay")
        alive = h <= rem_q
        marg_pd = np.where(alive, surv * lam_d, 0.0)
        ead = amortized_balance(port.current_upb.to_numpy(), port.current_rate.to_numpy(),
                                port.remaining_months.to_numpy(), 3 * h)
        ltv_h = 100 * ead / (value_now * np.exp(row.hpi_cum))
        lgd_h = lgd_mod.predict_lgd(lgd_model, ltv_h, has_mi, np.full(n, row.hpi_growth_12m))
        disc = (1 + port.current_rate.to_numpy() / 1200) ** (-3 * h)
        loss = marg_pd * lgd_h * ead * disc
        ecl_life += loss
        if h <= 4:
            ecl12 += loss
        cum_pd[:, h - 1] = (cum_pd[:, h - 2] if h > 1 else 0) + marg_pd
        surv = np.where(alive, surv * (1 - lam_d - lam_p), surv)
    lgd_now = lgd_mod.predict_lgd(lgd_model, 100 * port.current_upb / value_now, has_mi,
                                  np.full(n, path.hpi_growth_12m.iloc[0]))
    return {"ecl12": ecl12, "ecl_life": ecl_life, "cum_pd": cum_pd,
            "ecl_defaulted": lgd_now * port.current_upb.to_numpy()}


def ecl_by_stage(port: pd.DataFrame, res: dict, stage: np.ndarray) -> np.ndarray:
    return np.select([stage == 1, stage == 2], [res["ecl12"], res["ecl_life"]],
                     res["ecl_defaulted"])
