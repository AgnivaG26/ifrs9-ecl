"""Lifetime PD: cause-specific quarterly hazards for default and prepayment.

For each annual snapshot (December) of a performing loan, we follow it for up
to H quarters and record, per quarter it is still on the books, whether it
defaulted, prepaid, or carried on. Two logistic hazard models are fitted:

    P(default in quarter h | still alive)  = logit^-1( score + f(h) + macro at h )
    P(prepay  in quarter h | still alive)  = logit^-1( score + f(h) + rate gap at h + ... )

"macro at h" uses the actual economy when fitting, and a scenario path when
forecasting. That is what makes lifetime PD "forward-looking" under IFRS 9.
Survival to quarter h multiplies (1 - default hazard - prepay hazard) over the
quarters before it; marginal PD in quarter h = survival(h-1) x default hazard(h).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

H_FIT = 20  # quarters of follow-up used to fit (5 years)


def macro_features(macro_m: pd.DataFrame) -> pd.DataFrame:
    m = macro_m.set_index("t").sort_index()
    return m[["unemployment", "unemp_chg_12m", "hpi", "hpi_growth_12m", "mortgage_rate"]]


def hazard_dataset(snap: pd.DataFrame, events: pd.DataFrame, macro_m: pd.DataFrame,
                   H: int = H_FIT, max_snapshots: int = 250_000, seed: int = 3,
                   cutoff_t: int | None = None) -> pd.DataFrame:
    """snap: PD-dataset rows with score_lo; events: per-loan first_default_t,
    term_t, term_code, last_t. With `cutoff_t`, only information available at
    that month is used (no look-ahead): later events are treated as unseen."""
    s = snap[snap.period % 100 == 12].copy()
    ev = events.copy()
    if cutoff_t is not None:
        s = s[s.t <= cutoff_t - 3]
        ev.loc[ev.first_default_t > cutoff_t, "first_default_t"] = np.nan
        late = ev.term_t > cutoff_t
        ev.loc[late, ["term_t", "term_code"]] = [np.nan, None]
        ev["last_t"] = ev.last_t.clip(upper=cutoff_t)
    if len(s) > max_snapshots:
        s = s.sample(max_snapshots, random_state=seed)
    s = s.join(ev, on="loan_id")
    t = s["t"].to_numpy()
    big = 10 ** 6
    q_def = np.where(s.first_default_t.notna(),
                     np.ceil((s.first_default_t.fillna(0) - t) / 3), big).astype(int)
    pre = (s.term_code == "01") & s.term_t.notna()
    q_pre = np.where(pre, np.ceil((s.term_t.fillna(0) - t) / 3), big).astype(int)
    q_obs = np.floor((s.last_t.to_numpy() - t) / 3).astype(int)
    q_evt = np.minimum(q_def, q_pre)
    n_rows = np.where(q_evt <= H, q_evt, np.minimum(H, q_obs)).clip(min=0)

    rep = np.repeat(np.arange(len(s)), n_rows)
    h = np.concatenate([np.arange(1, k + 1) for k in n_rows]) if n_rows.sum() else np.array([], int)
    long = pd.DataFrame({
        "h": h, "t0": t[rep], "score_lo": s.score_lo.to_numpy()[rep],
        "current_rate": s.current_rate.to_numpy()[rep], "weight": s.weight.to_numpy()[rep],
        "default": (h == q_def[rep]).astype("int8"),
        "prepay": ((h == q_pre[rep]) & (q_pre[rep] < q_def[rep])).astype("int8"),
    })
    m = macro_features(macro_m)
    tau = long["t0"] + 3 * long["h"]
    mm = m.reindex(tau.to_numpy())
    for c in ["unemployment", "unemp_chg_12m", "hpi_growth_12m", "mortgage_rate"]:
        long[c] = mm[c].to_numpy()
    long["hpi_cum"] = np.log(mm["hpi"].to_numpy() / m["hpi"].reindex(long["t0"]).to_numpy())
    long["rate_gap"] = long["current_rate"] - long["mortgage_rate"]
    return long.dropna().reset_index(drop=True)


def _design(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    h = np.minimum(d["h"], H_FIT)  # beyond the fitted range, hold the age shape flat
    X = pd.DataFrame({"score_lo": d["score_lo"], "log_h": np.log(h), "h": h / 4.0},
                     index=d.index)
    if kind == "default":
        X["unemployment"] = d["unemployment"]
        X["unemp_chg_12m"] = d["unemp_chg_12m"]
        X["hpi_cum"] = d["hpi_cum"]
    else:
        X["rate_gap"] = d["rate_gap"]
        X["hpi_cum"] = d["hpi_cum"]
        X["hpi_growth_12m"] = d["hpi_growth_12m"]
    return sm.add_constant(X, has_constant="add")


def fit_hazards(long: pd.DataFrame) -> dict:
    out = {}
    for kind in ("default", "prepay"):
        out[kind] = sm.GLM(long[kind].astype(float), _design(long, kind),
                           family=sm.families.Binomial(),
                           freq_weights=long["weight"]).fit()
    return out


def predict_hazard(model, d: pd.DataFrame, kind: str) -> np.ndarray:
    return np.asarray(model.predict(_design(d, kind)))
