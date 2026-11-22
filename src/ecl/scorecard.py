"""Weight-of-evidence (WoE) scorecard: the standard bank PD model.

Why banks still use it: every variable is binned, each bin gets a WoE value and
a number of points, and the final model is a logistic regression on those WoE
values. Anyone can see why a loan scored what it did, monotonic risk patterns
are enforced by construction, and validators and regulators know how to test it.

Conventions
-----------
WoE_bin = ln( share of goods in bin / share of bads in bin ). Higher WoE = safer.
So in logit(PD) = b0 + sum_j b_j * WoE_j every b_j should be NEGATIVE; a positive
sign means the variable works against its own univariate story and is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

NUMERIC = ["credit_score", "orig_ltv", "orig_cltv", "orig_dti", "mtm_ltv", "loan_age",
           "rate_incentive", "paid_down_pct", "log_orig_upb", "dpd_now", "max_dpd_12m"]
CATEGORICAL = ["loan_purpose", "occupancy", "channel", "first_time_homebuyer",
               "multi_borrower", "property_type", "property_state", "has_mi", "short_term"]
# Features whose risk need not rise or fall steadily (age has a hump).
NON_MONOTONIC = {"loan_age"}


@dataclass
class Binning:
    feature: str
    kind: str                     # "numeric" | "categorical"
    edges: list = field(default_factory=list)     # numeric: inner cut points
    groups: dict = field(default_factory=dict)    # categorical: value -> group label
    table: pd.DataFrame | None = None             # per-bin counts, rates, WoE
    iv: float = 0.0

    def bin_labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            lab = pd.Series(np.searchsorted(self.edges, x.to_numpy(), side="right"),
                            index=x.index).astype(str)
            return lab.where(x.notna(), "Missing")
        g = x.astype(str).map(self.groups).fillna("Other")
        return g.where(x.notna(), "Missing")

    def woe(self, x: pd.Series) -> pd.Series:
        m = self.table.set_index("bin")["woe"]
        return self.bin_labels(x).map(m).fillna(0.0).astype(float)


def _woe_table(labels, y, w, order=None) -> pd.DataFrame:
    d = pd.DataFrame({"bin": labels, "bad": y * w, "good": (1 - y) * w, "n": w})
    t = d.groupby("bin", sort=False)[["n", "bad", "good"]].sum()
    if order is not None:
        t = t.reindex([b for b in order if b in t.index])
    B, G = t["bad"].sum(), t["good"].sum()
    # +0.5 smoothing so a bin with no defaults gets a finite WoE.
    t["bad_share"] = (t["bad"] + 0.5) / (B + 0.5 * len(t))
    t["good_share"] = (t["good"] + 0.5) / (G + 0.5 * len(t))
    t["woe"] = np.log(t["good_share"] / t["bad_share"])
    t["iv_part"] = (t["good_share"] - t["bad_share"]) * t["woe"]
    t["bad_rate"] = t["bad"] / t["n"]
    t["share"] = t["n"] / t["n"].sum()
    return t.reset_index()


def bin_numeric(x, y, w, feature, max_bins=8, init_bins=20, min_share=None,
                monotonic=True) -> Binning:
    ok = x.notna()
    uniq = np.unique(x[ok])
    if len(uniq) <= init_bins:
        # Few distinct values (e.g. days past due 0/1/2): cut between them.
        # Quantile cuts would collapse them into a single bin.
        edges = list((uniq[:-1] + uniq[1:]) / 2)
        min_share = 0.005 if min_share is None else min_share
    else:
        edges = list(np.unique(np.nanquantile(x[ok], np.linspace(0, 1, init_bins + 1)[1:-1])))
        min_share = 0.03 if min_share is None else min_share

    def stats(edges):
        lab = np.searchsorted(edges, x[ok].to_numpy(), side="right")
        d = pd.DataFrame({"b": lab, "bad": y[ok] * w[ok], "n": w[ok]})
        s = d.groupby("b")[["bad", "n"]].sum().reindex(range(len(edges) + 1), fill_value=0)
        return s

    def merge(edges, i):          # merge bin i with bin i+1 = drop edge i
        return edges[:i] + edges[i + 1:]

    direction = np.sign(np.corrcoef(x[ok].rank(), y[ok])[0, 1]) or 1.0
    for _ in range(200):
        s = stats(edges)
        rate = (s["bad"] / s["n"].clip(lower=1e-9)).to_numpy()
        share = (s["n"] / s["n"].sum()).to_numpy()
        if len(edges) == 0:
            break
        small = np.flatnonzero(share < min_share)
        if small.size:                         # 1) every bin holds >= min_share
            i = small[0]
            j = i if i < len(edges) else i - 1
            if 0 < i < len(edges):             # merge toward the closer rate
                j = i - 1 if abs(rate[i] - rate[i - 1]) < abs(rate[i] - rate[i + 1]) else i
            edges = merge(edges, j)
            continue
        if monotonic:                          # 2) default rate moves one way only
            diffs = np.diff(rate) * direction
            bad = np.flatnonzero(diffs < 0)
            if bad.size:
                edges = merge(edges, int(bad[np.argmin(s["n"].to_numpy()[bad] + s["n"].to_numpy()[bad + 1])]))
                continue
        if len(edges) + 1 > max_bins:          # 3) at most max_bins bins
            gaps = np.abs(np.diff(rate))
            edges = merge(edges, int(np.argmin(gaps)))
            continue
        break
    labels = pd.Series(np.searchsorted(edges, x.to_numpy(), side="right"),
                       index=x.index).astype(str).where(ok, "Missing")
    order = [str(i) for i in range(len(edges) + 1)] + ["Missing"]
    tab = _woe_table(labels, y, w, order)
    lo = [-np.inf] + edges
    hi = edges + [np.inf]
    rng_lab = {str(i): f"[{lo[i]:.4g}, {hi[i]:.4g})" for i in range(len(edges) + 1)}
    tab["range"] = tab["bin"].map(rng_lab).fillna("Missing")
    return Binning(feature, "numeric", edges=edges, table=tab, iv=float(tab["iv_part"].sum()))


def bin_categorical(x, y, w, feature, min_share=0.02) -> Binning:
    share = w.groupby(x.astype(str)).sum() / w.sum()
    groups = {v: (v if share[v] >= min_share else "Other") for v in share.index}
    labels = x.astype(str).map(groups).where(x.notna(), "Missing")
    tab = _woe_table(labels, y, w)
    tab = tab.sort_values("bad_rate").reset_index(drop=True)
    tab["range"] = tab["bin"]
    return Binning(feature, "categorical", groups=groups, table=tab,
                   iv=float(tab["iv_part"].sum()))


def fit_binnings(df, target="default_12m", weight="weight") -> dict[str, Binning]:
    y, w = df[target].astype(float), df[weight].astype(float)
    out = {}
    for f in NUMERIC:
        out[f] = bin_numeric(df[f].astype(float), y, w, f,
                             monotonic=f not in NON_MONOTONIC)
    for f in CATEGORICAL:
        out[f] = bin_categorical(df[f], y, w, f)
    return out


def woe_frame(df, binnings, features) -> pd.DataFrame:
    return pd.DataFrame({f: binnings[f].woe(df[f]) for f in features}, index=df.index)


def select_features(df, binnings, min_iv=0.02, max_corr=0.7, target="default_12m",
                    weight="weight"):
    """IV filter -> correlation filter -> logistic fit, dropping wrong signs and
    insignificant terms until every coefficient is negative and p < 0.01."""
    ivs = pd.Series({f: b.iv for f, b in binnings.items()}).sort_values(ascending=False)
    cand = [f for f in ivs.index if ivs[f] >= min_iv]
    X = woe_frame(df, binnings, cand)
    kept = []
    for f in cand:                      # greedy: keep the higher-IV of correlated pairs
        if all(abs(np.corrcoef(X[f], X[k])[0, 1]) < max_corr for k in kept):
            kept.append(f)
    log = []
    while True:
        res = fit_logit(df, binnings, kept, target, weight)
        coefs = res.params.drop("const")
        pvals = res.pvalues.drop("const")
        wrong = coefs[coefs > 0]
        weak = pvals[pvals > 0.01]
        if len(wrong):
            f = wrong.idxmax(); log.append((f, "wrong sign")); kept.remove(f); continue
        if len(weak):
            f = weak.idxmax(); log.append((f, f"p = {weak.max():.3f}")); kept.remove(f); continue
        break
    return kept, res, ivs, log


def fit_logit(df, binnings, features, target="default_12m", weight="weight"):
    X = sm.add_constant(woe_frame(df, binnings, features))
    # freq_weights reproduces the full sample when non-defaults were subsampled.
    model = sm.GLM(df[target].astype(float), X, family=sm.families.Binomial(),
                   freq_weights=df[weight].astype(float))
    return model.fit()


def predict_logit(res, df, binnings, features) -> np.ndarray:
    X = sm.add_constant(woe_frame(df, binnings, features), has_constant="add")
    return np.asarray(res.predict(X))


def points_table(res, binnings, features, pdo=20, base_score=600, base_odds=50):
    """Scorecard points per bin. Score = offset + factor * ln(odds of good):
    +20 points (PDO) doubles the odds of being good; 600 points = 50:1 odds."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    b0, k = res.params["const"], len(features)
    rows = []
    for f in features:
        t = binnings[f].table
        bj = res.params[f]
        pts = -(bj * t["woe"] + b0 / k) * factor + offset / k
        rows.append(pd.DataFrame({"feature": f, "bin": t["range"], "share": t["share"],
                                  "bad_rate": t["bad_rate"], "woe": t["woe"],
                                  "points": pts.round().astype(int)}))
    return pd.concat(rows, ignore_index=True), factor, offset


def score_from_pd(pd_hat, factor, offset):
    pd_hat = np.clip(pd_hat, 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - pd_hat) / pd_hat)
