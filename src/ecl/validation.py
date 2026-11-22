"""Model validation tests used for bank PD models.

Three questions, three families of tests:
- Discrimination: does the model rank bad loans above good ones?   (Gini, KS)
- Calibration:    are the predicted PDs the right size?             (binomial,
                  Jeffreys and Hosmer-Lemeshow tests by rating grade)
- Stability:      is today's population like the development one?    (PSI, CSI)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini(y, score, w=None) -> float:
    """Gini = 2*AUC - 1. Pass a *risk* score (higher = riskier), e.g. the PD."""
    return 2 * roc_auc_score(y, score, sample_weight=w) - 1


def ks(y, score, w=None) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative score
    distributions of defaulters and non-defaulters."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    y = np.asarray(y)
    o = np.argsort(score)
    yb, wb = y[o], w[o]
    cb = np.cumsum(wb * yb) / np.sum(wb * yb)
    cg = np.cumsum(wb * (1 - yb)) / np.sum(wb * (1 - yb))
    return float(np.max(np.abs(cb - cg)))


def make_grades(pd_dev, n_grades=10) -> np.ndarray:
    """Rating-grade cut points on PD, fixed on development data (deciles)."""
    return np.unique(np.quantile(pd_dev, np.linspace(0, 1, n_grades + 1)[1:-1]))


def assign_grades(pd_hat, cuts) -> np.ndarray:
    return np.searchsorted(cuts, pd_hat, side="right") + 1


def calibration_table(y, pd_hat, grades, w=None) -> pd.DataFrame:
    """Per grade: obligors, defaults, observed default rate, mean predicted PD,
    and p-values for 'the PD is too low' (one-sided)."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    d = pd.DataFrame({"grade": grades, "y": np.asarray(y) * w, "pd": np.asarray(pd_hat) * w,
                      "n": w})
    t = d.groupby("grade")[["n", "y", "pd"]].sum()
    t["observed_dr"] = t["y"] / t["n"]
    t["mean_pd"] = t["pd"] / t["n"]
    n, k, p = t["n"].round().astype(int), t["y"].round().astype(int), t["mean_pd"]
    # Binomial: P(D >= observed | PD is right). Small = PD underestimates risk.
    t["binomial_p"] = stats.binom.sf(k - 1, n, p)
    # Jeffreys (ECB TRIM): posterior Beta(D + 1/2, N - D + 1/2); p = its CDF at PD.
    t["jeffreys_p"] = stats.beta.cdf(p, k + 0.5, n - k + 0.5)
    return t.rename(columns={"y": "defaults"}).drop(columns="pd").reset_index()


def hosmer_lemeshow(table: pd.DataFrame) -> tuple[float, float]:
    """Chi-square over grades of (observed - expected)^2 / (expected (1 - PD))."""
    e = table["n"] * table["mean_pd"]
    chi = (((table["defaults"] - e) ** 2) / (e * (1 - table["mean_pd"]))).sum()
    dof = max(len(table) - 2, 1)
    return float(chi), float(stats.chi2.sf(chi, dof))


def psi(expected_share, actual_share) -> float:
    """Population Stability Index. < 0.10 stable, 0.10-0.25 watch, > 0.25 shift."""
    e = np.clip(np.asarray(expected_share, float), 1e-6, None)
    a = np.clip(np.asarray(actual_share, float), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def shares(labels, w=None, categories=None) -> pd.Series:
    w = pd.Series(np.ones(len(labels)) if w is None else np.asarray(w, float))
    s = w.groupby(np.asarray(labels)).sum()
    s = s / s.sum()
    return s.reindex(categories, fill_value=0) if categories is not None else s
