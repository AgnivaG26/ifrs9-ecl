"""PD development: dataset -> WoE scorecard -> point-in-time layer -> challenger
-> validation. Writes tables, figures and a model file used by the ECL stage.

    python scripts/run_pd.py --data data/sim      # stand-in data
    python scripts/run_pd.py --data data/real     # real Freddie Mac files
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from ecl import plots  # noqa: E402
from ecl import scorecard as sc  # noqa: E402
from ecl import validation as va  # noqa: E402
from ecl.config import DEV_END, OOT_START, PIT_MACRO, VINTAGES  # noqa: E402
from ecl.io import load_macro, load_origination, load_performance  # noqa: E402
from ecl.panel import build_pd_dataset  # noqa: E402

MACRO = PIT_MACRO                      # point-in-time layer drivers


def pit_design(df, ttc_pd):
    """Point-in-time layer: scorecard log-odds plus the macro cycle."""
    lo = np.log(np.clip(ttc_pd, 1e-9, 1) / np.clip(1 - ttc_pd, 1e-9, 1))
    return sm.add_constant(pd.DataFrame({"score_logodds": lo, **{m: df[m].to_numpy() for m in MACRO}},
                                        index=df.index), has_constant="add")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-rows", type=int, default=1_000_000,
                    help="cap on PD development rows: all defaults kept, non-defaults sampled and weighted")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="share of loans to use; 0.5 helps the real data fit in laptop memory")
    args = ap.parse_args()
    data, out = ROOT / args.data, ROOT / args.out
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading ...", flush=True)
    orig = load_origination(data / "raw", VINTAGES, sample_frac=args.sample_frac)
    perf = load_performance(data / "raw", VINTAGES, loan_ids=orig.loan_id)
    macro = load_macro(data / "macro")
    df = build_pd_dataset(orig, perf, macro, max_rows=args.max_rows)
    print(f"  {len(df):,} snapshot rows, {int(df.default_12m.sum()):,} defaults "
          f"({time.time() - t0:.0f}s)", flush=True)
    del perf                       # not needed from here on; frees most of the memory
    import gc; gc.collect()

    # --- splits: development (to 2015) split 70/30 by loan; out-of-time from 2016
    dev = df[df.period <= DEV_END]
    oot = df[df.period >= OOT_START].copy()
    loans = dev.loan_id.unique()
    rng = np.random.default_rng(7)
    hold_ids = set(rng.choice(loans, size=int(0.3 * len(loans)), replace=False))
    is_hold = dev.loan_id.isin(hold_ids)
    train, hold = dev[~is_hold].copy(), dev[is_hold].copy()

    # --- scorecard
    print("Binning and selecting features ...", flush=True)
    binnings = sc.fit_binnings(train)
    feats, res, ivs, drop_log = sc.select_features(train, binnings)
    points, factor, offset = sc.points_table(res, binnings, feats)
    for d in (train, hold, oot):
        d["pd_ttc"] = sc.predict_logit(res, d, binnings, feats)
        d["score"] = sc.score_from_pd(d["pd_ttc"], factor, offset)

    # --- point-in-time layer (IFRS 9 needs PDs that move with the economy)
    pit = sm.GLM(train.default_12m.astype(float), pit_design(train, train.pd_ttc),
                 family=sm.families.Binomial(), freq_weights=train.weight).fit()
    for d in (train, hold, oot):
        d["pd_pit"] = np.asarray(pit.predict(pit_design(d, d.pd_ttc)))

    # --- challenger: gradient boosting on raw features + macro
    print("Training challenger ...", flush=True)
    raw = sc.NUMERIC + sc.CATEGORICAL + ["unemployment", "unemp_chg_12m", "hpi_growth_12m"]
    def to_lgb(d):
        X = d[raw].copy()
        for c in sc.CATEGORICAL:
            X[c] = X[c].astype("category")
        return X
    Xtr = to_lgb(train)
    # Kept deliberately small: each loan appears in many snapshots, so a flexible
    # model can memorise loans instead of learning risk.
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=2000, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, verbose=-1,
                             random_state=7)
    gbm.fit(Xtr, train.default_12m, sample_weight=train.weight)
    cats = {c: Xtr[c].cat.categories for c in sc.CATEGORICAL}
    def to_lgb_aligned(d):
        X = to_lgb(d)
        for c in sc.CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(object), categories=cats[c])
        return X
    for d in (train, hold, oot):
        d["pd_gbm"] = gbm.predict_proba(to_lgb_aligned(d))[:, 1]

    # --- validation
    print("Validating ...", flush=True)
    metrics = {}
    for name, d in [("train", train), ("holdout", hold), ("out_of_time", oot)]:
        y, w = d.default_12m, d.weight
        metrics[name] = {
            "rows": int(len(d)), "defaults": int(d.default_12m.sum()),
            "observed_dr": float(np.average(y, weights=w)),
            "gini_scorecard": va.gini(y, d.pd_ttc, w), "ks_scorecard": va.ks(y, d.pd_ttc, w),
            "gini_pit": va.gini(y, d.pd_pit, w),
            "gini_challenger": va.gini(y, d.pd_gbm, w),
            "mean_pd_ttc": float(np.average(d.pd_ttc, weights=w)),
            "mean_pd_pit": float(np.average(d.pd_pit, weights=w)),
        }
    cuts = va.make_grades(train.pd_pit)
    calib = {}
    for name, d in [("holdout", hold), ("out_of_time", oot)]:
        t = va.calibration_table(d.default_12m, d.pd_pit, va.assign_grades(d.pd_pit, cuts), d.weight)
        t.to_csv(out / f"calibration_{name}.csv", index=False)
        chi, p = va.hosmer_lemeshow(t)
        calib[name] = t
        metrics[name].update({"hosmer_lemeshow_chi2": chi, "hosmer_lemeshow_p": p,
                              "grades_jeffreys_p_below_0.05": int((t.jeffreys_p < 0.05).sum())})
    # Stability: scorecard-score deciles and each feature's bins, development vs
    # out-of-time. Measured on the scorecard (no macro), not the point-in-time PD,
    # whose shifts mostly reflect the economy rather than the population.
    s_cuts = va.make_grades(train.pd_ttc)
    g_cats = list(range(1, len(s_cuts) + 2))
    metrics["psi_score"] = va.psi(va.shares(va.assign_grades(train.pd_ttc, s_cuts), train.weight, g_cats),
                                  va.shares(va.assign_grades(oot.pd_ttc, s_cuts), oot.weight, g_cats))
    csi = {}
    for f in feats:
        lab_tr, lab_oot = binnings[f].bin_labels(train[f]), binnings[f].bin_labels(oot[f])
        cats_f = sorted(set(lab_tr) | set(lab_oot))
        csi[f] = va.psi(va.shares(lab_tr, train.weight, cats_f), va.shares(lab_oot, oot.weight, cats_f))
    metrics["csi"] = csi

    # --- outputs
    points.to_csv(out / "scorecard_points.csv", index=False)
    ivs.rename("iv").to_frame().join(pd.Series({f: f in feats for f in ivs.index}, name="selected")) \
        .to_csv(out / "information_value.csv")
    coef = pd.DataFrame({"coef": res.params, "std_err": res.bse, "p_value": res.pvalues})
    coef.to_csv(out / "scorecard_coefficients.csv")
    pd.DataFrame({"coef": pit.params, "std_err": pit.bse, "p_value": pit.pvalues}) \
        .to_csv(out / "pit_layer_coefficients.csv")
    metrics.update({"features": feats, "dropped_in_selection": drop_log,
                    "pdo": 20, "base_score": 600, "base_odds": 50})
    (out / "pd_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    with open(out / "pd_model.pkl", "wb") as fh:
        pickle.dump({"binnings": binnings, "features": feats, "scorecard": res,
                     "pit": pit, "grade_cuts": cuts, "factor": factor, "offset": offset}, fh)

    # Figure 1: PIT vs TTC vs observed, by snapshot quarter.
    allq = pd.concat([train, hold, oot])
    allq["q"] = pd.to_datetime((allq.period // 100).astype(str) + "-" + (allq.period % 100).astype(str) + "-01")
    agg = allq.groupby("q").apply(lambda g: pd.Series({
        "observed": np.average(g.default_12m, weights=g.weight),
        "ttc": np.average(g.pd_ttc, weights=g.weight),
        "pit": np.average(g.pd_pit, weights=g.weight)}))
    fig, ax = plots.new((10, 4.2))
    ax.plot(agg.index, agg.observed * 100, color=plots.INK, lw=1.6, label="Observed 12-month default rate")
    ax.plot(agg.index, agg.pit * 100, color=plots.BLUE, lw=2, label="Point-in-time PD (with macro)")
    ax.plot(agg.index, agg.ttc * 100, color=plots.ORANGE, lw=2, ls=(0, (4, 2)), label="Scorecard PD (no macro)")
    ax.axvline(pd.Timestamp("2016-01-01"), color=plots.INK_2, lw=0.8, ls=":")
    ax.text(pd.Timestamp("2016-03-01"), ax.get_ylim()[1] * 0.92, "out-of-time", color=plots.INK_2, fontsize=9)
    ax.set_ylabel("% of performing loans", color=plots.INK_2)
    plots.title(ax, "Adding the economy makes PDs track the cycle")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="upper right")
    plots.save(fig, figs / "pd_vs_observed_by_quarter.png")

    # Figure 2: calibration by grade, out-of-time, with Jeffreys 90% intervals.
    t = calib["out_of_time"]
    from scipy import stats as st
    k, n = t.defaults.round(), t.n.round()
    lo = st.beta.ppf(0.05, k + 0.5, n - k + 0.5); hi = st.beta.ppf(0.95, k + 0.5, n - k + 0.5)
    fig, ax = plots.new((8, 4.2))
    # Jeffreys interval drawn as a line: with 0 defaults its lower end sits above 0.
    ax.vlines(t.grade, lo * 100, hi * 100, color=plots.INK_2, lw=1.2)
    ax.plot(t.grade, np.maximum(t.observed_dr, 1e-5) * 100, "o", color=plots.INK, ms=6,
            label="Observed (line = 90% Jeffreys interval)")
    ax.plot(t.grade, t.mean_pd * 100, "D", color=plots.BLUE, ms=7, label="Predicted point-in-time PD")
    ax.set_yscale("log"); ax.set_xticks(t.grade)
    ax.set_xlabel("rating grade (1 = safest)", color=plots.INK_2)
    ax.set_ylabel("12-month default rate, % (log)", color=plots.INK_2)
    plots.title(ax, "Out-of-time calibration by rating grade")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="upper left")
    plots.save(fig, figs / "calibration_by_grade_oot.png")

    # Figure 3: challenger explained with SHAP (mean |SHAP| per feature).
    sample = to_lgb_aligned(hold.sample(min(20000, len(hold)), random_state=1))
    sv = shap.TreeExplainer(gbm).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values().tail(12)
    fig, ax = plots.new((8, 4.6))
    ax.barh([c.replace("_", " ") for c in imp.index], imp.values, color=plots.BLUE, height=0.7)
    ax.set_xlabel("mean |SHAP value| (log-odds of default)", color=plots.INK_2)
    plots.title(ax, "What drives the challenger model")
    plots.save(fig, figs / "challenger_shap_importance.png")

    print(json.dumps({k: v for k, v in metrics.items() if k in ("train", "holdout", "out_of_time", "psi_score")},
                     indent=2, default=lambda x: round(float(x), 4)))
    print("features:", feats)
    print("dropped:", drop_log)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
