"""IFRS 9 ECL: lifetime PD hazards, LGD, staging, scenario-weighted ECL, stress.

Run after scripts/run_pd.py (it reads outputs/pd_model.pkl).

    python scripts/run_ecl.py --data data/real --sample-frac 0.3 --report 201712
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from ecl import engine, lgd as lgd_mod, lifetime as lt, plots  # noqa: E402
from ecl import scorecard as sc  # noqa: E402
from ecl.config import PIT_MACRO, VINTAGES  # noqa: E402
from ecl.io import load_macro, load_origination, load_performance, yyyymm_to_index  # noqa: E402
from ecl.panel import add_features, build_pd_dataset, loan_events, rolling_max_dpd  # noqa: E402

MACRO = PIT_MACRO                      # point-in-time layer drivers


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def pit_pd(model, ttc_pd, macro_df):
    X = sm.add_constant(pd.DataFrame({"score_logodds": logit(ttc_pd),
                                      **{m: np.asarray(macro_df[m]) for m in MACRO}}),
                        has_constant="add")
    return np.asarray(model["pit"].predict(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-rows", type=int, default=1_000_000,
                    help="cap on PD development rows: all defaults kept, non-defaults sampled and weighted")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="share of loans to use; 0.5 helps the real data fit in laptop memory")
    ap.add_argument("--report", type=int, default=201712,
                    help="reporting date YYYYMM; models use only data up to it")
    args = ap.parse_args()
    data, out = ROOT / args.data, ROOT / args.out
    figs = out / "figures"
    t0 = time.time()
    model = pickle.load(open(out / "pd_model.pkl", "rb"))
    bins, feats = model["binnings"], model["features"]

    print("Loading ...", flush=True)
    orig = load_origination(data / "raw", VINTAGES, sample_frac=args.sample_frac)
    perf = load_performance(data / "raw", VINTAGES, loan_ids=orig.loan_id)
    macro = load_macro(data / "macro")

    # ------------------------------------- everything needed from `perf`
    # Taken first, then `perf` is freed: on the real data it is ~13 million rows.
    tR = int(yyyymm_to_index(args.report))
    snaps = build_pd_dataset(orig, perf, macro, max_rows=args.max_rows)  # adds max_dpd_12m
    events = loan_events(perf)
    lgd_d = lgd_mod.lgd_dataset(orig, perf[perf.t <= tR], macro)
    R = args.report
    port = perf[(perf.period == R) & (perf.zero_balance_code == "")].copy()
    t_last = int(perf.t.max())
    del perf
    import gc; gc.collect()

    # ---------------------------------------------------------- lifetime PD
    print("Fitting lifetime hazards ...", flush=True)
    snaps["score_lo"] = logit(sc.predict_logit(model["scorecard"], snaps, bins, feats))
    long = lt.hazard_dataset(snaps, events, macro, cutoff_t=tR)
    haz = lt.fit_hazards(long)
    print(f"  hazard rows {len(long):,}: defaults {int(long.default.sum()):,}, "
          f"prepayments {int(long.prepay.sum()):,}", flush=True)

    # ------------------------------------------------------------------ LGD
    lgd_model = lgd_mod.fit_lgd(lgd_d)
    lgd_d["pred"] = lgd_mod.predict_lgd(lgd_model, lgd_d.ltv_liq, lgd_d.has_mi, lgd_d.hpi_growth_12m)
    lgd_d["ltv_band"] = pd.cut(lgd_d.ltv_liq, [0, 60, 80, 100, 120, 1000],
                               labels=["<60", "60-80", "80-100", "100-120", ">120"])
    lgd_d["year"] = lgd_d.period // 100
    lgd_by_year = lgd_d.groupby("year").agg(n=("lgd", "size"), observed_lgd=("lgd", "mean"),
                                            predicted_lgd=("pred", "mean"))
    lgd_check = lgd_d.groupby("ltv_band", observed=True).agg(
        n=("lgd", "size"), observed_lgd=("lgd", "mean"), predicted_lgd=("pred", "mean"))

    # ------------------------------------------------------ reporting date
    port = add_features(port, orig, macro)
    port["dpd"] = port["dpd"].fillna(0)
    # Property value today: from Freddie Mac's ELTV where reported, else marked to
    # market with the national index; original CLTV stands in for a missing LTV.
    value_nat = port.orig_upb / (port.orig_ltv.fillna(port.orig_cltv) / 100) * port.hpi / port.hpi_orig
    port["value_now"] = (port.current_upb / (port.eltv / 100)).where(port.eltv > 0, value_nat)
    port["value_now"] = port["value_now"].fillna(port.current_upb / 0.6)
    port["pd_ttc"] = sc.predict_logit(model["scorecard"], port, bins, feats)
    port["score_lo"] = logit(port.pd_ttc)
    port["pd12_now"] = pit_pd(model, port.pd_ttc, port)

    # PD at origination: the same models applied to what was known on day one.
    at_orig = port.copy()
    m_idx = macro.set_index("t")
    fp_t = yyyymm_to_index(at_orig.first_payment_date)
    om = m_idx.reindex(fp_t - 1)
    at_orig[["dpd_now", "max_dpd_12m", "paid_down_pct", "loan_age"]] = 0
    at_orig["mtm_ltv"] = at_orig.orig_ltv
    at_orig["rate_incentive"] = at_orig.orig_rate - om.mortgage_rate.to_numpy()
    for c in MACRO:
        at_orig[c] = om[c].to_numpy()
    port["pd12_orig"] = pit_pd(model, sc.predict_logit(model["scorecard"], at_orig, bins, feats), at_orig)
    port["stage"] = engine.assign_stage(port)

    # ------------------------------------------------------------ scenarios
    start = {"unemployment": m_idx.at[tR, "unemployment"],
             "unemp_q-1": m_idx.at[tR - 3, "unemployment"],
             "unemp_q-2": m_idx.at[tR - 6, "unemployment"],
             "unemp_q-3": m_idx.at[tR - 9, "unemployment"],
             "hpi_growth_12m": m_idx.at[tR, "hpi_growth_12m"],
             "mortgage_rate": m_idx.at[tR, "mortgage_rate"]}
    print("Running scenarios ...", flush=True)
    res, paths = {}, {}
    for s in engine.SCENARIOS:
        paths[s.name] = engine.scenario_path(s, start)
        res[s.name] = engine.run_scenario(port, haz, lgd_model, paths[s.name])
        port[f"ecl_{s.name}"] = engine.ecl_by_stage(port, res[s.name], port.stage)
    weights = {s.name: s.weight for s in engine.SCENARIOS}
    port["ecl"] = sum(port[f"ecl_{k}"] * w for k, w in weights.items())
    # pandas sums skip missing values silently; an ECL must never do that.
    assert port["ecl"].notna().all(), f"{port['ecl'].isna().sum()} loans have no ECL"
    avg_path = engine.weighted_average_scenario(start)
    ecl_avg_path = engine.ecl_by_stage(port, engine.run_scenario(port, haz, lgd_model, avg_path), port.stage).sum()

    # ---------------------------------------------- backtest: realized macro
    # Run the engine on the economy that actually followed the reporting date,
    # then compare predicted and observed cumulative default rates of this book.
    # Model error and scenario error separate cleanly this way.
    H_bt = int(min(20, (t_last - tR) // 3))
    mm = m_idx.reindex([tR + 3 * h for h in range(1, H_bt + 1)])
    realized = pd.DataFrame({
        "h": np.arange(1, H_bt + 1), "unemployment": mm.unemployment.to_numpy(),
        "unemp_chg_12m": mm.unemp_chg_12m.to_numpy(),
        "hpi_cum": np.log(mm.hpi.to_numpy() / m_idx.at[tR, "hpi"]),
        "hpi_growth_12m": mm.hpi_growth_12m.to_numpy(), "mortgage_rate": mm.mortgage_rate.to_numpy()})
    res_real = engine.run_scenario(port, haz, lgd_model, realized, H=H_bt)
    live = (port.stage < 3).to_numpy()
    fd = port.loan_id.map(events.first_default_t).to_numpy()
    # Only defaults AFTER the reporting date count (a cured loan's first default
    # can lie in the past).
    new_def = fd[live] > tR
    obs_cum = np.array([np.mean(new_def & (fd[live] <= tR + 3 * h)) for h in range(1, H_bt + 1)])
    pred_real = res_real["cum_pd"][live].mean(0)[:H_bt]
    pred_base = res["Base"]["cum_pd"][live].mean(0)[:H_bt]
    backtest = pd.DataFrame({"quarter": np.arange(1, H_bt + 1), "observed_cum_dr": obs_cum,
                             "predicted_realized_macro": pred_real, "predicted_base": pred_base})

    # --------------------------------------------------------------- tables
    by_stage = port.groupby("stage").agg(loans=("loan_id", "size"), exposure=("current_upb", "sum"),
                                         ecl=("ecl", "sum"))
    by_stage.loc["Total"] = by_stage.sum()
    by_stage["coverage_pct"] = 100 * by_stage.ecl / by_stage.exposure
    by_stage["exposure_share_pct"] = 100 * by_stage.exposure / by_stage.loc["Total", "exposure"]
    scen = pd.DataFrame({k: {"weight": weights[k], "ecl": port[f"ecl_{k}"].sum()} for k in weights}).T
    scen["coverage_pct"] = 100 * scen.ecl / port.current_upb.sum()
    scen.loc["Probability-weighted"] = [1.0, port.ecl.sum(), 100 * port.ecl.sum() / port.current_upb.sum()]
    scen.loc["Weighted-average path (one run)"] = [np.nan, ecl_avg_path, 100 * ecl_avg_path / port.current_upb.sum()]

    sens = []
    for rel in (2.0, 3.0, 4.0):
        for abs_ in (0.001, 0.003, 0.005):
            st = engine.assign_stage(port, rel=rel, abs_=abs_)
            e = sum(engine.ecl_by_stage(port, res[k], st) * w for k, w in weights.items()).sum()
            sens.append({"relative_threshold": rel, "absolute_threshold_pct": 100 * abs_,
                         "stage2_share_pct": 100 * (st == 2).mean(),
                         "ecl": e, "coverage_pct": 100 * e / port.current_upb.sum()})
    sens = pd.DataFrame(sens)

    # Stress: everyone in the severe scenario, and the stage migration it causes.
    sev = paths["Severe"]
    stress_pd12 = res["Severe"]["cum_pd"][:, 3]
    stress_stage = np.where((port.stage == 1) & (stress_pd12 >= 3 * port.pd12_orig)
                            & (stress_pd12 - port.pd12_orig >= 0.005), 2, port.stage)
    stress_ecl = engine.ecl_by_stage(port, res["Severe"], stress_stage).sum()

    backtest.to_csv(out / "lifetime_pd_backtest.csv", index=False)
    by_stage.to_csv(out / "ecl_by_stage.csv")
    scen.to_csv(out / "ecl_by_scenario.csv")
    sens.to_csv(out / "staging_sensitivity.csv", index=False)
    lgd_check.to_csv(out / "lgd_by_ltv_band.csv")
    lgd_by_year.to_csv(out / "lgd_by_year.csv")
    for kind in ("default", "prepay"):
        m_ = haz[kind]
        pd.DataFrame({"coef": m_.params, "std_err": m_.bse, "p_value": m_.pvalues}) \
            .to_csv(out / f"hazard_{kind}_coefficients.csv")
    pd.DataFrame({"coef": lgd_model.params, "std_err": lgd_model.bse}).to_csv(out / "lgd_coefficients.csv")
    summary = {
        "reporting_date": R, "loans": int(len(port)), "exposure": float(port.current_upb.sum()),
        "ecl_weighted": float(port.ecl.sum()),
        "coverage_pct": float(100 * port.ecl.sum() / port.current_upb.sum()),
        "ecl_weighted_average_path": float(ecl_avg_path),
        "nonlinearity_uplift_pct": float(100 * (port.ecl.sum() / ecl_avg_path - 1)),
        "stress_ecl": float(stress_ecl),
        "stress_uplift_pct": float(100 * (stress_ecl / port.ecl.sum() - 1)),
        "stress_stage2_share_pct": float(100 * (stress_stage == 2).mean()),
        "lgd_observed_mean": float(lgd_d.lgd.mean()), "lgd_n": int(len(lgd_d)),
        "backtest_quarters": H_bt,
        "backtest_observed_cum_dr": float(obs_cum[-1]),
        "backtest_predicted_realized_macro": float(pred_real[-1]),
        "backtest_predicted_base": float(pred_base[-1]),
        "start_macro": start,
    }
    (out / "ecl_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # --------------------------------------------------------------- figures
    colors = {"Upside": plots.AQUA, "Base": plots.BLUE, "Downside": plots.ORANGE, "Severe": plots.INK}
    s1 = (port.stage == 1).to_numpy()
    fig, ax = plots.new((8.5, 4.2))
    yrs = np.arange(1, engine.H_MAX + 1) / 4
    for k in weights:
        ax.plot(yrs, 100 * res[k]["cum_pd"][s1].mean(0), color=colors[k], lw=2,
                ls="-" if k != "Severe" else (0, (4, 2)), label=f"{k} ({weights[k]:.0%})")
    ax.set_xlabel("years after reporting date", color=plots.INK_2)
    ax.set_ylabel("cumulative PD, % (Stage 1 average)", color=plots.INK_2)
    plots.title(ax, "Lifetime PD depends on the economic scenario")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="upper left")
    plots.save(fig, figs / "lifetime_pd_by_scenario.png")

    fig, ax = plots.new((8.5, 3.8))
    names = list(weights) + ["Probability-weighted"]
    vals = [scen.loc[n, "ecl"] / 1e6 for n in names]
    cols = [colors.get(n, plots.YELLOW) for n in names]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f"  {v:,.2f}m", va="center", color=plots.INK_2, fontsize=9)
    ax.set_xlabel("ECL, $ millions", color=plots.INK_2)
    plots.title(ax, "ECL by scenario and the probability-weighted result")
    plots.save(fig, figs / "ecl_by_scenario.png")

    fig, ax = plots.new((8.5, 4.2))
    yrs_bt = backtest.quarter / 4
    ax.plot(yrs_bt, 100 * backtest.observed_cum_dr, color=plots.INK, lw=2, label="Observed")
    ax.plot(yrs_bt, 100 * backtest.predicted_realized_macro, color=plots.BLUE, lw=2,
            label="Predicted, with the economy that actually happened")
    ax.plot(yrs_bt, 100 * backtest.predicted_base, color=plots.ORANGE, lw=2, ls=(0, (4, 2)),
            label="Predicted, base scenario at the reporting date")
    ax.set_xlabel("years after reporting date", color=plots.INK_2)
    ax.set_ylabel("cumulative default rate, %", color=plots.INK_2)
    plots.title(ax, f"Lifetime PD backtest, book as at {str(R)[:4]}-{str(R)[4:]}")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="upper left")
    plots.save(fig, figs / "lifetime_pd_backtest.png")

    print(backtest.round(4).iloc[[3, 7, 11, 15, -1]].to_string())
    print(by_stage.round(2).to_string())
    print(scen.round(4).to_string())
    print(sens.round(3).to_string())
    print(lgd_check.round(3).to_string())
    print(lgd_by_year.round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "start_macro"}, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
