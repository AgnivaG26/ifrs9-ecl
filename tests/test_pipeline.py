"""Fast checks on a small simulation. Run with: pytest -q"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecl import engine, scorecard as sc, validation as va  # noqa: E402
from ecl.io import load_macro, load_origination, load_performance, yyyymm_to_index  # noqa: E402
from ecl.lifetime import hazard_dataset  # noqa: E402
from ecl.panel import build_pd_dataset, loan_events  # noqa: E402
from ecl.simulate import simulate  # noqa: E402

YEARS = [2005, 2006, 2007, 2008]


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    simulate(d, n_per_vintage=400, vintages=YEARS, seed=5)
    orig = load_origination(d / "raw", YEARS)
    perf = load_performance(d / "raw", YEARS)
    macro = load_macro(d / "macro")
    return d, orig, perf, macro


def test_loader_shapes_and_codes(sim):
    _, orig, perf, macro = sim
    assert len(orig) == 1600 and orig.loan_id.is_unique
    assert set(perf.loan_id) <= set(orig.loan_id)
    assert perf.groupby("loan_id").t.apply(lambda s: s.is_monotonic_increasing).all()
    assert {"unemployment", "hpi", "mortgage_rate", "unemp_chg_12m"} <= set(macro.columns)


def test_loader_accepts_older_svcg_name(sim, tmp_path):
    d, _, _, _ = sim
    shutil.copy(d / "raw" / "sample_orig_2005.txt", tmp_path / "sample_orig_2005.txt")
    shutil.copy(d / "raw" / "sample_perf_2005.txt", tmp_path / "sample_svcg_2005.txt")
    assert len(load_performance(tmp_path, [2005])) > 0


def test_default_flag_and_censoring(sim):
    _, orig, perf, macro = sim
    df = build_pd_dataset(orig, perf, macro)
    ev = loan_events(perf)
    row = df[df.default_12m == 1].iloc[0]
    fd = ev.loc[row.loan_id, "first_default_t"]
    assert row.t < fd <= row.t + 12
    # No snapshot is taken while a loan is already in default.
    assert (df.dpd_now < 3).all()


def test_amortization():
    b = engine.amortized_balance(np.array([100_000.0]), np.array([6.0]), np.array([360]), 0)
    assert b[0] == pytest.approx(100_000)
    end = engine.amortized_balance(np.array([100_000.0]), np.array([6.0]), np.array([360]), 360)
    assert end[0] == pytest.approx(0, abs=1e-6)


def test_scenarios():
    assert sum(s.weight for s in engine.SCENARIOS) == pytest.approx(1.0)
    start = {"unemployment": 4.0, "unemp_q-1": 4.0, "unemp_q-2": 4.0, "unemp_q-3": 4.0,
             "hpi_growth_12m": 0.04, "mortgage_rate": 4.0}
    p = engine.scenario_path(engine.SCENARIOS[2], start)
    assert p.unemp_chg_12m.iloc[4] == pytest.approx(p.unemployment.iloc[4] - p.unemployment.iloc[0])
    assert p.unemp_chg_12m.iloc[0] == pytest.approx(p.unemployment.iloc[0] - 4.0)


def test_woe_binning_is_monotonic_for_credit_score(sim):
    _, orig, perf, macro = sim
    df = build_pd_dataset(orig, perf, macro)
    b = sc.bin_numeric(df.credit_score.astype(float), df.default_12m.astype(float),
                       df.weight.astype(float), "credit_score")
    rates = b.table[b.table.bin != "Missing"].bad_rate.to_numpy()
    assert (np.diff(rates) <= 1e-12).all()      # higher score, lower default rate
    assert b.iv > 0


def test_calibration_tests_accept_a_correct_pd():
    rng = np.random.default_rng(0)
    pd_true = rng.uniform(0.005, 0.05, 200_000)
    y = rng.random(len(pd_true)) < pd_true
    g = va.assign_grades(pd_true, va.make_grades(pd_true))
    t = va.calibration_table(y, pd_true, g)
    assert (t.jeffreys_p > 0.001).all()
    assert va.hosmer_lemeshow(t)[1] > 0.001


def test_hazard_cutoff_has_no_look_ahead(sim):
    _, orig, perf, macro = sim
    df = build_pd_dataset(orig, perf, macro)
    df["score_lo"] = 0.0
    cut = int(yyyymm_to_index(201012))
    long = hazard_dataset(df, loan_events(perf), macro, cutoff_t=cut)
    assert (long.t0 + 3 * long.h <= cut + 3).all()
