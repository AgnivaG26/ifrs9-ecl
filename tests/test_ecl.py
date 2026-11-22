"""Unit tests on small hand-written inputs, each checked against a value worked
out by hand. No licensed data is needed. Run with: pytest -q"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecl import engine, lgd, scorecard as sc, validation as va  # noqa: E402
from ecl.config import PERF_COLS  # noqa: E402
from ecl.io import load_performance, yyyymm_to_index  # noqa: E402
from ecl.panel import default_flag, loan_events, rolling_max_dpd  # noqa: E402


# --------------------------------------------------------------------------- data
def _perf_line(**fields) -> str:
    """One pipe-delimited performance record; unnamed fields are left blank."""
    return "|".join(str(fields.get(c, "")) for c in PERF_COLS)


def test_loader_parses_freddie_mac_codes(tmp_path):
    # Older releases named the file sample_svcg_YYYY.txt; the loader accepts both.
    lines = [
        _perf_line(loan_id="L1", period=202001, current_upb=1000, delinquency_status="0", eltv=80),
        _perf_line(loan_id="L1", period=202002, current_upb=990, delinquency_status="RA", eltv=999),
        _perf_line(loan_id="L2", period=202001, current_upb=500, delinquency_status="3",
                   borrower_assistance="F", zero_balance_code=""),
    ]
    (tmp_path / "sample_svcg_2020.txt").write_text("\n".join(lines) + "\n")
    perf = load_performance(tmp_path, [2020])
    l1 = perf[perf.loan_id == "L1"]
    assert list(l1.dpd) == [0, 99]                 # "RA" (REO acquisition) = default
    assert l1.eltv.iloc[0] == 80 and np.isnan(l1.eltv.iloc[1])   # 999 = not available
    assert perf.loc[perf.loan_id == "L2", "forbearance"].item()  # assistance plan "F"
    assert list(perf.t) == list(yyyymm_to_index([202001, 202002, 202001]))


def test_month_index_round_trip():
    # Months since year 0, so that Dec 2019 -> Jan 2020 is one step.
    assert yyyymm_to_index(202001) - yyyymm_to_index(201912) == 1


# ------------------------------------------------------------- default definition
def test_default_flag():
    perf = pd.DataFrame({
        "dpd":               [0,     3,     3,     0,     0],
        "forbearance":       [False, False, True,  False, False],
        "zero_balance_code": ["",    "",    "",    "09",  "01"],
    })
    # current | 90+ DPD | 90+ DPD in forbearance | REO disposition | voluntary prepayment
    assert list(default_flag(perf)) == [False, True, False, True, False]


def test_loan_events():
    perf = pd.DataFrame({
        "loan_id": ["A", "A", "A", "B", "B"],
        "t":       [10, 11, 12, 10, 11],
        "dpd":     [0, 3, 4, 0, 0],
        "forbearance": False,
        "zero_balance_code": ["", "", "", "", "01"],
    })
    ev = loan_events(perf)
    assert ev.loc["A", "first_default_t"] == 11 and np.isnan(ev.loc["B", "first_default_t"])
    assert ev.loc["B", "term_t"] == 11 and ev.loc["B", "term_code"] == "01"
    assert ev.loc["A", "last_t"] == 12


def test_rolling_max_dpd_stays_within_each_loan():
    perf = pd.DataFrame({"loan_id": ["A", "A", "A", "A", "B", "B"],
                         "dpd":     [0, 2, 0, 0, 1, 0]})
    # Window of 2 months; loan B's first month must not see loan A's last month.
    assert list(rolling_max_dpd(perf, window=2)) == [0, 2, 2, 0, 1, 1]


# ------------------------------------------------------------------- scorecard
def test_woe_and_iv_match_hand_calculation():
    # Bin a: 2 bad, 8 good. Bin b: 8 bad, 2 good. Totals: 10 bad, 10 good.
    labels = ["a"] * 10 + ["b"] * 10
    y = np.array([1] * 2 + [0] * 8 + [1] * 8 + [0] * 2, float)
    t = sc._woe_table(pd.Series(labels), y, np.ones(20)).set_index("bin")
    # With +0.5 smoothing per bin: shares are (count + 0.5) / (10 + 0.5 * 2 bins).
    woe_a = math.log((8.5 / 11) / (2.5 / 11))
    assert t.loc["a", "woe"] == pytest.approx(woe_a)
    assert t.loc["b", "woe"] == pytest.approx(-woe_a)
    iv = 2 * ((8.5 - 2.5) / 11) * woe_a
    assert t["iv_part"].sum() == pytest.approx(iv)


def test_numeric_binning_forces_a_monotonic_default_rate():
    # Ten score values, ten loans each; defaults fall with score except one bump.
    scores = np.repeat(np.arange(600, 800, 20), 10).astype(float)
    bads_per_value = [6, 5, 6, 3, 3, 2, 2, 1, 0, 0]
    y = np.concatenate([[1] * b + [0] * (10 - b) for b in bads_per_value]).astype(float)
    b = sc.bin_numeric(pd.Series(scores), pd.Series(y), pd.Series(np.ones(100)), "credit_score")
    rates = b.table[b.table.bin != "Missing"].bad_rate.to_numpy()
    assert len(rates) >= 3
    assert (np.diff(rates) <= 0).all()             # higher score, lower default rate
    assert b.iv > 0


def test_points_scale():
    # PDO 20 and 600 points at odds 50:1: factor = 20 / ln 2, offset = 600 - factor ln 50.
    factor = 20 / math.log(2)
    offset = 600 - factor * math.log(50)
    pd_at_600 = 1 / 51                              # odds good:bad = 50:1
    assert sc.score_from_pd(pd_at_600, factor, offset) == pytest.approx(600)
    assert sc.score_from_pd(1 / 101, factor, offset) == pytest.approx(620, abs=0.3)  # odds doubled


# ------------------------------------------------------------------ validation
def test_gini_and_ks_on_a_hand_example():
    y = np.array([0, 1, 0, 1])
    score = np.array([1, 2, 3, 4])
    # Bad-good pairs ranked correctly: (2>1), (4>1), (4>3) = 3 of 4 -> AUC 0.75.
    assert va.gini(y, score) == pytest.approx(0.5)
    assert va.ks(y, score) == pytest.approx(0.5)
    assert va.gini(np.array([0, 0, 1, 1]), score) == pytest.approx(1.0)


def test_psi():
    assert va.psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # (0.25 - 0.5) ln(0.5) + (0.75 - 0.5) ln(1.5) = 0.25 ln 3
    assert va.psi([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.25 * math.log(3))


def test_calibration_tests():
    # Grade 1: 100 loans, PD 1%, 1 default (as expected).
    # Grade 2: 100 loans, PD 1%, 10 defaults (PD far too low).
    y = np.r_[np.repeat([1, 0], [1, 99]), np.repeat([1, 0], [10, 90])]
    pd_hat = np.full(200, 0.01)
    grades = np.repeat([1, 2], 100)
    t = va.calibration_table(y, pd_hat, grades).set_index("grade")
    assert t.loc[1, "binomial_p"] == pytest.approx(1 - 0.99 ** 100)   # P(D >= 1)
    assert t.loc[1, "jeffreys_p"] > 0.05
    assert t.loc[2, "binomial_p"] < 1e-5 and t.loc[2, "jeffreys_p"] < 1e-5


def test_hosmer_lemeshow_is_zero_when_defaults_equal_expected():
    t = pd.DataFrame({"n": [100, 200, 100], "mean_pd": [0.01, 0.05, 0.20],
                      "defaults": [1, 10, 20]})
    chi, p = va.hosmer_lemeshow(t)
    assert chi == pytest.approx(0.0) and p == pytest.approx(1.0)


# ---------------------------------------------------------------- IFRS 9 engine
def test_amortization():
    upb, rate, n = np.array([100_000.0]), np.array([6.0]), np.array([360])
    assert engine.amortized_balance(upb, rate, n, 0)[0] == pytest.approx(100_000)
    # 12 payments of 599.55 on 100,000 at 6% leave 98,771.99 (worked month by month).
    assert engine.amortized_balance(upb, rate, n, 12)[0] == pytest.approx(98_771.99, abs=0.01)
    assert engine.amortized_balance(upb, rate, n, 360)[0] == pytest.approx(0, abs=1e-6)


def test_staging_rules():
    port = pd.DataFrame({
        "dpd":         [0,    1,    0,     0,      3,     3],
        "forbearance": [False, False, False, False, False, True],
        "pd12_orig":   [0.01, 0.01, 0.005, 0.0005, 0.01,  0.01],
        "pd12_now":    [0.01, 0.01, 0.030, 0.0030, 0.50,  0.50],
    })
    # performing | 30+ DPD backstop | PD x6 and +2.5pp (SICR) | PD x6 but only +0.25pp
    # | 90+ DPD (default) | 90+ DPD inside forbearance (Stage 2, not default)
    assert list(engine.assign_stage(port)) == [1, 2, 2, 1, 3, 2]


def test_scenarios():
    assert sum(s.weight for s in engine.SCENARIOS) == pytest.approx(1.0)
    start = {"unemployment": 4.0, "unemp_q-1": 4.0, "unemp_q-2": 4.0, "unemp_q-3": 4.0,
             "hpi_growth_12m": 0.04, "mortgage_rate": 4.0}
    p = engine.scenario_path(engine.SCENARIOS[2], start)
    assert p.unemp_chg_12m.iloc[4] == pytest.approx(p.unemployment.iloc[4] - p.unemployment.iloc[0])
    assert p.unemp_chg_12m.iloc[0] == pytest.approx(p.unemployment.iloc[0] - 4.0)


def test_lgd_rises_with_ltv_and_stays_in_bounds():
    d = pd.DataFrame({"ltv_liq": [60, 80, 100, 120, 140, 70, 110],
                      "has_mi": [0, 0, 0, 0, 0, 1, 1],
                      "hpi_growth_12m": 0.0,
                      "lgd": [0.10, 0.20, 0.35, 0.50, 0.65, 0.05, 0.30]})
    m = lgd.fit_lgd(d)
    ltv = np.array([50, 90, 130, 200])
    pred = lgd.predict_lgd(m, ltv, np.zeros(4), np.zeros(4))
    assert (np.diff(pred) > 0).all() and ((pred > 0) & (pred < 1)).all()
