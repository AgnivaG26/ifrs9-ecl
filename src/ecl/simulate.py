"""Stand-in data in exactly the Freddie Mac SFLLD sample-file format.

Used to build and test the pipeline before the real files arrive. Everything
downstream reads these files through the same loader as the real ones.

What the simulation captures (so the models have something real to find):
- Borrower risk: credit score, LTV, DTI, occupancy, purpose, channel.
- Negative equity: current LTV moves with *state* house prices; sand states
  (CA, FL, AZ, NV) boom and bust harder than the national index. The model only
  sees the national index, so this is realistic unobserved heterogeneity.
- Unemployment shocks raise delinquency; COVID-era forbearance damps them.
- Delinquency rolls month by month: current -> 30 -> 60 -> 90+ (default), with
  cures. Defaulted loans liquidate after 6+ months with losses driven by house
  prices, costs and mortgage insurance.
- Prepayment driven by the refinancing incentive (loan rate - market rate).

Macro paths are stylised approximations of 2003-2025 US history, NOT real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ORIG_COLS, PERF_COLS, VINTAGES
from .io import index_to_yyyymm

END_YYYYMM = 202506  # performance cutoff for the simulation
SAND_STATES = {"CA": 1.7, "FL": 1.6, "AZ": 1.8, "NV": 2.0}
STATES = ["CA", "FL", "TX", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "AZ", "NV",
          "WA", "VA", "NJ", "MA", "CO", "MN"]
STATE_W = np.array([14, 7, 7, 5, 5, 4, 4, 4, 4, 4, 3, 2, 4, 4, 4, 3, 3, 3], float)

_ANCHORS = {
    "unemployment": [(200301, 6.0), (200501, 5.3), (200703, 4.4), (200806, 5.6),
                     (200910, 10.0), (201106, 9.1), (201306, 7.5), (201506, 5.3),
                     (201706, 4.3), (201909, 3.5), (202002, 3.5), (202004, 14.8),
                     (202012, 6.7), (202112, 3.9), (202304, 3.4), (202408, 4.2),
                     (202512, 4.3)],
    "hpi": [(200301, 130), (200607, 184), (200905, 140), (201202, 134), (201601, 175),
            (201912, 213), (202206, 305), (202302, 295), (202512, 328)],
    "mortgage_rate": [(200301, 5.9), (200306, 5.2), (200406, 6.2), (200607, 6.8),
                      (200812, 5.3), (201012, 4.7), (201211, 3.35), (201309, 4.5),
                      (201607, 3.45), (201811, 4.85), (202012, 2.7), (202112, 3.1),
                      (202210, 6.9), (202310, 7.6), (202409, 6.1), (202512, 6.7)],
}
_FRED_ID = {"unemployment": "UNRATE", "hpi": "CSUSHPISA", "mortgage_rate": "MORTGAGE30US"}


def _ym_index(yyyymm: int) -> int:
    return (yyyymm // 100) * 12 + yyyymm % 100 - 1


def macro_paths(rng) -> pd.DataFrame:
    t = np.arange(_ym_index(200301), _ym_index(202512) + 1)
    out = {"t": t}
    for name, pts in _ANCHORS.items():
        xs = [_ym_index(a) for a, _ in pts]
        ys = [b for _, b in pts]
        base = np.interp(t, xs, ys)
        noise = {"unemployment": 0.08, "hpi": 0.004, "mortgage_rate": 0.05}[name]
        if name == "hpi":
            base = base * np.exp(np.cumsum(rng.normal(0, noise, len(t))) * 0.2)
        else:
            base = base + rng.normal(0, noise, len(t))
        out[name] = np.round(base, 3)
    return pd.DataFrame(out)


def write_fred_csvs(macro: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in
                            zip(macro.t // 12, macro.t % 12 + 1)])
    for name, sid in _FRED_ID.items():
        pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"),
                      sid: macro[name]}).to_csv(out_dir / f"{sid}.csv", index=False)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _origination(rng, vintages, n_per_vintage, macro) -> pd.DataFrame:
    n = n_per_vintage * len(vintages)
    v = np.repeat(vintages, n_per_vintage)
    boom = np.isin(v, [2005, 2006, 2007]).astype(float)   # looser underwriting
    fp_t = np.array([_ym_index(y * 100 + 1) for y in v]) + rng.integers(1, 13, n)
    fico = np.clip(rng.normal(752 - 22 * boom, 48 - 6 * boom), 560, 832).round()
    ltv = np.where(rng.random(n) < 0.45, 80.0,
                   rng.uniform(35, 97, n) + 3 * boom).clip(20, 97).round()
    cltv = (ltv + np.where(rng.random(n) < 0.18 * boom, rng.uniform(8, 18, n), 0)).clip(max=105).round()
    dti = np.clip(rng.normal(34 + 4 * boom, 9), 8, 65).round()
    upb = (np.exp(rng.normal(np.log(210_000), 0.45, n)) / 1000).round() * 1000
    term = np.where(rng.random(n) < 0.15, 180, 360)
    purpose = rng.choice(["P", "C", "N"], n, p=[0.45, 0.30, 0.25])
    occ = rng.choice(["P", "I", "S"], n, p=[0.88, 0.08, 0.04])
    channel = rng.choice(["R", "B", "C"], n, p=[0.5, 0.2, 0.3])
    fthb = np.where((purpose == "P") & (rng.random(n) < 0.3), "Y", "N")
    state = rng.choice(STATES, n, p=STATE_W / STATE_W.sum())
    prop = rng.choice(["SF", "PU", "CO"], n, p=[0.68, 0.22, 0.10])
    nborr = rng.choice([1, 2], n, p=[0.45, 0.55])
    mi = np.where(ltv > 80, np.where(ltv > 90, 30, 25), 0)
    mkt = np.interp(fp_t - 1, macro.t, macro.mortgage_rate)
    rate = (mkt + 0.25 + 0.4 * (740 - fico) / 100 + 0.3 * (ltv > 80)
            + rng.normal(0, 0.22, n)).round(3)
    df = pd.DataFrame({
        "credit_score": fico.astype(int), "first_payment_date": index_to_yyyymm(fp_t),
        "first_time_homebuyer": fthb, "maturity_date": index_to_yyyymm(fp_t + term - 1),
        "msa": "", "mi_pct": mi, "num_units": 1, "occupancy": occ,
        "orig_cltv": cltv.astype(int), "orig_dti": dti.astype(int),
        "orig_upb": upb.astype(int), "orig_ltv": ltv.astype(int), "orig_rate": rate,
        "channel": channel, "ppm_flag": "N", "amortization_type": "FRM",
        "property_state": state, "property_type": prop,
        "postal_code": [f"{rng.integers(10, 99)}000" for _ in range(n)],
        "loan_id": [f"F{y % 100:02d}S{i:07d}" for i, y in enumerate(v)],
        "loan_purpose": purpose, "orig_term": term, "num_borrowers": nborr,
        "seller_name": "Other sellers", "super_conforming": "", "pre_harp_loan_id": "",
        "program_indicator": "9", "harp_indicator": "", "valuation_method": "9",
        "io_indicator": "N", "vantage_score": "",
    })
    # A few "not available" codes, as in the real files.
    df.loc[rng.random(n) < 0.002, "credit_score"] = 9999
    df.loc[rng.random(n) < 0.01, "orig_dti"] = 999
    df["_vintage"], df["_fp_t"] = v, fp_t
    return df


def simulate(out_dir: str | Path, n_per_vintage: int = 3000, vintages=VINTAGES,
             seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    raw, mac = out_dir / "raw", out_dir / "macro"
    raw.mkdir(parents=True, exist_ok=True)
    macro = macro_paths(rng)
    write_fred_csvs(macro, mac)

    orig = _origination(rng, list(vintages), n_per_vintage, macro)
    n = len(orig)
    fp_t = orig["_fp_t"].to_numpy()
    fico = orig["credit_score"].where(orig["credit_score"] < 9999, 740).to_numpy(float)
    dti = orig["orig_dti"].where(orig["orig_dti"] < 999, 36).to_numpy(float)
    ltv0 = orig["orig_ltv"].to_numpy(float) / 100
    second_lien = (orig["orig_cltv"].to_numpy(float) - orig["orig_ltv"].to_numpy(float)) / 100
    upb0 = orig["orig_upb"].to_numpy(float)
    value0 = upb0 / ltv0
    rate = orig["orig_rate"].to_numpy(float)
    term = orig["orig_term"].to_numpy()
    r_m = rate / 1200
    pmt = upb0 * r_m / (1 - (1 + r_m) ** (-term))
    mi = orig["mi_pct"].to_numpy(float)
    inv = (orig["occupancy"] == "I").to_numpy(float)
    cash = (orig["loan_purpose"] == "C").to_numpy(float)
    broker = (orig["channel"] == "B").to_numpy(float)
    fthb = (orig["first_time_homebuyer"] == "Y").to_numpy(float)
    amp = orig["property_state"].map(SAND_STATES).fillna(1.0).to_numpy()
    amp = np.where(orig["property_state"] == "TX", 0.6, amp)
    frailty = rng.normal(0, 0.55, n)

    mt = macro.set_index("t")
    log_hpi = np.log(mt["hpi"])
    t0_hpi = log_hpi.reindex(fp_t - 1).to_numpy()

    upb = upb0.copy()
    status = np.zeros(n, int)          # 0 current, 1 = 30dpd, 2 = 60dpd, >=3 default
    in_default = np.zeros(n, int)      # months since 90+ first reached
    active = np.zeros(n, bool)
    done = np.zeros(n, bool)

    rows = []
    t_end = _ym_index(END_YYYYMM)
    for t in range(fp_t.min(), t_end + 1):
        start = (fp_t == t) & ~done
        active |= start
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue
        m = len(idx)
        age = t - fp_t[idx]
        u = mt.at[t, "unemployment"]
        du = u - mt.at[t - 12, "unemployment"]
        if 202003 <= index_to_yyyymm(t) <= 202112:   # forbearance damps the shock
            u, du = min(u, 6.0), min(du, 0.8)
        hpi_ratio = np.exp((log_hpi.at[t] - t0_hpi[idx]) * amp[idx])
        hg = log_hpi.at[t] - log_hpi.at[t - 12]
        cltv = upb[idx] / (value0[idx] * hpi_ratio) + second_lien[idx]
        eltv = upb[idx] / (value0[idx] * hpi_ratio)
        mkt = mt.at[t, "mortgage_rate"]
        st = status[idx]
        r = rng.random(m)

        # --- delinquency transitions
        x = (-6.6 + 0.85 * (740 - fico[idx]) / 50 + 2.2 * np.clip(cltv - 0.8, -0.5, 1.0)
             + 0.28 * (dti[idx] - 36) / 10 + 0.13 * (u - 5) + 0.28 * du
             + 0.35 * inv[idx] + 0.15 * cash[idx] + 0.2 * broker[idx] + 0.1 * fthb[idx]
             + 0.6 * np.minimum(age, 24) / 24 - 0.3 * np.maximum(age - 60, 0) / 60
             + frailty[idx])
        p01 = _sigmoid(x)
        neg = (cltv > 1.0).astype(float)
        p10 = 0.50 - 0.15 * neg - 0.05 * (du > 1)
        p12 = 0.30 + 0.10 * neg + 0.05 * (fico[idx] < 680)
        p20 = 0.15 - 0.05 * neg
        p23 = 0.45 + 0.15 * neg
        new = st.copy()
        c0, c1, c2, cd = st == 0, st == 1, st == 2, st >= 3
        new[c0 & (r < p01)] = 1
        new[c1 & (r < p10)] = 0
        new[c1 & (r >= p10) & (r < p10 + p12)] = 2
        new[c2 & (r < p20)] = 0
        new[c2 & (r >= p20) & (r < p20 + 0.08)] = 1
        new[c2 & (r >= p20 + 0.08) & (r < p20 + 0.08 + p23)] = 3
        cure = cd & (r < 0.02 + 0.06 * (cltv < 0.7))   # equity makes cures likelier
        new[cd & ~cure] = np.minimum(st[cd & ~cure] + 1, 99)
        new[cure] = 0
        in_def = in_default[idx]
        in_def = np.where(new >= 3, in_def + 1, 0)

        # --- terminations
        zb = np.full(m, "", dtype=object)
        r2 = rng.random(m)
        incentive = rate[idx] - mkt
        p_pre = _sigmoid(-5.0 + 1.3 * incentive + 0.5 * np.minimum(age, 36) / 36
                         + 2.0 * hg - 1.5 * neg)
        prepay = (new == 0) & (st == 0) & (r2 < p_pre)
        matured = age >= term[idx] - 1
        zb[prepay | matured] = "01"
        p_liq = 0.06 + 0.04 * neg
        liq = (in_def >= 6) & (r2 < p_liq) & ~prepay
        kind = rng.choice(["09", "03", "02"], m, p=[0.70, 0.25, 0.05])
        zb[liq] = kind[liq]

        # --- balances: amortize if current this month
        cur_upb = upb[idx]
        interest = cur_upb * r_m[idx]
        princ = np.clip(pmt[idx] - interest, 0, cur_upb)
        cur_upb = np.where(new == 0, cur_upb - princ, cur_upb)
        removal = np.where(zb != "", cur_upb, np.nan)

        # --- loss fields for credit events (new sign convention:
        #     recoveries/proceeds negative, expenses/losses positive)
        nsp = np.full(m, np.nan); exp_ = np.full(m, np.nan); mir = np.full(m, np.nan)
        nmr = np.full(m, np.nan); dai = np.full(m, np.nan); loss = np.full(m, np.nan)
        if liq.any():
            j = np.flatnonzero(liq)
            disc = np.select([kind[j] == "09", kind[j] == "03"],
                             [rng.uniform(0.12, 0.32, j.size), rng.uniform(0.05, 0.20, j.size)],
                             rng.uniform(0.0, 0.10, j.size))
            value = value0[idx][j] * hpi_ratio[j]
            b = cur_upb[j]
            months_dq = in_def[j] + 3
            dai[j] = b * r_m[idx][j] * months_dq
            exp_[j] = b * rng.uniform(0.04, 0.10, j.size) + months_dq * b * 0.0015
            # Any surplus over what is owed goes to the borrower, so the lender's
            # proceeds are capped at the debt plus costs (losses are rarely gains).
            proceeds = np.minimum(value * (1 - disc), b + dai[j] + exp_[j])
            gross = b + dai[j] + exp_[j] - proceeds
            mir[j] = -np.minimum(np.maximum(gross, 0), mi[idx][j] / 100 * (b + dai[j]))
            nmr[j] = -b * rng.uniform(0, 0.01, j.size)
            nsp[j] = -proceeds
            loss[j] = (b + nsp[j]) + dai[j] + exp_[j] + mir[j] + nmr[j]

        reported_upb = np.where(zb != "", 0.0, cur_upb)
        dq_str = np.char.zfill(new.astype(str), 2)
        rows.append(pd.DataFrame({
            "loan_id": orig["loan_id"].to_numpy()[idx],
            "period": index_to_yyyymm(t), "current_upb": reported_upb.round(2),
            "delinquency_status": dq_str, "loan_age": age,
            "remaining_months": term[idx] - age - 1,
            "zero_balance_code": zb,
            "zero_balance_date": np.where(zb != "", str(index_to_yyyymm(t)), ""),
            "current_rate": rate[idx], "mi_recoveries": mir.round(2),
            "net_sales_proceeds": nsp.round(2), "non_mi_recoveries": nmr.round(2),
            "total_expenses": exp_.round(2), "actual_loss": loss.round(2),
            "eltv": np.round(eltv * 100), "zero_balance_removal_upb": removal.round(2),
            "delinquent_accrued_interest": dai.round(2),
            "_vintage": orig["_vintage"].to_numpy()[idx],
        }))

        upb[idx] = cur_upb
        status[idx] = new
        in_default[idx] = in_def
        ended = idx[zb != ""]
        active[ended] = False
        done[ended] = True

    perf = pd.concat(rows, ignore_index=True)
    for c in PERF_COLS:
        if c not in perf:
            perf[c] = ""
    for y in vintages:
        o = orig[orig["_vintage"] == y][ORIG_COLS]
        o.to_csv(raw / f"sample_orig_{y}.txt", sep="|", header=False, index=False)
        p = perf[perf["_vintage"] == y][PERF_COLS]
        p.to_csv(raw / f"sample_perf_{y}.txt", sep="|", header=False, index=False,
                 float_format="%.2f")
    print(f"Simulated {n:,} loans, {len(perf):,} loan-months -> {raw}")


if __name__ == "__main__":
    import sys
    simulate(sys.argv[1] if len(sys.argv) > 1 else "data/sim")
