"""Load Freddie Mac SFLLD sample files and FRED macro CSVs.

Real and simulated data go through exactly the same functions, so switching to
the real files only changes the input folder.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (FORBEARANCE_CODES, MACRO_SERIES, MISSING_CODES, ORIG_COLS,
                     ORIG_USE, PERF_COLS, PERF_USE)


def yyyymm_to_index(s) -> np.ndarray:
    """YYYYMM -> months since year 0, so dates can be subtracted."""
    s = np.asarray(s, dtype="int64")
    return (s // 100) * 12 + (s % 100) - 1


def index_to_yyyymm(i) -> np.ndarray:
    i = np.asarray(i, dtype="int64")
    return (i // 12) * 100 + (i % 12) + 1


def _find(raw_dir: Path, kind: str, year: int) -> Path | None:
    """Find sample_orig_YYYY.txt / sample_perf_YYYY.txt (older releases used
    sample_svcg_YYYY.txt), unzipping sample_YYYY.zip first if needed."""
    pats = {"orig": [f"sample_orig_{year}.txt"],
            "perf": [f"sample_perf_{year}.txt", f"sample_svcg_{year}.txt"]}[kind]
    for p in pats:
        hits = list(raw_dir.rglob(p))
        if hits:
            return hits[0]
    # Not found as text: unzip any zip for that year (Freddie Mac's sample_YYYY.zip,
    # or a file zipped by hand such as sample_perf_YYYY.zip), then look again.
    for z in sorted(raw_dir.glob(f"*{year}*.zip")):
        target = raw_dir / f"unzipped_{z.stem}"
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
            return _find(raw_dir, kind, year)
    return None


TEXT_COLS = {"loan_id", "delinquency_status", "zero_balance_code", "first_time_homebuyer",
             "borrower_assistance", "disaster_delinquency",
             "occupancy", "channel", "property_state", "property_type", "loan_purpose"}


def _read_pipe(path: Path, names: list[str], use: list[str]) -> pd.DataFrame:
    # Count the columns actually present: releases differ (new fields get added).
    with open(path, "r") as f:
        n = f.readline().count("|") + 1
    cols = names[:n] + [f"extra_{i}" for i in range(max(0, n - len(names)))]
    use = [c for c in use if c in cols]
    # pyarrow's reader directly: pandas' pyarrow engine mangles names + usecols.
    import pyarrow as pa
    from pyarrow import csv as pacsv
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=cols),
        parse_options=pacsv.ParseOptions(delimiter="|"),
        convert_options=pacsv.ConvertOptions(
            # Numbers are parsed as numbers straight away (text would need several
            # times the memory on ~40 million rows); codes stay text ("RA", "01").
            include_columns=use,
            column_types={c: (pa.string() if c in TEXT_COLS else
                              pa.int64() if c == "period" else pa.float32()) for c in use},
            strings_can_be_null=True),
    )
    return table.to_pandas()


def load_origination(raw_dir: str | Path, years, sample_frac: float = 1.0,
                     seed: int = 0) -> pd.DataFrame:
    """sample_frac < 1 keeps a random share of loans (whole histories), to fit
    the real data in a laptop's memory."""
    frames = []
    for y in years:
        p = _find(Path(raw_dir), "orig", y)
        if p is None:
            print(f"  [skip] no origination file for {y}")
            continue
        df = _read_pipe(p, ORIG_COLS, ORIG_USE)
        if sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=seed + y)
        df["vintage"] = y
        frames.append(df)
    orig = pd.concat(frames, ignore_index=True)
    num = ["credit_score", "first_payment_date", "mi_pct", "orig_cltv", "orig_dti",
           "orig_upb", "orig_ltv", "orig_rate", "orig_term", "num_borrowers"]
    for c in num:
        orig[c] = pd.to_numeric(orig[c], errors="coerce")
    for c, code in MISSING_CODES.items():
        if c in orig:
            orig.loc[orig[c] == code, c] = np.nan
    return orig


def load_performance(raw_dir: str | Path, years, loan_ids=None) -> pd.DataFrame:
    """loan_ids: keep only these loans (pass orig.loan_id after sampling)."""
    keep = None if loan_ids is None else set(loan_ids)
    frames = []
    for y in years:
        p = _find(Path(raw_dir), "perf", y)
        if p is None:
            print(f"  [skip] no performance file for {y}")
            continue
        df = _read_pipe(p, PERF_COLS, PERF_USE)
        if keep is not None:
            df = df[df["loan_id"].isin(keep)]
        frames.append(df)
    perf = pd.concat(frames, ignore_index=True)
    perf["period"] = perf["period"].astype("int64")
    perf["t"] = yyyymm_to_index(perf["period"])
    # Delinquency: "00".."99" numeric, "RA" = REO acquisition (treated as default),
    # "XX" / blank = unknown.
    dq = perf["delinquency_status"].str.strip()
    perf["dpd"] = pd.to_numeric(dq, errors="coerce")
    perf.loc[dq == "RA", "dpd"] = 99
    perf["dpd"] = perf["dpd"].astype("float32")
    for c in ["current_upb", "loan_age", "remaining_months", "current_rate",
              "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries",
              "total_expenses", "actual_loss", "eltv", "zero_balance_removal_upb",
              "delinquent_accrued_interest"]:
        perf[c] = pd.to_numeric(perf[c], errors="coerce").astype("float32")
    perf.loc[perf["eltv"] >= 999, "eltv"] = np.nan          # 999 = not available
    fb = pd.Series(False, index=perf.index)
    if "borrower_assistance" in perf:
        fb |= perf["borrower_assistance"].isin(FORBEARANCE_CODES)
        perf = perf.drop(columns="borrower_assistance")
    if "disaster_delinquency" in perf:
        fb |= perf["disaster_delinquency"].eq("Y")
        perf = perf.drop(columns="disaster_delinquency")
    perf["forbearance"] = fb.to_numpy()
    perf["zero_balance_code"] = perf["zero_balance_code"].fillna("").str.strip()
    perf = perf.drop(columns=["delinquency_status"])
    return perf.sort_values(["loan_id", "t"], kind="stable").reset_index(drop=True)


def load_macro(macro_dir: str | Path) -> pd.DataFrame:
    """Monthly macro table from FRED CSVs (UNRATE.csv, CSUSHPISA.csv,
    MORTGAGE30US.csv). Weekly series are averaged to months."""
    out = None
    for sid, name in MACRO_SERIES.items():
        hits = [p for p in Path(macro_dir).glob("*.csv") if re.search(sid, p.name, re.I)]
        if not hits:
            raise FileNotFoundError(f"{sid}.csv not found in {macro_dir}")
        df = pd.read_csv(hits[0])
        date_col = df.columns[0]  # FRED uses "observation_date" (older: "DATE")
        df[date_col] = pd.to_datetime(df[date_col])
        val = pd.to_numeric(df[df.columns[1]], errors="coerce")
        s = pd.Series(val.to_numpy(), index=df[date_col]).resample("MS").mean()
        s.name = name
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out = out.ffill()
    out["t"] = out.index.year * 12 + out.index.month - 1
    out["unemp_chg_12m"] = out["unemployment"].diff(12)
    out["hpi_growth_12m"] = out["hpi"].pct_change(12, fill_method=None)
    return out.reset_index(drop=True)
