"""Freddie Mac Single-Family Loan-Level Dataset (SFLLD) schema and project settings.

Column order follows the SFLLD General User Guide (July 2026 release). The
files are pipe-delimited with no header row, so these names are attached on
load. Freddie Mac has added fields over time (VantageScore 4.0 is the newest
origination field), so the loader tolerates files with fewer or more columns.
"""

from __future__ import annotations

ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy", "orig_cltv", "orig_dti", "orig_upb",
    "orig_ltv", "orig_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_id", "loan_purpose",
    "orig_term", "num_borrowers", "seller_name", "super_conforming",
    "pre_harp_loan_id", "program_indicator", "harp_indicator", "valuation_method",
    "io_indicator", "vantage_score",
]

PERF_COLS = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age",
    "remaining_months", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_date", "current_rate",
    "non_interest_bearing_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "cum_modification_cost",
    "step_rate_indicator", "payment_deferral", "eltv", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "disaster_delinquency", "borrower_assistance",
    "current_mod_cost", "interest_bearing_upb", "mi_cancellation", "servicer_name",
    "cramdown_costs",
]

# Columns the pipeline actually uses (reading fewer columns saves a lot of memory
# on the real files: 50,000 loans x 8 vintages is roughly 25-30 million rows).
ORIG_USE = [
    "loan_id", "credit_score", "first_payment_date", "first_time_homebuyer",
    "mi_pct", "occupancy", "orig_cltv", "orig_dti", "orig_upb", "orig_ltv",
    "orig_rate", "channel", "property_state", "property_type", "loan_purpose",
    "orig_term", "num_borrowers",
]
PERF_USE = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age",
    "remaining_months", "zero_balance_code", "zero_balance_date", "current_rate",
    "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "total_expenses",
    "actual_loss", "eltv", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "borrower_assistance", "disaster_delinquency",
]

# Freddie Mac codes for "not available".
MISSING_CODES = {"credit_score": 9999, "orig_ltv": 999, "orig_cltv": 999,
                 "orig_dti": 999, "mi_pct": 999, "num_borrowers": 99}

# ---------------------------------------------------------------- definitions
# Default = 90+ days past due (delinquency status >= 3), REO acquisition ("RA"),
# or a zero balance caused by a credit event. This matches the Basel / IFRS 9
# 90-days-past-due backstop.
DEFAULT_DPD = 3
CREDIT_EVENT_ZB_CODES = {"02", "03", "09", "15"}  # third-party sale, short sale /
# charge-off, REO disposition, whole-loan sale
PREPAY_ZB_CODES = {"01"}
# Delinquency while in forbearance (Borrower Assistance Plan = "F", e.g. COVID
# payment holidays) or flagged as disaster-related is NOT counted as default:
# in the real data, COVID forbearance shows up as 90+ days past due. This follows
# the regulatory treatment of payment moratoria (e.g. EBA/GL/2020/02). Such loans
# go to Stage 2 instead (a qualitative SICR trigger).
FORBEARANCE_CODES = {"F"}

PD_HORIZON = 12  # months, for the 12-month PD
SNAPSHOT_MONTHS = (3, 6, 9, 12)  # quarter-end observation points

# Development / out-of-time split by snapshot date (YYYYMM).
DEV_END = 201512
OOT_START = 201603

MACRO_SERIES = {"UNRATE": "unemployment", "CSUSHPISA": "hpi",
                "MORTGAGE30US": "mortgage_rate"}

# Macro drivers of the point-in-time PD layer. The unemployment LEVEL was tried
# and dropped: next to its 12-month change it took a counter-intuitive negative sign.
PIT_MACRO = ["unemp_chg_12m", "hpi_growth_12m"]

VINTAGES = list(range(2005, 2017))  # 2013-2016 keep the out-of-time population realistic
