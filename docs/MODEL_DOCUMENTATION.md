# IFRS 9 Expected Credit Loss Model: Development and Validation Report

**Portfolio:** US prime residential mortgages, Freddie Mac Single-Family Loan-Level Dataset (sample files, vintages 2005–2016, performance to June 2026)
**Sample used:** a random 30% of loans (about 180,000 loans, 13 million loan-months), chosen to fit in 7 GB of memory
**Reporting date for ECL:** December 2017 (73,468 loans, $12.5bn balance)

The structure follows SR 11-7 (US Federal Reserve) and SS1/23 (UK PRA): purpose, data, methodology, results, validation, findings, limitations, monitoring. Every number regenerates from `scripts/run_pd.py` and `scripts/run_ecl.py`.

---

## 1. Purpose and use

The model estimates IFRS 9 impairment allowances for a mortgage portfolio:

- a **12-month PD** to rank loans and detect a significant increase in credit risk (SICR);
- a **lifetime PD** that responds to forward-looking macroeconomic scenarios;
- **LGD** and **EAD** for each future quarter;
- **staging** and the **probability-weighted ECL** across four scenarios;
- a **stress test** under a severe recession.

Out of scope: regulatory capital (through-the-cycle PDs and downturn LGD), pricing, origination decisions.

## 2. Data

| Item | Detail |
|---|---|
| Source | Freddie Mac SFLLD sample files: 50,000 random loans per vintage; origination (31 fields) and monthly performance (35 fields) |
| Macro | FRED: unemployment rate (UNRATE), Case-Shiller national house price index (CSUSHPISA), 30-year mortgage rate (MORTGAGE30US) |
| Default definition | First month 90+ days past due **while not in forbearance or disaster relief**, or a credit-event zero balance (02 third-party sale, 03 short sale/charge-off, 09 REO disposition, 15 whole-loan sale) |
| Forbearance treatment | In the raw data, COVID payment holidays appear as 90+ days past due: in 2020, 3,803 of 5,123 such loan-months in the 2014 vintage carried Freddie Mac's forbearance flag. Counting them as defaults would contradict the regulatory treatment of payment moratoria (for example EBA/GL/2020/02). They count instead as a Stage 2 trigger |
| PD development sample | Quarter-end snapshots of performing, never-defaulted loans whose 12-month window lies inside the data. All 39,092 defaults kept, non-defaults sampled to 1,000,000 rows in total and weighted back |
| Splits | Development: snapshots to Dec 2015, split 70/30 by loan into train and holdout. Out-of-time: Mar 2016 onwards |
| Missing values | Freddie Mac codes 9999 (credit score), 999 (LTV, DTI, ELTV) set to missing. DTI is missing for 25,631 sampled loans; missing gets its own WoE bin |
| Licence | Freddie Mac terms allow internal use only: no raw data in the repository |

## 3. Methodology

### 3.1 12-month PD: WoE scorecard

Variables are binned (20 starting quantile bins, or cuts between distinct values for discrete variables) and merged until every bin holds enough rows and default rates move one way (loan age may have a hump). Selection: IV ≥ 0.02, correlation filter (|ρ| < 0.7 between WoE variables), then logistic regression requiring every coefficient to be negative and significant at 1%. Points: 600 at 50:1 odds, 20 points to double the odds.

| Variable | IV | Coefficient |
|---|---:|---:|
| Max days past due, last 12 months | 1.68 | −0.80 |
| Credit score at origination | 0.82 | −0.46 |
| Marked-to-market LTV | 0.65 | −0.58 |
| Refinancing incentive (loan rate − market rate) | 0.52 | −0.30 |
| Origination channel | 0.36 | −0.34 |
| DTI at origination | 0.26 | −0.42 |
| Short term (≤ 15 years) | 0.24 | −0.11 |
| Loan age | 0.17 | −0.11 |
| Property state | 0.12 | −0.96 |
| Multiple borrowers | 0.08 | −0.63 |
| Share of balance paid down | 0.08 | −0.66 |
| Loan purpose | 0.07 | −0.29 |

Removed: days past due now (IV 1.45, correlated with the 12-month maximum), original LTV and CLTV (correlated with marked-to-market LTV), mortgage insurance (wrong sign once LTV is in).

### 3.2 Point-in-time layer

logit(PD) = −0.18 + 0.98 × scorecard log-odds + 0.10 × (12-month change in unemployment) − 2.92 × (12-month house-price growth)

The unemployment *level* was tested and dropped: next to its change it took a counter-intuitive negative sign.

### 3.3 Lifetime PD: competing-risk hazards

From each December snapshot, loans are followed for up to 20 quarters. Two quarterly logistic hazards (default, prepayment) use the scorecard log-odds, time since the snapshot, and the economy in that future quarter. Survival multiplies (1 − default hazard − prepayment hazard); marginal PD in quarter h = survival to h−1 × default hazard in h. Fitted **only on information available at the reporting date** (1.65 million loan-quarters to Dec 2017; 13,127 defaults, 72,251 prepayments).

| Default hazard driver | Coefficient | Prepayment hazard driver | Coefficient |
|---|---:|---|---:|
| Scorecard log-odds | +0.75 | Scorecard log-odds | −0.16 |
| Unemployment rate | +0.14 | Rate gap (loan rate − market rate) | +0.41 |
| 12-month change in unemployment | +0.11 | 12-month house-price growth | +0.55 |
| House-price change since reporting date (log) | −2.28 | House-price change since reporting date (log) | −0.62 |

### 3.4 LGD

Realised LGD = actual loss ÷ balance removed, for 3,942 liquidations up to Dec 2017 (mean 50.5%). Fractional logit on LTV at liquidation (Freddie Mac's ELTV where reported, otherwise marked to market with the national index), a separate slope above 100% LTV, and mortgage insurance (coefficient −0.70). House-price growth at liquidation was tested and dropped (wrong sign, p = 0.07). LGD is forward-looking through LTV, which moves along each scenario's house-price path.

| LTV at liquidation | Liquidations | Observed LGD | Predicted LGD |
|---|---:|---:|---:|
| < 60% | 282 | 48.0% | 48.9% |
| 60–80% | 903 | 52.8% | 51.9% |
| 80–100% | 1,840 | 53.1% | 53.2% |
| 100–120% | 772 | 44.3% | 44.6% |
| > 120% | 145 | 41.4% | 42.4% |

LGD falls above 100% LTV in the raw data because high-LTV loans usually carry mortgage insurance; ELTV, where available (12% of liquidations), rises cleanly from 30% to 70% LGD. See finding F5.

### 3.5 EAD, discounting, staging, ECL

- **EAD:** scheduled amortising balance. **Discounting:** at each loan's own rate.
- **Stage 3:** 90+ days past due, not in forbearance.
- **Stage 2:** 30+ days past due, in forbearance, or 12-month PD now ≥ 3× its value at origination *and* at least 0.5 percentage points higher.
- **ECL:** sum over quarters of marginal PD × LGD × EAD × discount factor; 4 quarters for Stage 1, remaining life (capped at 10 years) for Stage 2; LGD × EAD for Stage 3.

### 3.6 Scenarios

| Scenario | Weight | Unemployment | House prices | Mortgage rate |
|---|---:|---|---|---|
| Upside | 15% | to 3.8% in 6 quarters | +10% over 2 years, then +4% a year | 5.5% |
| Base | 55% | to 4.6% in 4 quarters | +6% over 2 years, then +3% a year | 6.0% |
| Downside | 25% | to 7.0% in 8 quarters, 5.0% by year 5 | −8% over 2 years, then +3% a year | 6.5% |
| Severe | 5% | to 10.0% in 8 quarters, 6.0% by year 5 | −25% over 3 years, then +2.5% a year | 5.0% |

Illustrative judgements; in a bank they come from the economics team and a governance committee.

## 4. Results (reporting date Dec 2017)

| Stage | Loans | Exposure | Share | ECL | Coverage |
|---|---:|---:|---:|---:|---:|
| 1 | 69,848 | $11,889m | 95.0% | $19.8m | 0.17% |
| 2 | 3,102 | $534m | 4.3% | $69.1m | 12.96% |
| 3 | 518 | $88m | 0.7% | $42.7m | 48.54% |
| **Total** | **73,468** | **$12,510m** | 100% | **$131.7m** | **1.05%** |

| Scenario | ECL | Coverage |
|---|---:|---:|
| Upside | $111.1m | 0.89% |
| Base | $122.6m | 0.98% |
| Downside | $152.1m | 1.22% |
| Severe | $192.1m | 1.54% |
| **Probability-weighted** | **$131.7m** | **1.05%** |
| One run on the weighted-average path | $130.3m | 1.04% |

- **Non-linearity:** weighting scenarios adds 1.1% over a single average-scenario run. It is small here because most of the allowance sits in Stage 2 and 3, where LGD barely moves with the scenario.
- **Stress:** everything in the severe scenario, with loans migrating to Stage 2 where the stressed PD meets the SICR rule: ECL $199.1m (+51%), Stage 2 rises from 4.2% to 5.4% of loans.
- **Staging sensitivity:** SICR thresholds from 2× to 4× move Stage 2 between 4.0% and 4.6% of loans and ECL between 1.04% and 1.06% of exposure.

## 5. Validation

### 5.1 Discrimination

| Sample | Scorecard Gini | Point-in-time Gini | Challenger (LightGBM) Gini | KS |
|---|---:|---:|---:|---:|
| Train | 0.799 | 0.804 | 0.859 | 0.630 |
| Holdout | 0.805 | 0.811 | 0.837 | 0.642 |
| Out-of-time | 0.712 | 0.705 | 0.750 | 0.570 |

On real data **the challenger adds 3–4 Gini points**, holding up out-of-time. Its SHAP profile shows the same drivers as the scorecard (credit score, marked-to-market LTV, refinancing incentive, delinquency), with interactions the additive scorecard cannot capture. **Decision:** keep the scorecard as champion for transparency and for its stability under the bank's validation standards; run the challenger in parallel to monitor, and test whether two or three interaction terms (for example credit score × LTV) close the gap (finding F6).

### 5.2 Calibration

- **Holdout:** mean PD 1.58% vs observed 1.56%. **Out-of-time:** 0.40% vs 0.39%. Level calibration is good in both.
- **By grade (Jeffreys, out-of-time):** grades 2–7 and 10 pass; **grades 8 and 9 are under-predicted** (1.35% vs 1.17% and 3.49% vs 2.26%). Grade 1 is marginally under (0.06% vs 0.06%, p = 0.005). Grades 4–7 are slightly conservative. Hosmer-Lemeshow rejects on both samples; with 450,000+ weighted rows small gaps are significant, so the grade pattern matters more than the p-value.
- **Through time:** the point-in-time PD tracks the 2009 peak (about 4%), which the scorecard alone misses (2.8%). In 2020 it predicts a spike to about 1.1% that did not happen (observed about 0.3%; finding F2). From 2023 it under-predicts by about half (finding F3).

### 5.3 Stability

PSI of the scorecard (development vs out-of-time) is **0.66**. Characteristic stability: share paid down (1.36), loan age (1.08) and marked-to-market LTV (1.01) moved with seasoning and the 2020–22 house-price surge; **origination channel (0.52)** shifted in composition; refinancing incentive (0.17) and DTI (0.11) moved modestly; credit score, delinquency, state and purpose are stable (≤ 0.04).

### 5.4 Lifetime PD backtest

| Years after Dec 2017 | 1 | 2 | 3 | 5 |
|---|---:|---:|---:|---:|
| Observed cumulative default rate | 0.46% | 0.80% | 1.07% | 1.59% |
| Predicted, base scenario | 0.53% | 1.07% | 1.56% | 2.41% |
| Predicted, given the economy that actually followed | 0.46% | 0.89% | 1.93% | 2.26% |

Year 1 is exact on the realised path. From 2020 the model over-predicts, because forbearance kept unemployment from turning into defaults (finding F2). The base scenario is conservative by about half at 5 years.

## 6. Findings

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| F1 | Grades 8–9 under-predicted out-of-time and on holdout (grade 9: 3.5% vs 2.3%); mainly loans with recent, cured delinquency | High | Segment PD for recently delinquent loans, or add a months-since-last-delinquency variable |
| F2 | COVID forbearance: (a) would have been counted as default under a pure 90-days-past-due rule; (b) the unemployment–default link broke in 2020: PIT PD spiked to 1.1% vs 0.3% observed, and the realised-path backtest over-predicts 3-year defaults by 80% | High | (a) Definition aligned to moratoria guidance (done); (b) documented post-model adjustment for policy-driven periods; payment-relief indicator in the hazard model |
| F3 | 2023–25 PDs under-predict by about half while unemployment stays low | Medium | Investigate affordability drivers (payment shock from rates, insurance and tax costs); recalibrate the PIT intercept if it persists |
| F4 | Lifetime PD conservative at long horizons (base 2.41% vs observed 1.59% at 5 years) | Medium | Recalibrate the hazard time shape; test a richer prepayment model, since the 2020–21 refinancing wave removed risk faster than assumed |
| F5 | LGD model weakly discriminating, calibrated to 2009–16 crisis liquidations (about 50%); observed LGD fell to the high 30s–40s by 2019–20 | Medium | Use ELTV as coverage grows; add judicial-state and time-to-liquidation drivers; consider a downturn/PIT LGD split. ECL is likely conservative, mainly in Stage 3 |
| F6 | Challenger beats the scorecard by 3–4 Gini points | Low | Keep champion; test interaction terms; monitor the gap |
| F7 | Prepayment hazard: cumulative house-price change takes a negative sign alongside a positive 12-month growth term (collinear) | Low | Keep one house-price term |
| F8 | Origination channel CSI 0.52 | Low | Confirm whether this is a mix change or a coding change in the data |

## 7. Limitations

- 30% random sample of the Freddie Mac sample files (for memory); results move slightly with the seed.
- National house prices except where ELTV is reported; regional busts are understated.
- Lifetime horizon capped at 10 years; hazard time shape held flat beyond 5 fitted years.
- EAD assumes scheduled amortisation for delinquent loans too.
- SICR compares 12-month PDs, a permitted simplification.
- Scenario paths and weights are illustrative.

## 8. Ongoing monitoring

Quarterly: Gini and KS on the latest outcomes; Jeffreys test by grade; PSI and CSI; stage-transfer rates; ECL movement analysis (new loans, repayments, stage transfers, model and macro changes). Triggers: Gini down more than 5 points from development; any grade failing Jeffreys two quarters running; PSI above 0.25 not explained by the economy; the challenger's lead widening beyond 5 points.
