# fraud-risk

Card-fraud detection and risk scoring, built to practise the parts of fraud work that are easy to get wrong:
extreme class imbalance, leakage, label delay, cost-based thresholds and calibration.

Two public datasets, deliberately different:

| Dataset | What it is | Why it is here |
|---|---|---|
| **ULB credit-card fraud** | Real European card transactions, anonymised to 28 PCA components, 48 hours, 0.17% fraud | Real data, but no card or customer id, so no behavioural features are possible |
| **Sparkov** (Kaggle, CC0) | **Simulated** card transactions: 1.85M rows, ~1,000 cards, 693 merchants, 2019-2020, 0.5% fraud | Has card and merchant ids, so per-customer behaviour can be engineered |

> Sparkov is **simulated**. Its fraud patterns come from a generator, so the numbers below show that the
> pipeline works and which ideas help *on that generator*. They are not a forecast for real bank data.

## Results on Sparkov (test file = later in time, 2,145 frauds, no-skill PR-AUC 0.0039)

| model | features | test PR-AUC (95% CI) | frauds found in top 1% of alerts |
|---|---|---|---|
| logreg_base | single transaction | 0.178 (0.168-0.187) | 66.9% |
| hgb_base | single transaction | 0.791 (0.772-0.805) | 88.8% |
| hgb_geo | + distance to home | 0.790 (0.772-0.806) | 88.9% |
| **hgb_behaviour** (champion) | + per-card behaviour | **0.895 (0.883-0.905)** | **95.2%** |
| hgb_behaviour_risk | + label-delayed merchant risk | 0.891 (0.879-0.902) | 94.9% |
| logreg_behaviour_risk | same features, linear model | 0.221 (0.209-0.230) | 67.0% |
| iforest_behaviour | unsupervised anomaly score | 0.134 (0.123-0.148) | 35.2% |
| kmeans_behaviour | unsupervised cluster distance | 0.049 (0.043-0.055) | 19.9% |

The champion is chosen on **validation** PR-AUC only; test is read once, at the end.

What the experiments say:
- **Per-card behaviour is what helps**: PR-AUC 0.79 to 0.89 and the intervals do not overlap. The useful signals are
  velocity (transactions in the last hour/24h/7d), how unusual the amount is versus that card's own 30-day history,
  time and distance since the previous transaction (implied speed), and first-time merchant/category.
- **Distance to home and delayed merchant risk did nothing here** (differences inside the interval). Reported, not hidden.
- **A linear model fails** (PR-AUC about 0.2): the signal lives in interactions, for example large amount at night on a new category.
- **Unsupervised methods are far weaker than supervised** (about 34x and 12x better than chance, but far below 0.89).
  They are useful as a triage signal when there are no labels, not as a replacement.

Operating point (fitted on validation, applied unchanged to test). Assumption: one manual review costs 5 USD and a missed
fraud costs its amount:

| | value |
|---|---|
| alerts to review | 5,403 of 555,719 (0.97%) |
| precision / recall | 37.7% / 95.1% |
| total cost | 47,389 USD vs 1,133,325 USD if nothing is flagged (95.8% lower) |
| risk bands | LOW 0.008% fraud (2.1% of all frauds), MEDIUM 0.87%, HIGH 37.7% (95.1% of all frauds) |

The cost saving is large because simulated frauds are big (about 528 USD on average) against a 5 USD review. With a
20 USD review the saving drops to 89.9%; see `reports/sparkov/results.md` for 1, 5 and 20 USD.

Probabilities are calibrated (Brier 0.0010; predicted versus observed rate per bin is in the report). Feature drift
between train and test is small (largest PSI 0.08) and the score PSI is 0.011, which is expected for simulated data.

## Results on ULB (real, anonymised; test has only 74 frauds)

Best on validation: logistic regression, test PR-AUC 0.742 (0.636-0.847). Hand-made features (log amount, hour,
outlier counts) gave **no measurable gain**: the intervals overlap completely, and with 74 frauds a single number
means little, which is why every table carries an interval. Unsupervised methods reached PR-AUC 0.04-0.06.
Full tables: `reports/ulb/results.md`.

## Design decisions

- **Temporal split, never shuffled.** Fraud happens in time order; random splits leak. Exact duplicate rows are dropped first
  (1,081 in ULB) so a copy cannot sit in both train and test.
- **Past-only features.** Every per-card feature uses earlier transactions of that card only. Tests prove it: features
  of the first rows are identical whether or not the future exists, and tampering with future amounts or labels does not
  change past features (`tests/test_sparkov.py`).
- **Label delay.** Merchant and category risk use only labels at least 7 days old, because chargebacks arrive late in real life.
- **Time bug found by sanity check.** Sparkov's `unix_time` is inconsistent with its date string, and pandas 3 parses dates
  with a different unit than nanoseconds. A hard-coded divisor once made a "1 hour" window span weeks; the symptom was
  an impossible 104 transactions per hour. There is now a regression test.
- **Costs, not accuracy.** The threshold minimises expected cost on validation (review cost + missed amount) and is then
  frozen for test. MEDIUM = 0.2 x HIGH is a documented heuristic, not tuned.
- **No demographic features.** Gender, age and job exist in Sparkov and are deliberately not used.
- **MLflow** records every experiment (parameters, metrics) and registers the champion.

## Limitations

- Sparkov is simulated; ULB is real but anonymised and short (48 hours). No real bank data with customer ids is public.
- One temporal split, default hyper-parameters, no tuning. Intervals come from bootstrap on one test period, not from
  repeated time folds.
- Review cost is an assumption (1, 5 and 20 USD are reported), not a measured business number.
- The API serves ULB only. Sparkov scoring needs each card's recent history, which would require a feature store; that is not built.
- No cloud deployment yet. Drift monitoring is a PSI report, not a live monitor.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt        # or bin/pip on Linux/macOS
python -m pytest -q                                   # 24 offline tests, ~15 s
python -m fraudrisk.train                             # ULB: downloads ~150 MB, ~3 min
# Sparkov: download archive.zip from kaggle.com/datasets/kartik2112/fraud-detection (login required),
# unzip fraudTrain.csv and fraudTest.csv into data/sparkov/, then:
python -m fraudrisk.train_sparkov                     # ~7-10 min on a 4-core laptop (RAM not measured on the full run;
                                                      # a 400k-row sample needed under 300 MB)
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse experiments and the registered champion
uvicorn fraudrisk.serve:app                           # ULB scoring API: POST /score, GET /health
```

Outputs: `reports/<dataset>/results.md` (all tables), `reports/<dataset>/pr_curve.png`, `artifacts/<dataset>/model.joblib`.
