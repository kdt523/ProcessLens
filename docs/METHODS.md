# Methods

How ProcessLens ranks **suspect** sensors: the theory behind each method, its assumptions, where it fails, and why it was chosen. All tunable values live in `configs/rootcause.yaml` and `configs/model.yaml`; this file cites them instead of repeating them.

> **What a ranking means.** SECOM is observational. A high-ranked sensor is *associated with* failure in the analysed window. It may be a cause, a symptom, or a proxy for something unmeasured. Rankings tell engineers where to look first, not what to change.

## 0. Analysis window and inputs

- The analyst picks a time window (`processlens rootcause --start … --end …`). The default is the Phase 2 training window, so the test window is never touched.
- Sensors with more than `max_missing_frac` missing values, or with a single distinct value **inside the window**, are excluded. Everything is decided from the window alone.

## 1. Univariate screen: Mann–Whitney U + rank-biserial effect + BH-FDR

**What.** For each sensor, compare failing runs with passing runs using the two-sided Mann–Whitney U test. Effect size is the rank-biserial correlation r = 2U/(n_fail·n_pass) − 1, which equals Cliff's δ: the probability that a random failing run reads higher than a random passing run, minus the reverse. It ranges from −1 to +1, and its sign gives the direction. p-values are adjusted with Benjamini–Hochberg to control the false discovery rate across ~450 tests.

**Why not a t-test?** SECOM sensors are heavy-tailed: the audit finds robust-z outliers in most usable sensors, and many sensors are skewed or multimodal. A t-test compares means and assumes approximately normal sampling distributions; with ~70 failures and extreme values, a few outliers can drive it. Mann–Whitney depends only on ranks, so it is robust to outliers and monotone transformations, and it detects distribution shifts, not just mean shifts.

**Why BH and not Bonferroni?** Bonferroni controls the chance of *any* false positive, which is too strict for screening hundreds of correlated sensors and throws away real signal. BH controls the expected *share* of false positives among flagged sensors, which is the quantity an engineer cares about when given a list. BH stays valid under positive dependence, which is typical of correlated sensors.

**Assumptions / failure modes.** Runs are treated as independent; temporal autocorrelation inflates significance. The test sees one sensor at a time, so it misses interactions and effects that only appear conditionally. Missing values are dropped per sensor, so missing-not-at-random sensors can be biased. The separate missingness test (Fisher's exact on missing vs failure, BH-adjusted) covers that case.

## 2. Stability selection with L1-logistic regression

**What.** Draw `n_subsamples` stratified subsamples of `subsample_frac` of the runs. On each, fit an L1-penalised logistic regression (penalty `C`, class-balanced) and record which sensors get a non-zero coefficient. A sensor's **selection frequency** is the share of subsamples that select it (Meinshausen & Bühlmann, 2010).

**Why not a single lasso fit?** One lasso fit on ~450 correlated sensors and ~70 failures is unstable. Small perturbations of the data change which member of a correlated group gets picked, and the chosen penalty decides how many survive. Selection frequency averages over that instability: sensors chosen in most subsamples are robustly useful, and sensors chosen by luck are not. It is also multivariate, so it can surface sensors whose signal only appears after adjusting for others, which the univariate screen misses.

**Implementation notes.** The window is median-imputed and robust-scaled once, with values clipped to ±10 robust units for solver stability. This is not leakage: no out-of-sample prediction is made. A clipped value still marks an extreme reading.

**Failure modes.** Within a correlated cluster, L1 spreads selection across members, so each may look moderately stable while the cluster as a whole is strongly selected. This is the main reason suspects are reported per cluster. The method only captures linear effects in the log-odds, and frequencies depend on `C`.

## 3. SHAP importance from LightGBM in time-ordered folds

**What.** Fit a small LightGBM model (shallow trees, `configs/rootcause.yaml → shap.lightgbm`, class-weighted) on expanding rolling-origin folds of the window. On each held-out block, compute exact TreeSHAP values (LightGBM `pred_contrib`, identical to `shap.TreeExplainer`), take mean |SHAP| per sensor, and average over folds. The **SHAP share** is a sensor's fraction of the total.

**Why.** Trees capture non-linear and threshold effects (for example "fails when the sensor exceeds its 95th percentile") and interactions that both linear methods miss. LightGBM handles missing values natively, so no imputation is involved. Scoring on held-out, *later* blocks rewards sensors whose association generalises forward in time, not sensors the model memorised.

**Failure modes.** SHAP explains the *model*, not the process: credit among correlated sensors is split somewhat arbitrarily. On weak-signal data the model itself is weak (see `reports/model_report.md`), so SHAP rankings are noisy. Importance does not give direction.

## 4. Clustering correlated sensors

**What.** Average-linkage hierarchical clustering on the distance 1 − |Spearman ρ|, cut at `clusters.min_abs_spearman`. Spearman is computed by ranking each sensor over its observed values and taking pairwise-complete Pearson correlation of the ranks. This approximates exact pairwise Spearman closely and is much faster; see `processlens/stats.py`.

**Why report clusters.** When two sensors are strongly correlated, observational data cannot tell which one is associated with failure, or whether both proxy a third, unmeasured variable. Ranking them separately would either split the evidence (stability, SHAP) or double-count it (univariate). A cluster is reported with its **representative**, the member with the best consensus score, plus its full member list. The engineer should treat every member as equally suspect.

**Why average linkage, why |ρ|.** Single linkage chains weakly related sensors together; complete linkage fragments real groups. Average linkage sits between them. The absolute value is used because a negative correlation is equally confounding.

## 5. Consensus ranking and evidence grade

**Consensus.** Each method ranks sensors: univariate by p, stability by frequency, SHAP by mean |SHAP|. Ties share the average rank. The consensus score is the **mean reciprocal rank** (MRR) across methods. MRR rewards a sensor that any method places near the top, while Borda (also available) rewards consistent mid-table placement. For root cause the question is "did a credible method put this at the top?", so MRR is the default. A cluster's score is its best member's score.

**Evidence grade.** Graded per cluster from the strongest member statistics (`configs/rootcause.yaml → evidence_rules`):

| grade | rule |
|---|---|
| **strong** | q ≤ `strong.max_q` **and** \|effect\| ≥ `strong.min_abs_effect` **and** stability ≥ `strong.min_stability` |
| **moderate** | q ≤ `moderate.max_q` **and** \|effect\| ≥ `moderate.min_abs_effect` **and** stability ≥ `moderate.min_stability` |
| **weak** | anything else |

The rule needs agreement between a marginal test with FDR control and practically meaningful size, and a multivariate method that selects the cluster repeatedly. A tiny but significant difference, or a sensor that one method loves and the others ignore, cannot be graded "strong". How often these grades fire when there is *no* real effect is measured in the Phase 4 benchmark (null scenarios), not assumed.

## 6. Temporal views: when did it start?

- **Failure rate by sensor-quantile bin over time.** The sensor is cut into `temporal.n_bins` quantile bins over the window, and the failure rate is shown per bin per `bin_freq` period. A stable gradient across periods supports a persistent association; a gradient in one period only suggests an episode.
- **Onset.** Binary segmentation (`ruptures.Binseg`, L2 cost, minimum segment `onset_min_size`) finds the single most likely change point in the sensor's `onset_freq` mean. The output is the first period after the change, the means before and after, and the shift in units of the sensor's standard deviation. A change point is always found, even in pure noise, so read it together with the shift size. A small shift means no meaningful onset.

## 7. What is deliberately *not* done

- No causal claims, no causal discovery. Interventions or domain knowledge are needed for that.
- No resampling (SMOTE etc.) anywhere: class imbalance is handled by class weights.
- Method reliability is not asserted here; it is **measured** by the planted-fault benchmark (`reports/benchmark.md`, `docs/LIMITS.md`).
