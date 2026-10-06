# Known limits of ProcessLens

_Generated from the planted-fault benchmark (`reports/metrics/benchmark.json`) by `processlens benchmark`. Do not edit by hand._

ProcessLens ranks sensor clusters **associated with** failure. It does not prove causes. The limits below are measured, not assumed.

1. **Linear faults:** the consensus ranks the planted cluster in the top 5 in ≥ 80% of scenarios once the effect is at least **β = 1** log-odds per SD (94% at β = 2). Smaller effects are missed more often.
2. **Threshold (> q95) faults:** the consensus ranks the planted cluster in the top 5 in ≥ 80% of scenarios once the effect is at least **β = 1** log-odds per SD (98% at β = 2). Smaller effects are missed more often.
3. **Drift window (30% of time) faults:** never reaches 80% top-5 recovery on this grid; even at β = 2 it finds the planted cluster in only 67% of scenarios.
4. **Correlated sensors cannot be separated.** At β ≥ 1 the consensus puts the planted *cluster* first in 48% of scenarios but the exact planted *sensor* first in 41%. Treat every member of a suspect cluster as equally suspect.
5. **Several simultaneous causes dilute each other.** Averaged over the grid, top-5 recovery is 69% with one planted cause and 52% per cause with three.
6. **False alarms with no real cause** (30 null scenarios): at least one sensor passes q < 0.05 in 13%, a cluster is graded moderate or strong in 7%, and graded strong in 3%. A ranking always has a #1; only the evidence grade says whether it means anything.
7. **Missing data in the cause sensor** (consensus top-5 recovery at β ≥ 1): drift window (30% of time): 0% missing → 67% (n=3), 0–5% missing → 58% (n=79), >5% missing → 60% (n=8); linear: 0% missing → 100% (n=5), 0–5% missing → 90% (n=79), >5% missing → 89% (n=6); threshold (> q95): 0% missing → 100% (n=2), 0–5% missing → 96% (n=78), >5% missing → 97% (n=10).
8. **Consensus vs single methods.** Averaged over all planted scenarios, consensus top-5 recovery is 62% vs 64% for the best single method (univariate). See the ablation figure per mechanism.
9. **Benchmark realism.** Planted faults are additive effects on the log-odds of one to three real sensors. Real defects may involve interactions, lags, sensors that were never measured, or label noise, so these numbers are an upper bound for comparable effect sizes, not a guarantee.
