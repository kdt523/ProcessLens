# ProcessLens: Problem Spec

**Business question.** Which process parameters are associated with product defects, and how much can we trust that answer?

**Data.** UCI SECOM (id 179): one row per production run, ~590 anonymous sensor readings, a pass/fail label and a timestamp. Failures are rare (about 6.6%), many sensors have missing values, and sensors are strongly correlated.

## Users
- **Process engineers:** need a short, ranked list of parameters to investigate first.
- **Quality leads:** need to decide how many runs to inspect and what that buys.

## Decisions supported
1. **Which runs to inspect.** A calibrated risk score and a cost-based inspection policy ("inspect the top x% → catch y% of failures").
2. **Which parameters to investigate first.** Suspect clusters of correlated sensors, ranked by a consensus of statistical and ML evidence.
3. **Whether the evidence is strong enough to act on.** Each suspect carries an evidence grade (strong / moderate / weak), and the copilot abstains when nothing meets the rules.

## Success metrics
| Metric | Why it matters |
|---|---|
| PR-AUC with 95% bootstrap CI, shown next to the base rate | Honest measure of ranking quality on rare failures |
| Share of failures caught at a fixed inspection budget | What the quality lead actually buys |
| Root-cause hit@k (cluster level) on the planted-fault benchmark | Does the ranking recover known planted faults? |
| False-alarm rate on null scenarios | How often it "finds" something when nothing is there |
| Minimum detectable effect per fault mechanism | Where the method stops being reliable |
| Copilot numeric faithfulness and abstention accuracy | Can the written report be trusted? |

## Constraints
- Time-ordered validation; every preprocessing step fit on training data only; test set used once.
- Every reported number is produced by code and stored in `reports/metrics/*.json`.
- The LLM never sees raw rows and never chooses the suspect; it only explains tool outputs.

## Non-goals
- **Proving causality.** This is observational data. ProcessLens ranks *suspects associated with* failures for engineers to test; it never claims a sensor *causes* defects.
- Identifying the single responsible sensor inside a highly correlated cluster (the data cannot separate them).
- Real-time streaming, plant-historian integration, or high predictive accuracy for its own sake.
