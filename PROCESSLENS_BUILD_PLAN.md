# ProcessLens: Build Plan (hand this file to Claude Code)

> **Scope override (owner, 2026-10-06):** Docker, docker compose and the public demo deploy are skipped for now. The project runs directly via `uv` + `make`. Every other part of this plan stays as written.

> **ProcessLens** finds which manufacturing process parameters drive product defects. It proves how reliable those findings are with a planted-fault benchmark, and a guarded LLM copilot writes an engineer-ready root-cause report. It's built on the public UCI SECOM factory dataset and packaged like a production service.

---

## 0. How to use this plan (instructions for Claude Code)

You are building this project with the owner (kd) in his IDE. Follow these rules for the whole build:

1. **Work one phase at a time.** Start each phase in plan mode, show the plan, then implement. At the end of a phase, run its **Acceptance checks**, report the results, commit, and **stop for review** before the next phase.
2. **Never invent or hand-type numbers.** Every metric goes through code into `reports/metrics/*.json`. The README tables and resume bullets are generated from those files by `scripts/render_readme.py`. If a result is weak, report it honestly. Weak but honest is the point.
3. **No leakage, ever.** Every preprocessing step (imputation, scaling, feature selection) is fit on training data only, inside a scikit-learn `Pipeline`. Splits are time-ordered. The tests in Phase 2 enforce this.
4. **Reproducible.** Seed every random step. Put all parameters in `configs/*.yaml`. `make all` rebuilds every artifact from a clean clone.
5. **Small, typed, tested code.** Use Python 3.11, type hints, docstrings on public functions, `ruff` and `pytest` from day one. Prefer simple functions over frameworks.
6. **Ask before** adding heavy dependencies not listed in §4, or before changing the evaluation design in §6/§7.
7. Pin library versions in `pyproject.toml` (managed with `uv`). For LangGraph and LangChain, check the installed version's docs (`docs.langchain.com`) before writing agent code, because those APIs change often.

---

## 1. The story (what this project proves to a Michelin interviewer)

Michelin runs root-cause analysis on factory process data at scale. Its Parameters Analyzer (built on Dataiku) helps 600+ process engineers rank the variables linked to defects, and it cut that analysis from months to about an hour. ProcessLens is a small, honest version of that problem class:

- **Predict** which production runs will fail, and turn that into a cost-based inspection policy.
- **Explain** which sensors are linked to failures: statistics plus ML, at the level of groups of correlated sensors.
- **Prove** the method works with a **planted-fault benchmark**. Real data has no ground truth, so we inject known causes into a copy of the data and measure whether the method recovers them, including the smallest effect it can detect and its false-alarm rate when there's no cause at all.
- **Communicate** with a LangGraph copilot that writes the report only from tool outputs. A verifier checks every number, and the copilot abstains when the evidence is weak.
- **Industrialize** with data contracts, MLflow, FastAPI, a Streamlit dashboard, drift monitoring, Docker, CI quality gates and a model card.

**30-second pitch:** "I built a small version of the root-cause problem your process engineers solve. Real factory data has no ground truth, so I planted known faults to measure what my method can and can't detect. Then I put a guarded LLM copilot on top that only reports numbers it can verify."

---

## 2. JD gap → where the project proves it

| Michelin JD line | Phase | Artifact that proves it |
|---|---|---|
| Understand customer needs; formalize them in a datasheet/spec | 0 | `docs/PROBLEM_SPEC.md` (business question, users, decisions, success metrics) |
| Determine relevant data, identify missing data | 1 | `reports/data_audit.md`: missingness map, informative-missingness test |
| Prepare data (clean, enrich, aggregate), document the approach | 1–2 | pandera data contract + leakage-safe `Pipeline` + audit report |
| Methods selected on theoretical basis, pros and cons | 3 | `docs/METHODS.md`: why each test/model, assumptions, failure modes |
| Build predictive models | 2 | Baselines → LightGBM, calibrated, bootstrap CIs |
| Statistics | 2–4 | Mann–Whitney + effect sizes, BH-FDR, bootstrap CIs, power/MDE curves |
| Present results with visualization, performance **and limits** | 4, 6 | Streamlit dashboard + `MODEL_CARD.md` + MDE/limits page |
| Build and automate ML industrialization workflow | 2, 7 | MLflow tracking + registry, `make all`, CI gates, Docker |
| Promote models to production with data/platform engineers | 6–7 | FastAPI service, drift monitor, containerized deploy |
| Peer reviews, quality | 7 | PR template, CI (ruff, mypy, pytest), reproducible metrics |
| **Generative AI, LLMs, LangChain/LangGraph, prompt engineering, AI agents** | 5 | LangGraph copilot, versioned prompts, structured output, verifier, eval harness |
| RAG (light touch, not the focus) | 5 (optional) | Retrieval over `docs/METHODS.md` and sensor metadata for explanations |

---

## 3. Data facts (verified)

- **Source:** UCI ML Repository, SECOM (dataset id **179**), CC BY 4.0.
  - Fetch with `ucimlrepo.fetch_ucirepo(id=179)`, or fall back to the zip at `https://archive.ics.uci.edu/static/public/179/secom.zip`.
- **Files:**
  - `secom.data`: space-separated, `NaN` for missing.
  - `secom_labels.data`: label plus timestamp per row.
  - `secom.names`.
- **Shape:** 1,567 production runs. UCI lists 591 attributes. **Assert the actual shape at load time and record it; don't hardcode it.**
- **Labels:** `-1 = pass`, `1 = fail`. Map to `0/1`. 104 fails, about 6.6% base rate. The data is heavily imbalanced.
- **Timestamps:** parse with `dayfirst=True`, sort by time, and assert they're monotonic after sorting.
- **Expectation:** the predictive signal in SECOM is known to be weak. Many public notebooks report high scores because of random splits or resampling before the split, which leaks. **Ours will be honest and modest. The benchmark and the limits are what impress.**

---

## 4. Stack and repo layout

**Core:** python 3.11, uv, pandas, numpy, scipy, statsmodels, scikit-learn, lightgbm, shap, pandera, mlflow, plotly, streamlit, fastapi, uvicorn, pydantic v2, pyyaml, typer (CLI)
**GenAI:** langgraph, langchain-core, langchain-google-genai (Gemini Flash; model name in config)
**Quality:** pytest, pytest-cov, ruff, mypy, pre-commit
**Ops:** Docker, docker compose, GitHub Actions
**Optional:** `ruptures` (change-point detection), `evidently` (only if PSI by hand isn't enough)

```
processlens/
├── configs/            # data.yaml, model.yaml, benchmark.yaml, agent.yaml, costs.yaml
├── data/               # raw/ (gitignored), processed/ (parquet)
├── docs/               # PROBLEM_SPEC.md, METHODS.md, MODEL_CARD.md, LIMITS.md, ARCHITECTURE.md
├── src/processlens/
│   ├── data/           # ingest.py, contract.py (pandera), audit.py, split.py
│   ├── features/       # pipeline.py (sklearn ColumnTransformer / Pipeline)
│   ├── models/         # train.py, evaluate.py, calibrate.py, policy.py
│   ├── rootcause/      # univariate.py, stability.py, shap_rank.py, clusters.py, aggregate.py, temporal.py
│   ├── benchmark/      # synth.py (planted faults), run.py, metrics.py
│   ├── monitoring/     # drift.py (PSI)
│   ├── agent/          # tools.py, graph.py, schemas.py, verifier.py, prompts/ (v1.md, v2.md)
│   ├── api/            # main.py (FastAPI), schemas.py
│   └── cli.py          # `processlens ingest|audit|train|rootcause|benchmark|agent-eval|drift`
├── app/                # Streamlit multipage dashboard
├── evals/              # agent scenarios (jsonl), recorded LLM responses for replay
├── reports/            # figures/, metrics/*.json, data_audit.md, benchmark.md
├── scripts/            # render_readme.py
├── tests/
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml, .github/workflows/ci.yml
└── README.md
```

---

## 5. Phases

Each phase lists **Skills showcased**, **Tasks**, **Design details** and **Acceptance checks**. Time estimates assume focused days.

---

### Phase 0: Problem framing and repo scaffold (0.5 day)

**Skills showcased:** turning customer needs into a spec, engineering hygiene

**Tasks**
1. Scaffold the repo (layout above), `uv` project, `ruff`, `pytest`, `pre-commit`, and a `Makefile` with targets: `setup ingest audit train rootcause benchmark agent-eval app api test lint all`.
2. Write `docs/PROBLEM_SPEC.md`:
   - **Users:** process engineers and quality leads.
   - **Decisions:**
     1. Which runs to inspect.
     2. Which parameters to investigate first.
     3. Whether the evidence is strong enough to act on.
   - **Success metrics:** PR-AUC vs the base rate; failures caught at a fixed inspection budget; root-cause hit@k on the benchmark; false-alarm rate on null scenarios; report faithfulness.
   - **Non-goals:** proving causality, which observational data can't do. Say this explicitly.
3. Add `configs/*.yaml` with every tunable value: thresholds, seeds, split fractions, cost ratios, model names.

**Acceptance checks:** `make lint test` passes on an empty test suite; `PROBLEM_SPEC.md` exists and is under one page.

---

### Phase 1: Ingest, data contract and data audit (1 day)

**Skills showcased:** finding relevant and missing data, data cleaning, pandas, documentation

**Tasks**
1. `ingest.py`: download SECOM (ucimlrepo, falling back to the zip), store the raw files with a SHA-256 checksum, merge features + label + timestamp, sort by time, and write `data/processed/secom.parquet`.
2. `contract.py`: a pandera schema with numeric sensor columns, label in {0,1}, a non-null, monotonic timestamp, and expected row count. Validate on load. Add contract tests.
3. `audit.py`: generate `reports/data_audit.md` plus figures:
   - % missing per sensor (histogram and a list of sensors over 50% missing), and a missingness heatmap over time
   - constant and near-constant sensors, and exact-duplicate columns
   - correlation structure: the share of sensor pairs with |Spearman| > 0.9, and a preview of the clustering
   - failure rate over time (weekly), runs per week, time coverage
   - **informative missingness:** for each sensor, test whether *being missing* is linked to failure (Fisher's exact test + BH-FDR). Report the sensors where missingness itself carries signal. This is a real finding.
   - outliers per sensor (robust z-score > 5 count)
4. Write an "audit conclusions" section listing the cleaning decisions and the reason for each. These drive Phase 2.

**Acceptance checks:** `make ingest audit` runs end to end from a clean clone; the contract tests pass; `data_audit.md` renders with figures; every number in it comes from code.

---

### Phase 2: Leakage-safe predictive model and inspection policy (1.5 days)

**Skills showcased:** predictive modeling, statistics (bootstrap CIs), calibration, MLflow, business framing

**Tasks**
1. `split.py`: **time-ordered** split, 60% train / 20% validation / 20% test by timestamp. Use rolling-origin CV inside train for model selection. Touch the test set **once**, at the end.
2. `pipeline.py`: one sklearn `Pipeline` per model:
   - drop sensors with missing % above the threshold (computed on train only) and zero-variance sensors (on train)
   - median impute + `add_indicator=True` (keeps the informative missingness found in Phase 1)
   - `RobustScaler` for the linear models
3. `train.py`: models, in order:
   - `DummyClassifier`
   - logistic regression (L2 and L1, `class_weight="balanced"`)
   - random forest
   - LightGBM (`scale_pos_weight`, small trees, early stopping on validation)

   **No SMOTE before the split.** If resampling is used at all, it goes inside the CV folds and is documented. Log every run to MLflow (params, metrics, artifacts, git SHA, data checksum) and register the best model.
4. `evaluate.py`:
   - primary metric **PR-AUC**, shown next to the base rate; ROC-AUC; recall at precision ≥ X; Brier score; ECE
   - **bootstrap 95% CIs** for each
   - **paired bootstrap** of the PR-AUC difference between the best model and logistic regression, so we can say whether the "better model" is actually better
5. `calibrate.py`: isotonic or Platt calibration on validation, with a reliability plot.
6. `policy.py`, the business layer:
   - **gains/lift curve:** "inspect the top x% riskiest runs → catch y% of failures"
   - **cost model** from `configs/costs.yaml`: cost of a missed defect vs cost of an inspection. Choose the threshold that minimizes expected cost, and compare against *inspect all*, *inspect none* and *random inspection at the same budget*.
   - **sensitivity analysis** over cost ratios from 2:1 to 50:1. State the assumptions in the report.
7. Leakage tests (`tests/test_leakage.py`):
   - every test-set timestamp is after every train timestamp
   - imputer and scaler statistics equal those computed on train rows only
   - feature dropping is decided on train only (inject a column that's all-NaN in train → it must be dropped even if it's present in test)
   - no feature is derived from the label

**Acceptance checks:** `make train` writes `reports/metrics/model.json` (metrics with CIs) and figures for the PR curve, calibration, gains and cost sensitivity; the MLflow UI shows the runs; leakage tests pass. Results are reported honestly even if modest.

---

### Phase 3: Root-cause ranking engine (1 day)

**Skills showcased:** choosing methods on theoretical grounds, statistics, explainability, documentation

**Tasks** (all on the train window, or on an analyst-chosen time window passed as a parameter)
1. `univariate.py`: per sensor, a Mann–Whitney U test (fail vs pass) plus effect size (Cliff's delta / rank-biserial) and BH-FDR q-values. Also the missingness test from Phase 1.
2. `stability.py`: L1-logistic **stability selection**. Fit on 100 subsamples of 50% each (stratified, seeded) and record each sensor's selection frequency.
3. `shap_rank.py`: LightGBM fit inside time-ordered CV folds; mean |SHAP| per sensor, averaged across folds.
4. `clusters.py`: hierarchical clustering on `1 − |Spearman ρ|`, cut at a configurable threshold (default ρ ≥ 0.8). **Report suspects at the cluster level**, with a representative sensor. Among highly correlated sensors the true cause can't be identified from the data, and we say so.
5. `aggregate.py`: combine the method ranks into a consensus score (Borda or mean reciprocal rank). The output is a table with columns cluster, representative, consensus rank, q-value, effect size, stability frequency, SHAP share and evidence strength (strong/moderate/weak, using the rules in `METHODS.md`).
6. `temporal.py`: for the top suspects, failure rate by sensor quantile bin over time, plus "when did it start" using CUSUM or `ruptures` change-point detection on the sensor's weekly mean.
7. `docs/METHODS.md`: for each method, the theoretical basis, assumptions, strengths, failure modes, and why it was chosen over alternatives (for example, why Mann–Whitney rather than a t-test, and why stability selection rather than a single lasso fit).

**Acceptance checks:** `processlens rootcause --start ... --end ...` prints a ranked JSON and writes `reports/metrics/rootcause.json`; unit tests check each method on a tiny synthetic dataset with a known answer; `METHODS.md` exists.

---

### Phase 4: Planted-fault benchmark (1.5 days), the centerpiece

**Skills showcased:** experimental design, statistics (power, false discovery), ablation, presenting performance and limits

**Idea:** keep the **real** SECOM sensor matrix, so the missingness, correlations and drift are all realistic. Replace the labels with **synthetic labels generated from known causes**. Run the root-cause engine and measure whether it recovers them.

**Tasks**
1. `synth.py`: scenario generator (seeded):
   - choose `n_causes ∈ {1, 2, 3}` cause sensors from those with less than 20% missing
   - mechanism ∈:
     - **linear:** `logit p = b0 + β·z(x)`
     - **threshold:** fails when `x > q95`, with a set probability
     - **drift-window:** the effect only applies inside one time window, as when a machine degrades
   - effect size β ∈ {0.25, 0.5, 1.0, 1.5, 2.0}, in log-odds per SD
   - calibrate `b0` so the base rate stays near 6.6%
   - **null scenarios:** no cause at all (labels permuted, or drawn at the base rate)
   - at least 10 seeds per cell
2. `run.py`: for each scenario, run each method separately (univariate, stability, SHAP) **and** the consensus. Parallelize with joblib. Cache the results.
3. `metrics.py`:
   - **hit@k** (k = 1, 3, 5) at the **sensor** level and at the **cluster** level (a hit counts if the planted sensor's cluster is ranked)
   - **precision@k**
   - **false-alarm rate on null scenarios:** how often any suspect passes q < 0.05 or is labelled "strong" evidence
   - **minimum detectable effect (MDE):** the smallest β with at least 80% cluster-level hit@5, per mechanism
   - **runtime** per full analysis
4. `reports/benchmark.md` plus figures:
   - hit@5 vs effect size, one line per method (ablation)
   - MDE per mechanism
   - null false-alarm rates
   - consensus vs best single method
   - a "what this means" section in plain English, which becomes `docs/LIMITS.md`. For example: "reliably finds causes of effect ≥ Xσ; struggles with threshold faults on sensors with heavy missingness; cannot separate sensors within a correlated cluster."
5. Add a fast smoke configuration (few cells, 2 seeds) used by CI as a **quality gate**: consensus cluster-level hit@5 at a large effect must be at least a configured threshold, and the null false-alarm rate at most a configured threshold.

**Acceptance checks:** `make benchmark` writes `reports/metrics/benchmark.json` and `benchmark.md`; the CI smoke config runs in under 3 minutes; `LIMITS.md` is generated from the results.

---

### Phase 5: GenAI root-cause copilot (LangGraph) and its eval harness (1.5 days)

**Skills showcased:** Generative AI, LLMs, LangGraph/LangChain, prompt engineering, AI agents, structured output, LLM evaluation

**Design principle:** the LLM **never** sees raw data and **never** decides the cause. It calls typed tools over the analysis outputs, writes a structured report, and a programmatic verifier checks every number.

**Tasks**
1. `tools.py`: pure-Python functions wrapped as LangChain tools, each with a pydantic-typed input and output:
   - `get_data_health(window)`
   - `rank_suspects(window, top_k)`
   - `inspect_sensor(sensor_id, window)`: distributions, effect size, q-value, failure rate by bin
   - `get_cluster(sensor_id)`
   - `when_did_it_start(sensor_id)`
   - `get_model_performance()`
   - `get_known_limits()`: reads `LIMITS.md`, so the report states the limits
2. `schemas.py`: pydantic `RootCauseReport` with fields:
   - `summary`
   - `findings[]`: cluster, representative sensor, evidence strength, the supporting numbers each with the tool call that produced them, onset date
   - `recommended_checks[]`
   - `limits[]`
   - `decision`: one of `investigate` / `monitor` / `insufficient_evidence`
3. `graph.py`: an explicit LangGraph `StateGraph` with these nodes:
   - `plan`: decide which tools to call
   - `gather`: tool calls, using a ToolNode or explicit calls
   - `draft`: produce the report via `with_structured_output(RootCauseReport)`
   - `verify`
   - `revise`: at most 2 loops
   - `abstain`: taken if no suspect meets the evidence rules
   - `finalize`

   Gemini Flash through `langchain-google-genai`, with the model name, temperature and limits in `configs/agent.yaml`. Optional: fallback to a second model.
4. `verifier.py`: extract every numeric claim from the report. Each must match a value in the tool-call log within a tolerance, and each finding must cite a tool call. On failure, send specific feedback to `revise`. Log the faithfulness score.
5. **Prompt engineering:** keep `prompts/v1.md` (baseline) and `prompts/v2.md` (improved: explicit evidence rules, an abstention instruction, few-shot examples of a good finding and a correct abstention). Compare them in the eval and keep both in the repo.
6. **Eval harness** (`evals/`):
   - build 40 scenarios from the Phase 4 generator with known ground truth: 30 with a planted cause across effect sizes and mechanisms, and 10 null
   - metrics: hit@3 of the planted cluster in the findings; **numeric faithfulness** (% of numbers verified); **abstention accuracy** on null scenarios and on low-effect scenarios; schema-valid rate; latency p50/p95; tokens and cost per report
   - write the results to `reports/metrics/agent.json` for v1 vs v2
   - **record/replay:** store LLM responses so tests and CI run deterministically without an API key; `make agent-eval` runs live
7. Optional light RAG: retrieve paragraphs from `METHODS.md`/`LIMITS.md` so the copilot explains *why* evidence is weak. Keep it small, since RAG is already shown in the owner's other projects.

**Acceptance checks:** `make agent-eval` produces the v1 vs v2 table; the verifier has unit tests, including a deliberately wrong number that must be caught; replay-mode tests pass offline; the copilot abstains on null scenarios at the measured rate (whatever it is, reported honestly).

---

### Phase 6: Serving, dashboard and drift monitoring (1 day)

**Skills showcased:** presenting results with visualization, putting models into production, monitoring

**Tasks**
1. `api/main.py` (FastAPI, pydantic v2), with input validated against the pandera contract:
   - `GET /health`
   - `POST /score`: risk score for runs, plus the top contributing sensors (SHAP)
   - `POST /rootcause`: ranked suspects for a time window
   - `POST /report`: copilot report (live or replay)
   - `GET /drift`: PSI summary
2. `monitoring/drift.py`: PSI per sensor between the reference window (train) and a recent window, with alert levels at 0.1 and 0.25, and a missingness-rate drift check. Add a CLI command and a scheduled job script.
3. Streamlit app (`app/`), Plotly charts, multipage:
   1. **Overview:** data health, failure-rate trend, drift alerts
   2. **Model & limits:** PR curve with CI band, calibration, gains chart, cost-sensitivity slider (move the cost ratio and see the threshold and savings change)
   3. **Root-cause explorer:** suspects table, cluster view, pass vs fail distributions, onset timeline
   4. **Benchmark:** hit@k vs effect size, MDE, null false alarms (the "how much to trust this" page)
   5. **Copilot:** generate a report and show each claim with its verified badge and evidence
4. The app reads only from `reports/` and the API. No retraining inside the UI.

**Acceptance checks:** `make api` and `make app` run locally; the API has tests through FastAPI's TestClient; the dashboard loads in under 3 seconds on cached artifacts.

---

### Phase 7: Productionization, CI and documentation (1 day)

**Skills showcased:** ML industrialization, peer review and quality, documentation

**Tasks**
1. ~~Multi-stage `Dockerfile` (non-root user, slim image); `docker-compose.yml` with `api`, `app` and `mlflow`.~~ *(skipped per scope override)*
2. `.github/workflows/ci.yml`:
   - `ruff` + `mypy src/`
   - `pytest` with coverage (contract, leakage, method, verifier and API tests)
   - **benchmark smoke gate** (Phase 4)
   - agent tests in **replay** mode
   - ~~Docker build~~ *(skipped)*

   Add a PR template with a review checklist.
3. `docs/MODEL_CARD.md`: intended use, data, metrics with CIs, calibration, limits (from the benchmark), ethical and operational risks (false alarms cost inspection time; not a causal proof).
4. `docs/ARCHITECTURE.md`: one diagram (Mermaid) covering data → contract → pipeline → models/rootcause → benchmark → API/UI → copilot → verifier.
5. `scripts/render_readme.py` fills the README result tables from `reports/metrics/*.json`.
6. ~~Public demo deploy~~ *(skipped per scope override)*
7. README structure:
   - one-line value proposition + demo GIF
   - problem (Michelin-style process root cause), approach diagram
   - **results table** (generated)
   - **limits** (generated)
   - how to run (`make all`), repo map, what I'd do next

**Acceptance checks:** CI is green on GitHub; a fresh clone + `make all` reproduces every number in the README.

---

### Phase 8: Resume and interview kit (0.5 day)

**Tasks**
1. Generate resume bullets from `reports/metrics/*.json` (templates below). Every bracket gets a real value.
2. Write `docs/INTERVIEW_NOTES.md` with the 2-minute walkthrough and answers to the questions below.

**Resume bullet templates**
- *ProcessLens: Manufacturing Root-Cause Analyzer + LLM Copilot* | Python, pandas, scikit-learn, LightGBM, SHAP, statsmodels, MLflow, LangGraph, Gemini, FastAPI, Streamlit
- Built a leakage-safe defect-prediction pipeline on UCI SECOM ([N] runs, [S] sensors, [B]% failure rate) with time-ordered validation and pandera data contracts. Reached PR-AUC [X] (95% CI [a–b]) vs a [B] base rate; a cost-based inspection policy catches [Y]% of failures while inspecting [Z]% of runs.
- Designed a planted-fault benchmark over [M] scenarios to validate root-cause ranking (Mann–Whitney + BH-FDR, stability selection, SHAP, correlated-sensor clustering). Consensus ranking found [H]% of planted causes in the top 5 at ≥[E]σ effect, with a [F]% false-alarm rate on null scenarios; documented a minimum detectable effect of [MDE]σ.
- Built a LangGraph copilot that writes structured root-cause reports only from tool outputs. A numeric verifier and abstention rules reached [V]% numeric faithfulness and [A]% correct abstention across [K] eval scenarios (prompt v2 vs v1: [+Δ]). Served via FastAPI + Streamlit with drift monitoring and CI quality gates.

**Interview questions to prepare**
1. Why PR-AUC and not accuracy or ROC-AUC on a 6.6% base rate?
2. Why a time-ordered split? Show me how you proved there's no leakage.
3. Your PR-AUC is modest. Why is the project still useful? (Point to the inspection policy, the benchmark and the honest limits.)
4. How do you handle 100+ highly correlated sensors? Why report clusters?
5. Can this prove causality? (No. Observational data; it ranks suspects for engineers to test.)
6. How realistic is the planted-fault benchmark, and where could it mislead you?
7. Why doesn't the LLM pick the root cause? How do you stop it from making up numbers?
8. What would change in a real Michelin plant: data volume, streaming, the plant historian, people in the loop?

---

## 6. Schedule and cut lines

| Plan | Phases | Days |
|---|---|---|
| **Minimum strong version (about 5 days)** | 0, 1, 2, 3, 4 + a README with the generated results table | ~5 |
| **Full version (about 8–9 days)** | All phases | ~8.5 |

If time runs short, cut in this order: the public demo deploy → the drift page → the optional RAG → API endpoints beyond `/rootcause`. **Never cut** the leakage tests, the benchmark or the limits. Those are what make this a data science project and not a demo.

## 7. Pitfalls to avoid (Claude: check these at every phase)

- Resampling (SMOTE/undersampling) or feature selection before the split.
- Tuning on the test set, or looking at it more than once.
- Calling sensor rankings "causes". Use "suspects" or "associated with".
- Reporting a single number without a CI or the base rate.
- Letting the LLM see raw rows, or allowing unverified numbers into the report.
- Hand-written metrics in the README.
- Over-engineering: no Kubernetes or Terraform here. The owner already shows those in other projects. Keep this one about data science rigor and clear communication.
