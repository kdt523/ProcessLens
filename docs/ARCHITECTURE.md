# Architecture

```mermaid
flowchart LR
    subgraph Data
        UCI[(UCI SECOM<br/>ucimlrepo / zip)] --> ING[ingest.py<br/>checksums, sort by time]
        ING --> PQ[(secom.parquet)]
        PQ --> CON{{pandera contract<br/>validated on every load}}
        CON --> AUD[audit.py<br/>missingness, constants,<br/>correlation, drift in time]
    end

    subgraph Modelling["Defect prediction (Phase 2)"]
        CON --> SPL[time-ordered split<br/>train / val / test]
        SPL --> PIPE[sklearn Pipeline<br/>drop → impute+indicators → scale → model<br/>fit on train only]
        PIPE --> SEL[rolling-origin CV<br/>model selection]
        SEL --> CAL[Platt calibration<br/>on validation]
        CAL --> EVAL[test once:<br/>bootstrap CIs, paired bootstrap]
        EVAL --> POL[inspection policy<br/>gains, cost model, sensitivity]
        CAL --> MLF[(MLflow<br/>runs + registry)]
    end

    subgraph RootCause["Root-cause engine (Phase 3)"]
        CON --> UNI[Mann–Whitney + BH-FDR]
        CON --> STB[L1 stability selection]
        CON --> SHP[LightGBM SHAP<br/>time-ordered folds]
        CON --> CLU[Spearman clustering]
        UNI & STB & SHP & CLU --> AGG[consensus MRR<br/>cluster roll-up<br/>evidence grade]
        AGG --> TMP[onset / bin trends]
    end

    subgraph Bench["Planted-fault benchmark (Phase 4)"]
        CON --> SYN[synthetic labels from<br/>known planted causes]
        SYN --> AGG
        AGG --> BM[hit@k, MDE,<br/>null false alarms]
        BM --> LIM[(docs/LIMITS.md)]
    end

    subgraph Copilot["Copilot (Phase 5)"]
        AGG --> TOOLS[typed tools<br/>summaries only, no raw rows]
        LIM --> TOOLS
        EVAL --> TOOLS
        TOOLS --> G[LangGraph<br/>plan → gather → draft]
        G --> LLM{{Gemini Flash<br/>record / replay}}
        G --> VER[verifier<br/>every number traced<br/>to a tool call]
        VER -->|problems| REV[revise ≤ 2]
        REV --> VER
        TOOLS -->|no strong/moderate suspect| ABS[abstain]
        VER --> REP[(verified report)]
        ABS --> REP
    end

    subgraph Serving["Serving (Phase 6)"]
        API[FastAPI<br/>/score /rootcause /report /drift]
        UI[Streamlit dashboard]
        DRIFT[PSI drift monitor]
    end

    POL & AGG & BM & REP & DRIFT --> JSON[(reports/metrics/*.json)]
    JSON --> UI
    JSON --> README[render_readme.py<br/>README, model card, resume]
    CAL --> API
    AGG --> API
    REP --> API
    API --> UI
```

## Design rules the diagram encodes

- **One source of truth for numbers.** Every metric is written by code to `reports/metrics/*.json`. The README, model card, dashboard and resume bullets are rendered from those files.
- **The contract guards every entry point.** The parquet is validated on load, and API scoring requests are validated against the sensor contract.
- **Leakage boundaries.** Everything learned from data in the predictive path sits inside the `Pipeline` and is fit on training rows. The test window is evaluated once.
- **The LLM is downstream of the statistics.** It only reads tool summaries and writes text. It cannot change rankings or evidence grades, and the verifier blocks any number that is not in the tool log.
- **Measured trust.** The benchmark measures the root-cause engine's recall, false alarms and minimum detectable effect, and the copilot reports those limits.
