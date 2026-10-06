## What & why

<!-- One paragraph: what changes and the reason. -->

## Review checklist

- [ ] **No leakage:** every new preprocessing step is fit on training rows only, inside a `Pipeline`; splits stay time-ordered; the test set is not used for any choice.
- [ ] **No hand-typed numbers:** results come from code into `reports/metrics/*.json`; README / model card regenerated with `uv run python scripts/render_readme.py`.
- [ ] **Language:** rankings are "suspects" / "associated with", never "causes".
- [ ] **Copilot:** the LLM still sees only tool outputs; the verifier still rejects unverified numbers (tests updated if prompts or tools changed).
- [ ] **Config:** new tunables live in `configs/*.yaml`; random steps are seeded.
- [ ] **Quality:** `make lint test` passes; new modules have tests and docstrings.
- [ ] **Benchmark:** if the root-cause engine changed, `make benchmark` re-run and `docs/LIMITS.md` regenerated; smoke gate still passes.
- [ ] **Dependencies:** none added, or listed and justified here.
