"""Generated documents render from the metrics files without gaps."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_readme  # noqa: E402
import render_resume  # noqa: E402

NEEDED = ["audit", "model", "rootcause", "benchmark"]
pytestmark = pytest.mark.skipif(
    not all((ROOT / f"reports/metrics/{n}.json").exists() for n in NEEDED),
    reason="metrics not generated",
)
BAD = re.compile(r"\bnan\b|\bNone\b|\{[a-zA-Z_]+\}|\[[A-Z]{1,3}\]")


def test_readme_has_every_section_and_no_gaps() -> None:
    text = render_readme.render_readme()
    for heading in [
        "## Results",
        "### Defect prediction",
        "planted-fault benchmark",
        "## Known limits",
        "## How to run",
    ]:
        assert heading in text
    assert not BAD.search(text), BAD.search(text)


def test_model_card_renders() -> None:
    text = render_readme.render_model_card()
    assert "## Intended use" in text and "Not causal" in text
    assert not BAD.search(text)


def test_resume_brackets_all_filled() -> None:
    f = render_resume.facts()
    text = render_resume.bullets(f)
    assert not BAD.search(text), BAD.search(text)
    assert f["N"] in text and f["X"] in text


def test_interview_notes_cover_all_questions() -> None:
    text = render_resume.notes(render_resume.facts())
    assert text.count("### ") == 8 and "Two-minute walkthrough" in text
