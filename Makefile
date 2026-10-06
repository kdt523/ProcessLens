
.PHONY: docs setup ingest audit train rootcause benchmark benchmark-smoke agent-eval app api drift mlflow-ui test lint format all

PL = uv run processlens

setup:
	uv sync
	uv run pre-commit install

ingest:
	$(PL) ingest

audit:
	$(PL) audit

train:
	$(PL) train

rootcause:
	$(PL) rootcause

benchmark:
	$(PL) benchmark

benchmark-smoke:
	$(PL) benchmark --smoke

agent-eval:
	$(PL) agent-eval

drift:
	$(PL) drift

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

api:
	uv run uvicorn processlens.api.main:app --reload

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

all: setup ingest audit train rootcause benchmark drift docs

docs:
	uv run python scripts/render_readme.py
	uv run python scripts/render_resume.py
