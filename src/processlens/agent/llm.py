"""LLM access with record/replay so tests and CI run without an API key."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from processlens.config import PROJECT_ROOT

T = TypeVar("T", bound=BaseModel)


class MissingRecordingError(RuntimeError):
    """Replay mode found no stored response for this exact prompt."""


class LLMClient(Protocol):
    """Anything that turns (schema, system, user) into a parsed object plus usage."""

    model: str

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]:
        """Return the parsed response and a usage dict (tokens, seconds)."""
        ...


def load_env(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load ``KEY=value`` lines from ``.env`` into the environment (no override)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class GeminiClient:
    """Gemini via ``langchain-google-genai`` with JSON-schema structured output."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        load_env()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        llm = cfg["llm"]
        self.model = llm["model"]
        self._chat = ChatGoogleGenerativeAI(
            model=llm["model"],
            temperature=llm["temperature"],
            max_tokens=llm["max_output_tokens"],
            request_timeout=llm["timeout_s"],
            api_key=key,
        )

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]:
        """Call Gemini and parse into ``schema``; raises if the output does not parse."""
        t0 = time.perf_counter()
        out = self._chat.with_structured_output(schema, include_raw=True).invoke(
            [("system", system), ("human", user)]
        )
        assert isinstance(out, dict)
        if out.get("parsing_error") or out.get("parsed") is None:
            raise ValueError(f"Unparseable LLM output: {out.get('parsing_error')}")
        meta = getattr(out["raw"], "usage_metadata", None) or {}
        usage = {
            "input_tokens": int(meta.get("input_tokens", 0)),
            "output_tokens": int(meta.get("output_tokens", 0)),
            "seconds": time.perf_counter() - t0,
        }
        parsed = out["parsed"]
        return (parsed if isinstance(parsed, schema) else schema.model_validate(parsed)), usage


class RecordReplayClient:
    """Wraps a live client. ``replay`` reads stored responses; ``record`` stores new ones."""

    def __init__(
        self, mode: str, recordings: Path, live: LLMClient | None = None, model: str = "replay"
    ) -> None:
        if mode not in {"replay", "record", "live"}:
            raise ValueError(f"Unknown mode {mode!r}")
        if mode != "replay" and live is None:
            raise ValueError(f"mode {mode!r} needs a live client")
        self.mode, self.dir, self.live = mode, recordings, live
        self.model = live.model if live else model

    def key(self, schema: type[BaseModel], system: str, user: str) -> str:
        """Content hash identifying one exact request."""
        blob = json.dumps([self.model, schema.__name__, system, user]).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]:
        """Replay, record or pass through one structured call."""
        path = self.dir / f"{self.key(schema, system, user)}.json"
        if self.mode == "replay":
            if not path.exists():
                raise MissingRecordingError(f"No recording {path.name} for {schema.__name__}")
            rec = json.loads(path.read_text(encoding="utf-8"))
            return schema.model_validate(rec["parsed"]), {**rec["usage"], "replayed": True}
        assert self.live is not None
        obj, usage = self.live.structured(schema, system, user)
        if self.mode == "record":
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema": schema.__name__,
                        "model": self.model,
                        "parsed": obj.model_dump(),
                        "usage": usage,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        return obj, usage


def make_client(
    cfg: dict[str, Any], mode: str | None = None, root: Path = PROJECT_ROOT
) -> RecordReplayClient:
    """Build the configured client; ``mode`` overrides ``configs/agent.yaml``."""
    mode = mode or cfg["mode"]
    live = None if mode == "replay" else GeminiClient(cfg)
    return RecordReplayClient(mode, root / cfg["recordings_dir"], live, cfg["llm"]["model"])
