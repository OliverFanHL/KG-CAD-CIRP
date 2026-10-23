from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .semantic_state import QuantileSchema, build_compact_semantic_state, build_semantic_state
from .utils import append_jsonl, ensure_parent, read_jsonl, sha256_json

LOGGER = logging.getLogger(__name__)


class SQLiteResponseCache:
    def __init__(self, path: str | Path):
        self.path = ensure_parent(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT response_json FROM responses WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, cache_key: str, model: str, response: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO responses(cache_key, model, response_json) VALUES(?,?,?)",
            (cache_key, model, json.dumps(response, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def _build_sdk_questions(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        from typesafe_sdk import Choice, Noul
    except ImportError as exc:
        raise ImportError("Install the current SDK with `pip install typesafe-sdk`.") from exc

    classification = spec["classification"]
    questions: dict[str, Any] = {
        classification.get("id", "part_class"): Choice(
            instructions=classification["instructions"],
            criteria=classification["criteria"],
        )
    }
    for qid, cfg in spec.get("semantic_signature", {}).items():
        questions[qid] = Noul(instructions=cfg["instructions"])
    return questions


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    result = {}
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = int(value)
    return result


def _response_to_dict(response: Any, spec: dict[str, Any]) -> dict[str, Any]:
    classification_id = spec["classification"].get("id", "part_class")
    class_answer = response.answers[classification_id]
    class_probs = {str(k): float(v) for k, v in dict(class_answer.probabilities).items()}
    signature: dict[str, float] = {}
    for qid in spec.get("semantic_signature", {}):
        signature[qid] = float(response.answers[qid].noul)
    return {
        "model": str(getattr(response, "model", "unknown")),
        "predicted_class": str(class_answer.choice),
        "class_confidence": float(class_answer.confidence),
        "class_probabilities": class_probs,
        "semantic_signature": signature,
        "usage": _usage_to_dict(getattr(response, "usage", None)),
    }


def _call_typesafe(
    state: dict[str, Any],
    spec: dict[str, Any],
    model: str,
    retry_attempts: int,
    retry_base_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set the environment variable before running."
        )
    try:
        from typesafe_sdk import (
            RetryPolicy, TypeSafeAPIConnectionError, TypeSafeAPIError,
            TypeSafeAPIResponseValidationError, TypeSafeAPITimeoutError,
            TypeSafeClient, TypeSafeError,
        )
    except ImportError as exc:
        raise ImportError("Install the current SDK with `pip install typesafe-sdk`.") from exc

    questions = _build_sdk_questions(spec)
    # Let the SDK retry transient transport errors, 408/429, and 5xx only.
    # This avoids multiplying SDK retries with a second application retry loop.
    retry = RetryPolicy(
        max_retries=max(0, retry_attempts - 1),
        backoff_initial=retry_base_seconds,
        backoff_max=max(5.0, retry_base_seconds * (2 ** max(0, retry_attempts - 2))),
        timeout=None,
    )
    started = time.perf_counter()
    try:
        with TypeSafeClient(timeout=timeout_seconds, retry=retry) as client:
            response = client.system_one(model=model, state=state, questions=questions)
    except TypeSafeAPITimeoutError as exc:
        LOGGER.error("JEV request timed out after retries (%s): %s", type(exc).__name__, exc)
        raise
    except TypeSafeAPIConnectionError as exc:
        LOGGER.error("JEV connection failed after retries (%s): %s", type(exc).__name__, exc)
        raise
    except TypeSafeAPIResponseValidationError as exc:
        LOGGER.error("JEV returned a malformed response (%s, field=%s): %s", type(exc).__name__, exc.field_path, exc)
        raise
    except TypeSafeAPIError as exc:
        LOGGER.error("JEV API error (HTTP %s, request_id=%s, %s): %s", exc.status, exc.request_id, type(exc).__name__, exc)
        raise
    except TypeSafeError as exc:
        LOGGER.error("Invalid JEV client configuration or request (%s): %s", type(exc).__name__, exc)
        raise
    try:
        result = _response_to_dict(response, spec)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        LOGGER.error("JEV response has missing or invalid answer fields (%s): %s", type(exc).__name__, exc)
        raise
    result["latency_seconds"] = float(time.perf_counter() - started)
    result["sdk_version"] = version("typesafe-sdk")
    return result


def run_jev_for_dataframe(
    df,
    schema: QuantileSchema,
    question_spec: dict[str, Any],
    output_jsonl: str | Path,
    cache_db: str | Path,
    model: str = "jev-latest",
    retry_attempts: int = 5,
    retry_base_seconds: float = 1.5,
    timeout_seconds: float = 30.0,
    request_pause_seconds: float = 0.0,
    include_numeric_context: bool = True,
    include_profile_arrays: bool = True,
    force: bool = False,
    dry_run_states: str | Path | None = None,
    state_variant: str = "full",
    dry_run_stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if state_variant not in {"full", "compact"}:
        raise ValueError(f"Unknown state variant: {state_variant}")
    output_jsonl = Path(output_jsonl)
    cache = SQLiteResponseCache(cache_db)
    experiment_spec = {
            "model": model,
            "questions": question_spec,
            "semantic_schema": schema.to_dict(),
            "include_numeric_context": include_numeric_context,
            "include_profile_arrays": include_profile_arrays,
        }
    if state_variant != "full":
        experiment_spec["state_variant"] = state_variant
    experiment_hash = sha256_json(experiment_spec)

    existing = read_jsonl(output_jsonl)
    completed = {(str(x.get("sample_id")), str(x.get("request_hash"))) for x in existing}
    dry_path = Path(dry_run_states) if dry_run_states else None
    previewed = {(str(x.get("sample_id")), str(x.get("request_hash")))
                 for x in read_jsonl(dry_path)} if dry_path is not None else set()
    stats = {"selected": len(df), "invalid_descriptors": 0, "completed_responses": 0,
             "pending_responses": 0, "cached_responses": 0,
             "uncached_requests": 0, "previews_added": 0}

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Jev evaluation"):
            error = row.get("extract_error")
            if pd.notna(error) and str(error).strip():
                stats["invalid_descriptors"] += 1
                LOGGER.warning(
                    "Skipping sample %s: CAD descriptor extraction failed for %s: %s",
                    row["sample_id"], row.get("absolute_path", "unknown path"), error,
                )
                continue

            builder = build_compact_semantic_state if state_variant == "compact" else build_semantic_state
            state = builder(
                row,
                schema=schema,
                include_numeric_context=include_numeric_context,
                include_profile_arrays=include_profile_arrays,
            )
            request_payload = {"model": model, "state": state, "questions": question_spec}
            request_hash = sha256_json(request_payload)
            key = (str(row["sample_id"]), request_hash)
            if not force and key in completed:
                stats["completed_responses"] += 1
                continue

            if dry_path is not None:
                stats["pending_responses"] += 1
                if force or cache.get(request_hash) is None:
                    stats["uncached_requests"] += 1
                else:
                    stats["cached_responses"] += 1
                if key not in previewed:
                    append_jsonl(dry_path, {
                        "sample_id": str(row["sample_id"]),
                        "split": str(row["split"]),
                        "true_label": str(row["label"]),
                        "source_label": str(row.get("source_label", "")),
                        "relative_path": str(row.get("relative_path", "")),
                        "absolute_path": str(row.get("absolute_path", "")),
                        "state": state,
                        "request_hash": request_hash,
                        "experiment_hash": experiment_hash,
                    })
                    previewed.add(key)
                    stats["previews_added"] += 1
                continue

            cached = None if force else cache.get(request_hash)
            if cached is None:
                result = _call_typesafe(
                    state=state,
                    spec=question_spec,
                    model=model,
                    retry_attempts=retry_attempts,
                    retry_base_seconds=retry_base_seconds,
                    timeout_seconds=timeout_seconds,
                )
                cache.put(request_hash, model=model, response=result)
                source = "api"
            else:
                result = cached
                source = "cache"

            record = {
                "sample_id": str(row["sample_id"]),
                "true_label": str(row["label"]),
                "source_label": str(row.get("source_label", str(row.get("relative_path", "")).replace("\\", "/").split("/")[0])),
                "relative_path": str(row.get("relative_path", "")),
                "absolute_path": str(row.get("absolute_path", "")),
                "split": str(row["split"]),
                "experiment_hash": experiment_hash,
                "request_hash": request_hash,
                "cache_source": source,
                **result,
            }
            append_jsonl(output_jsonl, record)
            completed.add(key)
            if request_pause_seconds > 0 and source == "api":
                time.sleep(request_pause_seconds)
    finally:
        cache.close()

    if dry_run_stats is not None:
        dry_run_stats.update(stats)
    return read_jsonl(output_jsonl)
