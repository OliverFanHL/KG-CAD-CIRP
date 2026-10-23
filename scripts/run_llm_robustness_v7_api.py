"""Durable, budget-bounded Responses API runner for a frozen visual study.

Default invocation validates and freezes every job without reading an API key.
Only --execute enables network requests. Resume skips every terminal record,
including failed calls, and refuses attempts without a durable response record.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.openai.com/v1/responses"
INPUT_LIMIT = 16_000
PRICING = {
    "uncached_input_per_million": 2.0,
    "cached_input_per_million": 0.1,
    "cache_write_input_per_million": 2.5,
    "reservation_input_per_million": 2.5,
    "output_per_million": 10.0,
}

MODEL_PRICING = {
    "gpt-6.1-sol": dict(PRICING),
    "gpt-6-luna": {
        "uncached_input_per_million": .1, "cached_input_per_million": .01,
        "cache_write_input_per_million": .125, "reservation_input_per_million": .125,
        "output_per_million": .5,
    },
}
FORBIDDEN_PAYLOAD_KEYS = {
    "job_id", "family_id", "part_id", "rank", "ranks", "ranking",
    "rankings", "grade", "grades", "relevance_grade", "ground_truth",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input(value: str, study_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_path = (ROOT / path).resolve()
    local_path = (study_dir / path).resolve()
    if root_path.is_file() and local_path.is_file() and root_path != local_path:
        raise ValueError(f"Ambiguous relative input path: {value}")
    return root_path if root_path.is_file() else local_path


def inspect_payload(value: Any, location: str = "instruction_payload") -> None:
    """Keep local identity, result rank and prior judgment metadata off the API."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"Forbidden API payload metadata: {location}.{key}")
            inspect_payload(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            inspect_payload(child, f"{location}[{index}]")


def request_bytes(study: dict, job: dict, image_path: Path) -> bytes:
    image = image_path.read_bytes()
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Expected PNG source: {image_path}")
    if digest(image) != job["image_sha256"].lower():
        raise ValueError(f"Changed image source for job {job['job_id']}")
    body = {
        "model": job["model"],
        "store": False,
        "max_output_tokens": study.get("max_output_tokens", 1200),
        "reasoning": {"effort": study.get("reasoning_effort", "low")},
        "instructions": job.get("instructions", study["instructions"]),
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": canonical(job["instruction_payload"]).decode("utf-8")},
            {"type": "input_image", "detail": "auto",
             "image_url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")},
        ]}],
        "text": {"format": {
            "type": "json_schema", "name": "predicate_evidence", "strict": True,
            "schema": job.get("schema", study["schema"]),
        }},
    }
    return canonical(body)


def schema_errors(value: Any, schema: dict, root: dict | None = None,
                  location: str = "$") -> list[str]:
    """Validate the strict JSON-schema subset used for structured evidence.

    Unsupported assertion keywords fail visibly rather than silently passing.
    Annotation keywords (including format) do not assert semantic correctness.
    """
    root = schema if root is None else root
    if isinstance(schema, bool):
        return [] if schema else [f"{location}: false schema"]
    supported = {
        "$ref", "$defs", "definitions", "$schema", "$id", "title", "description",
        "default", "examples", "format", "type", "enum", "const", "properties",
        "required", "additionalProperties", "items", "prefixItems", "minItems",
        "maxItems", "uniqueItems", "minProperties", "maxProperties", "minimum",
        "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "anyOf", "oneOf", "allOf", "not",
    }
    unsupported = sorted(set(schema) - supported)
    if unsupported:
        return [f"{location}: unsupported schema assertions {unsupported}"]
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return [f"{location}: only local schema references are supported"]
        target: Any = root
        try:
            for item in ref[2:].split("/"):
                target = target[item.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            return [f"{location}: unresolved schema reference"]
        return schema_errors(value, target, root, location)
    errors: list[str] = []
    for option in ("anyOf", "oneOf", "allOf"):
        if option in schema:
            matches = sum(not schema_errors(value, item, root, location) for item in schema[option])
            if ((option == "anyOf" and matches == 0)
                    or (option == "oneOf" and matches != 1)
                    or (option == "allOf" and matches != len(schema[option]))):
                errors.append(f"{location}: {option} mismatch")
    if "not" in schema and not schema_errors(value, schema["not"], root, location):
        errors.append(f"{location}: forbidden schema match")
    types = schema.get("type")
    checks = {
        "null": value is None, "boolean": isinstance(value, bool),
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }
    if types is not None:
        types = [types] if isinstance(types, str) else types
        if not any(checks.get(item, False) for item in types):
            return errors + [f"{location}: expected type {types}"]
    if "enum" in schema and not any(canonical(value) == canonical(item) for item in schema["enum"]):
        errors.append(f"{location}: value outside enum")
    if "const" in schema and canonical(value) != canonical(schema["const"]):
        errors.append(f"{location}: const mismatch")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{location}: missing required key {key}")
        for key, child in value.items():
            if key in properties:
                errors.extend(schema_errors(child, properties[key], root, f"{location}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{location}: unexpected key {key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(schema_errors(child, schema["additionalProperties"], root, f"{location}.{key}"))
        for key, predicate in (("minProperties", lambda n: len(value) < n),
                               ("maxProperties", lambda n: len(value) > n)):
            if key in schema and predicate(schema[key]):
                errors.append(f"{location}: {key} mismatch")
    elif isinstance(value, list):
        for key, predicate in (("minItems", lambda n: len(value) < n),
                               ("maxItems", lambda n: len(value) > n)):
            if key in schema and predicate(schema[key]):
                errors.append(f"{location}: {key} mismatch")
        if schema.get("uniqueItems") and len({canonical(item) for item in value}) != len(value):
            errors.append(f"{location}: duplicate array items")
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(value):
            item_schema = prefix[index] if index < len(prefix) else schema.get("items", {})
            errors.extend(schema_errors(child, item_schema, root, f"{location}[{index}]"))
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            errors.append(f"{location}: string length mismatch")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{location}: pattern mismatch")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, predicate in (("minimum", lambda n: value < n), ("maximum", lambda n: value > n),
                               ("exclusiveMinimum", lambda n: value <= n),
                               ("exclusiveMaximum", lambda n: value >= n)):
            if key in schema and predicate(schema[key]):
                errors.append(f"{location}: {key} mismatch")
        if "multipleOf" in schema and not math.isclose(value / schema["multipleOf"], round(value / schema["multipleOf"]), abs_tol=1e-10):
            errors.append(f"{location}: multipleOf mismatch")
    return errors


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete durable ledger; manual audit required: {path.name}")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"Blank ledger line {path.name}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid ledger line {path.name}:{line_number}")
        rows.append(value)
    return rows


class Ledger:
    def __init__(self, directory: Path):
        self.attempts_path = directory / "api_attempts.jsonl"
        self.responses_path = directory / "api_responses.jsonl"
        self.lock = threading.Lock()

    def append(self, path: Path, value: dict) -> None:
        data = canonical(value) + b"\n"
        with self.lock:
            with path.open("ab", buffering=0) as stream:
                remaining = memoryview(data)
                while remaining:
                    written = stream.write(remaining)
                    if not written:
                        raise OSError("Short durable-ledger write")
                    remaining = remaining[written:]
                os.fsync(stream.fileno())


def validate_resume(ledger: Ledger, job_map: dict, freeze_sha: str) -> tuple[dict, float]:
    attempts: dict[str, dict] = {}
    for row in load_jsonl(ledger.attempts_path):
        job_id = row["job_id"]
        if job_id in attempts or job_id not in job_map:
            raise ValueError("Duplicate or unknown attempted job; refusing resampling")
        if row["protocol_sha256"] != freeze_sha or row["request_sha256"] != job_map[job_id]["request_sha256"]:
            raise ValueError(f"Attempt does not match frozen protocol: {job_id}")
        attempts[job_id] = row
    responses: dict[str, dict] = {}
    charged = 0.0
    for row in load_jsonl(ledger.responses_path):
        job_id = row["job_id"]
        if job_id in responses or job_id not in attempts:
            raise ValueError("Duplicate or unattempted response in ledger")
        attempt = attempts[job_id]
        if (row["attempt_id"] != attempt["attempt_id"]
                or row["request_sha256"] != attempt["request_sha256"]
                or row["protocol_sha256"] != freeze_sha):
            raise ValueError(f"Response/attempt identity mismatch: {job_id}")
        charge = row["cost_usd"]
        if not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0:
            raise ValueError("Invalid saved budget charge")
        charged += charge
        responses[job_id] = row
    ambiguous = sorted(set(attempts) - set(responses))
    if ambiguous:
        raise ValueError(f"Interrupted/ambiguous API attempts; refusing resampling: {ambiguous[:10]}")
    return responses, charged


def prepare(study_path: Path, output: Path) -> tuple[dict, list[dict], str]:
    raw = study_path.read_bytes()
    study = json.loads(raw)
    jobs = study.get("jobs")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 20000:
        raise ValueError("Study must contain 1..20000 jobs")
    if study.get("model") != "per_job_frozen":
        raise ValueError("This protocol requires frozen per-job model identities")
    if not isinstance(study.get("instructions"), str) or not study["instructions"].strip():
        raise ValueError("Missing nonempty instructions")
    if not isinstance(study.get("schema"), dict):
        raise ValueError("Missing global JSON schema")
    max_tokens = study.get("max_output_tokens", 1200)
    concurrency = study.get("concurrency", 6)
    budget = study.get("max_usd", 75)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 20_000:
        raise ValueError("max_output_tokens must be in 1..20000")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be in 1..16")
    if not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 0 < budget <= 75:
        raise ValueError("max_usd must be positive and at most 75")
    effort = study.get("reasoning_effort", "low")
    if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning_effort")
    source_files = []
    for item in study.get("required_files", []):
        path = resolve_input(item["path"], study_path.parent)
        actual = file_digest(path)
        if actual != item["sha256"].lower():
            raise ValueError(f"Changed required source: {path}")
        source_files.append({"path": str(path), "sha256": actual})
    prepared: list[dict] = []
    seen: set[str] = set()
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in seen:
            raise ValueError("Job identifiers must be nonempty and unique")
        seen.add(job_id)
        if not isinstance(job.get("instruction_payload"), dict):
            raise ValueError(f"Missing instruction_payload for {job_id}")
        inspect_payload(job["instruction_payload"])
        if job.get("model") not in MODEL_PRICING:
            raise ValueError("Unapproved per-job model")
        expected = job.get("predicate_ids")
        if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
            raise ValueError(f"Missing/duplicate predicate_ids for {job_id}")
        if not all(isinstance(item, str) and item for item in expected):
            raise ValueError(f"Invalid predicate identifier for {job_id}")
        schema = job.get("schema", study["schema"])
        if not isinstance(schema, dict):
            raise ValueError(f"Missing schema for {job_id}")
        path = resolve_input(job["image_path"], study_path.parent)
        request_sha = digest(request_bytes(study, job, path))
        prepared.append({
            "job_id": job_id, "family_id": job.get("family_id"), "part_id": job.get("part_id"),
            "image_path": str(path), "image_sha256": job["image_sha256"].lower(),
            "predicate_ids": expected,
            "instruction_payload_sha256": digest(canonical(job["instruction_payload"])),
            "instructions_sha256": digest(job.get("instructions", study["instructions"]).encode("utf-8")), "model": job["model"],
            "schema_sha256": digest(canonical(schema)), "request_sha256": request_sha,
        })
    expected_freeze = {
        "format": "llm_geometry_api_freeze_v1", "study_path": str(study_path),
        "study_sha256": digest(raw), "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_digest(Path(__file__).resolve()), "endpoint": ENDPOINT,
        "model": study["model"], "max_output_tokens": max_tokens,
        "reasoning_effort": effort, "concurrency": concurrency, "max_usd": budget,
        "input_token_reservation": INPUT_LIMIT, "pricing_usd": PRICING, "model_pricing_usd": MODEL_PRICING,
        "evaluation_freeze_sha256": file_digest(output / "evaluation_freeze.json"),
        "required_files": source_files, "jobs": prepared,
    }
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "protocol_freeze.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_bytes())
        comparison = {key: value for key, value in frozen.items() if key != "created_utc"}
        if canonical(comparison) != canonical(expected_freeze):
            raise ValueError("Changed study, runner, request, image or protocol; refusing continuation")
    else:
        if any((output / name).exists() for name in ("api_attempts.jsonl", "api_responses.jsonl")):
            raise ValueError("Ledgers exist without a protocol freeze; manual audit required")
        frozen = {**expected_freeze, "created_utc": now()}
        with freeze_path.open("xb", buffering=0) as stream:
            stream.write(canonical(frozen) + b"\n")
            os.fsync(stream.fileno())
    return study, prepared, file_digest(freeze_path)


def usage_cost(usage: Any, reserve: float, model: str = "gpt-6.1-sol") -> tuple[float, dict, str]:
    prices = MODEL_PRICING[model]
    if not isinstance(usage, dict) or not all(key in usage for key in ("input_tokens", "output_tokens")):
        return reserve, {}, "reservation_usage_unavailable"
    count_in, count_out = usage["input_tokens"], usage["output_tokens"]
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    # Cache writes are charged separately only if the API reports their token count.
    written = details.get("cache_write_tokens", details.get("cache_creation_tokens", 0))
    for value in (count_in, count_out, cached, written):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return reserve, {}, "reservation_invalid_usage"
    if cached + written > count_in:
        return reserve, {}, "reservation_invalid_usage"
    uncached = count_in - cached - written
    cost = (uncached * prices["uncached_input_per_million"]
            + cached * prices["cached_input_per_million"]
            + written * prices["cache_write_input_per_million"]
            + count_out * prices["output_per_million"]) / 1_000_000
    return cost, {"input_tokens": count_in, "cached_input_tokens": cached,
                  "cache_write_input_tokens": written, "output_tokens": count_out}, "reported_usage_estimate"


def output_text(response: dict) -> str:
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    texts.append(block.get("text", ""))
    return "".join(texts)


def redact(message: str, api_key: str) -> str:
    message = message.replace(api_key, "[REDACTED]") if api_key else message
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    return message[:200]


class Budget:
    def __init__(self, maximum: float, charged: float, reservation: float):
        self.maximum, self.charged, self.reservation = maximum, charged, reservation
        self.pending = 0.0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_reason: str | None = None
        self.consecutive_transport_errors = 0

    def halt(self, reason: str) -> None:
        with self.lock:
            if not self.stop_reason:
                self.stop_reason = reason
            self.stop.set()

    def finish(self, cost: float, transport_error: bool) -> None:
        with self.lock:
            self.pending = max(0.0, self.pending - self.reservation)
            self.charged += cost
            self.consecutive_transport_errors = self.consecutive_transport_errors + 1 if transport_error else 0
            if self.consecutive_transport_errors >= 3 and not self.stop_reason:
                self.stop_reason = "three_consecutive_transport_errors"
                self.stop.set()


def run_job(study: dict, job: dict, meta: dict, protocol_sha: str,
            ledger: Ledger, budget: Budget, api_key: str) -> dict | None:
    if budget.stop.is_set():
        return None
    body = request_bytes(study, job, Path(meta["image_path"]))
    if digest(body) != meta["request_sha256"]:
        budget.halt("request_source_changed")
        raise ValueError("Request changed after preflight")
    attempt = {
        "job_id": meta["job_id"], "family_id": meta["family_id"], "part_id": meta["part_id"],
        "attempt_id": str(uuid.uuid4()), "started_utc": now(),
        "protocol_sha256": protocol_sha, "request_sha256": meta["request_sha256"],
        "image_sha256": meta["image_sha256"], "model_requested": job["model"],
        "reserved_usd": budget.reservation,
    }
    with budget.lock:
        if budget.stop.is_set():
            return None
        if budget.charged + budget.pending + budget.reservation > budget.maximum + 1e-12:
            budget.stop_reason = "budget_reservation_limit"
            budget.stop.set()
            return None
        # The reservation and durable write-ahead precede every possible API call.
        ledger.append(ledger.attempts_path, attempt)
        budget.pending += budget.reservation
    started = time.monotonic()
    record = {**attempt, "status": "api_error", "model_resolved": None,
              "response_id": None, "response_status": None, "output_text": "",
              "output_json": None, "output_schema_valid": False, "validation_errors": [],
              "usage": None, "usage_tokens": {}, "cost_usd": budget.reservation,
              "cost_basis": "reservation_failed_call", "raw_response": None,
              "error": None, "http_status": None}
    transport_error = False
    try:
        request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": "Bearer " + api_key, "Content-Type": "application/json",
        })
        with urllib.request.urlopen(request, timeout=study.get("timeout_seconds", 180)) as stream:
            record["http_status"] = stream.status
            response = json.loads(stream.read())
        record["raw_response"] = response
        record["model_resolved"] = response.get("model")
        record["response_id"] = response.get("id")
        record["response_status"] = response.get("status")
        record["usage"] = response.get("usage")
        cost, tokens, basis = usage_cost(record["usage"], budget.reservation, job["model"])
        record.update(cost_usd=cost, usage_tokens=tokens, cost_basis=basis)
        record["output_text"] = output_text(response)
        try:
            parsed = json.loads(record["output_text"], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            record["output_json"] = parsed
            record["validation_errors"] = schema_errors(parsed, job.get("schema", study["schema"]))
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            record["validation_errors"] = ["JSON/schema validation: " + redact(str(error), api_key)]
        record["output_schema_valid"] = not record["validation_errors"]
        record["status"] = ("completed" if record["output_schema_valid"] and response.get("status") == "completed"
                            else "model_output_invalid")
        if tokens.get("input_tokens", 0) > INPUT_LIMIT:
            record["input_token_reservation_exceeded"] = True
            budget.halt("input_token_reservation_exceeded")
        if tokens.get("output_tokens", 0) > study.get("max_output_tokens", 1200):
            record["output_token_reservation_exceeded"] = True
            budget.halt("output_token_reservation_exceeded")
    except urllib.error.HTTPError as error:
        transport_error = True
        record["http_status"] = error.code
        record["error"] = redact(error.read(2048).decode("utf-8", errors="replace"), api_key)
        if 400 <= error.code < 500:
            budget.halt(f"http_{error.code}_no_retry")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        transport_error = True
        record["error"] = redact(str(error), api_key)
    except Exception as error:
        # No retry follows an exception after the durable attempt record.
        record["status"] = "runner_error"
        record["error"] = redact(type(error).__name__ + ": " + str(error), api_key)
        budget.halt("runner_error_after_attempt")
    record["finished_utc"] = now()
    record["elapsed_seconds"] = time.monotonic() - started
    # If this durable append fails, resume will see the unresolved attempt and stop.
    ledger.append(ledger.responses_path, record)
    budget.finish(record["cost_usd"], transport_error)
    return record


def summary(rows: dict, total: int, budget: Budget | None, freeze_sha: str,
            mode: str, charged: float = 0.0) -> dict:
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"mode": mode, "total_jobs": total, "terminal_jobs": len(rows),
            "remaining_jobs": total - len(rows), "status_counts": counts,
            "charged_usd": budget.charged if budget else charged,
            "outstanding_reservations_usd": budget.pending if budget else 0,
            "stop_reason": budget.stop_reason if budget else None,
            "protocol_sha256": freeze_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/llm_robustness_v7/study.json")
    parser.add_argument("--output", type=Path, help="Default: study directory")
    parser.add_argument("--preflight", action="store_true", help="Validate/freeze all jobs without network (default)")
    parser.add_argument("--execute", action="store_true", help="Enable paid API calls")
    parser.add_argument("--limit", type=int, help="Maximum remaining jobs to attempt this invocation")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name; no .env files are read")
    args = parser.parse_args(argv)
    if args.preflight and args.execute:
        parser.error("Choose either --preflight or --execute")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    study_path = args.study.resolve()
    output = (args.output or study_path.parent).resolve()
    # One runner owns each output directory, including concurrent preflights.
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".api_runner.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Runner lock exists; verify no runner is active before manual lock removal")
    try:
        os.write(lock_fd, canonical({"pid": os.getpid(), "created_utc": now()}))
        os.fsync(lock_fd)
        study, metas, freeze_sha = prepare(study_path, output)
        ledger = Ledger(output)
        job_map = {item["job_id"]: item for item in metas}
        rows, charged = validate_resume(ledger, job_map, freeze_sha)
        if charged > study.get("max_usd", 75) + 1e-12:
            raise ValueError("Saved charges already exceed study budget")
        if not args.execute:
            print(json.dumps(summary(rows, len(metas), None, freeze_sha, "preflight", charged), sort_keys=True), flush=True)
            return 0
        # API credentials are accessed only after explicit --execute and all checks.
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
        reservation = (INPUT_LIMIT * PRICING["reservation_input_per_million"]
                       + study.get("max_output_tokens", 1200) * PRICING["output_per_million"]) / 1_000_000
        budget = Budget(study.get("max_usd", 75), charged, reservation)
        source_jobs = {item["job_id"]: item for item in study["jobs"]}
        remaining = [meta for meta in metas if meta["job_id"] not in rows]
        if args.limit is not None:
            remaining = remaining[:args.limit]
        cursor = 0
        completed_here = 0
        print(json.dumps(summary(rows, len(metas), budget, freeze_sha, "execute_started"), sort_keys=True), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=study.get("concurrency", 6)) as executor:
            active: dict[concurrent.futures.Future, str] = {}
            while active or (cursor < len(remaining) and not budget.stop.is_set()):
                while (len(active) < study.get("concurrency", 6) and cursor < len(remaining)
                       and not budget.stop.is_set()):
                    meta = remaining[cursor]
                    cursor += 1
                    future = executor.submit(run_job, study, source_jobs[meta["job_id"]], meta,
                                             freeze_sha, ledger, budget, api_key)
                    active[future] = meta["job_id"]
                if not active:
                    break
                finished, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    del active[future]
                    try:
                        row = future.result()
                    except BaseException:
                        budget.halt("worker_or_ledger_failure")
                        raise
                    if row is not None:
                        rows[row["job_id"]] = row
                        completed_here += 1
                        if completed_here % 10 == 0:
                            print(json.dumps(summary(rows, len(metas), budget, freeze_sha, "progress"), sort_keys=True), flush=True)
        result = summary(rows, len(metas), budget, freeze_sha, "execute_finished")
        result["terminal_jobs_this_invocation"] = completed_here
        print(json.dumps(result, sort_keys=True), flush=True)
        return 3 if budget.stop.is_set() else 0
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        # Never print environment values or request headers.
        print(json.dumps({"status": "stopped", "error": str(error)[:300]}), file=sys.stderr, flush=True)
        raise SystemExit(2)
