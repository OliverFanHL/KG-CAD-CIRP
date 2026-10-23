"""Official CLM typed-decision inference on the frozen CAD semantic protocol."""
from __future__ import annotations

import hashlib
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .laya_client import question_dict
from .semantic_state import QuantileSchema, build_compact_semantic_state, build_semantic_state
from .utils import append_jsonl, read_jsonl, sha256_json, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_clm_for_dataframe(
    df: pd.DataFrame, schema: QuantileSchema, spec: dict, output_jsonl: Path,
    *, checkpoint: Path, model: str = "clm-latest",
    encoder_backend: str = "vllm",
    encoder_model: str = "Qwen/Qwen3-8B",
    encoder_device: str = "auto", encoder_batch_size: int = 4,
    emb_url: str = "http://127.0.0.1:8090/v1/embeddings",
    emb_model: str = "qwen3-8b", encoder_revision: str = "",
    timeout_seconds: float = 300,
    retry_attempts: int = 3, include_numeric: bool = True,
    include_profiles: bool = False, state_variant: str = "compact",
    dry_run: bool = False, max_new_samples: int | None = None,
    max_runtime_seconds: float | None = None,
) -> dict:
    if state_variant not in {"compact", "full"}:
        raise ValueError(f"Unknown state variant: {state_variant}")
    if model != "clm-latest":
        raise ValueError("The matched CLM candidate requires the trained clm-latest head")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be at least 1")
    if encoder_backend not in {"vllm", "transformers"}:
        raise ValueError(f"Unknown CLM encoder backend: {encoder_backend}")
    if max_new_samples is not None and max_new_samples < 1:
        raise ValueError("max_new_samples must be positive")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLM head checkpoint missing: {checkpoint}")
    checkpoint_hash = file_sha256(checkpoint)
    head_stamp = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    questions = question_dict(spec)
    class_id = spec["classification"].get("id", "part_class")
    existing = read_jsonl(output_jsonl)
    if len({str(r["sample_id"]) for r in existing}) != len(existing):
        raise ValueError("Existing CLM results contain duplicate sample IDs")
    if any(r.get("split") not in {"train", "val", "test"} for r in existing):
        raise ValueError("Existing CLM results have invalid splits")
    if any(r.get("model") != model or r.get("checkpoint_sha256") != checkpoint_hash or
           r.get("encoder_revision") != encoder_revision or
           r.get("encoder_backend", "vllm") != encoder_backend or
           r.get("encoder_model", "Qwen/Qwen3-8B") != encoder_model for r in existing):
        raise ValueError("Existing CLM results use a different model, encoder, or head checkpoint")
    done = {str(r["sample_id"]): r["request_hash"] for r in existing}
    reusable = {r["request_hash"]: r for r in existing}
    rows = []
    builder = build_compact_semantic_state if state_variant == "compact" else build_semantic_state
    for _, row in df.iterrows():
        state = builder(row, schema, include_numeric_context=include_numeric,
                        include_profile_arrays=include_profiles)
        request_hash = sha256_json({"model": model, "state": state, "questions": spec})
        sid = str(row["sample_id"])
        if sid in done and done[sid] != request_hash:
            raise ValueError(f"Existing CLM request differs for {sid}")
        rows.append((row, state, request_hash))
    if set(done) - {str(row["sample_id"]) for row, _, _ in rows}:
        raise ValueError("Existing CLM results contain samples outside the frozen cohort")
    identity = {str(row["sample_id"]): row for row, _, _ in rows}
    for record in existing:
        row = identity[str(record["sample_id"])]
        if (str(record.get("split")) != str(row["split"]) or
                str(record.get("true_label")) != str(row["label"])):
            raise ValueError(f"Existing CLM split or label differs for {record['sample_id']}")
    pending = sum(str(row["sample_id"]) not in done for row, _, _ in rows)
    metadata = {"cohort_size": len(rows), "completed": len(done), "pending": pending,
                "unique_pending_requests": len({h for row, _, h in rows if str(row["sample_id"]) not in done and h not in reusable}),
                "model": model, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash, "state_variant": state_variant,
                "encoder_backend": encoder_backend, "encoder_model": encoder_model,
                "served_encoder_model": emb_model, "encoder_revision": encoder_revision,
                "embedding_url": emb_url,
                "question_ids": list(questions)}
    if dry_run:
        return metadata
    try:
        from clm import Engine
    except ImportError as exc:
        raise ImportError("Install the official client with `pip install contrastive-lm`.") from exc
    package_version = version("contrastive-lm")
    if encoder_backend == "transformers":
        from .clm_transformers_embedder import TransformersEmbedder
        embedder = TransformersEmbedder(model=encoder_model, revision=encoder_revision,
                                       device=encoder_device, batch_size=encoder_batch_size)
    else:
        from clm.embedder import Embedder
        embedder = Embedder(url=emb_url, model=emb_model, timeout=timeout_seconds)
        if not embedder.healthy():
            raise RuntimeError(f"Qwen3-8B pooling server is unavailable at {emb_url}")
    engine = Engine(embedder=embedder, checkpoint=str(checkpoint),
                    device=embedder.device if encoder_backend == "transformers" else None,
                    action_cache=0)
    if model not in {entry["name"] for entry in engine.models()}:
        raise ValueError(f"CLM engine does not serve {model}")
    processed = 0
    run_started = time.monotonic()
    try:
        for row, state, request_hash in tqdm(rows, desc="CLM decisions"):
            if max_new_samples is not None and processed >= max_new_samples:
                break
            if max_runtime_seconds is not None and time.monotonic() - run_started >= max_runtime_seconds:
                break
            if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != head_stamp:
                raise ValueError("CLM head checkpoint changed during inference")
            sid = str(row["sample_id"])
            if sid in done:
                continue
            origin = reusable.get(request_hash)
            if origin is None:
                for attempt in range(retry_attempts):
                    start = time.perf_counter()
                    try:
                        response = engine.answer(state=state, questions=questions, model=model)
                        elapsed = time.perf_counter() - start
                        break
                    except Exception as exc:
                        # Only transient transport/server failures are retried.
                        transient = type(exc).__name__ in {"EmbedderError", "ConnectionError", "Timeout"}
                        if not transient or attempt == retry_attempts - 1:
                            raise
                        time.sleep(min(2 ** attempt, 8))
                if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != head_stamp:
                    raise ValueError("CLM head checkpoint changed during inference")
                if response["model"] != model or set(response["answers"]) != set(questions):
                    raise ValueError("CLM returned another model or incomplete answers")
                answer = response["answers"][class_id]
                probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
                if set(probs) != set(spec["classification"]["criteria"]):
                    raise ValueError("CLM Choice probabilities do not match the frozen classes")
                signature = {qid: float(response["answers"][qid]["noul"])
                             for qid in spec.get("semantic_signature", {})}
                if any(not 0 <= value <= 1 for value in [*probs.values(), *signature.values()]):
                    raise ValueError("CLM returned invalid probabilities")
                result = {"predicted_class": str(answer["choice"]),
                          "class_confidence": float(answer["confidence"]),
                          "class_probabilities": probs, "semantic_signature": signature,
                          "usage": response["usage"],
                          "latency_seconds": elapsed,
                          "cache_source": "local_inference"}
            else:
                result = {k: origin[k] for k in ("predicted_class", "class_confidence",
                                                   "class_probabilities", "semantic_signature", "usage")}
                result.update({"latency_seconds": None,
                               "cache_source": "payload_reuse",
                               "reused_from_sample_id": origin["sample_id"]})
            record = {"sample_id": sid, "split": str(row["split"]),
                      "true_label": str(row["label"]),
                      "source_label": str(row.get("source_label", "")),
                      "relative_path": str(row.get("relative_path", "")),
                      "absolute_path": str(row.get("absolute_path", "")),
                      "request_hash": request_hash, "model": model,
                      "checkpoint_sha256": checkpoint_hash,
                      "encoder_revision": encoder_revision,
                      "encoder_backend": encoder_backend,
                      "encoder_model": encoder_model,
                      "clm_package_version": package_version, **result}
            append_jsonl(output_jsonl, record)
            done[sid] = request_hash
            processed += 1
            reusable.setdefault(request_hash, record)
    finally:
        if encoder_backend == "transformers":
            embedder.close()
        else:
            embedder.session.close()
    metadata["completed"] = len(done)
    metadata["pending"] = len(rows) - len(done)
    write_json(output_jsonl.with_suffix(".meta.json"), metadata)
    return metadata
