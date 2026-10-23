"""Optional Laya typed-decision baseline on the exact JEV state/questions."""
from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .semantic_state import QuantileSchema, build_compact_semantic_state, build_semantic_state
from .utils import append_jsonl, read_jsonl, sha256_json


def _check_laya_context(agent, state: dict, questions: dict) -> None:
    """Fail closed if Laya would silently truncate evidence or option descriptions."""
    from laya.common import build_sequence, render_options, serialize_state

    tok = agent.tok
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "),
                    add_special_tokens=False)["input_ids"]
    for qid, question in questions.items():
        internal = {"t": question["type"], "ins": question["instructions"],
                    "crit": question.get("criteria", {})}
        options = render_options(internal)
        sizes = [1 + len(tok(" " + option.replace(tok.mask_token, " "),
                             add_special_tokens=False)["input_ids"]) for option in options]
        if max(sizes) > 49 or sum(sizes) > head_max_len - 16:
            raise ValueError(f"Laya would truncate options for {qid}; use the compact question protocol")
        sequence, markers = build_sequence(tok, state, internal, max_len, head_max_len)
        if len(markers) != len(options) or sequence[-len(state_ids) - 1:-1] != state_ids:
            raise ValueError(f"Laya would truncate CAD state for {qid} ({len(state_ids)} state tokens); "
                             "use configs/study_compact.yaml")


def question_dict(spec: dict[str, Any]) -> dict[str, Any]:
    c = spec["classification"]
    result = {c.get("id", "part_class"): {"type": "choice", "instructions": c["instructions"],
                                             "criteria": c["criteria"]}}
    result.update({qid: {"type": "noul", "instructions": cfg["instructions"]}
                  for qid, cfg in spec.get("semantic_signature", {}).items()})
    return result


def run_laya_for_dataframe(df: pd.DataFrame, schema: QuantileSchema, spec: dict[str, Any],
                           output_jsonl: Path, *, repository: str, subfolder: str,
                           revision: str | None = None, dry_run: bool = False,
                           include_numeric: bool = True, include_profiles: bool = True,
                           state_variant: str = "full") -> list[dict]:
    if not dry_run:
        try:
            import laya
        except ImportError as exc:
            raise ImportError("Laya baseline requires `pip install -e .[laya]`.") from exc
    questions = question_dict(spec)
    existing = read_jsonl(output_jsonl)
    if not dry_run and revision is None:
        if existing:
            revision = str(existing[0].get("model", "")).rsplit("@", 1)[-1]
            if not revision or revision == "None":
                raise ValueError("Existing Laya results are not pinned to a checkpoint revision")
        else:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(repository).sha
    done = {(str(r.get("sample_id")), str(r.get("request_hash"))) for r in existing}
    reusable = {str(r.get("request_hash")): r for r in existing}
    agent = None
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Laya decisions"):
        builder = build_compact_semantic_state if state_variant == "compact" else build_semantic_state
        state = builder(row, schema, include_numeric_context=include_numeric,
                        include_profile_arrays=include_profiles)
        request_hash = sha256_json({"model": f"{repository}/{subfolder}@{revision}",
                                    "state": state, "questions": spec})
        key = (str(row["sample_id"]), request_hash)
        if key in done:
            continue
        if dry_run:
            continue
        if request_hash in reusable:
            origin = reusable[request_hash]
            record = {**origin,
                      "sample_id": str(row["sample_id"]),
                      "split": str(row["split"]),
                      "true_label": str(row["label"]),
                      "source_label": str(row.get("source_label", str(row.get("relative_path", "")).replace("\\", "/").split("/")[0])),
                      "relative_path": str(row.get("relative_path", "")),
                      "absolute_path": str(row.get("absolute_path", "")),
                      "cache_source": "payload_reuse",
                      "reused_from_sample_id": str(origin["sample_id"]),
                      "latency_seconds": None,
                      "preflight_seconds": None,
                      "model_load_seconds": None}
            append_jsonl(output_jsonl, record)
            done.add(key)
            continue
        model_load_seconds = None
        if agent is None:
            load_start = time.perf_counter()
            if revision:
                from huggingface_hub import snapshot_download
                snapshot = snapshot_download(repository, revision=revision,
                                             allow_patterns=[f"{subfolder}/{name}" for name in (
                                                 "rl_agent_config.json", "model.safetensors",
                                                 "tokenizer/*", "encoder/*",
                                             )])
                agent = laya.load(str(Path(snapshot) / subfolder))
            else:
                agent = laya.load(repository, subfolder=subfolder)
            model_load_seconds = time.perf_counter() - load_start
        preflight_start = time.perf_counter()
        _check_laya_context(agent, state, questions)
        preflight_seconds = time.perf_counter() - preflight_start
        start = time.perf_counter()
        response = agent.predict(state, questions)
        elapsed = time.perf_counter() - start
        answers = response["answers"]
        class_id = spec["classification"].get("id", "part_class")
        choice = answers[class_id]
        signature = {qid: float(answers[qid]["noul"]) for qid in spec.get("semantic_signature", {})}
        record = {"sample_id": str(row["sample_id"]), "split": str(row["split"]),
                  "true_label": str(row["label"]),
                  "source_label": str(row.get("source_label", str(row.get("relative_path", "")).replace("\\", "/").split("/")[0])),
                  "relative_path": str(row.get("relative_path", "")),
                  "absolute_path": str(row.get("absolute_path", "")),
                  "request_hash": request_hash,
                  "model": f"{repository}/{subfolder}@{revision}",
                  "predicted_class": str(choice["choice"]),
                  "class_confidence": float(choice.get("confidence", 0)),
                  "class_probabilities": {str(k): float(v) for k, v in choice["probabilities"].items()},
                  "semantic_signature": signature, "latency_seconds": elapsed,
                  "preflight_seconds": preflight_seconds,
                  "model_load_seconds": model_load_seconds,
                  "device": str(agent.device),
                  "laya_package_version": version("laya"),
                  "cache_source": "local_inference"}
        append_jsonl(output_jsonl, record)
        done.add(key)
        reusable[request_hash] = record
    return read_jsonl(output_jsonl)
