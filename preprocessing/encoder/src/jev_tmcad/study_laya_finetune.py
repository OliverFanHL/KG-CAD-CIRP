"""Train-only typed Choice adaptation of Laya, with frozen CAD provenance.

The official Laya RLCD/proper-scoring objective is applied only to the class
Choice, whose labels exist. Unlabeled Noul questions remain inference probes.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from .dataset import TMCAD_CLASSES
from .laya_client import _check_laya_context, question_dict
from .semantic_state import build_compact_semantic_state, build_semantic_state
from .study_data import cohort_hash, valid_descriptors
from .study_ground_truth import file_sha256
from .utils import append_jsonl, read_jsonl, sha256_json, write_json


def _rows(cohort, descriptor_path, schema, variant, numeric, profiles):
    desc = valid_descriptors(descriptor_path)
    joined = cohort.merge(desc.drop(columns=[c for c in ("split", "label", "source_label", "relative_path", "absolute_path")
                                           if c in desc]), on="sample_id", validate="one_to_one", sort=False)
    if len(joined) != len(cohort):
        raise ValueError("Fine-tuning descriptors do not cover the frozen cohort")
    builder = build_compact_semantic_state if variant == "compact" else build_semantic_state
    for _, row in joined.iterrows():
        state = builder(row, schema, include_numeric_context=numeric, include_profile_arrays=profiles)
        yield {"sample_id": str(row["sample_id"]), "split": str(row["split"]),
               "label": str(row["label"]), "source_label": str(row["source_label"]),
               "relative_path": str(row["relative_path"]), "absolute_path": str(row["absolute_path"]),
               "state": state, "state_hash": sha256_json(state)}


def prepare_finetune_data(cohort, descriptor_path, schema, spec, directory, *,
                          state_variant, numeric, profiles, seed, calibration_fraction):
    """Write train/calibration/validation data; never write test labels to training files."""
    if set(spec["classification"]["criteria"]) != set(TMCAD_CLASSES):
        raise ValueError("Choice criteria must match the ten frozen CAD classes")
    targets = [directory / f"{s}.jsonl" for s in ("train", "calibration", "validation")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Prepared fine-tuning data already exists; preserve the frozen split")
    rows = list(_rows(cohort, descriptor_path, schema, state_variant, numeric, profiles))
    by_hash: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == "train":
            by_hash.setdefault(row["state_hash"], []).append(row)
    rng = random.Random(seed)
    groups = sorted(by_hash)
    rng.shuffle(groups)
    n_cal = max(1, round(len(groups) * calibration_fraction))
    if n_cal >= len(groups):
        raise ValueError("Calibration split leaves no training states")
    calibration_hashes = set(groups[:n_cal])
    options = list(spec["classification"]["criteria"])
    for row in rows:
        if row["split"] == "test":
            continue
        partition = ("calibration" if row["state_hash"] in calibration_hashes else "train") if row["split"] == "train" else "validation"
        # Ambiguous compact states receive the empirical train-only class distribution.
        # This avoids asking an identical input to predict contradictory hard targets.
        matching = by_hash[row["state_hash"]] if row["split"] == "train" else [row]
        counts = {name: sum(item["label"] == name for item in matching) for name in options}
        record = {**row, "target": [counts[name] / len(matching) for name in options],
                  "partition": partition}
        append_jsonl(directory / f"{partition}.jsonl", record)
    meta = {"cohort_hash": cohort_hash(cohort), "descriptor_sha256": file_sha256(descriptor_path),
            "question_sha256": sha256_json(spec), "state_variant": state_variant,
            "numeric": numeric, "profiles": profiles, "seed": seed,
            "calibration_fraction": calibration_fraction, "options": options,
            "objective": "Choice only; empirical target distribution for identical compact states; no Noul pseudo-labels",
            "counts": {p.stem: len(read_jsonl(p)) for p in targets},
            "partition_hashes": {p.stem: file_sha256(p) for p in targets}}
    write_json(directory / "metadata.json", meta)
    return meta


def _base_checkpoint(repository, subfolder, revision):
    from huggingface_hub import HfApi, snapshot_download
    resolved = revision or HfApi().model_info(repository).sha
    snapshot = snapshot_download(repository, revision=resolved,
                                 allow_patterns=[f"{subfolder}/{name}" for name in
                                                 ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    return Path(snapshot) / subfolder, resolved


def _tokenize(rows, tok, cfg, question):
    from laya.common import QTYPES, build_sequence, render_options, serialize_state
    internal = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
    count = len(render_options(internal))
    items = []
    for row in rows:
        ids, markers = build_sequence(tok, row["state"], internal,
                                      int(cfg.get("max_len", 512)), int(cfg.get("head_max_len", 192)))
        if len(markers) != count:
            raise ValueError(f"Laya truncated Choice options for {row['sample_id']}")
        state_ids = tok(serialize_state(row["state"]).replace(tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"]
        if ids[-len(state_ids) - 1:-1] != state_ids:
            raise ValueError(f"Laya truncated descriptor state for {row['sample_id']}")
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                      "target": row["target"], "label": list(question["criteria"]).index(row["label"]),
                      "sample_id": row["sample_id"]})
    return items


def _logits(model, items, pad_id, batch_size, device, precision="fp32"):
    import torch
    from laya.common import collate_items
    model.eval()
    collected = []
    amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            b = collate_items([[it] for it in items[start:start + batch_size]], pad_id)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                z, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            collected.append(z.float().cpu().numpy())
    return np.concatenate(collected) if collected else np.empty((0, 0))


def _validation_macro_f1(tokenized_items, logits, n_options: int) -> float:
    """Compare tokenized integer Choice IDs to integer predictions.

    `_tokenize` stores option indices in `label`, while the original JSONL
    stores class-name strings. Mixing those two representations caused Kaggle's
    first-epoch F1 calculation to fail after the model had already trained.
    """
    true_ids = np.asarray([int(item["label"]) for item in tokenized_items], dtype=int)
    predicted_ids = np.asarray(logits).argmax(axis=1).astype(int)
    if len(true_ids) != len(predicted_ids) or np.any((true_ids < 0) | (true_ids >= n_options)):
        raise ValueError("Validation Choice indices or logit coverage are invalid")
    return float(f1_score(true_ids, predicted_ids, labels=list(range(n_options)),
                          average="macro", zero_division=0))


def _calibrate_choice_temperature(tokenized_items, logits, n_options: int) -> tuple[float, float]:
    """Fit Choice temperature on train-only calibration option IDs."""
    y = np.asarray([int(item["label"]) for item in tokenized_items], dtype=int)
    z = np.asarray(logits, dtype=float)
    if z.shape != (len(y), n_options) or np.any((y < 0) | (y >= n_options)) or not np.isfinite(z).all():
        raise ValueError("Calibration Choice indices or logits are invalid")
    temperatures = np.geomspace(.5, 5., 81)

    def nll(temperature: float) -> float:
        scaled = z / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -float(np.mean(logp[np.arange(len(y)), y]))

    selected = float(min(temperatures, key=nll))
    return selected, nll(selected)


def train_laya_finetune(directory, checkpoint_dir, spec, *, repository, subfolder, revision,
                        seed, device, epochs, batch_size, encoder_lr, head_lr,
                        weight_decay, group_size, sigma_start, sigma_end, rl_weight,
                        accumulation_steps, precision="fp32", save_epoch_snapshots=False):
    """Official group-relative RLCD plus soft CE; select on validation macro F1."""
    import torch
    import torch.nn.functional as F
    import laya
    from laya.common import collate_items, proper_reward
    from safetensors.torch import save_file

    if checkpoint_dir.exists():
        raise FileExistsError(f"Fine-tuned checkpoint already exists: {checkpoint_dir}")
    progress_path = checkpoint_dir.with_suffix(".progress.json")
    started_at = time.perf_counter()

    def progress(stage: str, **fields) -> None:
        """Write an atomic, human-readable heartbeat without touching the checkpoint."""
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"stage": stage, "pid": os.getpid(),
                  "updated_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": round(time.perf_counter() - started_at, 1), **fields}
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    progress("loading_training_data")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_rows, cal_rows, val_rows = (read_jsonl(directory / f"{name}.jsonl") for name in
                                     ("train", "calibration", "validation"))
    data_meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if data_meta["question_sha256"] != sha256_json(spec):
        raise ValueError("Fine-tuning questions differ from the frozen training data")
    if data_meta["options"] != list(spec["classification"]["criteria"]):
        raise ValueError("Fine-tuning Choice option order changed")
    for name in ("train", "calibration", "validation"):
        if data_meta["partition_hashes"][name] != file_sha256(directory / f"{name}.jsonl"):
            raise ValueError(f"Frozen fine-tuning {name} records changed")
    if not train_rows or not cal_rows or not val_rows:
        raise ValueError("Training, calibration, and validation partitions are required")
    if set(r["state_hash"] for r in train_rows) & set(r["state_hash"] for r in cal_rows):
        raise ValueError("Train and calibration compact states overlap")
    if any(r["split"] != "train" for r in train_rows + cal_rows) or any(r["split"] != "val" for r in val_rows):
        raise ValueError("Fine-tuning partitions mix frozen train/validation/test splits")
    progress("loading_base_checkpoint", repository=repository, revision=revision)
    base, resolved = _base_checkpoint(repository, subfolder, revision)
    agent = laya.load(str(base), device=device)
    progress("tokenizing", device=str(agent.device), train_samples=len(train_rows),
             calibration_samples=len(cal_rows), validation_samples=len(val_rows))
    for row in (train_rows[0], cal_rows[0], val_rows[0]):
        _check_laya_context(agent, row["state"], {spec["classification"].get("id", "part_class"):
                                                   question_dict(spec)[spec["classification"].get("id", "part_class")]})
    model, tok = agent.model, agent.tok
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    model.train()
    train = _tokenize(train_rows, tok, agent.cfg, spec["classification"])
    cal = _tokenize(cal_rows, tok, agent.cfg, spec["classification"])
    val = _tokenize(val_rows, tok, agent.cfg, spec["classification"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": head_lr},
    ], weight_decay=weight_decay)
    device = agent.device
    best, best_state, history = -1.0, None, []
    start_time = time.perf_counter()
    if group_size < 2 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("group_size must be at least 2; batch_size and accumulation_steps positive")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed-precision Laya training requires a CUDA GPU")
    amp = precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    history_path = checkpoint_dir.with_suffix(".history.json")
    snapshot_dir = checkpoint_dir.parent / "laya_epoch_checkpoints"
    if save_epoch_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(epochs), desc="Laya fine-tuning"):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        losses = []
        sigma = sigma_start + (sigma_end - sigma_start) * epoch / max(epochs - 1, 1)
        optimizer.zero_grad(set_to_none=True)
        batch_starts = range(0, len(order), batch_size)
        progress("training", epoch=epoch + 1, epochs_total=epochs, batches_done=0,
                 batches_total=len(batch_starts), train_samples=len(train))
        batch_bar = tqdm(batch_starts, desc=f"Epoch {epoch + 1}/{epochs} batches",
                         leave=False, mininterval=1)
        for batch_number, start in enumerate(batch_bar, 1):
            b = collate_items([[train[int(i)]] for i in order[start:start + batch_size]], tok.pad_token_id)
            ids, att, pos, mask, qt, target = (b[key].to(device) for key in
                                               ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target"))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits, _ = model(ids, att, pos, mask, qt)
            logits = logits.float()
            ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            # Group-relative noisy-logit policy gradient from the official Laya recipe.
            eps = torch.randn((group_size, *logits.shape), device=device) * sigma
            eps = eps - eps.mean(dim=-1, keepdim=True)
            q = torch.softmax(logits.detach()[None, :, :] + eps, dim=-1)
            rewards = proper_reward(q, target, qt, mask, w_sph=.75, w_rps=1.)
            advantages = (rewards - rewards.mean(0, keepdim=True)) / rewards.std(0, keepdim=True).clamp_min(1e-6)
            sampled = torch.multinomial(q.reshape(-1, q.shape[-1]), 1).reshape(group_size, -1)
            logp = F.log_softmax(logits, -1).unsqueeze(0).expand(group_size, -1, -1)
            rl = -(advantages.detach() * logp.gather(-1, sampled[..., None]).squeeze(-1)).mean()
            loss = ce + rl_weight * rl
            scaler.scale(loss / accumulation_steps).backward()
            if ((start // batch_size + 1) % accumulation_steps == 0 or start + batch_size >= len(order)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(batch_starts):
                batch_bar.set_postfix(loss=f"{np.mean(losses[-25:]):.4f}")
                progress("training", epoch=epoch + 1, epochs_total=epochs,
                         batches_done=batch_number, batches_total=len(batch_starts),
                         samples_seen=min(start + batch_size, len(train)), train_samples=len(train),
                         recent_loss=round(float(np.mean(losses[-25:])), 5), device=str(device))
        progress("validating", epoch=epoch + 1, epochs_total=epochs,
                 validation_samples=len(val))
        # Preserve the completed epoch before any validation metric can fail.
        if save_epoch_snapshots:
            snapshot = snapshot_dir / f"epoch_{epoch + 1:02d}.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, snapshot)
        z = _logits(model, val, tok.pad_token_id, batch_size, device, precision)
        score = _validation_macro_f1(val, z, len(spec["classification"]["criteria"]))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(history_path, {"epochs": history, "best_validation_macro_f1": best,
                                  "precision": precision, "batch_size": batch_size,
                                  "accumulation_steps": accumulation_steps})
        if save_epoch_snapshots:
            write_json(snapshot.with_suffix(".json"), {"epoch": epoch + 1,
                       "validation_macro_f1": score, "train_loss": history[-1]["train_loss"],
                       "model_sha256": file_sha256(snapshot)})
        progress("epoch_complete", epoch=epoch + 1, epochs_total=epochs,
                 validation_macro_f1=score, best_validation_macro_f1=best,
                 train_loss=round(float(np.mean(losses)), 5))
    model.load_state_dict(best_state)
    progress("calibrating", calibration_samples=len(cal))
    z = _logits(model, cal, tok.pad_token_id, batch_size, device, precision)
    temperature, calibration_nll = _calibrate_choice_temperature(
        cal, z, len(spec["classification"]["criteria"]))
    progress("saving_checkpoint", best_validation_macro_f1=best)
    checkpoint_dir.mkdir(parents=True)
    model.encoder.save_pretrained(checkpoint_dir / "encoder", safe_serialization=True)
    tok.save_pretrained(checkpoint_dir / "tokenizer")
    save_file({k: v.contiguous() for k, v in best_state.items()}, checkpoint_dir / "model.safetensors")
    cfg = dict(agent.cfg)
    cfg.pop("temperature_by_options", None)
    cfg["temperature"] = [temperature, 1.0, 1.0]
    write_json(checkpoint_dir / "rl_agent_config.json", cfg)
    meta = {"base_repository": repository, "base_subfolder": subfolder, "base_revision": resolved,
            "laya_version": version("laya"), "data_metadata": str(directory / "metadata.json"),
            "data_hashes": {name: file_sha256(directory / f"{name}.jsonl") for name in
                            ("train", "calibration", "validation")},
            "best_validation_macro_f1": best, "best_epoch": 1 + max(range(len(history)), key=lambda i: history[i]["validation_macro_f1"]),
            "calibration_temperature": temperature, "calibration_nll": calibration_nll,
            "training_seconds": time.perf_counter() - start_time, "accumulation_steps": accumulation_steps,
            "batch_size": batch_size, "precision": precision,
            "save_epoch_snapshots": save_epoch_snapshots, "epochs": history,
            "checkpoint_sha256": file_sha256(checkpoint_dir / "model.safetensors")}
    write_json(checkpoint_dir / "study_metadata.json", meta)
    progress("complete", checkpoint=str(checkpoint_dir),
             best_validation_macro_f1=best, best_epoch=meta["best_epoch"])
    return meta


def run_finetuned_laya(cohort, descriptor_path, schema, spec, checkpoint_dir, output_jsonl, *,
                       state_variant, numeric, profiles, device):
    rows = _rows(cohort, descriptor_path, schema, state_variant, numeric, profiles)
    return _run_finetuned_laya_rows(rows, len(cohort), spec, checkpoint_dir, output_jsonl,
                                   device=device)


def run_finetuned_laya_states(states_jsonl: Path, spec: dict, checkpoint_dir: Path,
                             output_jsonl: Path, *, device: str) -> int:
    """Run the identical local inference path from frozen, precomputed CAD states.

    This is the Kaggle path: raw STEP files and descriptor extraction are not
    needed remotely. The upload builder verifies states against the local
    descriptor cache and keeps sample/split/label provenance intact.
    """
    rows = read_jsonl(states_jsonl)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Kaggle inference-state bundle contains duplicate sample IDs")
    if any(row.get("state_hash") != sha256_json(row["state"]) for row in rows):
        raise ValueError("Kaggle inference-state hash mismatch")
    return _run_finetuned_laya_rows(rows, len(rows), spec, checkpoint_dir, output_jsonl,
                                   device=device)


def _run_finetuned_laya_rows(rows, n_rows, spec, checkpoint_dir, output_jsonl, *, device):
    import laya
    if not (checkpoint_dir / "study_metadata.json").exists():
        raise FileNotFoundError("Fine-tuned Laya study checkpoint metadata missing")
    digest = file_sha256(checkpoint_dir / "model.safetensors")
    meta = json.loads((checkpoint_dir / "study_metadata.json").read_text(encoding="utf-8"))
    if meta["checkpoint_sha256"] != digest:
        raise ValueError("Fine-tuned Laya checkpoint changed")
    model_name = f"laya-ft:{digest}"
    questions = question_dict(spec)
    existing = read_jsonl(output_jsonl)
    done = {(str(r["sample_id"]), str(r["request_hash"])) for r in existing}
    reusable = {str(r["request_hash"]): r for r in existing}
    agent = None
    for row in tqdm(rows, total=n_rows, desc="Fine-tuned Laya decisions"):
        request_hash = sha256_json({"model": model_name, "state": row["state"], "questions": spec})
        key = (row["sample_id"], request_hash)
        if key in done:
            continue
        if request_hash in reusable:
            origin = reusable[request_hash]
            record = {**origin, "sample_id": row["sample_id"], "split": row["split"],
                      "true_label": row["label"], "source_label": row["source_label"],
                      "relative_path": row["relative_path"], "absolute_path": row["absolute_path"],
                      "cache_source": "payload_reuse", "reused_from_sample_id": origin["sample_id"],
                      "latency_seconds": None, "preflight_seconds": None, "model_load_seconds": None}
        else:
            load_seconds = None
            if agent is None:
                start = time.perf_counter(); agent = laya.load(str(checkpoint_dir), device=device)
                load_seconds = time.perf_counter() - start
            start = time.perf_counter(); _check_laya_context(agent, row["state"], questions)
            preflight = time.perf_counter() - start
            start = time.perf_counter(); answers = agent.predict(row["state"], questions)["answers"]
            elapsed = time.perf_counter() - start
            choice = answers[spec["classification"].get("id", "part_class")]
            record = {"sample_id": row["sample_id"], "split": row["split"],
                      "true_label": row["label"], "source_label": row["source_label"],
                      "relative_path": row["relative_path"], "absolute_path": row["absolute_path"],
                      "request_hash": request_hash, "model": model_name,
                      "predicted_class": choice["choice"], "class_confidence": choice.get("confidence"),
                      "class_probabilities": choice["probabilities"],
                      "semantic_signature": {qid: float(answers[qid]["noul"]) for qid in spec["semantic_signature"]},
                      "latency_seconds": elapsed, "preflight_seconds": preflight,
                      "model_load_seconds": load_seconds, "device": str(agent.device),
                      "laya_package_version": version("laya"), "cache_source": "local_inference"}
            reusable[request_hash] = record
        append_jsonl(output_jsonl, record)
        done.add(key)
    return len(read_jsonl(output_jsonl))


def evaluate_finetuned_laya_choice(states_jsonl: Path, results_jsonl: Path,
                                   output_dir: Path) -> dict:
    """Report the same direct-Choice metrics as the local matrix, with source IDs."""
    from sklearn.metrics import confusion_matrix
    from .study_evaluation import _metrics

    states = read_jsonl(states_jsonl)
    results = read_jsonl(results_jsonl)
    truth = {str(row["sample_id"]): row for row in states}
    predicted = {str(row["sample_id"]): row for row in results}
    if len(truth) != len(states) or len(predicted) != len(results) or set(truth) != set(predicted):
        raise ValueError("Fine-tuned Laya direct evaluation requires exactly one response per frozen sample")
    rows = []
    for sample_id, row in truth.items():
        result = predicted[sample_id]
        if result["true_label"] != row["label"] or result["split"] != row["split"]:
            raise ValueError(f"Fine-tuned Laya split/label mismatch for {sample_id}")
        if set(result["class_probabilities"]) != set(TMCAD_CLASSES):
            raise ValueError(f"Incomplete Choice probabilities for {sample_id}")
        rows.append({"sample_id": sample_id, "split": row["split"], "label": row["label"],
                     "source_label": row["source_label"], "relative_path": row["relative_path"],
                     "absolute_path": row.get("absolute_path", ""),
                     "predicted_class": result["predicted_class"],
                     **{f"prob__{c}": result["class_probabilities"][c] for c in TMCAD_CLASSES},
                     "request_hash": result["request_hash"], "model": result["model"]})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "direct_choice_predictions.csv", index=False)
    mapping = {name: sorted(frame.loc[frame["label"] == name, "source_label"].dropna().astype(str).unique().tolist())
               for name in TMCAD_CLASSES}
    write_json(output_dir / "label_mapping.json", {"canonical_to_source": mapping,
                                                     "class_order": TMCAD_CLASSES})
    summary = {"note": "Exploratory: existing test labels were inspected before this extension",
               "class_order": TMCAD_CLASSES, "methods": {}}
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        y = part["label"].to_numpy()
        pred = part["predicted_class"].to_numpy()
        p = part[[f"prob__{c}" for c in TMCAD_CLASSES]].to_numpy(dtype=float)
        summary["methods"][split] = _metrics(y, pred, p)
        confusion = pd.DataFrame(confusion_matrix(y, pred, labels=TMCAD_CLASSES),
                                 index=TMCAD_CLASSES, columns=TMCAD_CLASSES)
        confusion.index.name = "true_canonical_class"
        confusion.to_csv(output_dir / f"confusion_{split}.csv")
    write_json(output_dir / "direct_choice_metrics.json", summary)
    return summary
