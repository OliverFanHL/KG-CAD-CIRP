"""Matched classification experiments; validation selects every probe setting."""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cad_features import numeric_feature_columns
from .dataset import TMCAD_CLASSES
from .provenance import class_label_mapping, source_manifest, with_source_labels
from .reporting import prediction_frame, save_confusion
from .semantic_state import QuantileSchema, build_compact_semantic_state, build_semantic_state
from .study_data import check_cohort_hash, cohort_hash, valid_descriptors
from .study_experiments import CLASSIFICATION_PROBES, classification_ids, feature_blocks
from .utils import read_jsonl, sha256_json, write_json


def _backend_frame(path: Path, cohort: pd.DataFrame, spec: dict, schema: QuantileSchema,
                   *, backend: str, model: str, required: bool, include_numeric: bool,
                   include_profiles: bool, state_variant: str,
                   checkpoint_sha256: str | None = None) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required {backend} results missing: {path}")
        return None
    raw = read_jsonl(path)
    if not raw:
        raise ValueError(f"Empty {backend} results: {path}")
    resolved_models = {str(record.get("model", "")) for record in raw}
    if len(resolved_models) != 1 or not next(iter(resolved_models)):
        raise ValueError(f"{backend} results mix resolved model versions or omit model identity")
    builder = build_compact_semantic_state if state_variant == "compact" else build_semantic_state
    expected = {}
    for _, row in cohort.iterrows():
        state = builder(row, schema, include_numeric_context=include_numeric,
                        include_profile_arrays=include_profiles)
        sample_id = str(row["sample_id"])
        if sample_id in expected:
            raise ValueError(f"Duplicate cohort sample ID: {sample_id}")
        expected[sample_id] = sha256_json({"model": model, "state": state, "questions": spec})
    by_pair = {}
    by_hash = {}
    for record in raw:
        sample_id = str(record.get("sample_id"))
        request_hash = str(record.get("request_hash"))
        if sample_id not in expected or expected[sample_id] != request_hash:
            raise ValueError(f"{backend} result has an unknown sample or incompatible request hash: {sample_id}")
        # JEV records the resolved server model (for example jev-1.13.0)
        # while the request may use the jev-latest alias. Its request hash
        # already validates the configured request model and entire payload.
        if (backend.startswith("laya") or backend == "clm") and str(record.get("model")) != model:
            raise ValueError(f"{backend} model differs for {sample_id}")
        if backend == "clm" and record.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"CLM checkpoint differs for {sample_id}")
        key = (sample_id, request_hash)
        if key in by_pair:
            raise ValueError(f"Duplicate {backend} result for sample and request: {sample_id}")
        by_pair[key] = record
        by_hash.setdefault(request_hash, record)
    records = []
    for _, row in cohort.iterrows():
        sample_id = str(row["sample_id"])
        request_hash = expected[sample_id]
        r = by_pair.get((sample_id, request_hash))
        reused_from = None
        if r is None and backend.startswith("laya"):
            # Older Laya runs saved one row per payload hash, although several
            # samples can have the same compact state. Reuse only after both
            # payload hashes have been independently checked above.
            r = by_hash.get(request_hash)
            if r is not None:
                reused_from = str(r["sample_id"])
        if r is None:
            raise ValueError(f"{backend} result missing or incompatible for {sample_id}; "
                             "check model, questions, state settings, and cohort")
        if reused_from is None and (str(r.get("split")) != str(row["split"]) or
                                    str(r.get("true_label")) != str(row["label"])):
            raise ValueError(f"{backend} cached split/label differs for {sample_id}")
        sig = r.get("semantic_signature", {})
        if set(sig) != set(spec.get("semantic_signature", {})):
            raise ValueError(f"Incomplete {backend} signature for {row['sample_id']}")
        if any(not np.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in sig.values()):
            raise ValueError(f"Invalid {backend} signature values for {row['sample_id']}")
        probs = r.get("class_probabilities", {})
        if set(probs) != set(TMCAD_CLASSES) or r.get("predicted_class") not in TMCAD_CLASSES:
            raise ValueError(f"Incomplete {backend} Choice answer for {row['sample_id']}")
        if any(not np.isfinite(float(v)) or float(v) < 0 for v in probs.values()) or sum(float(v) for v in probs.values()) <= 0:
            raise ValueError(f"Invalid {backend} Choice probabilities for {row['sample_id']}")
        rec = {"sample_id": sample_id, "split": row["split"], "label": row["label"],
               f"{backend}__request_hash": request_hash,
               f"{backend}__response_sample_id": str(r["sample_id"]),
               f"{backend}__choice": r["predicted_class"],
               f"{backend}__latency_seconds": None if reused_from else r.get("latency_seconds"),
               f"{backend}__cache_source": "payload_reuse" if reused_from else r.get("cache_source", "unknown"),
               f"{backend}__preflight_seconds": None if reused_from else r.get("preflight_seconds"),
               f"{backend}__model_load_seconds": None if reused_from else r.get("model_load_seconds"),
               f"{backend}__reused_from_sample_id": reused_from if reused_from is not None else r.get("reused_from_sample_id")}
        rec.update({f"{backend}__sig__{k}": float(v) for k, v in sig.items()})
        rec.update({f"{backend}__prob__{c}": float(probs.get(c, 0.0)) for c in TMCAD_CLASSES})
        records.append(rec)
    return pd.DataFrame(records)


def prepare_features(cohort: pd.DataFrame, descriptor_path: Path, embeddings_path: Path,
                     jev_path: Path, laya_path: Path, schema_path: Path, spec: dict,
                     *, jev_model: str, laya_model: str, include_numeric: bool,
                     include_profiles: bool, require_laya: bool,
                     state_variant: str, laya_ft_path: Path | None = None,
                     laya_ft_model: str | None = None,
                     clm_path: Path | None = None, clm_model: str | None = None,
                     clm_checkpoint_sha256: str | None = None) -> pd.DataFrame:
    cohort = with_source_labels(cohort)
    desc = valid_descriptors(descriptor_path)
    state_rows = cohort[["sample_id"]].merge(desc, on="sample_id", validate="one_to_one")
    if len(state_rows) != len(cohort) or any(
        not state_rows[col].astype(str).equals(cohort[col].reset_index(drop=True).astype(str))
        for col in ("split", "label", "source_label", "relative_path", "absolute_path")
    ):
        raise ValueError("Descriptor state rows or original metadata do not match the frozen cohort")
    desc_cols = [c for c in numeric_feature_columns(list(desc)) if pd.api.types.is_numeric_dtype(desc[c])]
    frame = cohort.merge(desc[["sample_id"] + desc_cols], on="sample_id", validate="one_to_one")
    if len(frame) != len(cohort):
        raise ValueError("Descriptor coverage differs from cohort")
    frame = frame.rename(columns={c: f"desc__{c}" for c in desc_cols})
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Geometry embeddings missing: {embeddings_path}")
    check_cohort_hash(cohort, embeddings_path.with_suffix(".meta.json"))
    emb_meta = json.loads(embeddings_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    checkpoint = Path(emb_meta["checkpoint"])
    if not checkpoint.exists() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != emb_meta["checkpoint_sha256"]:
        raise ValueError("Geometry embedding file is stale relative to its checkpoint")
    emb = pd.read_csv(embeddings_path, dtype={"sample_id": str})
    joined_meta = cohort[["sample_id", "split", "label"]].merge(
        emb[["sample_id", "split", "label"]], on="sample_id", suffixes=("_cohort", "_emb"),
        validate="one_to_one")
    if len(joined_meta) != len(cohort) or any(
        joined_meta[f"{col}_cohort"].astype(str).ne(joined_meta[f"{col}_emb"].astype(str)).any()
        for col in ("split", "label")
    ):
        raise ValueError("Geometry embedding labels or splits differ from the frozen cohort")
    for col in ("source_label", "relative_path", "absolute_path"):
        if col in emb:
            values = cohort[["sample_id", col]].merge(emb[["sample_id", col]], on="sample_id",
                                                        suffixes=("_cohort", "_emb"), validate="one_to_one")
            if values[f"{col}_cohort"].astype(str).ne(values[f"{col}_emb"].astype(str)).any():
                raise ValueError(f"Geometry embedding {col} differs from the frozen cohort")
    geom_cols = [c for c in emb if c.startswith("geom__") or c.startswith("geom_logit__")]
    frame = frame.merge(emb[["sample_id"] + geom_cols], on="sample_id", validate="one_to_one")
    if len(frame) != len(cohort):
        raise ValueError("Geometry embeddings do not cover cohort")
    schema = QuantileSchema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
    for backend, path, model, required in [
        ("jev", jev_path, jev_model, True), ("laya", laya_path, laya_model, require_laya),
        *(([("laya_ft", laya_ft_path, laya_ft_model, True)] if laya_ft_path is not None else [])),
        *(([("clm", clm_path, clm_model, True)] if clm_path is not None else [])),
    ]:
        backend_frame = _backend_frame(path, state_rows, spec, schema, backend=backend, model=model,
                                       required=required, include_numeric=include_numeric,
                                       include_profiles=include_profiles, state_variant=state_variant,
                                       checkpoint_sha256=clm_checkpoint_sha256 if backend == "clm" else None)
        if backend_frame is not None:
            frame = frame.merge(backend_frame.drop(columns=["split", "label"]), on="sample_id",
                                validate="one_to_one")
    return frame


def _metrics(y, pred, probabilities) -> dict:
    out = {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, labels=TMCAD_CLASSES, average="macro", zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    if probabilities is not None:
        p = np.clip(probabilities, 1e-12, 1)
        p /= p.sum(axis=1, keepdims=True)
        out["nll"] = float(log_loss(y, p, labels=TMCAD_CLASSES))
        target = np.zeros_like(p)
        for i, label in enumerate(y):
            target[i, TMCAD_CLASSES.index(label)] = 1
        out["brier"] = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    return out


def _pipeline(c: float) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("classifier", LogisticRegression(C=c, max_iter=4000, solver="lbfgs"))])


def _expand_probabilities(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    observed = pipe.predict_proba(x)
    full = np.zeros((len(x), len(TMCAD_CLASSES)), dtype=float)
    for j, name in enumerate(pipe.classes_):
        full[:, TMCAD_CLASSES.index(name)] = observed[:, j]
    return full


def _save_method(test: pd.DataFrame, pred, probabilities, output_dir: Path,
                 name: str, mapping: dict[str, list[str]]) -> None:
    prediction_frame(test, pred, probabilities, TMCAD_CLASSES, mapping).to_csv(
        output_dir / f"{name}.csv", index=False)
    save_confusion(test["label"].to_numpy(), pred, TMCAD_CLASSES, mapping,
                   name.replace("_", " "), output_dir / "figures" / f"confusion_{name}.png")


def _direct(frame: pd.DataFrame, backend: str, output_dir: Path,
            mapping: dict[str, list[str]]) -> dict:
    test = frame[frame["split"] == "test"]
    y = test["label"].to_numpy()
    pred = test[f"{backend}__choice"].to_numpy()
    p = test[[f"{backend}__prob__{c}" for c in TMCAD_CLASSES]].to_numpy(dtype=float)
    result = _metrics(y, pred, p)
    result["selection"] = "none; frozen typed Choice"
    _save_method(test, pred, p, output_dir, f"{backend}_direct_choice", mapping)
    return result


def evaluate_classification(frame: pd.DataFrame, output_dir: Path, *, c_grid: list[float],
                            only: str | None = None) -> dict:
    if only is not None and only not in classification_ids():
        raise ValueError(f"Unknown classification experiment: {only}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = with_source_labels(frame)
    source_manifest(frame).to_csv(output_dir / "source_manifest.csv", index=False)
    mapping = class_label_mapping(frame, TMCAD_CLASSES)
    write_json(output_dir / "label_mapping.json", {
        "canonical_to_source": mapping, "class_order": TMCAD_CLASSES,
        "source": "original class-folder component of relative_path",
    })
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("Classification requires train, validation, and untouched test splits")
    blocks = feature_blocks(frame)
    groups = {name: [col for block in recipe for col in blocks[block]]
              for name, recipe in CLASSIFICATION_PROBES.items()
              if all(blocks[block] for block in recipe)}
    if only is not None and only in CLASSIFICATION_PROBES and only not in groups:
        raise ValueError(f"Required features are unavailable for classification/{only}")
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    combined = frame[frame["split"].isin(["train", "val"])]
    results = {}
    logits_cols = [f"geom_logit__{name}" for name in TMCAD_CLASSES]
    if (only is None or only == "geometry_encoder_head") and all(col in frame for col in logits_cols):
        logits = test[logits_cols].to_numpy(dtype=float)
        stable = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        pred = np.asarray(TMCAD_CLASSES)[probabilities.argmax(axis=1)]
        results["geometry_encoder_head"] = _metrics(test["label"].to_numpy(), pred, probabilities)
        results["geometry_encoder_head"]["selection"] = "checkpoint epoch selected on validation macro F1"
        _save_method(test, pred, probabilities, output_dir, "geometry_encoder_head", mapping)
    for backend in ("jev", "laya", "laya_ft", "clm"):
        if (only is None or only == f"{backend}_direct_choice") and f"{backend}__choice" in frame:
            results[f"{backend}_direct_choice"] = _direct(frame, backend, output_dir, mapping)
    for name, cols in groups.items():
        if only is not None and name != only:
            continue
        if not cols:
            continue
        best = None
        selection = []
        for c in c_grid:
            pipe = _pipeline(float(c))
            start = time.perf_counter()
            pipe.fit(train[cols], train["label"])
            fit_seconds = time.perf_counter() - start
            val_pred = pipe.predict(val[cols])
            val_score = _metrics(val["label"].to_numpy(), val_pred, None)["macro_f1"]
            selection.append({"C": float(c), "val_macro_f1": val_score, "fit_seconds": fit_seconds})
            if best is None or val_score > best[0]:
                best = (val_score, float(c))
        assert best is not None
        pipe = _pipeline(best[1])
        start = time.perf_counter()
        pipe.fit(combined[cols], combined["label"])
        refit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = pipe.predict(test[cols])
        probabilities = _expand_probabilities(pipe, test[cols])
        predict_seconds = time.perf_counter() - start
        result = _metrics(test["label"].to_numpy(), pred, probabilities)
        result.update({"selected_C": best[1], "val_macro_f1": best[0],
                       "selection": selection, "refit_seconds": refit_seconds,
                       "test_predict_batch_seconds": predict_seconds,
                       "test_predict_per_sample_seconds": predict_seconds / len(test),
                       "n_features": len(cols), "n_train": len(train), "n_val": len(val)})
        results[name] = result
        _save_method(test, pred, probabilities, output_dir, name, mapping)
    if only is not None and only not in results:
        raise ValueError(f"Required features are unavailable for classification/{only}")
    write_json(output_dir / "summary.json", {"cohort_hash": cohort_hash(frame),
                                              "experiment": only, "methods": results})
    return results
