"""Study commands registered on the single run.py entry point."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import Config, load_questions
from .efficiency import summarize_efficiency
from .geometry_encoder import extract_embeddings, mesh_cache_hash, train_geometry
from .jev_client import run_jev_for_dataframe
from .laya_client import run_laya_for_dataframe
from .clm_client import run_clm_for_dataframe, file_sha256 as clm_file_sha256
from .study_ground_truth import prepare_annotation_files, finalize_annotation_files, materialize_blinded_assets, file_sha256
from .study_laya_finetune import prepare_finetune_data, train_laya_finetune, run_finetuned_laya
from .mesh_data import build_mesh_cache, load_mesh_manifest
from .semantic_state import QuantileSchema
from .study_audit import audit_exact_duplicates
from .study_data import create_cohort, load_cohort, valid_descriptors, cohort_hash
from .study_evaluation import evaluate_classification, prepare_features
from .study_experiments import experiment_catalog
from .study_retrieval import evaluate_retrieval, load_judgments, make_pair_pool
from .utils import write_json
from .utils import read_jsonl


def _context(args):
    cfg = Config.load(args.config)
    load_dotenv(cfg.base_dir / ".env")
    def path(key):
        raw = Path(cfg.get(key))
        return raw if raw.is_absolute() else cfg.base_dir / raw
    return cfg, path


def _cohort(cfg, path):
    return load_cohort(path("study.cohort_path"))


def _schema(path):
    return QuantileSchema.from_dict(json.loads(path("study.schema_path").read_text(encoding="utf-8")))


def _spec(path):
    return load_questions(path("study.questions_path"))


def cmd_plan(args):
    cfg, path = _context(args)
    target = path("study.cohort_path")
    if target.exists() and not args.force:
        raise FileExistsError(f"Frozen cohort already exists: {target}. Use --force only to start a new study.")
    cohort = create_cohort(path("study.descriptor_path"), path("study.jev_results_path"), target,
                           source=cfg.get("cohort.source", "existing_jev"),
                           per_class=cfg.get("cohort.per_class", {}),
                           seed=int(cfg.get("project.seed", 42)))
    print(f"Frozen cohort: {target} ({len(cohort)} samples)")
    print(cohort.groupby(["split", "label"]).size().to_string())


def cmd_mesh(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    result = build_mesh_cache(cohort, path("study.mesh_dir"),
                              points=int(cfg.get("mesh.points", 512)),
                              neighbors=int(cfg.get("mesh.neighbors", 16)),
                              tolerance=float(cfg.get("mesh.tolerance", 0.003)),
                              seed=int(cfg.get("project.seed", 42)),
                              workers=args.workers if args.workers is not None else int(cfg.get("mesh.workers", 1)),
                              checkpoint_every=int(cfg.get("mesh.checkpoint_every", 100)))
    print(result["status"].value_counts().to_string())


def cmd_audit(args):
    cfg, path = _context(args)
    result = audit_exact_duplicates(_cohort(cfg, path), path("study.evaluation_dir") / "duplicate_audit.json")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))


def cmd_geometry_train(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    meshes = load_mesh_manifest(path("study.mesh_dir") / "manifest.csv", cohort)
    result = train_geometry(cohort, meshes, path("study.geometry_checkpoint"),
                            seed=int(cfg.get("project.seed", 42)),
                            device=cfg.get("geometry.device", "auto"),
                            epochs=int(cfg.get("geometry.epochs", 80)),
                            batch_size=int(cfg.get("geometry.batch_size", 16)),
                            learning_rate=float(cfg.get("geometry.learning_rate", 0.001)),
                            weight_decay=float(cfg.get("geometry.weight_decay", 0.0001)),
                            embedding_dim=int(cfg.get("geometry.embedding_dim", 128)),
                            patience=int(cfg.get("geometry.patience", 12)),
                            cache_in_memory=bool(cfg.get("geometry.cache_in_memory", False)))
    print(f"Training: {result['completed_epochs']}/{result['configured_epochs']} epochs; "
          f"stop reason: {result['stop_reason']}; "
          f"best checkpoint: epoch {result['best_epoch']} "
          f"(validation macro F1 {result['best_val_macro_f1']:.4f}).")
    if result["stop_reason"] == "early_stopping":
        print(f"Validation did not improve for {result['patience']} consecutive epochs.")


def cmd_geometry_embed(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    meshes = load_mesh_manifest(path("study.mesh_dir") / "manifest.csv", cohort)
    result = extract_embeddings(cohort, meshes, path("study.geometry_checkpoint"),
                                path("study.geometry_embeddings"),
                                device=cfg.get("geometry.device", "auto"),
                                batch_size=int(cfg.get("geometry.batch_size", 16)))
    print(f"Geometry embeddings: {len(result)} rows")


def cmd_study_jev(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    desc = valid_descriptors(path("study.descriptor_path"))
    rows = cohort[["sample_id"]].merge(desc, on="sample_id", validate="one_to_one")
    dry_stats = {}
    result = run_jev_for_dataframe(rows, _schema(path), _spec(path),
                                   path("study.jev_results_path"), path("study.jev_cache_path"),
                                   model=cfg.get("jev.model", "jev-latest"),
                                   retry_attempts=int(cfg.get("jev.retry_attempts", 5)),
                                   retry_base_seconds=float(cfg.get("jev.retry_base_seconds", 1.5)),
                                   timeout_seconds=float(cfg.get("jev.timeout_seconds", 30.0)),
                                   include_numeric_context=bool(cfg.get("cad.include_numeric_context", True)),
                                   include_profile_arrays=bool(cfg.get("cad.include_profile_arrays", True)),
                                   dry_run_states=path("study.jev_dry_run_path") if args.dry_run else None,
                                   state_variant=cfg.get("cad.state_variant", "full"),
                                   dry_run_stats=dry_stats if args.dry_run else None)
    if args.dry_run:
        print("JEV DRY RUN ONLY: no API requests were sent and no result records were created.")
        print(f"Samples still needing JEV results: {dry_stats['pending_responses']}; "
              f"compatible cached responses: {dry_stats['cached_responses']}; "
              f"new API requests needed for a live run: {dry_stats['uncached_requests']}.")
        print(f"Request previews: {dry_stats['pending_responses'] - dry_stats['previews_added']} "
              f"were already in {path('study.jev_dry_run_path')}; "
              f"{dry_stats['previews_added']} were added this time.")
        print(f"Saved JEV result records: {len(result)}. "
              "To obtain responses, run the same study-jev command without --dry-run.")
    else:
        print(f"JEV records in results file: {len(result)}")


def cmd_study_laya(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    desc = valid_descriptors(path("study.descriptor_path"))
    rows = cohort[["sample_id"]].merge(desc, on="sample_id", validate="one_to_one")
    result = run_laya_for_dataframe(rows, _schema(path), _spec(path), path("study.laya_results_path"),
                                    repository=cfg.get("laya.repository", "convaiinnovations/laya"),
                                    subfolder=cfg.get("laya.subfolder", "typed-decisions"),
                                    revision=cfg.get("laya.revision"), dry_run=args.dry_run,
                                    include_numeric=bool(cfg.get("cad.include_numeric_context", True)),
                                    include_profiles=bool(cfg.get("cad.include_profile_arrays", True)),
                                    state_variant=cfg.get("cad.state_variant", "full"))
    print(f"Laya records in results file: {len(result)}")


def cmd_study_clm(args):
    cfg, path = _context(args)
    cohort = _cohort(cfg, path)
    desc = valid_descriptors(path("study.descriptor_path"))
    rows = cohort[["sample_id"]].merge(desc, on="sample_id", validate="one_to_one")
    if len(rows) != len(cohort):
        raise ValueError("CLM descriptor coverage differs from the frozen cohort")
    for field in ("split", "label", "source_label", "relative_path", "absolute_path"):
        matched = cohort[["sample_id", field]].merge(rows[["sample_id", field]], on="sample_id",
                                                    suffixes=("_cohort", "_descriptor"), validate="one_to_one")
        if matched[f"{field}_cohort"].astype(str).ne(matched[f"{field}_descriptor"].astype(str)).any():
            raise ValueError(f"CLM descriptor {field} differs from the frozen cohort")
    result = run_clm_for_dataframe(
        rows, _schema(path), _spec(path), path("study.clm_results_path"),
        checkpoint=path("clm.checkpoint_path"), model=cfg.get("clm.model", "clm-latest"),
        encoder_backend=cfg.get("clm.encoder_backend", "vllm"),
        encoder_model=cfg.get("clm.encoder_model", "Qwen/Qwen3-8B"),
        encoder_device=cfg.get("clm.encoder_device", "auto"),
        encoder_batch_size=int(cfg.get("clm.encoder_batch_size", 4)),
        emb_url=cfg.get("clm.emb_url", "http://127.0.0.1:8090/v1/embeddings"),
        emb_model=cfg.get("clm.emb_model", "qwen3-8b"),
        encoder_revision=cfg.get("clm.encoder_revision", ""),
        timeout_seconds=float(cfg.get("clm.timeout_seconds", 300)),
        retry_attempts=int(cfg.get("clm.retry_attempts", 3)),
        include_numeric=bool(cfg.get("cad.include_numeric_context", True)),
        include_profiles=bool(cfg.get("cad.include_profile_arrays", True)),
        state_variant=cfg.get("cad.state_variant", "full"), dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


def _features(cfg, path):
    cohort = _cohort(cfg, path)
    meshes = load_mesh_manifest(path("study.mesh_dir") / "manifest.csv", cohort)
    embedding_meta = json.loads(path("study.geometry_embeddings").with_suffix(".meta.json").read_text(encoding="utf-8"))
    if embedding_meta["mesh_hash"] != mesh_cache_hash(meshes):
        raise ValueError("Geometry embeddings were extracted from a different mesh cache")
    laya_revision = cfg.get("laya.revision")
    if laya_revision is None and path("study.laya_results_path").exists():
        saved_laya = read_jsonl(path("study.laya_results_path"))
        if saved_laya:
            laya_revision = str(saved_laya[0]["model"]).rsplit("@", 1)[-1]
    laya_model = (f"{cfg.get('laya.repository', 'convaiinnovations/laya')}/"
                  f"{cfg.get('laya.subfolder', 'typed-decisions')}@{laya_revision}")
    ft_path = path("laya_finetune.results_path") if cfg.get("laya_finetune.results_path") and path("laya_finetune.results_path").exists() else None
    ft_model = f"laya-ft:{file_sha256(path('laya_finetune.checkpoint_dir') / 'model.safetensors')}" if ft_path else None
    clm_path = path("study.clm_results_path") if cfg.get("study.clm_results_path") and path("study.clm_results_path").exists() else None
    clm_hash = clm_file_sha256(path("clm.checkpoint_path")) if clm_path else None
    features = prepare_features(cohort, path("study.descriptor_path"),
                                path("study.geometry_embeddings"), path("study.jev_results_path"),
                                path("study.laya_results_path"), path("study.schema_path"),
                                _spec(path), jev_model=cfg.get("jev.model", "jev-latest"),
                                laya_model=laya_model,
                                include_numeric=bool(cfg.get("cad.include_numeric_context", True)),
                                include_profiles=bool(cfg.get("cad.include_profile_arrays", True)),
                                require_laya=bool(cfg.get("laya.required", False)),
                                state_variant=cfg.get("cad.state_variant", "full"),
                                laya_ft_path=ft_path, laya_ft_model=ft_model,
                                clm_path=clm_path, clm_model=cfg.get("clm.model", "clm-latest"),
                                clm_checkpoint_sha256=clm_hash)
    return cohort, features


def cmd_pool(args):
    cfg, path = _context(args)
    target = path("study.pair_pool_path")
    if target.exists():
        raise FileExistsError(f"Pair pool exists: {target}. Preserve annotated judgments.")
    _, features = _features(cfg, path)
    pool = make_pair_pool(features, target, top_k=int(cfg.get("retrieval.pool_top_k", 10)),
                          random_k=int(cfg.get("retrieval.pool_random_k", 10)),
                          seed=int(cfg.get("project.seed", 42)))
    print(f"Annotation pool: {target} ({len(pool)} pairs)")


def cmd_grade_prepare(args):
    cfg, path = _context(args)
    result = prepare_annotation_files(_cohort(cfg, path), path("study.pair_pool_path"),
                                      path("ground_truth.annotation_dir"),
                                      per_class=cfg.get("ground_truth.queries_per_class", {"val": 5, "test": 10}),
                                      seed=int(cfg.get("ground_truth.seed", 42)),
                                      copy_assets=args.copy_assets)
    print(f"Blinded annotation package: {result['n_pairs']} pairs, {result['n_assets']} assets")


def cmd_grade_finalize(args):
    _, path = _context(args)
    result = finalize_annotation_files(path("ground_truth.annotation_dir"), path("study.judgments_path"))
    print(json.dumps(result, indent=2))


def cmd_grade_assets(args):
    _, path = _context(args)
    print(json.dumps(materialize_blinded_assets(path("ground_truth.annotation_dir")), indent=2))


def cmd_laya_ft_prepare(args):
    cfg, path = _context(args)
    result = prepare_finetune_data(_cohort(cfg, path), path("study.descriptor_path"),
                                   _schema(path), _spec(path), path("laya_finetune.data_dir"),
                                   state_variant=cfg.get("cad.state_variant", "full"),
                                   numeric=bool(cfg.get("cad.include_numeric_context", True)),
                                   profiles=bool(cfg.get("cad.include_profile_arrays", True)),
                                   seed=int(cfg.get("project.seed", 42)),
                                   calibration_fraction=float(cfg.get("laya_finetune.calibration_fraction", .1)))
    print(json.dumps(result, indent=2))


def cmd_laya_ft_train(args):
    cfg, path = _context(args)
    data_meta = json.loads((path("laya_finetune.data_dir") / "metadata.json").read_text(encoding="utf-8"))
    if data_meta["cohort_hash"] != cohort_hash(_cohort(cfg, path)) or \
            data_meta["descriptor_sha256"] != file_sha256(path("study.descriptor_path")) or \
            data_meta["state_variant"] != cfg.get("cad.state_variant", "full"):
        raise ValueError("Fine-tuning data differs from the frozen cohort, descriptors, or state protocol")
    print(f"Laya fine-tuning progress: {path('laya_finetune.checkpoint_dir').with_suffix('.progress.json')}", flush=True)
    result = train_laya_finetune(path("laya_finetune.data_dir"), path("laya_finetune.checkpoint_dir"),
                                 _spec(path), repository=cfg.get("laya.repository", "convaiinnovations/laya"),
                                 subfolder=cfg.get("laya.subfolder", "typed-decisions"),
                                 revision=cfg.get("laya.revision"), seed=int(cfg.get("project.seed", 42)),
                                 device=cfg.get("laya_finetune.device", "cuda"),
                                 epochs=int(cfg.get("laya_finetune.epochs", 4)),
                                 batch_size=int(cfg.get("laya_finetune.batch_size", 8)),
                                 encoder_lr=float(cfg.get("laya_finetune.encoder_lr", 2.5e-5)),
                                 head_lr=float(cfg.get("laya_finetune.head_lr", 1e-4)),
                                 weight_decay=float(cfg.get("laya_finetune.weight_decay", .01)),
                                 group_size=int(cfg.get("laya_finetune.group_size", 4)),
                                 sigma_start=float(cfg.get("laya_finetune.sigma_start", .4)),
                                 sigma_end=float(cfg.get("laya_finetune.sigma_end", .1)),
                                 rl_weight=float(cfg.get("laya_finetune.rl_weight", 1.)),
                                 accumulation_steps=int(cfg.get("laya_finetune.accumulation_steps", 4)),
                                 precision=cfg.get("laya_finetune.precision", "fp32"),
                                 save_epoch_snapshots=bool(cfg.get("laya_finetune.save_epoch_snapshots", False)))
    print(json.dumps({k: v for k, v in result.items() if k != "epochs"}, indent=2))


def cmd_laya_ft_infer(args):
    cfg, path = _context(args)
    count = run_finetuned_laya(_cohort(cfg, path), path("study.descriptor_path"), _schema(path),
                               _spec(path), path("laya_finetune.checkpoint_dir"),
                               path("laya_finetune.results_path"),
                               state_variant=cfg.get("cad.state_variant", "full"),
                               numeric=bool(cfg.get("cad.include_numeric_context", True)),
                               profiles=bool(cfg.get("cad.include_profile_arrays", True)),
                               device=cfg.get("laya_finetune.device", "cuda"))
    print(f"Fine-tuned Laya result records: {count}")


def cmd_study_evaluate(args):
    cfg, path = _context(args)
    domain = method = None
    if args.experiment:
        parts = args.experiment.split("/", 1)
        if len(parts) != 2 or parts[0] not in experiment_catalog() or parts[1] not in experiment_catalog()[parts[0]]:
            raise ValueError("Unknown experiment ID. Run study-experiments to list valid domain/name IDs.")
        domain, method = parts
    if domain == "retrieval" and args.classification_only:
        raise ValueError("A retrieval experiment cannot use --classification-only")
    judgments = path("study.judgments_path") if cfg.get("study.judgments_path") else path("study.pair_pool_path")
    if domain != "classification" and not args.classification_only:
        if not judgments.exists():
            raise FileNotFoundError(f"Judged retrieval pool missing: {judgments}; use --classification-only for classification")
        load_judgments(judgments)  # Fail before loading features or creating output files.
    cohort, features = _features(cfg, path)
    out = path("study.evaluation_dir")
    if domain is not None:
        out = out / "experiments" / domain / method
    out.mkdir(parents=True, exist_ok=True)
    identity_cols = ["sample_id", "split", "label", "source_label", "relative_path", "absolute_path"]
    alignment_cols = identity_cols + [
        f"{backend}__{field}" for backend in ("jev", "laya", "laya_ft", "clm")
        for field in ("request_hash", "response_sample_id", "reused_from_sample_id", "cache_source")
        if f"{backend}__{field}" in features
    ]
    features[alignment_cols].to_csv(out / "semantic_response_alignment.csv", index=False)
    classification = None
    if domain != "retrieval":
        classification = evaluate_classification(features, out / "classification",
                                                  c_grid=[float(x) for x in cfg.get("classification.c_grid", [0.01, 0.1, 1, 10])],
                                                  only=method)
    retrieval = None
    if judgments.exists() and not args.classification_only and domain != "classification":
        retrieval = evaluate_retrieval(features, judgments, out / "retrieval",
                                       ks=[int(x) for x in cfg.get("retrieval.ks", [1, 5, 10])],
                                       weights=[float(x) for x in cfg.get("retrieval.fusion_weights", [0, .25, .5, .75, 1])],
                                       only=method if domain == "retrieval" else None,
                                       query_scope=cfg.get("retrieval.query_scope", "all"))
    write_json(out / "run_manifest.json", {
        "experiment": args.experiment or "all_available",
        "config": str(Path(args.config).resolve()),
        "semantic_response_alignment": str(out / "semantic_response_alignment.csv"),
        "laya_payload_reuses": int(features["laya__reused_from_sample_id"].notna().sum())
            if "laya__reused_from_sample_id" in features else 0,
        "cohort_path": str(path("study.cohort_path")), "cohort_size": len(cohort),
        "descriptor_path": str(path("study.descriptor_path")),
        "source_manifest": str(out / ("classification" if classification is not None else "retrieval") / "source_manifest.csv"),
        "label_mapping": str(out / ("classification" if classification is not None else "retrieval") / "label_mapping.json"),
        "geometry_checkpoint": str(path("study.geometry_checkpoint")),
        "jev_results": str(path("study.jev_results_path")),
        "laya_results": str(path("study.laya_results_path")) if path("study.laya_results_path").exists() else None,
        "laya_ft_results": str(path("laya_finetune.results_path")) if cfg.get("laya_finetune.results_path") and path("laya_finetune.results_path").exists() else None,
        "clm_results": str(path("study.clm_results_path")) if cfg.get("study.clm_results_path") and path("study.clm_results_path").exists() else None,
        "clm_checkpoint_sha256": clm_file_sha256(path("clm.checkpoint_path")) if cfg.get("study.clm_results_path") and path("study.clm_results_path").exists() else None,
        "clm_encoder_model": cfg.get("clm.emb_model") if cfg.get("study.clm_results_path") else None,
        "clm_encoder_revision": cfg.get("clm.encoder_revision") if cfg.get("study.clm_results_path") else None,
        "retrieval_judgments": str(judgments) if retrieval is not None else None,
    })
    summarize_efficiency(path("study.descriptor_path"), path("study.mesh_dir") / "manifest.csv",
                         path("study.geometry_embeddings"), path("study.geometry_checkpoint"), features,
                         out / "efficiency.json", set(cohort["sample_id"].astype(str)))
    print(json.dumps({"classification_methods": list(classification or {}),
                      "retrieval_methods": list(retrieval or {})}, indent=2))


def cmd_study_experiments(args):
    print(json.dumps(experiment_catalog(), indent=2))


def register_study_commands(sub):
    commands = {
        "study-plan": (cmd_plan, "Freeze the matched study cohort"),
        "study-mesh": (cmd_mesh, "Cache local STEP surface neighborhoods"),
        "study-audit": (cmd_audit, "Audit exact STEP duplicates across splits"),
        "study-train-geometry": (cmd_geometry_train, "Train the local geometry encoder"),
        "study-embed": (cmd_geometry_embed, "Extract geometry embeddings"),
        "study-jev": (cmd_study_jev, "Query JEV on the frozen cohort"),
        "study-laya": (cmd_study_laya, "Run Laya on the frozen cohort"),
        "study-clm": (cmd_study_clm, "Run official CLM on the frozen cohort"),
        "study-pool": (cmd_pool, "Make retrieval pairs for blind annotation"),
        "study-grade-prepare": (cmd_grade_prepare, "Freeze blinded expert rating package"),
        "study-grade-finalize": (cmd_grade_finalize, "Adjudicate complete expert ratings"),
        "study-grade-assets": (cmd_grade_assets, "Copy opaque STEP files for blinded rating"),
        "study-laya-ft-prepare": (cmd_laya_ft_prepare, "Prepare train-only Laya Choice fine-tuning data"),
        "study-laya-ft-train": (cmd_laya_ft_train, "Fine-tune Laya with validation checkpoint selection"),
        "study-laya-ft-infer": (cmd_laya_ft_infer, "Run fine-tuned Laya on the frozen cohort"),
        "study-evaluate": (cmd_study_evaluate, "Evaluate matched classification and judged retrieval"),
        "study-experiments": (cmd_study_experiments, "List exact experiment IDs and feature recipes"),
    }
    for name, (function, help_text) in commands.items():
        parser = sub.add_parser(name, help=help_text)
        parser.set_defaults(func=function)
        if name == "study-plan":
            parser.add_argument("--force", action="store_true")
        if name == "study-mesh":
            parser.add_argument("--workers", type=int, default=None,
                                help="Parallel STEP tessellation processes; start with 2, use 4 if memory allows")
        if name == "study-grade-prepare":
            parser.add_argument("--copy-assets", action="store_true", help="Copy opaque STEP files for blinded rating")
        if name in {"study-jev", "study-laya", "study-clm"}:
            parser.add_argument("--dry-run", action="store_true")
        if name == "study-evaluate":
            parser.add_argument("--classification-only", action="store_true")
            parser.add_argument("--experiment", help="Run one matrix row: classification/ID or retrieval/ID")
