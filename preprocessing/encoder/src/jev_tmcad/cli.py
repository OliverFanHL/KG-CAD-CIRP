from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .classification_eval import evaluate_classification
from .config import Config, load_questions
from .dataset import add_stratified_splits, save_manifest, scan_tmcad, select_subset
from .extraction import extract_dataset_descriptors
from .jev_client import run_jev_for_dataframe
from .qualitative import generate_qualitative_report
from .retrieval_eval import evaluate_retrieval
from .semantic_state import QuantileSchema, build_semantic_state, fit_quantile_schema
from .study_cli import register_study_commands
from .utils import setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _paths(cfg: Config) -> dict[str, Path]:
    return {
        "manifest": cfg.output_path("manifest.csv"),
        "descriptors": cfg.output_path("descriptors", "tmcad_descriptors.csv"),
        "schema": cfg.output_path("descriptors", "quantile_schema.json"),
        "jev": cfg.output_path("jev", "jev_results.jsonl"),
        "cache": cfg.output_path("cache", "jev_cache.sqlite"),
        "classification": cfg.output_path("evaluation", "classification"),
        "retrieval": cfg.output_path("evaluation", "retrieval"),
        "qualitative": cfg.output_path("qualitative"),
    }


def _load_cfg(args) -> Config:
    cfg = Config.load(args.config)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    load_dotenv(cfg.base_dir / ".env")
    return cfg


def cmd_scan(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    dataset = Path(args.dataset) if args.dataset else cfg.dataset_root
    df = scan_tmcad(dataset)
    df = add_stratified_splits(
        df,
        train_fraction=float(cfg.get("splits.train", 0.70)),
        val_fraction=float(cfg.get("splits.val", 0.10)),
        test_fraction=float(cfg.get("splits.test", 0.20)),
        seed=int(cfg.get("project.seed", 42)),
    )
    save_manifest(df, p["manifest"])
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Manifest: {p['manifest']}")
    print(counts)
    print(f"Total: {len(df)}")


def cmd_extract(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    if not p["manifest"].exists():
        raise FileNotFoundError(f"Manifest not found: {p['manifest']}. Run `scan` first.")
    manifest = pd.read_csv(p["manifest"])
    result = extract_dataset_descriptors(
        manifest,
        output_csv=p["descriptors"],
        relative_tolerance=float(cfg.get("cad.tessellation_relative_tolerance", 0.005)),
        angular_bins=int(cfg.get("cad.angular_profile_bins", 72)),
        axial_bins=int(cfg.get("cad.axial_profile_bins", 12)),
        resume=not args.no_resume,
        workers=args.workers,
    )
    failures = result["extract_error"].fillna("").astype(str).str.len().gt(0).sum()
    print(f"Descriptors: {p['descriptors']}")
    print(f"Extracted rows: {len(result)} | failures: {failures}")


def _load_or_fit_schema(desc: pd.DataFrame, schema_path: Path, force: bool = False) -> QuantileSchema:
    if schema_path.exists() and not force:
        return QuantileSchema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
    train = desc[(desc["split"] == "train") & (desc["extract_error"].fillna("").astype(str).str.len() == 0)]
    if train.empty:
        raise RuntimeError("No valid training descriptors available for fitting semantic quantile bins.")
    schema = fit_quantile_schema(train)
    write_json(schema_path, schema.to_dict())
    return schema


def cmd_inspect_state(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    desc = pd.read_csv(p["descriptors"])
    schema = _load_or_fit_schema(desc, p["schema"])
    valid = desc[desc["extract_error"].fillna("").astype(str).str.len() == 0]
    if args.sample_id:
        valid = valid[valid["sample_id"].astype(str) == args.sample_id]
    rows = valid.head(args.n)
    for _, row in rows.iterrows():
        state = build_semantic_state(
            row,
            schema,
            include_numeric_context=bool(cfg.get("cad.include_numeric_context", True)),
            include_profile_arrays=bool(cfg.get("cad.include_profile_arrays", True)),
        )
        print("=" * 80)
        print(f"sample_id={row['sample_id']} (ground truth shown here for local audit only: {row['label']})")
        print(json.dumps(state, indent=2, ensure_ascii=False))


def cmd_jev(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    desc = pd.read_csv(p["descriptors"])
    desc = desc[desc["extract_error"].fillna("").astype(str).str.len() == 0].copy()
    schema = _load_or_fit_schema(desc, p["schema"], force=args.refit_schema)
    question_path = Path(args.questions) if args.questions else (cfg.base_dir / "configs" / "questions.yaml")
    spec = load_questions(question_path)
    subset = select_subset(
        desc,
        splits=args.splits,
        max_per_class=args.max_per_class,
        seed=int(cfg.get("project.seed", 42)),
    )
    print(f"Selected {len(subset)} samples for Jev: {subset.groupby(['split','label']).size().to_dict()}")
    dry_path = cfg.output_path("jev", "dry_run_states.jsonl") if args.dry_run else None
    rows = run_jev_for_dataframe(
        subset,
        schema=schema,
        question_spec=spec,
        output_jsonl=p["jev"],
        cache_db=p["cache"],
        model=args.model or cfg.get("jev.model", "jev-latest"),
        retry_attempts=int(cfg.get("jev.retry_attempts", 5)),
        retry_base_seconds=float(cfg.get("jev.retry_base_seconds", 1.5)),
        timeout_seconds=float(cfg.get("jev.timeout_seconds", 30.0)),
        request_pause_seconds=float(cfg.get("jev.request_pause_seconds", 0.0)),
        include_numeric_context=not args.no_numeric_context,
        include_profile_arrays=not args.no_profile_arrays,
        force=args.force,
        dry_run_states=dry_path,
    )
    if args.dry_run:
        print(f"Dry-run states written to: {dry_path}")
        return
    print(f"Jev results: {p['jev']}")
    current = rows
    if rows and rows[-1].get("experiment_hash"):
        exp_hash = rows[-1]["experiment_hash"]
        current = [r for r in rows if r.get("experiment_hash") == exp_hash]
    else:
        exp_hash = None
    used = [r.get("usage", {}) for r in current]
    input_tokens = sum(int(u.get("input_tokens", 0)) for u in used)
    output_tokens = sum(int(u.get("output_tokens", 0)) for u in used)
    latencies = [float(r["latency_seconds"]) for r in current if r.get("cache_source") == "api" and r.get("latency_seconds") is not None]
    run_summary = {
        "experiment_hash": exp_hash,
        "responses": len(current),
        "resolved_models": sorted({str(r.get("model")) for r in current if r.get("model")}),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_rows": sum(r.get("cache_source") == "api" for r in current),
        "cache_rows": sum(r.get("cache_source") == "cache" for r in current),
        "mean_recorded_api_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
    }
    write_json(cfg.output_path("jev", "latest_run_summary.json"), run_summary)
    print(json.dumps(run_summary, indent=2))


def cmd_evaluate(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    c = evaluate_classification(
        descriptor_csv=p["descriptors"],
        jev_jsonl=p["jev"],
        output_dir=p["classification"],
        ece_bins=int(cfg.get("classification.ece_bins", 10)),
    )
    r = evaluate_retrieval(
        descriptor_csv=p["descriptors"],
        jev_jsonl=p["jev"],
        output_dir=p["retrieval"],
        ks=[int(x) for x in cfg.get("retrieval.ks", [1, 5, 10])],
        semantic_weight=float(args.semantic_weight if args.semantic_weight is not None else cfg.get("retrieval.semantic_weight", 0.5)),
    )
    print("Classification summary:")
    print(json.dumps(c, indent=2))
    print("Retrieval summary:")
    print(json.dumps(r, indent=2))


def cmd_qualitative(args) -> None:
    cfg = _load_cfg(args)
    p = _paths(cfg)
    report = generate_qualitative_report(
        descriptor_csv=p["descriptors"],
        classification_dir=p["classification"],
        retrieval_dir=p["retrieval"],
        output_dir=p["qualitative"],
        max_queries=int(args.max_queries or cfg.get("retrieval.max_queries_for_qualitative", 12)),
        top_k=int(args.top_k or cfg.get("retrieval.qualitative_top_k", 5)),
    )
    print(f"Open this file in a browser: {report}")


def cmd_all(args) -> None:
    # Reuse command functions while preserving common config values.
    scan_args = argparse.Namespace(config=args.config, dataset=args.dataset)
    cmd_scan(scan_args)
    extract_args = argparse.Namespace(config=args.config, no_resume=False, workers=args.workers)
    cmd_extract(extract_args)
    jev_args = argparse.Namespace(
        config=args.config,
        questions=None,
        splits=["train", "val", "test"],
        max_per_class=args.pilot_per_class,
        model=None,
        refit_schema=False,
        no_numeric_context=False,
        no_profile_arrays=False,
        force=False,
        dry_run=args.dry_run,
    )
    cmd_jev(jev_args)
    if args.dry_run:
        return
    eval_args = argparse.Namespace(config=args.config, semantic_weight=None)
    cmd_evaluate(eval_args)
    qual_args = argparse.Namespace(config=args.config, max_queries=None, top_k=None)
    cmd_qualitative(qual_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encoder-free Jev experiments on TMCAD")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="Scan TMCAD class folders and create stratified manifest")
    p.add_argument("--dataset", default=None, help="Override TMCAD root")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("extract", help="Extract deterministic descriptors from STEP models")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="OpenCASCADE extraction processes; start with 1, use 2-4 if memory allows")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("inspect-state", help="Print label-free JSON state(s) before API calls")
    p.add_argument("--sample-id", default=None)
    p.add_argument("-n", type=int, default=1)
    p.set_defaults(func=cmd_inspect_state)

    p = sub.add_parser("jev", help="Run Jev Choice + semantic Noul questions")
    p.add_argument("--questions", default=None, help="Override questions YAML")
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--max-per-class", type=int, default=None, help="Pilot cap within each selected split/class")
    p.add_argument("--model", default=None, help="e.g. jev-latest")
    p.add_argument("--refit-schema", action="store_true")
    p.add_argument("--no-numeric-context", action="store_true")
    p.add_argument("--no-profile-arrays", action="store_true")
    p.add_argument("--force", action="store_true", help="Ignore API response cache")
    p.add_argument("--dry-run", action="store_true", help="Write states but make no API requests")
    p.set_defaults(func=cmd_jev)

    p = sub.add_parser("evaluate", help="Run quantitative classification and retrieval evaluation")
    p.add_argument("--semantic-weight", type=float, default=None, help="Hybrid retrieval score weight for semantic signature")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("qualitative", help="Render classification/retrieval qualitative HTML report")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_qualitative)

    p = sub.add_parser("all", help="Scan -> extract -> Jev -> evaluate -> qualitative")
    p.add_argument("--dataset", default=None)
    p.add_argument("--pilot-per-class", type=int, default=20, help="Per split/class; 20 => at most 600 Jev calls")
    p.add_argument("--workers", type=int, default=1, help="Descriptor extraction processes")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_all)
    register_study_commands(sub)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
