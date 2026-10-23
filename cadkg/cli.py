from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import inventory, import_descriptors, import_embeddings, import_predictions, leakage_audit
from .classifier import fit_probe, predict_probe
from .demo import run_demo
from .evaluation import evaluate, evaluate_clearance_cases
from .geometry import measure_catalog
from .io import read_json, receipt, write_json
from .query import parse_text
from .reasoning import clearance_screen
from .retrieval import Retriever

from .bundle import ROOT, config_from, load_bundle, save_bundle, prepare_annotations


def parser():
    parser = argparse.ArgumentParser(description="CAD retrieval, visual expert review and reproducible experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    sub = subs.add_parser("annotate", help="Local visual review; separate task, knowledge and candidate decisions")
    sub.add_argument("--action", choices=["serve", "export", "compare"], default="serve")
    sub.add_argument("--run", default="outputs/intent_batch2")
    sub.add_argument("--workspace", default="outputs/expert_review")
    sub.add_argument("--tasks", help="Source-bound reviewer instructions, normally annotations/review_tasks.json")
    sub.add_argument("--port", type=int, default=8765)
    sub.add_argument("--open", action="store_true", help="Open the local browser interface")
    sub.add_argument("--reviewer", help="Reviewer ID for export")
    sub.add_argument("--reviewers", nargs=2, help="Two distinct reviewer IDs for agreement comparison")
    sub.add_argument("--output", help="Export/comparison directory; defaults inside the review workspace")
    sub.add_argument("--validation-fixture", action="store_true", help="Mark all saved decisions as synthetic UI validation, never expert truth")
    sub = subs.add_parser("expert-evaluate", help="Evaluate actual human records on frozen rankings; unresolved families stay blocked")
    sub.add_argument("--run", default="outputs/intent_batch2")
    sub.add_argument("--review", required=True, help="review.json downloaded or exported from the annotation app")
    sub.add_argument("--output", default="outputs/intent_batch2_expert")
    sub.add_argument("--tasks")
    sub.add_argument("--allow-partial", action="store_true", help="Evaluate only fully judged, task-approved families; report all exclusions")
    sub = subs.add_parser("inventory")
    sub.add_argument("--dataset", required=True)
    sub.add_argument("--manifest")
    sub.add_argument("--per-class-split", type=int)
    sub.add_argument("--seed", type=int, default=17)
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("import-legacy")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--descriptors")
    sub.add_argument("--embeddings")
    sub.add_argument("--predictions")
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("measure")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--limit", type=int)
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("build")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--annotations", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--knowledge-policy", choices=["curated_only", "include_source_supported"], default="curated_only")
    sub = subs.add_parser("query")
    sub.add_argument("--bundle", required=True)
    group = sub.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-file")
    group.add_argument("--text")
    sub.add_argument("--method", default="kg_full")
    sub.add_argument("--config")
    sub.add_argument("--k", type=int, default=5)
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("evaluate")
    sub.add_argument("--bundle", required=True)
    sub.add_argument("--queries", required=True)
    sub.add_argument("--judgments", required=True)
    sub.add_argument("--config")
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("annotate-init")
    sub.add_argument("--catalog")
    sub.add_argument("--queries")
    sub.add_argument("--config")
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("train-probe")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--output", required=True, help="Directory for model.json and predicted_catalog.json")
    sub = subs.add_parser("reason")
    sub.add_argument("--bundle", required=True)
    sub.add_argument("--case", required=True, help="JSON arguments to clearance_screen")
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("evaluate-reasoning")
    sub.add_argument("--bundle", required=True)
    sub.add_argument("--cases", required=True)
    sub.add_argument("--output", required=True)
    sub = subs.add_parser("demo")
    sub.add_argument("--output", default="outputs/demo")
    sub = subs.add_parser("pilot")
    sub.add_argument("--dataset", default="TMCAD")
    sub.add_argument("--legacy-root", default="../jepa_cad")
    sub.add_argument("--per-class-split", type=int, default=1)
    sub.add_argument("--measure-limit", type=int, default=3)
    sub.add_argument("--config")
    sub.add_argument("--output", default="outputs/pilot")
    sub = subs.add_parser("technical", help="Bounded real-CAD checks and a blank expert annotation pack")
    sub.add_argument("--dataset", default="TMCAD")
    sub.add_argument("--legacy-root", default="../jepa_cad")
    sub.add_argument("--config", default=str(ROOT / "configs/technical.json"))
    sub.add_argument("--reuse-import", action="store_true", help="Reuse this output directory's verified cohort import")
    sub.add_argument("--output", default="outputs/technical")
    sub = subs.add_parser("technical-replay", help="Reproduce saved technical rankings without another geometry run")
    sub.add_argument("--run", default="outputs/technical")
    sub.add_argument("--output", default="outputs/technical/reproducibility.json")
    sub = subs.add_parser("reference-prepare", help="Neutral detailed CAD renders for explicitly AI-authored reference inspection")
    sub.add_argument("--bundle", default="outputs/technical/bundle")
    sub.add_argument("--proposed-queries", default="outputs/technical/proposed_queries.json")
    sub.add_argument("--output", default="outputs/ai_reference/evidence")
    sub = subs.add_parser("reference-run", help="Evaluate explicitly AI-authored source-bound judgments; no expert truth claims")
    sub.add_argument("--bundle", default="outputs/technical/bundle")
    sub.add_argument("--annotations", default="annotations/ai_reference")
    sub.add_argument("--evidence", default="outputs/ai_reference/evidence")
    sub.add_argument("--output", default="outputs/ai_reference/results")
    sub = subs.add_parser("reference-report", help="Render quantitative plots and actual-CAD qualitative evidence for the AI reference study")
    sub.add_argument("--run", default="outputs/ai_reference/results")
    sub.add_argument("--evidence", default="outputs/ai_reference/evidence")
    sub.add_argument("--output", default="outputs/ai_reference/report")
    sub = subs.add_parser("intent-batch", help="Expanded source-bound intent-family experiment matrix")
    sub.add_argument("--config", default=str(ROOT / "configs/intent_batch2.json"))
    sub.add_argument("--output", default="outputs/intent_batch2")
    sub.add_argument("--stage", choices=["all", "search", "evidence", "evaluate", "report", "validate"], default="all")
    sub = subs.add_parser("full-scale", help="Full-corpus paired experiments; function truth remains separate from labels")
    sub.add_argument("--stage", choices=["all", "prepare", "measure", "build", "class", "functional-search", "pool-evidence", "functional-evaluate", "report", "validate"], default="all")
    sub.add_argument("--dataset", default="TMCAD")
    sub.add_argument("--legacy-root", default="../jepa_cad")
    sub.add_argument("--config", default="configs/full_scale.json")
    sub.add_argument("--output", default="outputs/full_scale")
    sub.add_argument("--annotations", default="annotations/full_scale_ai.json")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "annotate":
            from .review import main as review_main
            review_main(args)
        elif args.command == "expert-evaluate":
            from .experiments.expert import evaluate_expert
            print(evaluate_expert(args.run, args.review, args.output, args.allow_partial, args.tasks))
        elif args.command == "inventory":
            if args.per_class_split is not None and args.per_class_split < 1:
                raise ValueError("per-class-split must be positive")
            catalog = inventory(args.dataset, args.manifest, args.per_class_split, args.seed)
            write_json(args.output, catalog)
            print(f"Inventoried {catalog['audit']['available_total']} STEP files; selected {len(catalog['parts'])}")
        elif args.command == "import-legacy":
            catalog = read_json(args.catalog)
            for path, fn in [(args.descriptors, import_descriptors), (args.embeddings, import_embeddings), (args.predictions, import_predictions)]:
                if path:
                    fn(catalog, path)
            catalog["audit"]["leakage"] = leakage_audit(catalog)
            write_json(args.output, catalog)
            print(f"Imported artifacts for {len(catalog['parts'])} selected parts")
        elif args.command == "measure":
            if args.limit is not None and args.limit < 1:
                raise ValueError("limit must be positive")
            catalog = read_json(args.catalog)
            errors = measure_catalog(catalog, args.limit)
            write_json(args.output, catalog)
            print(f"STEP measurement errors: {len(errors)}; details in output")
            if errors:
                return 1
        elif args.command == "build":
            knowledge, graph = save_bundle(read_json(args.catalog), args.annotations, args.output, args.knowledge_policy)
            print(f"Built {len(graph['nodes'])} nodes, {len(graph['edges'])} edges; admitted priors: {len(knowledge['class_functions'])}; policy: {args.knowledge_policy}")
        elif args.command == "query":
            catalog, knowledge, graph = load_bundle(args.bundle)
            config = config_from(args.config)
            query = read_json(args.query_file) if args.query_file else parse_text(args.text, knowledge)
            result = Retriever(catalog, knowledge, graph, config).rank(query, args.method, args.k)
            write_json(args.output, result)
            print(f"{result['status']}: {len(result['results'])} candidates")
        elif args.command == "evaluate":
            catalog, knowledge, graph = load_bundle(args.bundle)
            if catalog.get("dataset_type") == "synthetic_fixture":
                raise ValueError("Use demo for synthetic fixtures; research evaluation requires real independent expert judgments")
            config = config_from(args.config)
            result = evaluate(Retriever(catalog, knowledge, graph, config), read_json(args.queries), args.judgments, config, args.output)
            write_json(Path(args.output) / "run_manifest.json", receipt("evaluate", [Path(args.bundle)/n for n in ["catalog.json", "knowledge.json", "graph.json"]] + [args.queries, args.judgments, args.config], config))
            print(f"Evaluated {len(result['methods'])} methods using independent closed-gallery judgments")
        elif args.command == "annotate-init":
            prepare_annotations(args.output, read_json(args.catalog) if args.catalog else None, read_json(args.queries) if args.queries else None, config_from(args.config))
            print(f"Annotation templates saved to {args.output}")
        elif args.command == "train-probe":
            catalog = read_json(args.catalog)
            state = fit_probe(catalog, Path(args.output) / "model.json")
            predict_probe(catalog, state, Path(args.output) / "model.json")
            write_json(Path(args.output) / "predicted_catalog.json", catalog)
            write_json(Path(args.output) / "run_manifest.json", receipt("train-probe", [args.catalog], {"selected_alpha": state["alpha"]}))
            print("Small ridge probe fitted on train; regularizer selected on val")
        elif args.command == "reason":
            catalog, _, _ = load_bundle(args.bundle)
            result = clearance_screen(catalog, **read_json(args.case))
            write_json(args.output, result)
            print(f"Diameter-clearance screen: {result['status']}; assembly compatibility: unknown")
        elif args.command == "evaluate-reasoning":
            catalog, _, _ = load_bundle(args.bundle)
            result = evaluate_clearance_cases(catalog, read_json(args.cases), args.output)
            write_json(Path(args.output).with_suffix(".manifest.json"), receipt("evaluate-reasoning", [Path(args.bundle)/"catalog.json", args.cases], vars(args)))
            print(f"Evaluated {len(result)} independent clearance cases")
        elif args.command == "demo":
            run_demo(args.output)
            print("Fictional fixture completed; outputs are plumbing checks only")
        elif args.command == "pilot":
            from .experiments.technical import run_pilot
            result = run_pilot(args)
            print(f"Real-data pilot: {result['audit']['selected_total']} parts, {result['step_measurements']} measured, {result['matched_gallery']} gallery candidates")
            if result["measurement_errors"]:
                print("Measurement failures preserved in validation.json")
                return 1
        elif args.command == "technical":
            from .experiments.technical import run_technical
            result = run_technical(args)
            print(f"Technical run: {result['selected_parts']} parts, {result['fresh_step_checks']} fresh geometric checks, {result['ranking_runs']} ranking runs")
            print(f"Invariants passed: {result['invariants']['passed']}/{result['invariants']['checks']}; human relevance judgments remain blank")
            if result["measurement_errors"] or result["invariants"]["passed"] != result["invariants"]["checks"]:
                return 1
        elif args.command == "technical-replay":
            from .experiments.technical import replay_technical
            result = replay_technical(args)
            print(f"Exact saved-input replay: {result['passed']}; {result['ranking_runs']} ranking runs, {result['ranking_rows']} rows")
            if not result["passed"]:
                return 1
        elif args.command == "reference-prepare":
            from .experiments.reference import prepare_evidence
            print(prepare_evidence(args))
        elif args.command == "reference-run":
            from .experiments.reference import run_reference
            result = run_reference(args)
            print(f"AI-authored reference study: {result['judgment_pairs']} pairs, {result['ranking_runs']} ranking runs; human expert reviewed: False")
        elif args.command == "reference-report":
            from .experiments.reference_report import make_report
            print(make_report(args))
        elif args.command == "intent-batch":
            from .experiments.intent import main as intent_main
            intent_main(args)
        elif args.command == "full-scale":
            from .experiments.full import main as full_main
            full_main(args)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(f"cadkg: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
