"""Bounded real-CAD diagnostics and a blank human review pack, never synthetic expert truth."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..catalog import CLASSES, leakage_audit, inventory, import_descriptors, import_embeddings
from ..evaluation import classification_metrics
from ..geometry import extract_step, measure_catalog
from ..graph import build_graph
from ..bundle import ROOT, config_from, save_bundle
from ..io import object_hash, read_csv, read_json, receipt, sha256, write_csv, write_json
from ..query import parse_text
from ..reasoning import check_constraints, clearance_screen
from ..retrieval import Retriever


def tessellation_check(path, deflection=0.05):
    """Cross-representation check using tessellation, sharing OCCT import (not independent engineering truth)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRep import BRep_Tool
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("Tessellation STEP import failed")
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() < 1:
        raise ValueError("Tessellation root transfer failed")
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.3, False)
    explorer, points = TopExp_Explorer(shape, TopAbs_FACE), []
    cast = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    while explorer.More():
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(cast(explorer.Current()), location)
        if mesh is not None:
            transform = location.Transformation()
            for i in range(1, mesh.NbNodes()+1):
                node = mesh.Node(i).Transformed(transform)
                points.append([node.X(), node.Y(), node.Z()])
        explorer.Next()
    if not points:
        raise ValueError("No tessellated surface nodes")
    xyz = np.unique(np.asarray(points), axis=0)
    dims = np.ptp(xyz, axis=0)
    return xyz, {f"aabb_{axis}_mm": float(d) for axis, d in zip("xyz", dims)} | {
        "aabb_small_mm": float(min(dims)), "aabb_mid_mm": float(sorted(dims)[1]), "aabb_large_mm": float(max(dims))}


def preview(points, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = points[::max(1, len(points)//5000)]
    center = (points.max(0)+points.min(0))/2
    radius = max(float(np.ptp(points, axis=0).max()/2), 1e-9)
    fig = plt.figure(figsize=(7, 3.4))
    for i, azimuth in enumerate([35, 135], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(*points.T, s=1, color="#256b91", alpha=0.6, linewidths=0)
        for axis, middle in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            axis(middle-radius, middle+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(25, azimuth)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.02, "Surface preview only. Use the original CAD/drawing to verify interfaces.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .92])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def proposed_queries(catalog, functions, split, bound):
    mappings = {"support_rotating_shaft": "bearing", "connect_shafts": "coupling", "participate_belt_drive": "pulley"}
    result = []
    for function in functions:
        fid = function["function_id"]
        ref = min(p["part_id"] for p in catalog["parts"] if p["split"] == "train" and p["label"] == mappings[fid])
        for variant in ["text", "envelope", "example"]:
            result.append({"query_id": f"{split}_{fid}_{variant}", "scenario_id": f"{split}_{fid}_{variant}", "split": split,
                           "text": f"{function['label']}" + (f"; aabb_large_mm <= {bound} mm" if variant != "text" else ""),
                           "function_ids": [fid], "category_ids": [], "reference_part_id": ref if variant == "example" else None,
                           "constraints": [] if variant == "text" else [{"attribute": "aabb_large_mm", "op": "<=", "value": bound, "unit": "mm"}]})
    return result


def signature(result):
    return [(r["part_id"], r["score"], r["reasoning"]["status"]) for r in result["results"]]


def record_bounds_crosscheck(part, bounds, deflection):
    """Quarantine a disputed box even when only a non-largest axis differs."""
    delta = abs(part["measurements"]["aabb_large_mm"]["value"]-bounds["aabb_large_mm"])
    axis_delta = max(abs(part["measurements"][f"aabb_{axis}_mm"]["value"]-bounds[f"aabb_{axis}_mm"]) for axis in "xyz")
    agreed = axis_delta <= 4*deflection
    part["geometry_audit"]["bbox_tessellation_crosscheck"] = {
        "status": "agreement" if agreed else "unresolved_disagreement", "max_axis_delta_mm": axis_delta,
        "largest_extent_delta_mm": delta, "threshold_mm": 4*deflection,
        "scope": "shared-kernel surface tessellation check; not independent engineering truth"}
    if not agreed:
        for key, measurement in part["measurements"].items():
            if key.startswith("aabb_"):
                measurement["status"] = "computed_unverified"
                measurement["uncertainty_reason"] = "analytic vs surface-mesh bounds disagree; inspect native STEP"
    return agreed, delta, axis_delta


def ranking_rows(result, kind, split, query_id, method):
    return [{"geometry_kind": kind, "split": split, "query_id": query_id, "method": method, "rank": row["rank"],
             "part_id": row["part_id"], "score": row["score"], "constraint_status": row["reasoning"]["status"],
             "intent_support": row["intent_support"], "path_count": len(row["paths"])} for row in result["results"]]


def function_prior_diagnostic(results, weak_threshold=0.1):
    """Describe weak ranking evidence without assigning relevance or a calibrated acceptance rule."""
    rows = []
    for result in results:
        if result["method"] != "kg_full" or not result["query"]["function_ids"]:
            continue
        supports = [min(sum(path["contribution"] for path in row["paths"]
                            if path["path"][-1] == f"function:{fid}") for fid in result["query"]["function_ids"])
                    for row in result["results"]]
        rows.append({"geometry_kind": result["geometry_kind"], "split": result["split"], "query_id": result["query"]["query_id"],
                     "returned": len(supports), "top1_prior_support": supports[0] if supports else None,
                     "largest_returned_prior_support": max(supports) if supports else None,
                     "smallest_returned_prior_support": min(supports) if supports else None})
    weak = [row for row in rows if row["largest_returned_prior_support"] is not None and row["largest_returned_prior_support"] < weak_threshold]
    return {"result_type": "uncalibrated_function_prior_strength_diagnostic_not_relevance_truth",
            "illustrative_weak_prior_threshold": weak_threshold, "threshold_is_calibrated": False,
            "full_method_query_runs": len(rows), "runs_all_returned_priors_below_threshold": len(weak),
            "weak_runs": weak, "per_query": rows,
            "interpretation": "Nonzero classifier mass can yield a ranking with weak functional evidence. Top-k after dimensional filtering is not suitable-part acceptance. Independent expert development judgments are needed to validate any acceptance/abstention rule."}


def replay_technical(args):
    """Replay saved rankings without measuring CAD or touching expert annotation files."""
    import csv
    import io
    from ..bundle import config_from, load_bundle
    run = Path(args.run).resolve()
    cfg, queries = read_json(run / "resolved_config.json"), read_json(run / "proposed_queries.json")
    base_settings = cfg["retrieval_settings"] if "retrieval_settings" in cfg else config_from(None)
    catalog, knowledge, graph = load_bundle(run / "bundle")
    ranks, cases, results = [], [], []
    for kind in cfg["geometry_kinds"]:
        for split in ["dev", "test"]:
            settings = {**base_settings, "geometry_kind": kind,
                        "gallery_splits": ["val" if split == "dev" else "test"]}
            retriever = Retriever(catalog, knowledge, graph, settings)
            for query in [q for q in queries if q["split"] == split]:
                for method in cfg["methods"]:
                    result = retriever.rank(query, method, cfg["k"])
                    results.append(result | {"split": split})
                    ranks += ranking_rows(result, kind, split, query["query_id"], method)
                    cases.append((kind, split, query["query_id"], method, len(result["results"])))
    saved = read_csv(run / "rankings.csv")
    # Compare the original CSV representation exactly, including float precision.
    stream = io.StringIO(newline="")
    fields = list(saved[0]) if saved else list(ranks[0]) if ranks else []
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(ranks)
    replayed = list(csv.DictReader(io.StringIO(stream.getvalue())))
    original_cases = [(r["geometry_kind"], r["split"], r["query_id"], r["method"], int(r["returned"])) for r in read_csv(run / "per_query.csv")]
    report = {"result_type": "saved_input_exact_ranking_replay", "ranking_runs": len(cases), "ranking_rows": len(ranks),
              "rankings_exact_match": replayed == saved, "answer_counts_exact_match": cases == original_cases,
              "saved_rankings_sha256": object_hash(saved), "replayed_rankings_sha256": object_hash(replayed),
              "scope": "Saved bundle/query/config replay; does not independently validate engineering relevance or rebuild upstream artifacts."}
    report["passed"] = report["rankings_exact_match"] and report["answer_counts_exact_match"]
    write_json(args.output, report)
    write_json(Path(args.output).parent / "function_prior_diagnostic.json", function_prior_diagnostic(results))
    write_json(Path(args.output).with_suffix(".manifest.json"), receipt("technical-replay", [run / name for name in
               ["resolved_config.json", "proposed_queries.json", "rankings.csv", "per_query.csv", "bundle/catalog.json", "bundle/knowledge.json", "bundle/graph.json"]], vars(args)))
    return report


def prepare_review_pack(catalog, queries, sources_dir, output):
    """Preserve previous expert work. Human grades and reviewers stay blank."""
    from ..bundle import prepare_annotations
    output = Path(output)
    prepare_annotations(output / "knowledge", catalog)
    functions = read_csv(Path(sources_dir) / "functions.csv")
    relations = read_csv(Path(sources_dir) / "class_functions.csv")
    # Authoring grid has all combinations, with suggestions but no asserted expert review.
    rows = []
    for label in CLASSES:
        for function in functions:
            suggestion = next((r for r in relations if r["class_id"] == label and r["function_id"] == function["function_id"]), None)
            rows.append({"class_id": label, "function_id": function["function_id"], "relevance": "", "confidence": "",
                         "scope": suggestion["scope"] if suggestion else "", "caveat": suggestion["caveat"] if suggestion else "",
                         "reviewer": "", "evidence": suggestion["evidence"] if suggestion else "", "status": "draft"})
    functions = [{k: ("draft" if k == "status" else v) for k, v in f.items() if k != "source_id"} for f in functions]
    # annotate-init created header-only files: populate once, never replace populated expert files.
    for name, prepared in [("functions.csv", functions), ("class_functions.csv", rows)]:
        target = output / "knowledge" / name
        if not read_csv(target):
            write_csv(target, prepared)
    for split in ["dev", "test"]:
        path = output / "grading" / split
        path.mkdir(parents=True, exist_ok=True)
        query_path = path / "queries.json"
        if not query_path.exists():
            write_json(query_path, [q for q in queries if q["split"] == split])
        gallery = [p for p in catalog["parts"] if p["split"] == ("val" if split == "dev" else "test")]
        judge_rows = [{"query_id": q["query_id"], "part_id": p["part_id"], "grade": "", "constraint_verdict": "",
                       "reviewer": "", "evidence": "", "source": "", "split": split}
                      for q in read_json(query_path) for p in gallery]
        target = path / "judgments.csv"
        if not target.exists():
            write_csv(target, judge_rows)
    candidates = [{"part_id": p["part_id"], "candidate_id": c["candidate_id"], "suggested_role": "unknown",
                   "candidate_diameter_mm": c["diameter_mm"], "reviewed_role": "", "reviewer": "", "evidence": ""}
                  for p in catalog["parts"] for c in p.get("cylinder_candidates", [])]
    path = output / "interface_candidates_to_review.csv"
    if not path.exists():
        write_csv(path, candidates, ["part_id", "candidate_id", "suggested_role", "candidate_diameter_mm", "reviewed_role", "reviewer", "evidence"])
    guide = output / "START_HERE.md"
    if not guide.exists():
        guide.write_text("""# Review pack

1. Review the three draft function definitions and the 30-row class/function grid in `knowledge/`. Fill prior weights, scope, caveats, evidence and your actual reviewer ID. Set `status=curated` only after review. Missing relationships are unknown, not negatives.
2. Review/edit `grading/dev/queries.json` and `grading/test/queries.json` before any grading. These are machine-proposed scenarios, not expert requirements. Current reference exemplars were selected by dataset class for technical testing; choose appropriate training exemplars independently. Keep development and test scenario families separate. Freeze the pack after this review.
3. Grade `grading/*/judgments.csv` by inspecting individual CADs. All relevance grades and verdicts are blank. Use grades 0/1/2, constraint verdict pass/fail/unknown/not_requested, and `source=independent_expert` only for your own independent assessment. Record reviewer and instance-specific evidence. Hide model outputs and priors during grading.
4. `previews/` provides neutral-ID surface previews; `curator_manifest.csv` maps IDs to actual CAD files. Keep this manifest separate from blinded grading. Open original STEP/drawings for uncertain or interface judgments. Previews are not fit verification.
5. Interface cylinder candidates have role unknown. Annotate selected verified bores/shafts in `knowledge/interfaces.csv` only if studying them. Do not infer production tolerances from nominal CAD.

No web app is required. CSV files work with spreadsheet editors and your usual CAD viewer. See the repository annotation guide for field definitions and exact evaluation commands.
""", encoding="utf-8")


def run_technical(args):
    from ..bundle import ROOT, config_from, save_bundle
    import argparse
    cfg = read_json(args.config)
    cfg["retrieval_settings"] = config_from(cfg.get("retrieval_config"))
    output = Path(args.output).resolve()
    started = time.monotonic()
    timeout = cfg.get("timeout_seconds", 300)
    if not 1 <= cfg["parts_per_class_split"] <= 2 or cfg["maximum_parts"] > 60 or cfg["maximum_geometry_parts"] > 40:
        raise ValueError("Technical run is bounded to 60 selected parts and 40 fresh geometric checks")
    def check_time():
        if time.monotonic()-started > timeout:
            raise RuntimeError("Technical run exceeded its configured timeout between parts/stages")
    imported_path = output / "catalog_import"
    if not args.reuse_import or not (imported_path / "catalog.json").exists():
        run_pilot(argparse.Namespace(dataset=args.dataset, legacy_root=args.legacy_root, per_class_split=cfg["parts_per_class_split"],
                                    seed=cfg["seed"], measure_limit=0, config=cfg.get("retrieval_config"), output=str(imported_path)))
    catalog = read_json(imported_path / "catalog.json")
    if catalog["audit"]["seed"] != cfg["seed"] or catalog["audit"]["per_class_split"] != cfg["parts_per_class_split"]:
        raise ValueError("Reused import has different cohort settings; rerun without --reuse-import in a new output directory")
    if Path(catalog["dataset_root"]).resolve() != Path(args.dataset).resolve():
        raise ValueError("Reused import references a different dataset root")
    if len(catalog["parts"]) > cfg["maximum_parts"] or catalog["audit"]["leakage"]["exact_byte_cross_split_duplicates"]:
        raise ValueError("Oversized cohort or exact cross-split duplicates")
    for part in catalog["parts"]:
        if sha256(part["absolute_path"]) != part["file_sha256"]:
            raise ValueError("STEP file changed after legacy import")
    legacy = Path(args.legacy_root).resolve() / "jev_tmcad_v0/outputs/study/full"
    producer = read_json(legacy / "geometry_embeddings.meta.json")
    if sha256(legacy / "geometry_encoder.pt") != producer["checkpoint_sha256"]:
        raise ValueError("Producer checkpoint hash mismatch")
    geometry_rows, mesh_bounds, errors = [], {}, []
    heldout = sorted([p for p in catalog["parts"] if p["split"] != "train"], key=lambda p: Path(p["absolute_path"]).stat().st_size)
    if len(heldout) > cfg["maximum_geometry_parts"]:
        raise ValueError("Too many selected geometry parts")
    for i, part in enumerate(heldout):
        check_time()
        if Path(part["absolute_path"]).stat().st_size > cfg["maximum_step_bytes"]:
            raise ValueError("Selected STEP exceeds technical-run size budget")
        t0 = time.monotonic()
        try:
            measured = extract_step(part["absolute_path"])
            part.update(measured)
            xyz, bounds = tessellation_check(part["absolute_path"], cfg["mesh_deflection_mm"])
            mesh_bounds[part["part_id"]] = bounds
            analytic = part["measurements"]["aabb_large_mm"]["value"]
            crosscheck_ok, delta, axis_delta = record_bounds_crosscheck(part, bounds, cfg["mesh_deflection_mm"])
            if not crosscheck_ok:
                mesh_bounds.pop(part["part_id"], None)
            geometry_rows.append({"part_id": part["part_id"], "split": part["split"], "shape_valid": part["geometry_audit"]["valid_shape"],
                                  "analytic_aabb_large_mm": analytic, "mesh_aabb_large_mm": bounds["aabb_large_mm"],
                                  "abs_delta_mm": delta, "max_axis_delta_mm": axis_delta, "crosscheck_within_4_deflections": int(crosscheck_ok),
                                  "cylinder_candidates": len(part["cylinder_candidates"]), "seconds": time.monotonic()-t0})
            if cfg["previews"]:
                preview(xyz, output / f"human_review/previews/{part['part_id']}.png", f"Part {part['part_id']}")
        except (ValueError, RuntimeError, TypeError) as exc:
            errors.append({"part_id": part["part_id"], "error": str(exc)})
        print(f"Geometry {i+1}/{len(heldout)}: {part['part_id']}", flush=True)
    # Training exemplars receive explicitly labeled cached normalized previews, not dimensional evidence.
    if cfg["previews"]:
        for part in [p for p in catalog["parts"] if p["split"] == "train"]:
            check_time()
            cache = legacy / f"mesh/{part['part_id']}.npz"
            if cache.exists():
                with np.load(cache, allow_pickle=False) as data:
                    preview(data["features"][:, :3], output / f"human_review/previews/{part['part_id']}.png", f"Training exemplar {part['part_id']} (normalized cache)")
    catalog["measurement_errors"] = errors
    catalog["audit"]["leakage"] = leakage_audit(catalog)
    sources = ROOT / "data/knowledge/priors"
    knowledge, graph = save_bundle(catalog, sources, output / "bundle", "include_source_supported")
    queries = []
    for split in ["dev", "test"]:
        queries += proposed_queries(catalog, knowledge["functions"], split, cfg["reference_bounds_mm"][split])
    write_json(output / "proposed_queries.json", queries)
    prepare_review_pack(catalog, queries, sources, output / "human_review")
    write_csv(output / "human_review/curator_manifest.csv", [{"part_id": p["part_id"], "split": p["split"], "dataset_label": p["label"],
              "cad_file": p["absolute_path"], "preview": f"previews/{p['part_id']}.png", "source_sha256": p["file_sha256"]} for p in catalog["parts"]])
    per_query, ranks, invariants, results = [], [], [], []
    for kind in cfg["geometry_kinds"]:
        for split in ["dev", "test"]:
            settings = {**cfg["retrieval_settings"], "geometry_kind": kind, "gallery_splits": ["val" if split == "dev" else "test"]}
            retriever = Retriever(catalog, knowledge, graph, settings)
            gallery = retriever.candidate_ids()
            poisoned = deepcopy(catalog)
            for part in poisoned["parts"]:
                part["label"] = CLASSES[(CLASSES.index(part["label"])+1) % len(CLASSES)]
            poisoned_retriever = Retriever(poisoned, knowledge, build_graph(poisoned, knowledge), settings)
            for query in [q for q in queries if q["split"] == split]:
                check_time()
                qid = query["query_id"]
                full = retriever.rank(query, "kg_full", len(gallery))
                flat = retriever.rank(query, "flat_full", len(gallery))
                altered = poisoned_retriever.rank(query, "kg_full", len(gallery))
                invariants += [{"check": name, "geometry_kind": kind, "split": split, "query_id": qid, "passed": int(passed)}
                               for name, passed in [("flat_graph_equivalence", signature(full) == signature(flat)),
                                                    ("dataset_labels_not_used_in_serving", signature(full) == signature(altered)),
                                                    ("deterministic_repeat", signature(full) == signature(retriever.rank(query, "kg_full", len(gallery))))]]
                if not query["reference_part_id"]:
                    parsed = parse_text(query["text"], knowledge)
                    invariants.append({"check": "text_json_same_ranking", "geometry_kind": kind, "split": split, "query_id": qid,
                                       "passed": int(signature(full) == signature(retriever.rank(parsed, "kg_full", len(gallery))))})
                for method in cfg["methods"]:
                    t0 = time.monotonic()
                    result = retriever.rank(query, method, cfg["k"])
                    elapsed = (time.monotonic()-t0)*1000
                    results.append(result | {"split": split})
                    failed, known, uncertain = 0, 0, 0
                    for row in result["results"]:
                        pid = row["part_id"]
                        for constraint in query["constraints"]:
                            bounds = mesh_bounds.get(pid)
                            if not bounds or abs(bounds[constraint["attribute"]]-constraint["value"]) <= 4*cfg["mesh_deflection_mm"]:
                                uncertain += 1
                                continue
                            known += 1
                            failed += int(bounds[constraint["attribute"]] > constraint["value"])
                    ranks += ranking_rows(result, kind, split, qid, method)
                    if result["constraints_enforced"]:
                        invariants.append({"check": "no_failed_or_unknown_constraints_returned", "geometry_kind": kind, "split": split, "query_id": qid,
                                           "passed": int(all(r["reasoning"]["status"] in {"pass", "not_requested"} for r in result["results"]))})
                    per_query.append({"geometry_kind": kind, "split": split, "query_id": qid, "method": method, "returned": len(result["results"]),
                                      "abstained": int(not result["results"]), "constraint_rejected": result["constraint_rejected"],
                                      "tessellation_known_checks": known, "tessellation_failed_checks": failed, "near_boundary_or_missing_checks": uncertain,
                                      "latency_ms": elapsed})
            print(f"Ranking matrix completed: {kind}/{split}", flush=True)
    summaries = []
    groups = defaultdict(list)
    for row in per_query:
        groups[row["geometry_kind"], row["split"], row["method"]].append(row)
    for (kind, split, method), rows in groups.items():
        known = sum(r["tessellation_known_checks"] for r in rows)
        summaries.append({"geometry_kind": kind, "split": split, "method": method, "queries": len(rows),
                          "answer_fraction": float(1-np.mean([r["abstained"] for r in rows])),
                          "returned_total": sum(r["returned"] for r in rows), "tessellation_known_checks": known,
                          "tessellation_failed_checks": sum(r["tessellation_failed_checks"] for r in rows),
                          "tessellation_violation_fraction": sum(r["tessellation_failed_checks"] for r in rows)/known if known else None,
                          "near_boundary_or_missing_checks": sum(r["near_boundary_or_missing_checks"] for r in rows),
                          "median_latency_ms": float(np.median([r["latency_ms"] for r in rows]))})
    # Exercise missing interface and standalone numerical discovery on real parts.
    settings = {**cfg["retrieval_settings"], "geometry_kind": "embedding", "gallery_splits": ["test"]}
    retriever = Retriever(catalog, knowledge, graph, settings)
    stress = {"query_id": "unannotated_bore", "function_ids": ["connect_shafts"],
              "constraints": [{"attribute": "bore_diameter_mm", "op": "between", "value": [19, 21], "unit": "mm"}]}
    result = retriever.rank(stress)
    invariants.append({"check": "missing_interface_abstains", "geometry_kind": "embedding", "split": "test", "query_id": stress["query_id"], "passed": int(result["status"] == "abstained")})
    numeric = {"query_id": "numeric_only", "constraints": [{"attribute": "aabb_large_mm", "op": "<=", "value": 150, "unit": "mm"}]}
    numeric_result = retriever.rank(numeric)
    invariants.append({"check": "numeric_only_can_discover", "geometry_kind": "embedding", "split": "test", "query_id": "numeric_only", "passed": int(numeric_result["status"] == "ranked")})
    test_ids = retriever.candidate_ids()
    clearance = clearance_screen(catalog, test_ids[0], "unreviewed", test_ids[-1], "unreviewed", 0, 0.1)
    invariants.append({"check": "no_unsupported_compatibility_claim", "geometry_kind": "embedding", "split": "test", "query_id": "missing_clearance_interfaces", "passed": int(clearance["status"] == "unknown" and clearance["assembly_compatibility"] == "unknown")})
    report = {"result_type": "real_CAD_technical_diagnostics_not_functional_relevance_evaluation", "selected_parts": len(catalog["parts"]),
              "part_splits": dict(Counter(p["split"] for p in catalog["parts"])), "fresh_step_checks": len(geometry_rows), "measurement_errors": errors,
              "geometry_crosschecks_within_tolerance": sum(r["crosscheck_within_4_deflections"] for r in geometry_rows),
              "max_geometry_delta_mm": max((r["abs_delta_mm"] for r in geometry_rows), default=None),
              "quarantined_bbox_part_ids": [r["part_id"] for r in geometry_rows if not r["crosscheck_within_4_deflections"]],
              "provisional_function_priors": len(knowledge["class_functions"]), "human_reviewed_relations": 0,
              "proposed_queries": len(queries), "ranking_runs": len(per_query), "invariants": {"checks": len(invariants), "passed": sum(r["passed"] for r in invariants)},
              "classification_diagnostic": classification_metrics(retriever), "methods": summaries, "elapsed_seconds": time.monotonic()-started,
              "limits": ["No human functional relevance grades were created. No functional nDCG or MAP is reported.",
                         "Tessellation check shares the CAD kernel/import; it is a numerical cross-check, not independent expert truth.",
                         "Disputed analytic/surface-mesh AABBs are quarantined as computed_unverified; they cannot pass mandatory dimensional filters.",
                         "Generic source-backed priors require expert scope review for TMCAD use. Scores do not verify part suitability.",
                         "Queries and exemplars are machine-proposed technical cases, not independently authored held-out engineering needs.",
                         "Reference-free geometry/category baselines may abstain; coverage is not an effectiveness comparison.",
                         "Near-duplicate family separation remains unverified."]}
    write_csv(output / "geometry_crosschecks.csv", geometry_rows)
    write_csv(output / "invariants.csv", invariants)
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "rankings.csv", ranks)
    write_csv(output / "summary.csv", summaries)
    write_json(output / "summary.json", report)
    write_json(output / "explanations.json", results)
    write_json(output / "function_prior_diagnostic.json", function_prior_diagnostic(results))
    write_json(output / "stress_cases.json", {"missing_interface": result, "numeric_only": numeric_result, "clearance_missing_evidence": clearance})
    write_json(output / "resolved_config.json", cfg)
    semantic = {name: read_csv(output / name) for name in ["rankings.csv", "invariants.csv"]}
    write_json(output / "semantic_results_digest.json", {"sha256": object_hash(semantic), "excludes": "runtime/latency/absolute path metadata"})
    write_json(output / "run_manifest.json", receipt("technical", [args.config, cfg.get("retrieval_config") or ROOT / "configs/core.json",
                                                               imported_path / "catalog.json", *sources.glob("*"),
                                                               legacy / "geometry_embeddings.meta.json", legacy / "geometry_encoder.pt"], vars(args)))
    return report


def run_pilot(args):
    if not 1 <= args.per_class_split <= 3 or not 0 <= args.measure_limit <= 10:
        raise ValueError("Pilot is bounded to <=3 parts per class/split and <=10 STEP measurements")
    base = Path(args.legacy_root) / "jev_tmcad_v0" / "outputs"
    catalog = inventory(args.dataset, base / "manifest.csv", args.per_class_split, getattr(args, "seed", 17))
    import_descriptors(catalog, base / "descriptors" / "tmcad_descriptors.csv")
    import_embeddings(catalog, base / "study" / "full" / "geometry_embeddings.csv")
    # Record checkpoint/producer metadata without loading or retraining a model.
    for path in [base / "study/full/geometry_encoder.pt", base / "study/full/geometry_encoder.json", base / "study/full/geometry_embeddings.meta.json"]:
        if path.exists():
            catalog["artifacts"][path.name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    candidates = sorted([p for p in catalog["parts"] if p["split"] == "test"], key=lambda p: Path(p["absolute_path"]).stat().st_size)[:args.measure_limit]
    errors = measure_catalog({"parts": candidates}) if candidates else []
    catalog["measurement_errors"] = errors
    catalog["audit"]["leakage"] = leakage_audit(catalog)
    knowledge, graph = save_bundle(catalog, ROOT / "data/annotations" / "templates", args.output)
    config = config_from(args.config)
    retriever = Retriever(catalog, knowledge, graph, config)
    train = [p["part_id"] for p in catalog["parts"] if p["split"] == "train" and p["part_id"] in retriever.vectors]
    if not train:
        raise ValueError("No matched training exemplar")
    query = {"query_id": "real_geometry_validation", "text": "Similar geometry to a training exemplar", "reference_part_id": train[0], "split": "test"}
    result = retriever.rank(query, "geometry_class", 5)
    write_json(Path(args.output) / "retrieval_example.json", result)
    measured = [p for p in candidates if p.get("geometry_audit")]
    if measured:
        pid = measured[0]["part_id"]
        value = measured[0]["measurements"]["aabb_large_mm"]["value"]
        # Numeric pass/fail boundary tests, not relevance ground truth.
        from ..reasoning import check_constraints
        checks = {name: check_constraints(measured[0], [{"attribute": "aabb_large_mm", "op": "<=", "value": bound, "unit": "mm"}])
                  for name, bound in [("looser_bound", value*1.05), ("tighter_bound", value*0.95)]}
        write_json(Path(args.output) / "measurement_validation.json", {"part_id": pid, "checks": checks,
                                                                      "bore_check": check_constraints(measured[0], [{"attribute": "bore_diameter_mm", "op": "<=", "value": 20, "unit": "mm"}])})
    report = {"result_type": "real_data_plumbing_only", "audit": catalog["audit"], "matched_gallery": len(retriever.candidate_ids()),
              "retrieval_status": result["status"], "step_measurements": len(measured), "measurement_errors": errors,
              "curated_engineering_relations": len(knowledge["class_functions"]),
              "notes": ["No human relevance ground truth supplied; no functional retrieval performance measured.",
                        "Reused supervised geometry embeddings and classifier logits; not CAD-JEPA/TMCAD evidence."]}
    write_json(Path(args.output) / "validation.json", report)
    inputs = [v["path"] for v in catalog["artifacts"].values() if "path" in v]
    write_json(Path(args.output) / "run_manifest.json", receipt("pilot", [base / "manifest.csv", *inputs], vars(args)))
    return report


