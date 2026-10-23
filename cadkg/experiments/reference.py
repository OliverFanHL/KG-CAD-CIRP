"""AI-authored evidence/reference study, explicitly separate from independent expert evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import object_hash, read_csv, read_json, receipt, sha256, write_csv, write_json


from ..geometry import surface_mesh, draw_surface


def prepare_evidence(args):
    from ..bundle import load_bundle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    catalog, _, _ = load_bundle(args.bundle)
    output = Path(args.output).resolve()
    gallery = sorted([p for p in catalog["parts"] if p["split"] == "test"], key=lambda p: p["part_id"])
    if len(gallery) > 20:
        raise ValueError("Reference inspection is bounded to 20 test candidates")
    refs = sorted({q["reference_part_id"] for q in read_json(args.proposed_queries) if q.get("reference_part_id")})
    selected = [(f"T{i:02}", p) for i, p in enumerate(gallery, 1)]
    selected += [(f"E{i:02}", next(p for p in catalog["parts"] if p["part_id"] == pid)) for i, pid in enumerate(refs, 1)]
    manifest, measurements, meshes = [], {}, {}
    for code, part in selected:
        if sha256(part["absolute_path"]) != part["file_sha256"]:
            raise ValueError("STEP source changed")
        triangles, evidence = surface_mesh(part["absolute_path"])
        meshes[code] = triangles
        measurements[part["part_id"]] = evidence | {"inspection_id": code, "file_sha256": part["file_sha256"],
            "aabb_quarantined": part.get("geometry_audit", {}).get("bbox_tessellation_crosscheck", {}).get("status") == "unresolved_disagreement"}
        npz = output / f"meshes/{code}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, triangles=triangles)
        fig = plt.figure(figsize=(11, 4))
        for i, (elev, azim) in enumerate([(25, 35), (15, 135), (90, 0)], 1):
            draw_surface(fig.add_subplot(1, 3, i, projection="3d"), triangles, elev, azim)
        fig.suptitle(f"{code} | {part['part_id']} | surface extent {evidence['surface_large_mm']:.2f} mm", fontsize=12)
        fig.text(.5, .02, "Rendered source geometry. Nominal shape only; no inferred material, tolerance or load rating.", ha="center", fontsize=9)
        fig.tight_layout(rect=[0, .05, 1, .92])
        target = output / f"views/{code}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160)
        plt.close(fig)
        manifest.append({"inspection_id": code, "part_id": part["part_id"], "split": part["split"],
                         "view": f"views/{code}.png", "mesh": f"meshes/{code}.npz", "source_sha256": part["file_sha256"]})
        print(f"Reference surface: {code}, {len(triangles)} triangles", flush=True)
    for start in range(0, len(gallery), 5):
        fig = plt.figure(figsize=(14, 5))
        for i, (code, _) in enumerate(selected[start:start+5], 1):
            ax = fig.add_subplot(1, 5, i, projection="3d")
            draw_surface(ax, meshes[code])
            ax.set_title(code, fontsize=13)
        fig.tight_layout()
        fig.savefig(output / f"contact_{start//5+1}.png", dpi=180)
        plt.close(fig)
    write_csv(output / "inspection_manifest.csv", manifest)
    write_json(output / "measurements.json", measurements)
    write_json(output / "preparation_manifest.json", receipt("reference-prepare", [Path(args.bundle)/"catalog.json", args.proposed_queries], vars(args)))
    return {"gallery_parts": len(gallery), "exemplars": len(refs), "status": "geometry_evidence_prepared_no_relevance_grades_generated"}


FUNCTION_SCOPES = {"support_rotating_shaft": "support", "connect_shafts": "connect", "participate_belt_drive": "belt"}
from ..retrieval import REFERENCE_METHODS


def numeric_reference(query, measurement, guard_mm=.2):
    """Explicitly provisional same-kernel surface reference; never copy a serving verdict."""
    if not query["constraints"]:
        return "not_requested"
    if measurement["aabb_quarantined"] or measurement["faces_missing_mesh"]:
        return "unknown"
    verdicts = []
    for c in query["constraints"]:
        if c["attribute"] != "aabb_large_mm" or c["unit"] != "mm" or c["op"] != "<=":
            raise ValueError("AI reference protocol only covers an upper largest-extent bound in mm")
        delta = measurement["surface_large_mm"]-c["value"]
        verdicts.append("unknown" if abs(delta) <= guard_mm else "pass" if delta < 0 else "fail")
    return "fail" if "fail" in verdicts else "unknown" if "unknown" in verdicts else "pass"


def reference_grade(observation, query, variant, verdict):
    scope = FUNCTION_SCOPES[query["function_ids"][0]]
    grade = observation[variant][scope]
    if query["query_id"] == "belt_v_profile_gap":
        grade = 2 if observation["inspection_id"] == "T17" else 0
    if query["query_id"] == "belt_timing_profile_gap":
        grade = 2 if observation["inspection_id"] == "T06" else (observation[variant]["belt"] if observation["inspection_id"] == "T13" else 0)
    if verdict == "fail":
        return 0
    return min(grade, 1) if verdict == "unknown" else grade


def reference_rank(retriever, query, method, k):
    """A category-lookup control receives the same generic intent/class mapping."""
    if method != "category_from_intent":
        return retriever.rank(query, method, k)
    categories = sorted({r["class_id"] for r in retriever.knowledge["class_functions"]
                         if r["function_id"] in query["function_ids"] and r["relevance"] > 0})
    transformed = {**query, "function_ids": [], "category_ids": categories}
    result = retriever.rank(transformed, "category", k)
    return result | {"method": method, "query": query,
                     "baseline_transform": "Same admitted intent-to-class prior, category probability only; no exemplar geometry or dimensional filtering."}


def run_reference(args):
    from ..bundle import load_bundle
    from ..evaluation import metrics
    from ..query import validate_query
    from ..reasoning import check_constraints
    from ..retrieval import Retriever
    annotations, evidence, output = Path(args.annotations), Path(args.evidence), Path(args.output)
    catalog, knowledge, graph = load_bundle(args.bundle)
    protocol = read_json(annotations / "protocol.json")
    if protocol["source"] != "ai_reference" or protocol["human_expert_reviewed"] is not False:
        raise ValueError("Reference runner cannot impersonate independent expert evaluation")
    observations = read_json(annotations / "inspected_parts.json")
    inspection = {r["inspection_id"]: r for r in read_csv(evidence / "inspection_manifest.csv")}
    measurements = read_json(evidence / "measurements.json")
    parts = {p["part_id"]: p for p in catalog["parts"]}
    gallery = sorted(p["part_id"] for p in catalog["parts"] if p["split"] == "test")
    if len(gallery) > 20 or len(observations) != len(gallery) or {r["part_id"] for r in observations} != set(gallery):
        raise ValueError("Reviewed candidates must exactly match the bounded test gallery")
    for row in observations:
        pid, code = row["part_id"], row["inspection_id"]
        if row["human_expert_reviewed"] is not False or row["status"] != "ai_reference":
            raise ValueError("Observation provenance must remain AI-authored")
        if row["file_sha256"] != parts[pid]["file_sha256"] or sha256(parts[pid]["absolute_path"]) != row["file_sha256"]:
            raise ValueError("Reference observation is bound to different source geometry")
        if inspection[code]["part_id"] != pid or inspection[code]["source_sha256"] != row["file_sha256"]:
            raise ValueError("Neutral inspection ID changed; do not transfer judgments to new parts")
        for scope in FUNCTION_SCOPES.values():
            values = [row[v][scope] for v in ["lower", "nominal", "upper"]]
            if any(type(v) is not int or v not in [0, 1, 2] for v in values) or values != sorted(values):
                raise ValueError("Invalid ordered reference sensitivity grades")
    queries = [validate_query(q, knowledge) for q in read_json(annotations / "scenarios.json")]
    metadata = {r["query_id"]: r for r in read_csv(annotations / "case_metadata.csv")}
    exemplars = {r["part_id"]: r for r in read_json(annotations / "exemplar_review.json")}
    if len(queries) > 12 or len({q["query_id"] for q in queries}) != len(queries) or set(metadata) != {q["query_id"] for q in queries}:
        raise ValueError("Reference case/metadata mismatch or oversized matrix")
    for query in queries:
        if query["split"] != "test" or len(query["function_ids"]) != 1 or query["function_ids"][0] not in FUNCTION_SCOPES:
            raise ValueError("Reference query outside the inspected scope")
        ref = query["reference_part_id"]
        if ref and (parts[ref]["split"] != "train" or ref not in exemplars or exemplars[ref]["file_sha256"] != parts[ref]["file_sha256"]):
            raise ValueError("Exemplar must be an inspected matching training source")
    source_files = sorted(annotations.glob("*"))
    freeze = {str(p): sha256(p) for p in source_files if p.is_file()}
    write_json(output / "authoring_freeze.json", {"source": "ai_reference", "human_expert_reviewed": False, "sha256_before_ranking": freeze})
    qrels, judgments = [], {v: {} for v in ["lower", "nominal", "upper"]}
    # Complete qrels are formed from individually inspected evidence before any rankings are computed.
    for query in queries:
        qid = query["query_id"]
        for variant in judgments:
            judgments[variant][qid] = {}
        scope = FUNCTION_SCOPES[query["function_ids"][0]]
        for observation in observations:
            pid = observation["part_id"]
            verdict = numeric_reference(query, measurements[pid])
            grades = {v: reference_grade(observation, query, v, verdict) for v in judgments}
            note = observation[f"{scope}_reason"]
            if "profile_gap" in qid:
                note += " Additional profile-specific inspection: fine tooth/flange form in T06, opposed conical groove flanks in T17; profile is not represented by the current retrieval query schema."
            qrels.append({"query_id": qid, "scenario_id": query["scenario_id"], "part_id": pid,
                          "inspection_id": observation["inspection_id"], "grade": grades["nominal"],
                          "grade_lower_sensitivity": grades["lower"], "grade_upper_sensitivity": grades["upper"],
                          "constraint_verdict": verdict, "reviewer": "Codex AI assistant", "source": "ai_reference", "split": "test",
                          "confidence": observation["confidence"], "human_expert_reviewed": False,
                          "evidence": observation["view"]+"; "+note+f"; fresh surface extent {measurements[pid]['surface_large_mm']:.4f} mm",
                          "source_sha256": observation["file_sha256"]})
            for variant in judgments:
                judgments[variant][qid][pid] = {"grade": grades[variant], "constraint_verdict": verdict}
    write_csv(output / "ai_reference_judgments.csv", qrels)
    write_json(output / "reference_protocol.json", protocol)
    write_json(output / "queries.json", queries)
    resolved = read_json(Path(args.bundle).parent / "resolved_config.json")["retrieval_settings"]
    resolved = {**resolved, "gallery_splits": ["test"], "evaluation_split": "test", "k": 5, "methods": REFERENCE_METHODS}
    per_query, rankings, explanations, reason_rows = [], [], [], []
    for kind in ["embedding", "descriptor"]:
        config = {**resolved, "geometry_kind": kind}
        retriever = Retriever(catalog, knowledge, graph, config)
        if retriever.candidate_ids() != gallery:
            raise ValueError("Reference methods require the same complete inspected matched gallery")
        for query in queries:
            qid = query["query_id"]
            for method in REFERENCE_METHODS:
                result = reference_rank(retriever, query, method, config["k"])
                explanations.append(result | {"geometry_kind": kind})
                ids = [r["part_id"] for r in result["results"]]
                for variant, by_query in judgments.items():
                    gold = by_query[qid]
                    score = metrics(ids, gold, config["k"])
                    known = [gold[pid]["constraint_verdict"] for pid in ids if gold[pid]["constraint_verdict"] in ["pass", "fail"]]
                    per_query.append({"geometry_kind": kind, "sensitivity": variant, "query_id": qid,
                                      "scenario_id": query["scenario_id"], "subset": metadata[qid]["subset"], "method": method,
                                      **score, "strong_precision@5": sum(gold[pid]["grade"] == 2 for pid in ids)/5,
                                      "returned": len(ids), "answered": int(bool(ids)), "empty_target_answer": int(not score["answerable"] and bool(ids)),
                                      "known_dimension_checks": len(known), "dimension_violations": sum(v == "fail" for v in known)})
                for row in result["results"]:
                    ranks = {v: judgments[v][qid][row["part_id"]]["grade"] for v in judgments}
                    rankings.append({"geometry_kind": kind, "query_id": qid, "method": method, "rank": row["rank"], "part_id": row["part_id"],
                                     "inspection_id": next(o["inspection_id"] for o in observations if o["part_id"] == row["part_id"]),
                                     "score": row["score"], "reference_grade": ranks["nominal"], "grade_lower_sensitivity": ranks["lower"],
                                     "grade_upper_sensitivity": ranks["upper"], "predicted_constraint": row["reasoning"]["status"],
                                     "reference_constraint": judgments["nominal"][qid][row["part_id"]]["constraint_verdict"]})
        print(f"AI reference matrix: {kind}, {len(queries)*len(REFERENCE_METHODS)} ranking runs", flush=True)
    # One representation-independent comparison of numeric rule outputs, not called expert accuracy.
    for query in queries:
        for pid, row in judgments["nominal"][query["query_id"]].items():
            reason_rows.append({"query_id": query["query_id"], "part_id": pid, "reference_status": row["constraint_verdict"],
                               "predicted_status": check_constraints(parts[pid], query["constraints"])["status"]})
    summaries = []
    for kind in ["embedding", "descriptor"]:
        for variant in judgments:
            for subset in ["primary", "matched_exemplar", "text_only", "empty_target_stress", "capability_gap"]:
                for method in REFERENCE_METHODS:
                    rows = [r for r in per_query if r["geometry_kind"] == kind and r["sensitivity"] == variant and r["method"] == method
                            and (r["subset"] != "capability_gap" if subset == "primary" else r["subset"] == subset)]
                    if not rows:
                        continue
                    answerable = [r for r in rows if r["answerable"]]
                    empty = [r for r in rows if not r["answerable"]]
                    checks = sum(r["known_dimension_checks"] for r in rows)
                    summaries.append({"geometry_kind": kind, "sensitivity": variant, "subset": subset, "method": method,
                                      "queries": len(rows), "answerable_queries": len(answerable), "empty_target_queries": len(empty),
                                      "mean_ndcg@5": float(np.mean([r["ndcg@5"] for r in rows])),
                                      "answerable_ndcg@5": float(np.mean([r["ndcg@5"] for r in answerable])) if answerable else None,
                                      "precision@5": float(np.mean([r["precision@5"] for r in rows])),
                                      "strong_precision@5": float(np.mean([r["strong_precision@5"] for r in rows])),
                                      "recall@5": float(np.mean([r["recall@5"] for r in rows])),
                                      "map@5": float(np.mean([r["map@5"] for r in rows])),
                                      "answer_fraction": float(np.mean([r["answered"] for r in rows])),
                                      "empty_target_answer_rate": float(np.mean([r["answered"] for r in empty])) if empty else None,
                                      "known_dimension_checks": checks,
                                      "dimension_violations": sum(r["dimension_violations"] for r in rows),
                                      "dimension_violation_fraction": sum(r["dimension_violations"] for r in rows)/checks if checks else None})
    known = [r for r in reason_rows if r["reference_status"] in ["pass", "fail"]]
    answered = [r for r in known if r["predicted_status"] in ["pass", "fail"]]
    uncertain = [r for r in reason_rows if r["reference_status"] == "unknown"]
    flat_equivalent = all(
        [(r["part_id"], r["score"]) for r in result["results"]] ==
        [(r["part_id"], r["score"]) for other in explanations
         if other["geometry_kind"] == result["geometry_kind"] and other["query"]["query_id"] == result["query"]["query_id"] and other["method"] == "flat_full"
         for r in other["results"]]
        for result in explanations if result["method"] == "kg_full")
    report = {"result_type": "AI_authored_provisional_reference_study_NOT_independent_expert_ground_truth",
              "human_expert_reviewed": False, "gallery_parts": len(gallery), "queries": len(queries),
              "scenario_families": len({q["scenario_id"] for q in queries}), "ranking_runs": len(explanations), "judgment_pairs": len(qrels),
              "methods": summaries, "flat_graph_rank_equivalent": flat_equivalent,
              "numeric_reference": {"known_cases": len(known), "answered_cases": len(answered), "unknown_reference_cases": len(uncertain),
                  "system_pass_on_unknown_reference_cases": sum(r["predicted_status"] == "pass" for r in uncertain),
                  "agreement_on_answered": sum(r["predicted_status"] == r["reference_status"] for r in answered)/len(answered) if answered else None,
                  "scope": "Fresh surface representation in the same kernel; shared-input diagnostic, not expert accuracy."},
              "notes": ["All relevance judgments are explicitly AI-authored and bound to inspected source geometry.",
                        "Nominal/lower/upper are grading sensitivity scenarios, not confidence intervals or ground-truth bounds.",
                        "Only three intent families; no statistical significance or corpus-wide generalization claim.",
                        "Category-from-intent receives the same generic mapping; graph/table use identical information.",
                        "Capability-gap cases contain requirements absent from the structured query vocabulary.",
                        "A returned top-k list is not a verified engineering solution."]}
    if any(sha256(p) != digest for p, digest in freeze.items()):
        raise ValueError("Authoring files changed during ranking")
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "rankings.csv", rankings)
    write_csv(output / "numeric_reasoning.csv", reason_rows)
    write_json(output / "summary.json", report)
    write_json(output / "explanations.json", explanations)
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "semantic_digest.json", {"sha256": object_hash({"per_query": per_query, "rankings": rankings, "summary": report})})
    write_json(output / "run_manifest.json", receipt("reference-run", [*source_files, evidence / "measurements.json", evidence / "inspection_manifest.csv",
               *[Path(args.bundle)/n for n in ["catalog.json", "knowledge.json", "graph.json"]]], vars(args)))
    return report
