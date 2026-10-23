"""Full-corpus paired matrix. Class-label tasks and functional judgments stay separate."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..catalog import CLASSES, leakage_audit
from ..bundle import load_bundle, save_bundle
from ..evaluation import classification_metrics
from ..io import object_hash, read_json, receipt, sha256, write_csv, write_json
from .reference import reference_rank, REFERENCE_METHODS
from ..retrieval import METHODS, Retriever, LABELS


def retrieval_config(settings, kind):
    return {"geometry_kind": kind, "gallery_splits": ["test"], "matched_cohort": True,
            "unknown_policy": "exclude", "weights": settings["weights"]}


def build_full(output, settings):
    output = Path(output)
    catalog = read_json(output / "measured_catalog.json")
    legacy = Path(catalog['artifacts']['embeddings']['path']).parent.parent.parent
    for path in [legacy / "manifest.csv", legacy / "study/full/geometry_encoder.pt", legacy / "study/full/geometry_embeddings.meta.json"]:
        if path.exists():
            catalog["artifacts"][path.name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    catalog["audit"]["leakage"] = leakage_audit(catalog)
    save_bundle(catalog, settings["knowledge"], output / "bundle", settings["knowledge_policy"])
    return catalog["audit"]


def normalized_vectors(retriever, ids):
    x = np.asarray([retriever.vectors[pid] for pid in ids], dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)


def probability_matrix(retriever, ids, flat=False):
    return np.asarray([[retriever.probabilities(pid, flat).get(c, 0) for c in CLASSES] for pid in ids])


def paired_scores(geometry, category, class_reference, method, weights):
    # No functional intent is present in this auxiliary class benchmark. The semantic
    # signal is unavailable rather than set to zero; unavailable signals have no weight.
    available = {"geometry": geometry, "category": category, "class_reference": class_reference}
    selected = [s for s in METHODS[method]["signals"] if s in available]
    denom = sum(weights.get(s, 1) for s in selected)
    return sum(weights.get(s, 1)*available[s] for s in selected)/denom


def topk_stable(scores, k):
    # Gallery columns are lexicographically ordered part IDs. Stable sorting matches
    # the serving tie rule even when probabilities saturate or many scores are equal.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def extent(part):
    m = part["measurements"].get("aabb_large_mm", {})
    return float(m["value"]) if m.get("status") == "computed" and m.get("unit") == "mm" else None


def byte_unique(ids, parts):
    seen, selected = set(), []
    for pid in sorted(ids):
        digest = parts[pid]["file_sha256"]
        if not digest:
            raise ValueError("Unhashed corpus part")
        if digest not in seen:
            selected.append(pid)
            seen.add(digest)
    return selected


def common_gallery(retrievers, catalog):
    parts = {p["part_id"]:p for p in catalog["parts"]}
    contaminated = {pid for group in leakage_audit(catalog)["exact_byte_cross_split_duplicates"] for pid in group
                    if parts[pid]["split"] != "train"}
    gallery = byte_unique(set.intersection(*(set(r.candidate_ids()) for r in retrievers.values()))-contaminated, parts)
    return gallery, contaminated


def paired_statistics(per_query, settings):
    lookup = {(r["geometry_kind"], r["query_id"], r["method"]): r for r in per_query}
    rows = []
    for kind in settings["geometry_kinds"]:
        for condition in ["unconstrained", "q25_envelope", "q75_envelope"]:
            full = [r for r in per_query if r["geometry_kind"] == kind and r["method"] == "kg_full" and r["condition"] == condition]
            for other in settings["methods"]:
                if other == "kg_full":
                    continue
                diffs = np.asarray([r["ndcg@5"]-lookup[kind, r["query_id"], other]["ndcg@5"] for r in full])
                by_class = defaultdict(list)
                for r, d in zip(full, diffs):
                    by_class[r["query_class"]].append(float(d))
                class_means = np.array([np.mean(by_class[c]) for c in sorted(by_class)])
                rng = np.random.default_rng(settings["seed"])
                boot = rng.choice(class_means, (settings["bootstrap_samples"], len(class_means))).mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                rows.append({"geometry_kind": kind, "condition": condition, "comparison": f"kg_full - {other}",
                             "metric": "ndcg@5", "n_paired_queries": len(diffs), "class_clusters": len(class_means),
                             "mean_delta": float(diffs.mean()), "class_balanced_delta": float(class_means.mean()),
                             "class_resampling_low": float(lo), "class_resampling_high": float(hi),
                             "wins": int((diffs > 1e-10).sum()), "ties": int((np.abs(diffs) <= 1e-10).sum()),
                             "losses": int((diffs < -1e-10).sum()),
                             "uncertainty_scope": "Descriptive resampling of 10 observed classes; no p-value or family-independent generalization"})
    return rows


def run_class_matrix(output, settings):
    output = Path(output)
    catalog, knowledge, graph = load_bundle(output / "bundle")
    parts = {p["part_id"]: p for p in catalog["parts"]}
    leakage = leakage_audit(catalog)
    contaminated = {pid for group in leakage["exact_byte_cross_split_duplicates"] for pid in group
                    if parts[pid]["split"] != "train"}
    retrievers = {kind: Retriever(catalog, knowledge, graph, retrieval_config(settings, kind)) for kind in settings["geometry_kinds"]}
    common, contaminated = common_gallery(retrievers, catalog)
    train = sorted(p["part_id"] for p in catalog["parts"] if p["split"] == "train" and p["prediction"] is not None
                   and all(p["part_id"] in r.vectors and np.linalg.norm(r.vectors[p["part_id"]]) > 1e-12 for r in retrievers.values()))
    train = byte_unique(train, parts)
    bounds = {}
    for c in CLASSES:
        values = [extent(p) for p in catalog["parts"] if p["split"] == "val" and p["label"] == c
                  and p["part_id"] not in contaminated and extent(p) is not None]
        if not values:
            raise ValueError(f"No trusted validation extent for {c}")
        bounds[c] = dict(zip(["q25_envelope", "q75_envelope"], np.quantile(values, settings["envelope_validation_quantiles"]).tolist()))
    rows, ranked_rows, parity = [], [], []
    discounts = 1/np.log2(np.arange(settings["k"])+2)
    settings_hash = object_hash(settings)
    write_json(output / "frozen_protocol.json", {"settings": settings, "settings_sha256": settings_hash,
             "validation_selected_bounds_mm": bounds, "class_gallery": common, "train_query_ids": train,
             "truth": "Existing part labels; numeric target uses fresh analytical extent. No function truth generated from KG.",
             "exact_cross_split_heldout_excluded": sorted(contaminated),
             "near_duplicate_limit": "STEP-byte duplicates checked; geometric families not independently annotated."})
    for kind, retriever in retrievers.items():
        for condition in ["unconstrained", "q25_envelope", "q75_envelope"]:
            gallery = common if condition == "unconstrained" else [pid for pid in common if extent(parts[pid]) is not None]
            gx = normalized_vectors(retriever, gallery)
            gp = probability_matrix(retriever, gallery)
            assert np.array_equal(gp, probability_matrix(retriever, gallery, True))
            gy = np.array([CLASSES.index(parts[pid]["label"]) for pid in gallery])
            ge = np.array([extent(parts[pid]) or 0 for pid in gallery])
            gh = np.array([parts[pid]["file_sha256"] for pid in gallery])
            for start in range(0, len(train), settings["batch_size"]):
                refs = train[start:start+settings["batch_size"]]
                labels = np.array([CLASSES.index(parts[pid]["label"]) for pid in refs])
                geom = np.clip(normalized_vectors(retriever, refs) @ gx.T, -1, 1)*.5+.5
                cp = probability_matrix(retriever, refs) @ gp.T
                category = gp[:, labels].T
                hashes = np.array([parts[pid]["file_sha256"] for pid in refs])
                eligible = hashes[:, None] != gh[None, :]
                limit = np.array([bounds[parts[pid]["label"]][condition] if condition != "unconstrained" else np.inf for pid in refs])
                feasible = (ge[None, :] <= limit[:, None]) | np.isclose(ge[None, :], limit[:, None], rtol=1e-12, atol=1e-9)
                relevant = (labels[:, None] == gy[None, :]) & feasible & eligible
                nrel = relevant.sum(1)
                idcg = np.array([discounts[:min(settings["k"], n)].sum() for n in nrel])
                for method in settings["methods"]:
                    scores = paired_scores(geom, category, cp, method, settings["weights"])
                    allowed = eligible & (feasible if METHODS[method]["constraints"] else True)
                    masked = np.where(allowed, scores, -np.inf)
                    order = topk_stable(masked, settings["k"])
                    valid = np.take_along_axis(allowed, order, 1)
                    hits = np.take_along_axis(relevant, order, 1) & valid
                    dcg = (hits*discounts).sum(1)
                    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
                    precision = hits.sum(1)/settings["k"]
                    fails = (valid & ~np.take_along_axis(feasible, order, 1)).sum(1)
                    for i, pid in enumerate(refs):
                        qid = f"class_{pid}_{condition}"
                        rows.append({"query_id": qid, "reference_part_id": pid, "query_class": parts[pid]["label"],
                            "geometry_kind": kind, "method": method, "condition": condition,
                            "gallery": len(gallery), "bound_mm": "" if condition == "unconstrained" else float(limit[i]),
                            "ndcg@5": float(ndcg[i]), "precision@5": float(precision[i]), "relevant_count": int(nrel[i]),
                            "returned": int(valid[i].sum()), "dimension_violations": int(fails[i]),
                            "truth_source": "dataset_class_plus_analytical_extent_auxiliary"})
                        # First reference per class gives an inspectable serving-parity witness.
                        if pid == next(t for t in train if parts[t]["label"] == parts[pid]["label"]):
                            query = {"query_id": qid, "category_ids": [parts[pid]["label"]], "reference_part_id": pid,
                                     "constraints": [] if condition == "unconstrained" else [{"attribute": "aabb_large_mm", "op": "<=", "value": float(limit[i]), "unit": "mm"}]}
                            served = retriever.rank(query, method, settings["k"], gallery)["results"]
                            vector_ids = [gallery[j] for j, ok in zip(order[i], valid[i]) if ok]
                            passed = vector_ids == [r["part_id"] for r in served] and all(np.isclose(masked[i, j], row["score"], rtol=1e-10, atol=1e-12) for j,row in zip(order[i],served))
                            parity.append({"geometry_kind": kind, "method": method, "query_id": qid, "passed": passed})
                            for rank, j in enumerate(order[i], 1):
                                if valid[i, rank-1]:
                                    ranked_rows.append({"geometry_kind": kind, "method": method, "query_id": qid,
                                        "rank": rank, "part_id": gallery[j], "score": float(masked[i,j]), "class_match": int(hits[i,rank-1])})
                if start % (settings['batch_size']*8) == 0 or start+len(refs)==len(train):
                    print(f"Class matrix: {kind}, {condition}, {min(start+len(refs), len(train))}/{len(train)} exemplars", flush=True)
    summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["geometry_kind"], row["condition"], row["method"]].append(row)
    for (kind, condition, method), selected in grouped.items():
        byclass = defaultdict(list)
        for row in selected:
            byclass[row["query_class"]].append(row["ndcg@5"])
        summaries.append({"geometry_kind": kind, "condition": condition, "method": method, "paired_queries": len(selected),
             "gallery": selected[0]["gallery"], "mean_ndcg@5": float(np.mean([r["ndcg@5"] for r in selected])),
             "class_balanced_ndcg@5": float(np.mean([np.mean(v) for v in byclass.values()])),
             "mean_precision@5": float(np.mean([r["precision@5"] for r in selected])),
             "returned": sum(r["returned"] for r in selected), "dimension_violations": sum(r["dimension_violations"] for r in selected),
             "answerable_queries": sum(r["relevant_count"] > 0 for r in selected)})
    write_csv(output / "class_per_query.csv", rows)
    write_csv(output / "class_summary.csv", summaries)
    write_csv(output / "class_paired_differences.csv", paired_statistics(rows, settings))
    write_csv(output / "class_rank_examples.csv", ranked_rows)
    write_json(output / "class_serving_parity.json", {"checks": len(parity), "passed": sum(r["passed"] for r in parity), "details": parity})
    report = {"parts": len(catalog["parts"]), "splits": dict(Counter(p["split"] for p in catalog["parts"])),
              "graph_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]), "matched_test_gallery": len(common),
              "trusted_dimension_gallery": sum(extent(parts[pid]) is not None for pid in common),
              "train_exemplars": len(train), "query_variants": len(train)*3, "ranking_runs": len(rows),
              "settings_sha256": settings_hash, "classifier_test": classification_metrics(retrievers["embedding"], common),
              "measurement": catalog["audit"]["full_measurement"], "leakage": leakage,
              "functional_truth": "not supplied by auxiliary class experiment"}
    write_json(output / "class_matrix.json", report)
    write_json(output / "class_semantic_digest.json", {"sha256": object_hash({"per_query":rows,"paired":paired_statistics(rows, settings),"examples":ranked_rows}),
               "scope":"Deterministic metrics, paired differences and fixed examples; excludes paths/timings"})
    write_json(output / "run_manifest.json", receipt("full-scale class matrix", [output / "bundle/bundle_manifest.json", output / "frozen_protocol.json"], settings))
    if not all(r["passed"] for r in parity):
        raise ValueError("Vectorized ranking disagrees with serving ranker")
    return report


def run_functional_search(output, settings):
    output = Path(output)
    catalog, knowledge, graph = load_bundle(output / "bundle")
    queries = read_json(settings["functional_queries"])
    rows, pool, full_results = [], set(), []
    retrievers = {kind: Retriever(catalog, knowledge, graph, retrieval_config(settings, kind)) for kind in settings["geometry_kinds"]}
    parts = {p["part_id"]: p for p in catalog["parts"]}
    contaminated = {pid for group in leakage_audit(catalog)["exact_byte_cross_split_duplicates"] for pid in group if parts[pid]["split"] != "train"}
    gallery, contaminated = common_gallery(retrievers, catalog)
    for kind, retriever in retrievers.items():
        for query in queries:
            for method in REFERENCE_METHODS:
                # The original reference helper derives intent category, while all other
                # methods receive the identical structured query and full held-out gallery.
                if method == "category_from_intent":
                    mapped = dict(query)
                    mapped["category_ids"] = sorted({r["class_id"] for r in knowledge["class_functions"] if r["function_id"] in query["function_ids"]})
                    result = retriever.rank(mapped, "category", settings["k"], gallery)
                else:
                    result = retriever.rank(query, method, settings["k"], gallery)
                full_results.append({"geometry_kind": kind, "query_id": query["query_id"], "method": method, "result": result})
                for row in result["results"]:
                    pool.add(row["part_id"])
                    rows.append({"geometry_kind": kind, "query_id": query["query_id"], "scenario_id": query["scenario_id"],
                        "method": method, "rank": row["rank"], "part_id": row["part_id"], "score": row["score"],
                        "signals": str(row["signals"]), "serving_constraint": row["reasoning"]["status"], "gallery": len(gallery)})
    # Also retain the prior blind reference cohort as independent-of-new-ranking controls.
    controls = {r["part_id"] for r in read_json("annotations/ai_reference/inspected_parts.json")}
    pool |= controls & set(gallery)
    ids = sorted(pool)
    manifest = [{"inspection_id": f"F{i:03}", "part_id": pid, "source_sha256": parts[pid]["file_sha256"],
                 "split": "test", "selection": "method_union_top5" if any(r["part_id"] == pid for r in rows) else "previous_blind_control"} for i,pid in enumerate(ids,1)]
    write_csv(output / "functional_rankings.csv", rows)
    write_json(output / "functional_explanations.json", full_results)
    write_csv(output / "functional_pool_manifest.csv", manifest)
    write_json(output / "functional_protocol.json", {"queries": queries, "query_sha256": object_hash(queries), "gallery": gallery,
         "methods": REFERENCE_METHODS, "pool_parts": len(pool), "ranking_runs": len(full_results), "k": settings["k"],
         "grade_policy": "All returned top5 candidates inspected. Pool nDCG uses union pool IDCG, not exhaustive full-gallery relevance. No full-gallery recall.",
         "limitations": "AI judgments authored after pooling can be selection-biased; evidence views conceal labels, scores and retrieval method."})
    return {"search_gallery": len(gallery), "pool_parts": len(pool), "ranking_runs": len(full_results)}


def main(args):
    settings = read_json(args.config)
    output = Path(args.output)
    if settings["k"] != 5:
        raise ValueError("This frozen matrix defines metrics at k=5")
    if args.stage in {"all", "prepare"}:
        from ..catalog import inventory, import_descriptors, import_embeddings
        imported = output / "imported_catalog.json"
        if not imported.exists():
            base = Path(args.legacy_root) / "jev_tmcad_v0/outputs"
            catalog = inventory(args.dataset, base / "manifest.csv")
            print("Importing all source-bound legacy representations", flush=True)
            import_descriptors(catalog, base / "descriptors/tmcad_descriptors.csv")
            import_embeddings(catalog, base / "study/full/geometry_embeddings.csv")
            write_json(imported, catalog)
        else:
            from ..io import sha256
            catalog=read_json(imported)
            if Path(catalog['dataset_root']).resolve()!=Path(args.dataset).resolve():
                raise ValueError('Cached import uses a different dataset root; use a new output directory')
            for name in ['descriptors','embeddings']:
                artifact=catalog['artifacts'][name]
                if sha256(artifact['path'])!=artifact['sha256']:
                    raise ValueError('Legacy export changed; reimport into a new output directory')
        # Copy earlier discovered quarantines with their source audit; source hashes are
        # checked in the original import and measurement cache, never assume CSV mm.
        previous = Path("outputs/technical/bundle/catalog.json")
        if previous.exists() and not (output / "previous_quarantine.json").exists():
            write_json(output / "previous_quarantine.json", {p["part_id"]: p.get("geometry_audit", {}) for p in read_json(previous)["parts"]
                       if any(k.startswith("aabb_") and m["status"] == "computed_unverified" for k,m in p["measurements"].items())})
    if args.stage in {"all", "measure"}:
        from .full_measure import measure_full
        print(measure_full(read_json(output / "imported_catalog.json"), output, settings["workers"], settings["step_timeout_seconds"]), flush=True)
    if args.stage in {"all", "build"}:
        audit=build_full(output, settings)
        print({"graph_parts":audit['selected_total'],"measurement":audit['full_measurement']['outcomes']},flush=True)
    if args.stage in {"all", "class"}:
        result=run_class_matrix(output, settings)
        print({k:result[k] for k in ['parts','graph_nodes','graph_edges','matched_test_gallery','train_exemplars','ranking_runs']},flush=True)
    if args.stage in {"all", "functional-search"}:
        print(run_functional_search(output, settings), flush=True)
    if args.stage in {"all", "pool-evidence", "functional-evaluate", "report"}:
        from .full_report import prepare_pool, evaluate_pool, make_report
        if args.stage in {"all", "pool-evidence"}:
            print(prepare_pool(output), flush=True)
        if args.stage in {"all", "functional-evaluate"}:
            if Path(args.annotations).exists():
                print(evaluate_pool(output, args.annotations), flush=True)
            elif args.stage == "functional-evaluate":
                raise ValueError("Missing source-bound inspected pool judgments")
            else:
                print("Pool evidence prepared. Functional effectiveness remains pending inspected judgments.", flush=True)
        if args.stage in {"all", "report"}:
            print(make_report(output), flush=True)
    if args.stage=='validate' or (args.stage=='all' and (output/'functional_evaluation.json').exists()):
        from .full_validation import validate_full
        print(validate_full(output),flush=True)
