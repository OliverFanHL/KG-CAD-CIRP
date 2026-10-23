"""Prespecified v6 analysis/report; no metric inputs are read by self-test.

The 264 new-reference queries define twelve family-paired primary comparisons.
Point inference is globally withheld unless every targeted per-query nDCG@5 is
exactly identified across all seven methods and both representations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from cadkg.experiments.verified_statistics import _bootstrap, _flip_p, _holm
from cadkg.io import read_csv, read_json, sha256, write_csv, write_json
from scripts.partial_feature_review import KINDS, require, _verify_hashes
from scripts.run_query_diversity import METHODS
from scripts.supplement_partial_metrics import metric_names


ROOT = Path("outputs/query_diversity_v6")
PLAN = Path("docs/QUERY_DIVERSITY_V6.md")
CONTROLS = tuple(m for m in METHODS if m != "kg_full")
SPRINGS = {"compression_return", "tension_return"}
METRICS = metric_names((5, 10, 20))
DISPLAY = {
    "kg_full": "Proposed Full",
    "geometry_class_filter": "Geometry + exemplar category + filter",
    "geometry_intent_filter": "Geometry + function prior + filter",
    "kg_no_geometry": "Class agreement + function prior + filter",
    "kg_no_constraints": "Full without size filter",
    "kg_no_class": "Geometry + same filter",
    "rrf_filter": "Rank fusion + same filter",
}
PANELS = ("new_exemplar264", "legacy110", "legacy_exemplar77", "all440", "pooled_exemplar341",
          "new_exemplar_constrained220", "new_exemplar_unconstrained44", "pooled_exemplar_constrained278",
          "text99", "new_text66", "legacy_text33", "all_constrained366")
EXPECTED_PANELS = dict(zip(PANELS, (264, 110, 77, 440, 341, 220, 44, 278, 99, 66, 33, 366)))


def _panel(q, meta, name):
    ex, co = bool(q["reference_part_id"]), bool(q["constraints"])
    origin = meta[q["query_id"]]["origin"]
    return {
        "new_exemplar264": origin == "new_exemplar",
        "legacy110": origin == "legacy110",
        "legacy_exemplar77": origin == "legacy110" and ex,
        "all440": True,
        "pooled_exemplar341": ex,
        "new_exemplar_constrained220": origin == "new_exemplar" and co,
        "new_exemplar_unconstrained44": origin == "new_exemplar" and not co,
        "pooled_exemplar_constrained278": ex and co,
        "text99": not ex,
        "new_text66": origin == "new_text",
        "legacy_text33": origin == "legacy110" and not ex,
        "all_constrained366": co,
    }[name]


def _parse_metrics(rows):
    for r in rows:
        for metric in METRICS:
            for edge in ("lower", "upper"):
                key = f"{metric}_{edge}"
                r[key] = float(r[key])
                require(np.isfinite(r[key]) and 0 <= r[key] <= 1 + 1e-12, f"Invalid metric: {key}")
            require(r[f"{metric}_lower"] <= r[f"{metric}_upper"] + 1e-12, "Reversed metric envelope")
    return rows


def _mean(rows, metrics=METRICS):
    require(bool(rows), "Empty analysis group")
    return {f"{m}_{e}": float(np.mean([r[f"{m}_{e}"] for r in rows])) for m in metrics for e in ("lower", "upper")}


def _difference(a, b, metrics=METRICS):
    return {f"{m}_{e}": a[f"{m}_lower"] - b[f"{m}_upper"] if e == "lower"
            else a[f"{m}_upper"] - b[f"{m}_lower"] for m in metrics for e in ("lower", "upper")}


def _reference_means(rows, queries, metadata, metrics=METRICS):
    grouped = defaultdict(list)
    for r in rows:
        q = queries[r["query_id"]]
        grouped[q["scenario_id"], q["reference_part_id"]].append(r)
    result = []
    for (family, reference), members in sorted(grouped.items()):
        require(reference and len(members) == 6 and len({r["query_id"] for r in members}) == 6,
                "Primary reference must have exactly six matched query conditions")
        meta = metadata[members[0]["query_id"]]
        grades = {int(metadata[r["query_id"]]["reference_grade"]) for r in members}
        review_ids = {metadata[r["query_id"]]["reference_review_id"] for r in members}
        require(len(grades) == len(review_ids) == 1 and grades <= {1, 2}, "Reference quality metadata inconsistent")
        result.append(dict(family_id=family, reference_part_id=reference,
                           reference_review_id=meta["reference_review_id"], reference_grade=next(iter(grades)),
                           queries=6, **_mean(members, metrics)))
    require(len(result) == 44 and len({r["reference_part_id"] for r in result}) == 44,
            "Primary panel must contain 44 unique new training references")
    counts = Counter(r["family_id"] for r in result)
    require(len(counts) == 11 and set(counts.values()) == {4}, "Four references per task family required")
    return result


def primary_analysis(rows, queries, metadata, seed):
    primary = [r for r in rows if r["cohort"] == "closed" and r["grading_scenario"] == "nominal"
               and metadata[r["query_id"]]["origin"] == "new_exemplar"]
    qids = {qid for qid, q in queries.items() if metadata[qid]["origin"] == "new_exemplar"}
    require(len(qids) == 264, "Prespecified primary query count is 264")
    lookup = {(r["geometry_kind"], r["method"], r["query_id"]): r for r in primary}
    require(len(primary) == len(lookup) == 264 * 7 * 2
            and set(lookup) == {(k, m, q) for k in KINDS for m in METHODS for q in qids},
            "Complete primary seven-method/two-representation pairing required")
    family_counts = Counter(queries[q]["scenario_id"] for q in qids)
    require(len(family_counts) == 11 and set(family_counts.values()) == {24}, "Primary family balance is 11 by 24")
    # Exact equality is the frozen plan's gate. Do not substitute a tolerance,
    # rounded equality, midpoint completion, or a per-comparison favorable gate.
    identified = all(r["ndcg@5_lower"] == r["ndcg@5_upper"] for r in primary)
    rng = np.random.default_rng(seed)
    reference_rows, absolute, comparisons, family_effects, reference_effects = [], [], [], [], []
    for kind in KINDS:
        per_method = {}
        for method in METHODS:
            refs = _reference_means([lookup[kind, method, q] for q in sorted(qids)], queries, metadata)
            per_method[method] = {(r["family_id"], r["reference_part_id"]): r for r in refs}
            reference_rows.extend(dict(geometry_kind=kind, method=method, **r) for r in refs)
            groups = defaultdict(list)
            for r in refs:
                groups[r["family_id"]].append(r)
            family = [_mean(members) for members in groups.values()]
            absolute.append(dict(geometry_kind=kind, method=method, queries=264, references=44, families=11,
                                 averaging="six_conditions_then_four_references_then_equal_family", **_mean(family)))
        for control in CONTROLS:
            a, b = per_method["kg_full"], per_method[control]
            require(set(a) == set(b), "Unpaired reference panel")
            groups = defaultdict(list)
            for key in sorted(a):
                diff = _difference(a[key], b[key])
                rec = dict(geometry_kind=kind, comparator=control, family_id=key[0],
                           reference_part_id=key[1], reference_review_id=a[key]["reference_review_id"],
                           reference_grade=a[key]["reference_grade"], conditions=6, **diff)
                reference_effects.append(rec)
                groups[key[0]].append(rec)
            families = sorted(groups)
            local = []
            for fid in families:
                require(len(groups[fid]) == 4, "Primary effect requires four reference effects per family")
                rec = dict(geometry_kind=kind, comparator=control, family_id=fid, references=4, queries=24,
                           **_mean(groups[fid]))
                local.append(rec)
                family_effects.append(rec)
            effect = _mean(local)
            lo, hi = effect["ndcg@5_lower"], effect["ndcg@5_upper"]
            paired_queries = [_difference(lookup[kind, "kg_full", qid], lookup[kind, control, qid], ("ndcg@5",))
                              for qid in sorted(qids)]
            direct = _mean(paired_queries, ("ndcg@5",))
            require(abs(direct["ndcg@5_lower"] - lo) < 1e-12 and abs(direct["ndcg@5_upper"] - hi) < 1e-12,
                    "Balanced query and hierarchical family averages diverge")
            nlo = np.asarray([r["ndcg@5_lower"] for r in local])
            nhi = np.asarray([r["ndcg@5_upper"] for r in local])
            rec = dict(geometry_kind=kind, proposed_method="kg_full", comparator=control,
                       comparison=f"{kind}:kg_full - {control}", queries=264, references=44, families=11,
                       endpoint="ndcg@5", delta_lower=lo, delta_upper=hi,
                       balanced_query_average_lower=direct["ndcg@5_lower"],
                       balanced_query_average_upper=direct["ndcg@5_upper"],
                       point_inference_eligible=identified, holm_family_size=12,
                       annotation_interval="conservative_outer_envelope_not_sampling_CI",
                       delta=None, ci95_lower=None, ci95_upper=None, exact_sign_flip_p=None, holm_p=None,
                       spring_joint_flip_p=None, spring_cluster_ci95_lower=None, spring_cluster_ci95_upper=None,
                       family_wins=None, family_ties=None, family_losses=None,
                       certain_family_wins=int(sum(nlo > 0)), certain_family_losses=int(sum(nhi < 0)),
                       certain_family_ties=int(sum((nlo == 0) & (nhi == 0))),
                       unresolved_family_signs=int(sum(~((nlo > 0) | (nhi < 0) | ((nlo == 0) & (nhi == 0))))),
                       leave_one_family_out_min=float(min((nlo.sum() - v) / 10 for v in nlo)),
                       leave_one_family_out_max=float(max((nhi.sum() - v) / 10 for v in nhi)),
                       bootstrap_replicates=20000, bootstrap_seed=seed,
                       inference_status="IDENTIFIED_EXPLORATORY_POINT" if identified else "WITHHELD_LABEL_UNCERTAINTY")
            if identified:
                require(np.array_equal(nlo, nhi), "Identified per-query metrics did not yield identified family effects")
                springs = [i for i, fid in enumerate(families) if fid in SPRINGS]
                require(len(springs) == 2, "Both spring families required for sensitivity")
                clusters = [[i] for i in range(11) if i not in springs] + [springs]
                ci, sci = _bootstrap(nlo, rng, 20000), _bootstrap(nlo, rng, 20000, clusters)
                rec.update(delta=float(nlo.mean()), ci95_lower=ci[0], ci95_upper=ci[1], exact_sign_flip_p=_flip_p(nlo),
                           spring_joint_flip_p=_flip_p(nlo, clusters), spring_cluster_ci95_lower=sci[0], spring_cluster_ci95_upper=sci[1],
                           family_wins=int(sum(nlo > 1e-12)), family_ties=int(sum(abs(nlo) <= 1e-12)),
                           family_losses=int(sum(nlo < -1e-12)))
            comparisons.append(rec)
    require(len(comparisons) == 12, "Exactly twelve primary comparisons required")
    if identified:
        for rec, p in zip(comparisons, _holm([r["exact_sign_flip_p"] for r in comparisons])):
            rec["holm_p"] = p
    return dict(identified=identified, primary_rows=primary, absolute=absolute, comparisons=comparisons,
                family_effects=family_effects, reference_rows=reference_rows, reference_effects=reference_effects)


def descriptive(rows, queries, metadata, fields, panels=PANELS):
    groups = defaultdict(list)
    for r in rows:
        for panel in panels:
            if _panel(queries[r["query_id"]], metadata, panel):
                key = tuple(r[f] for f in fields) + (panel, r["family_id"])
                groups[key].append(r)
    family, macros = [], defaultdict(list)
    for key, members in sorted(groups.items()):
        r = dict(zip(fields, key[:len(fields)]))
        r.update(panel=key[-2], family_id=key[-1], query_seed_cases=len(members),
                 unique_queries=len({m["query_id"] for m in members}), **_mean(members))
        family.append(r)
        macros[key[:-1]].append(r)
    summary = []
    for key, members in sorted(macros.items()):
        r = dict(zip(fields, key[:len(fields)]))
        r.update(panel=key[-1], families=len(members), query_seed_cases=sum(m["query_seed_cases"] for m in members),
                 unique_queries=sum(m["unique_queries"] for m in members),
                 averaging="query_seed_mean_within_family_then_equal_family", **_mean(members))
        summary.append(r)
    return family, summary


def quality_strata(primary):
    def aggregate(rows, fields):
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[f] for f in fields) + (r["reference_grade"], r["family_id"])].append(r)
        family, macros = [], defaultdict(list)
        for key, members in sorted(groups.items()):
            row = dict(zip(fields, key[:len(fields)]))
            row.update(reference_grade=key[-2], family_id=key[-1], references=len(members), **_mean(members))
            family.append(row)
            macros[key[:-1]].append(row)
        result = []
        for key, members in sorted(macros.items()):
            row = dict(zip(fields, key[:len(fields)]))
            row.update(reference_grade=key[-1], families=len(members), references=sum(r["references"] for r in members),
                       interpretation="Descriptive selected-reference quality stratum; family composition can differ; not a causal quality effect",
                       **_mean(members))
            result.append(row)
        return family, result
    qa, qs = aggregate(primary["reference_rows"], ("geometry_kind", "method"))
    qe, qes = aggregate(primary["reference_effects"], ("geometry_kind", "comparator"))
    return qa, qs, qe, qes


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
                      + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def _interval(row, metric="ndcg@5", digits=4):
    lo, hi = row[f"{metric}_lower"], row[f"{metric}_upper"]
    return f"{lo:.{digits}f}" if lo == hi else f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _effect(row):
    return f"{row['delta']:.4f}" if row["point_inference_eligible"] else f"[{row['delta_lower']:.4f}, {row['delta_upper']:.4f}]"


def _plot(primary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["geometry_kind"], r["method"]): r for r in primary["absolute"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, metric, title in zip(axes, ("ndcg@5", "strong_precision@5"),
                                  ("Ranking utility (nDCG@5)", "AI strong-match precision@5")):
        for offset, kind, color in ((-.17, "embedding", "#8798a8"), (.17, "descriptor", "#197c87")):
            lo = np.asarray([lookup[kind, m][f"{metric}_lower"] for m in METHODS])
            hi = np.asarray([lookup[kind, m][f"{metric}_upper"] for m in METHODS])
            # Endpoints, not midpoints or completed labels: line is the entire
            # annotation envelope; coincident endpoints are identified values.
            y = np.arange(len(METHODS)) + offset
            ax.hlines(y, lo, hi, color=color, linewidth=4)
            ax.plot(lo, y, "|", color=color, markersize=8, label=kind.capitalize())
            ax.plot(hi, y, "|", color=color, markersize=8)
        ax.set_xlim(0, 1); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(METHODS)), [DISPLAY[m] for m in METHODS]); axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.suptitle("264 new-reference queries · 44 training references · 11 equally weighted task families")
    fig.text(.5, .015, "Lines show admissible-annotation envelopes, not confidence intervals. AI reference; no human validity claim.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(output / "primary_method_bounds.png", dpi=180)
    fig.savefig(output / "primary_method_bounds.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for i, row in enumerate(primary["comparisons"]):
        if primary["identified"]:
            lo, hi, point = row["ci95_lower"], row["ci95_upper"], row["delta"]
            ax.plot(point, i, "o", color="#176b79", markersize=4)
        else:
            lo, hi = row["delta_lower"], row["delta_upper"]
        ax.hlines(i, lo, hi, color="#176b79", linewidth=2)
        ax.plot([lo, hi], [i, i], "|", color="#176b79", markersize=7)
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.set_yticks(range(12), [f"{r['geometry_kind']}: Full − {DISPLAY[r['comparator']]}" for r in primary["comparisons"]])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Equal-family nDCG@5 difference")
    ax.set_title("Unadjusted exploratory 95% family bootstrap intervals" if primary["identified"]
                 else "Conservative annotation effect envelopes · point significance withheld")
    fig.tight_layout(); fig.savefig(output / "primary_effects.png", dpi=180)
    fig.savefig(output / "primary_effects.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def fallacy_scan(primary, prevalence, zero_cases):
    uncertain = not primary["identified"]
    simpson = "Primary has 24 queries and four references per family, so query/family weighting agrees. Family effects retained; secondary strata may differ in family composition."
    if uncertain:
        simpson += " Annotation intervals prevent a general point-sign reversal assessment."
    records = [
        ("Simpson's paradox", "CAUTION" if uncertain else "NOTE", simpson),
        ("Ecological fallacy", "NOTE", "Inference unit is eleven authored families; no inference from a family mean to an individual manufactured part."),
        ("Berkson's paradox", "CAUTION", "Training references pass source suitability eligibility after descriptor-conditioned FPS; this selected sample is not a probability population sample."),
        ("Collider bias", "CAUTION", "Reference suitability and candidate cohort/size filters define selected analyses; effects are paired algorithm diagnostics, not causal adjustment estimates."),
        ("Base rate neglect", "NOTE", f"Raw grade/null prevalence retained for {len(prevalence)} cohort-family cells; {zero_cases} nominal closed method-query cases have known zero relevance. nDCG is not classification accuracy."),
        ("Regression to the mean", "CAUTION", "No v6 performance-based reference selection; historical outcomes were already known, so changes between legacy and new queries are not a causal improvement estimate."),
        ("Survivorship bias", "CAUTION", "All 440 authored queries, empty returns and null labels remain. Unsupported training references are excluded by the explicit source-only eligibility rule; no claim covers all training parts."),
        ("Look-elsewhere effect", "NOTE", "All six controls and both representations are reported as one twelve-comparison Holm family. Secondary panels/strata/robustness have no point significance tests."),
        ("Garden of forking paths", "CAUTION", "Design, queries, code and analysis sealed before v6 results, with validated replay. Sequential choices follow historical v5 exposure and remain exploratory."),
        ("Correlation is not causation", "NOTE", "Controlled algorithm output differences concern this fixed AI-graded benchmark; they do not establish engineering suitability or wider causal benefit."),
        ("Reverse causality", "NOTE", "v5 relevance labels are fixed before v6 and cannot be changed by v6 scores; historical exposure remains disclosed, without a causal direction claim."),
    ]
    return [dict(fallacy=name, severity=severity, assessment=detail,
                 scope="Code/data/reporting check, not an independent human methodological review") for name, severity, detail in records]


def self_test():
    families = ["compression_return", "tension_return"] + [f"synthetic_family_{i}" for i in range(9)]
    rows, queries, metadata = [], {}, {}
    for f, family in enumerate(families):
        for ref in range(4):
            for condition in range(6):
                qid = f"synthetic_{f}_{ref}_{condition}"
                queries[qid] = dict(query_id=qid, scenario_id=family, reference_part_id=f"synthetic_ref_{f}_{ref}", constraints=[])
                metadata[qid] = dict(origin="new_exemplar", reference_grade=1 if ref == 0 else 2,
                                     reference_review_id=f"synthetic_review_{f}_{ref}")
                for kind in KINDS:
                    for m, method in enumerate(METHODS):
                        value = .7 + f * .001 + ref * .002 + condition * .0001 - m * .01
                        r = dict(cohort="closed", grading_scenario="nominal", query_id=qid, family_id=family,
                                 geometry_kind=kind, method=method)
                        for metric in METRICS:
                            r[f"{metric}_lower"] = r[f"{metric}_upper"] = value
                        rows.append(r)
    result = primary_analysis(rows, queries, metadata, 20261020)
    require(result["identified"] and len(result["comparisons"]) == 12 and len(result["family_effects"]) == 132,
            "Synthetic primary comparison family failed")
    for r in result["comparisons"]:
        expected = METHODS.index(r["comparator"]) * .01
        require(abs(r["delta"] - expected) < 1e-12 and r["holm_family_size"] == 12 and r["holm_p"] is not None,
                "Reference/condition/family averaging or pooled Holm failed")
    rows[1]["ndcg@5_upper"] += .01
    uncertain = primary_analysis(rows, queries, metadata, 20261020)
    require(not uncertain["identified"] and all(r["delta"] is None and r["holm_p"] is None
                                                and r["ci95_lower"] is None for r in uncertain["comparisons"]),
            "Single uncertain case did not withhold all twelve point tests")
    return dict(status="VERIFIED_SYNTHETIC", primary_query_cases=264, method_representation_cases=3696,
                comparisons=12, family_effect_rows=132, global_uncertainty_gate_checked=True,
                averaging="Six conditions, four references, eleven equal families",
                experiment_metric_results_read=False)


def analyze(config_path="configs/query_diversity_v6.json", run_path=None, output=None):
    cfg = read_json(config_path)
    freeze_path = Path(cfg.get("freeze_path", ROOT / "experiment_freeze.json"))
    freeze = read_json(freeze_path)["input_sha256"]
    _verify_hashes(freeze)
    resolved = {Path(p).resolve() for p in freeze}
    for path in (Path(__file__), PLAN, Path(config_path), Path("cadkg/experiments/verified_statistics.py"),
                 cfg["queries_path"], cfg["query_manifest_path"], cfg["annotation_path"]):
        require(Path(path).resolve() in resolved, f"Analysis input/code absent from experiment freeze: {path}")
    require(tuple(cfg["methods"]) == METHODS and tuple(cfg["primary_comparators"]) == CONTROLS,
            "Fixed six primary controls and seven methods required")
    require(cfg["seed"] == 20261020 and cfg.get("bootstrap_replicates", 20000) == 20000,
            "Prespecified bootstrap seed/replicates changed")
    freeze_hash = sha256(freeze_path)
    run_path = Path(run_path or cfg["output"])
    validation_path = Path(cfg.get("validation_path", ROOT / "validation.json"))
    validation = read_json(validation_path)
    validation_hash = sha256(validation_path)
    manifest_path = run_path / "run_manifest.json"
    require(validation.get("status") == "VERIFIED"
            and validation.get("run_manifest_sha256") == sha256(manifest_path), "Target run lacks bound independent computational validation")
    manifest = read_json(manifest_path)
    require(manifest["input_sha256"] == freeze and manifest["experimental_freeze_sha256"] == freeze_hash,
            "Run and analysis use different experimental inputs")
    for name, digest in manifest["artifact_sha256"].items():
        require(Path(name).name == name and sha256(run_path / name) == digest, f"Run artifact changed: {name}")
    out = Path(output or ROOT / "analysis")
    require(out.resolve().is_relative_to(ROOT.resolve()) and not out.exists(), "Analysis needs a fresh folder in its experiment root")
    protocol, cohorts = read_json(run_path / "effective_protocol.json"), read_json(run_path / "cohorts.json")
    queries = {q["query_id"]: q for q in protocol["queries"]}
    require(len(queries) == len(protocol["queries"]) == 440 and len(protocol["families"]) == 11,
            "Fixed 440-query/eleven-family design required")
    raw_meta = read_csv(cfg["query_manifest_path"]) if Path(cfg["query_manifest_path"]).suffix.lower() == ".csv" else read_json(cfg["query_manifest_path"])
    if isinstance(raw_meta, dict):
        raw_meta = raw_meta.get("queries", raw_meta.get("rows"))
    metadata = {r["query_id"]: r for r in raw_meta}
    require(len(metadata) == len(raw_meta) and set(metadata) == set(queries), "Query-design metadata mismatch")
    for panel, count in EXPECTED_PANELS.items():
        require(sum(_panel(q, metadata, panel) for q in queries.values()) == count, f"Prespecified panel count mismatch: {panel}")
    counts = Counter(q["scenario_id"] for q in queries.values())
    require(len(counts) == 11 and set(counts.values()) == {40}, "Forty queries per task family required")
    rows = _parse_metrics(read_csv(run_path / "per_query_bounds.csv"))
    robust = _parse_metrics(read_csv(run_path / "robustness_per_query_bounds.csv"))
    lookup = {(r["geometry_kind"], r["query_id"], r["cohort"], r["method"], r["grading_scenario"]): r for r in rows}
    expected = {(k, q, c, m, s) for k in KINDS for q in queries for c in cohorts for m in METHODS
                for s in (("nominal", "conditional_zero", "conditional_two") if c == "closed" else ("nominal",))}
    require(len(lookup) == len(rows) and set(lookup) == expected, "Complete nominal/conditional/cohort matrix required")
    for r in rows + robust:
        require(r["family_id"] == queries[r["query_id"]]["scenario_id"]
                and r["origin"] == metadata[r["query_id"]]["origin"], "Metric/query design mismatch")
    primary = primary_analysis(rows, queries, metadata, cfg["seed"])
    families, secondary = descriptive(rows, queries, metadata, ("cohort", "geometry_kind", "method", "grading_scenario"))
    robust_families, robust_secondary = descriptive(robust, queries, metadata,
                                                     ("geometry_kind", "method", "robustness", "severity"))
    # Compare each robustness case to exactly the same query in nominal scoring.
    # Guard cases omit no-constraint queries; never subtract an unmatched 264-case
    # nominal average from a guard-only 220-case condition average.
    robust_changes = []
    for r in robust:
        base = lookup[r["geometry_kind"], r["query_id"], "closed", r["method"], "nominal"]
        robust_changes.append({**r, **_difference(r, base)})
    _, robust_change_summary = descriptive(robust_changes, queries, metadata,
                                              ("geometry_kind", "method", "robustness", "severity"))
    qa, qs, qe, qes = quality_strata(primary)
    condition_rows = [dict(r, condition=metadata[r["query_id"]].get("condition", "unspecified"))
                      for r in primary["primary_rows"]]
    _, condition_summary = descriptive(condition_rows, queries, metadata, ("geometry_kind", "method", "condition"), ("new_exemplar264",))
    annotation = read_json(cfg["annotation_path"])
    require(annotation.get("source") == "ai_reference" and annotation.get("human_expert_reviewed") is False,
            "AI source provenance must remain explicit")
    observed = {(r["family_id"], r["part_id"]): r["grade"] for r in annotation["judgments"]}
    prevalence = []
    for cohort, selected in cohorts.items():
        for family in sorted(counts):
            totals = Counter(observed[family, pid] for pid in selected)
            prevalence.append(dict(cohort=cohort, family_id=family, candidates=len(selected),
                                   reject=totals[0], conditional=totals[1], strong=totals[2], cannot_assess=totals[None],
                                   positive_fraction_lower=(totals[1] + totals[2]) / len(selected) if selected else 0.,
                                   positive_fraction_upper=(totals[1] + totals[2] + totals[None]) / len(selected) if selected else 0.))
    nominal_closed = [r for r in rows if r["cohort"] == "closed" and r["grading_scenario"] == "nominal"]
    known_zero = sum(int(r["relevant_total_upper"]) == 0 for r in nominal_closed)
    possible_zero = sum(int(r["relevant_total_lower"]) == 0 < int(r["relevant_total_upper"]) for r in nominal_closed)
    scan = fallacy_scan(primary, prevalence, known_zero)
    require(len(scan) == 11, "ARS fallacy scan must check all eleven categories")
    new_reference_quality = {r["reference_part_id"]: r["reference_grade"] for r in primary["reference_rows"]}
    quality_counts = Counter(new_reference_quality.values())
    null_pairs = sum(r["grade"] is None for r in annotation["judgments"])
    require(null_pairs == 2 and len(cohorts["closed"]) == 1816, "Immutable v5 candidates/null judgments changed")
    synthetic = self_test()
    out.mkdir(parents=True)
    for name, table in (("primary_comparisons.csv", primary["comparisons"]), ("primary_method_bounds.csv", primary["absolute"]),
                        ("primary_family_differences.csv", primary["family_effects"]), ("per_reference_method_bounds.csv", primary["reference_rows"]),
                        ("per_reference_effects.csv", primary["reference_effects"]), ("secondary_family_bounds.csv", families),
                        ("secondary_summary_bounds.csv", secondary), ("source_quality_family_bounds.csv", qa),
                        ("source_quality_summary_bounds.csv", qs), ("source_quality_family_effects.csv", qe),
                        ("source_quality_effects.csv", qes), ("robustness_family_bounds.csv", robust_families),
                        ("robustness_summary_bounds.csv", robust_secondary), ("robustness_matched_change_bounds.csv", robust_change_summary),
                        ("condition_summary_bounds.csv", condition_summary), ("cohort_grade_prevalence.csv", prevalence),
                        ("fallacy_scan.csv", scan)):
        write_csv(out / name, table)
    write_json(out / "synthetic_validation.json", synthetic)
    _plot(primary, out)

    abs_lookup = {(r["geometry_kind"], r["method"]): r for r in primary["absolute"]}
    main_table = _table(["Method", "Role", "Embedding nDCG@5", "Descriptor nDCG@5", "Descriptor strong P@5"],
                        [[DISPLAY[m], "**Proposed**" if m == "kg_full" else "Retained control / ablation",
                          _interval(abs_lookup["embedding", m]), _interval(abs_lookup["descriptor", m]),
                          _interval(abs_lookup["descriptor", m], "strong_precision@5")] for m in METHODS])
    stats_table = _table(["Representation", "Full minus control", "Effect / annotation envelope", "95% family CI", "Raw p", "Holm p (12)", "Family W/T/L"],
                         [[r["geometry_kind"], DISPLAY[r["comparator"]], _effect(r),
                           f"[{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]" if primary["identified"] else "Withheld",
                           f"{r['exact_sign_flip_p']:.6f}" if primary["identified"] else "Withheld",
                           f"{r['holm_p']:.6f}" if primary["identified"] else "Withheld",
                           f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}" if primary["identified"]
                           else f"certain +{r['certain_family_wins']} / −{r['certain_family_losses']}; unresolved {r['unresolved_family_signs']}" ]
                          for r in primary["comparisons"]])
    secondary_lookup = {(r["cohort"], r["geometry_kind"], r["method"], r["grading_scenario"], r["panel"]): r for r in secondary}
    panel_table = _table(["Descriptive panel", "Queries", "Full embedding", "Full descriptor", "Descriptor without function", "Descriptor without exemplar category"],
                         [[p, EXPECTED_PANELS[p], *[_interval(secondary_lookup["closed", k, m, "nominal", p])
                            for k, m in (("embedding", "kg_full"), ("descriptor", "kg_full"),
                                         ("descriptor", "geometry_class_filter"), ("descriptor", "geometry_intent_filter"))]] for p in PANELS])
    cohort_table = _table(["Same-source cohort", "CADs", "New-reference Full embedding", "New-reference Full descriptor"],
                          [[c, len(selected), *[_interval(secondary_lookup[c, k, "kg_full", "nominal", "new_exemplar264"])
                                                for k in KINDS]] for c, selected in cohorts.items()])
    quality_table = _table(["Representation", "Source reference grade", "References", "Families represented", "Full nDCG@5"],
                           [[r["geometry_kind"], r["reference_grade"], r["references"], r["families"], _interval(r)]
                            for r in qs if r["method"] == "kg_full"])
    grading_table = _table(["Representation", "Grading diagnostic", "Full new-reference nDCG@5"],
                           [[kind, scenario, _interval(secondary_lookup["closed", kind, "kg_full", scenario, "new_exemplar264"])]
                            for kind in KINDS for scenario in ("conditional_zero", "nominal", "conditional_two")])
    depth_table = _table(["Representation", "Method", "nDCG@10", "nDCG@20", "Recall@20", "AP", "AP@20"],
                         [[kind, DISPLAY[method], *[_interval(abs_lookup[kind, method], metric)
                           for metric in ("ndcg@10", "ndcg@20", "recall@20", "ap", "ap@20")]]
                          for kind in KINDS for method in METHODS])
    robust_table = _table(["Representation", "Existing robustness", "Severity", "Query-seed cases", "Full nDCG@5", "Change vs matched nominal"],
                          [[r["geometry_kind"], r["robustness"], r["severity"], r["query_seed_cases"], _interval(r),
                            _interval(next(d for d in robust_change_summary if (d["geometry_kind"], d["method"], d["robustness"], d["severity"], d["panel"])
                                           == (r["geometry_kind"], r["method"], r["robustness"], r["severity"], r["panel"])))]
                           for r in robust_secondary if r["method"] == "kg_full" and r["panel"] == "new_exemplar264"])
    fallacy_table = _table(["Check", "Assessment", "Evidence / limit"], [[r["fallacy"], r["severity"], r["assessment"]] for r in scan])
    significant = [r["comparison"] for r in primary["comparisons"] if primary["identified"] and r["holm_p"] < .05]
    inference_text = (
        "Every primary per-query nDCG@5 bound collapses exactly across all seven methods and both representations. "
        "The twelve prespecified point comparisons therefore use the exact family sign-flip test, 20,000 family bootstrap replicates "
        "with seed 20261020, and one pooled twelve-test Holm adjustment. Intervals are unadjusted exploratory 95% bootstrap intervals. "
        "Spring joint-flip p-values, cluster-bootstrap intervals and leave-one-family-out ranges are retained in the comparison CSV. "
        "Spring p-values are an unadjusted dependence sensitivity, not another favorable primary testing family. "
        + ("Holm-below-.05 comparisons: " + "; ".join(significant) + "." if significant else "No comparison has Holm p below .05.")
        if primary["identified"] else
        "At least one primary per-query nDCG@5 envelope has nonzero width. The frozen global gate therefore withholds all twelve "
        "point tests, bootstrap confidence intervals and Holm significance claims. All twelve effect envelopes remain reported; "
        "they are conservative annotation outer bounds and are not sampling confidence intervals. No midpoint or favorable completion is used."
    )
    report = f'''# Query diversity with seven controlled methods, v6

## Material Passport

- Origin skill/mode: ARS experiment-agent / validate; report assembled by Codex AI.
- Verification status: VERIFIED for the computational run/replay supplied by the validator; overall interpretation CAUTION.
- Source: frozen v5 AI geometric reference, unchanged 1,816 held-out CADs and eleven authored task families.
- Design: sequential exploratory extension after known v5 outcomes; no independent human validation.
- Frozen plan: [QUERY_DIVERSITY_V6.md](../../../docs/QUERY_DIVERSITY_V6.md); [experiment freeze](../experiment_freeze.json).

## Primary evidence

**kg_full is the proposed Full method.** The other six rows remain controls, including the strongest same-filter alternatives.
This run has 440 explicit structured queries: 110 legacy queries, 264 queries from 44 previously unused training references,
and 66 new text-bound queries. The primary panel is exactly 264 new-reference queries: six size conditions within each of four
references per family, then eleven families equally. It is not 264 independent tasks. The pooled 341 exemplar queries and every
other panel are descriptive secondary analyses. Query text is not a new language-parser evaluation.

{main_table}

![Primary method annotation bounds](primary_method_bounds.png)

{inference_text}

{stats_table}

![Primary paired effects](primary_effects.png)

Statistical uncertainty is conditional on these authored families and the exchangeability assumption of sign flips; no treatment
was randomized across industrial tasks. An nDCG difference measures graded ranking utility. Strong P@5 uses grade 2 after the
existing independent dimensional transform and does not certify material, load, fit, tolerance or production suitability. No
engineering success threshold for an nDCG value or effect size has been validated. A null or nonsignificant difference does not
establish equivalence, and a favorable component ablation result does not identify a universal representation or architecture winner.

The existing deeper-cutoff diagnostics on the same primary queries are:

{depth_table}

## Secondary query panels and reference quality

{panel_table}

The new references were selected in frozen descriptor-based farthest-point order from training-only candidates, excluding
validation/test source hashes and the old references. Separate-context source-view AI grading preceded family unblinding and
eligibility selection. The selected new references include {quality_counts[2]} strong and {quality_counts[1]} conditional sources.
These source grades and [per-reference effects](per_reference_effects.csv) remain explicit; unsupported source forms were not
silently replaced using retrieval outcomes. This is an AI source audit, not human reliability or human truth.

{quality_table}

Quality strata can contain different task families and source shapes. Their averages are descriptive, not a causal estimate of
source quality or a reason to remove conditional references. Descriptor-conditioned FPS is not a probability sample and can favor
descriptor diversity over embedding diversity. Primary comparisons are within each representation; this design cannot establish
which representation generalizes better. Additional reference shapes and numeric requirements probe variation inside the same
CAD source distribution and eleven role scopes, not new task families or unseen industrial domains.

## Candidate, grading and robustness diagnostics

{cohort_table}

These cohorts are subsets of the same held-out source gallery, not independent data domains or new candidate designs. Raw
grade/null prevalence is in [cohort_grade_prevalence.csv](cohort_grade_prevalence.csv). There are {known_zero} known-zero-relevance
and {possible_zero} possibly-zero-relevance nominal closed method-query cases; all stay in denominators. Counts include method
and representation repetitions and do not represent independent samples. Both frozen null judgments remain among all candidates.

{grading_table}

Conditional-zero/two are existing grading diagnostics, not selected alternative ground truths.

{robust_table}

Robustness changes use the same query IDs and seeds as their nominal counterparts. Guard cases contain only constrained queries;
they are never compared with an unmatched 264-query nominal mean. Classifier corruption preserves the old draw universe and adds
a separate fixed stream only for new references, shared across all methods. It is a synthetic sensitivity, not an observed deployment
error rate. Tie orders, dimensional guards and all retained methods are saved in the [robustness tables](robustness_summary_bounds.csv).

## Reproducibility, integrity and claim boundaries

The computational validator bound this report to the completed run manifest and checked the source/query design and run/replay.
The runner retains exact original-110 orders, scores and metric-bound continuity, along with common legacy robustness continuity.
The [continuity validation](../{run_path.name}/continuity_validation.json), [run manifest](../{run_path.name}/run_manifest.json) and
[root validation](../validation.json) retain the checks. Full/flat identity is a shared-algebra implementation check; no eighth
performance row is introduced, and equality provides no retrieval-quality evidence for a graph representation advantage.
Text cases with collapsed rankings are retained and logged rather than dropped as duplicate results.

The current data can assess relative shortlisting utility on this frozen AI-graded benchmark. It does not validate engineering
compatibility, natural-language understanding, human annotation truth, new source domains, or population generalization.
Historical test exposure and retrospective v5 rubric refinement remain limitations even after a successful computational replay.

## ARS statistical fallacy scan

Coverage: **11/11 checked**. These are explicit computational/design/reporting checks with residual cautions, not a claim that
an automated checklist proves methodological validity or that an independent human review occurred.

{fallacy_table}

## Reproduction

After the root validator writes a VERIFIED validation.json bound to the run manifest:

```powershell
& $cadkgPython -m scripts.analyze_query_diversity analyze --config configs/query_diversity_v6.json
```

All twelve primary comparisons, 132 family-effect rows, 528 new-reference effect rows, descriptive cohort/panel/condition/quality
tables, complete robustness summaries, the fallacy scan and code/input/result hashes are retained. Existing experiment and
annotation files are preserved. Numerical confidence statements, where allowed by the global identification gate, remain exploratory.
'''
    (out / "report.md").write_text(report, encoding="utf-8")
    _verify_hashes(freeze)
    for name in ("effective_protocol.json", "cohorts.json", "per_query_bounds.csv", "robustness_per_query_bounds.csv"):
        require(sha256(run_path / name) == manifest["artifact_sha256"][name], f"Consumed metric input changed: {name}")
    require(sha256(freeze_path) == freeze_hash and sha256(manifest_path) == validation["run_manifest_sha256"]
            and sha256(validation_path) == validation_hash,
            "Frozen experiment or validated run changed during analysis")
    write_json(out / "analysis_manifest.json", dict(status="ANALYZED_VALIDATED_RUN", source="ai_reference", human_expert_reviewed=False,
        experiment_freeze_sha256=freeze_hash, validation_sha256=validation_hash, run_manifest_sha256=sha256(manifest_path),
        analysis_code_sha256=sha256(__file__), primary_queries=264, primary_families=11, primary_references=44,
        primary_comparisons=12, point_tests_performed=primary["identified"], holm_family_size=12,
        fallacy_scan_coverage="11/11 checked; residual cautions documented", source_reference_grade_counts=dict(quality_counts),
        input_sha256=freeze, artifact_sha256={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        computational_verification_not_human_validity=True))
    return dict(status="ANALYZED_VALIDATED_RUN", report=str(out / "report.md"), primary_comparisons=12,
                point_tests_performed=primary["identified"], fallacy_checks=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "report", "self-test"), nargs="?", default="analyze")
    parser.add_argument("--config", default="configs/query_diversity_v6.json")
    parser.add_argument("--run", help="Validated experiment run folder; default config output")
    parser.add_argument("--output", help="Fresh analysis folder; default outputs/query_diversity_v6/analysis")
    args = parser.parse_args()
    result = self_test() if args.action == "self-test" else analyze(args.config, args.run, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
