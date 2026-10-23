"""Plots and an evidence-linked report for the explicitly provisional AI reference study."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import read_csv, read_json, receipt, write_json
from .reference import REFERENCE_METHODS, draw_surface


from ..retrieval import LABELS


def make_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, evidence, output = Path(args.run), Path(args.evidence), Path(args.output)
    report = read_json(root / "summary.json")
    if report.get("human_expert_reviewed") is not False:
        raise ValueError("This report renderer is for explicitly AI-authored reference data")
    output.mkdir(parents=True, exist_ok=True)
    summaries = report["methods"]
    def summary(kind, method, sensitivity="nominal", subset="matched_exemplar"):
        return next(r for r in summaries if (r["geometry_kind"], r["method"], r["sensitivity"], r["subset"]) == (kind, method, sensitivity, subset))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1.35, 1]})
    yy = np.arange(len(REFERENCE_METHODS))
    for kind, offset, color in [("embedding", -.17, "#286d9d"), ("descriptor", .17, "#d48a34")]:
        bars = axes[0].barh(yy+offset, [summary(kind, m)["mean_ndcg@5"] for m in REFERENCE_METHODS], .31, color=color, label=kind)
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        bars = axes[1].barh(yy+offset, [summary(kind, m)["strong_precision@5"] for m in REFERENCE_METHODS], .31, color=color)
        axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    for ax in axes[:2]:
        ax.set_yticks(yy, [LABELS[m] for m in REFERENCE_METHODS], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.13)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_title("Ranking quality: nDCG@5\n4 matched exemplar cases", fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .95), ncol=2, fontsize=9)
    axes[1].set_title("Strong candidate fraction at 5\nAI reference grade = 2", fontsize=11)
    methods = ["category_from_intent", "geometry_class", "kg_no_constraints", "kg_full"]
    known = [summary("embedding", m)["known_dimension_checks"] for m in methods]
    fails = [summary("embedding", m)["dimension_violations"] for m in methods]
    axes[2].bar(np.arange(4), fails, color=["#a46f58"]*3+["#3e8970"])
    axes[2].set_xticks(np.arange(4), ["Category", "Geom.\n+ class", "No filter", "Full"], fontsize=9)
    axes[2].set_title("Confirmed envelope violations\nEmbedding; constrained cases", fontsize=11)
    axes[2].set_ylim(0, max(fails)+1.5)
    for i, (f, n) in enumerate(zip(fails, known)):
        axes[2].text(i, f+.08, f"{f}/{n}", ha="center", fontsize=10)
    fig.suptitle("Provisional AI reference study — 20 CADs, no human expert validation", fontsize=14)
    fig.text(.5, .02, "Four related exemplar cases in two intent families. Counts are repeated query–candidate comparisons. No significance claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .88])
    fig.savefig(output / "quantitative.png", dpi=220)
    fig.savefig(output / "quantitative.pdf")
    plt.close(fig)
    ranks = read_csv(root / "rankings.csv")
    lookup = {r["part_id"]: r for r in read_csv(evidence / "inspection_manifest.csv")}
    cache = {}
    def mesh(pid):
        if pid not in cache:
            with np.load(evidence / lookup[pid]["mesh"], allow_pickle=False) as data:
                cache[pid] = data["triangles"]
        return cache[pid]
    cases = [
        ("support_envelope", ["geometry_class", "kg_full"], "Shaft support: largest stored-axis extent ≤220 mm", "The 327 mm carrier is removed; the 190 mm carrier remains. Low-relevance filler still appears."),
        ("connect_example", ["category_from_intent", "kg_full"], "Shaft connection: visible attachment evidence required", "Both hollow transitions receive nominal grade 0. Broad category priors and geometry do not establish shaft attachment."),
        ("belt_envelope", ["geometry_class", "kg_full"], "Belt wheel: largest stored-axis extent ≤80 mm", "Both belt-wheel forms remain; oversized unrelated forms are removed. Extra returned candidates can still be irrelevant."),
        ("belt_v_profile_gap", ["kg_full"], "Profile gap: V-groove request versus timing-tooth request", "Full ranking is identical for both requests: the structured vocabulary has no belt-profile constraint.")]
    qualitative = []
    for qid, methods, title, note in cases:
        fig = plt.figure(figsize=(12, 3.4*len(methods)+1))
        details = []
        for row_i, method in enumerate(methods):
            rows = sorted([r for r in ranks if r["geometry_kind"] == "embedding" and r["query_id"] == qid and r["method"] == method], key=lambda r: int(r["rank"]))[:3]
            for col, row in enumerate(rows):
                ax = fig.add_subplot(len(methods), 3, row_i*3+col+1, projection="3d")
                draw_surface(ax, mesh(row["part_id"]))
                grade = int(row["reference_grade"])
                color = ["#ad473c", "#a97925", "#2d7a58"][grade]
                ax.set_title(f"{LABELS[method]} · rank {row['rank']}\n{row['inspection_id']} · AI grade {grade} · dimension {row['reference_constraint']}", fontsize=10, color=color)
                details.append({**row, "title": title})
        if qid == "belt_v_profile_gap":
            timing = [r for r in ranks if r["geometry_kind"] == "embedding" and r["query_id"] == "belt_timing_profile_gap" and r["method"] == "kg_full"]
            v_rows = [r for r in ranks if r["geometry_kind"] == "embedding" and r["query_id"] == qid and r["method"] == "kg_full"]
            assert [(r["part_id"], r["score"]) for r in timing] == [(r["part_id"], r["score"]) for r in v_rows]
            note += " T17 has a conical-flank groove; T06 has fine teeth and flanges. Their required grades reverse."
        fig.suptitle(title+"\nActual STEP surfaces; AI-authored relevance", fontsize=13)
        fig.text(.5, .02, note, ha="center", fontsize=9, wrap=True)
        fig.tight_layout(rect=[0, .09, 1, .90])
        fig.savefig(output / f"qualitative_{qid}.png", dpi=170)
        plt.close(fig)
        qualitative.append({"query_id": qid, "note": note, "top3": details})
    write_json(output / "qualitative_cases.json", qualitative)
    header = "| Method | Embedding nDCG@5 | Descriptor nDCG@5 | Embedding P@5 | Strong P@5 | Embedding violations / known checks |\n|---|---:|---:|---:|---:|---:|\n"
    table = ""
    for method in REFERENCE_METHODS:
        a, b = summary("embedding", method), summary("descriptor", method)
        table += f"| {LABELS[method]} | {a['mean_ndcg@5']:.3f} | {b['mean_ndcg@5']:.3f} | {a['precision@5']:.3f} | {a['strong_precision@5']:.3f} | {a['dimension_violations']}/{a['known_dimension_checks']} |\n"
    sensitivities = "| AI grading scenario | Full embedding nDCG@5, matched cases | Full embedding nDCG@5, nine primary cases | Primary answerable cases |\n|---|---:|---:|---:|\n"
    for variant in ["lower", "nominal", "upper"]:
        a = summary("embedding", "kg_full", variant)
        p = summary("embedding", "kg_full", variant, "primary")
        sensitivities += f"| {variant} | {a['mean_ndcg@5']:.3f} | {p['mean_ndcg@5']:.3f} | {p['answerable_queries']}/9 |\n"
    text = f"""# What the CAD retrieval results mean

## Material Passport

- Origin: academic-research-suite, experiment validation; Codex AI-authored reference study.
- Date/version: 2026-10-08 / ai_reference_v1.
- Verification: source geometry inspected, matrix executed, provenance recorded. No independent human expert validation.

The reference annotations are **AI-authored, evidence-linked provisional judgments**, not human expert ground truth. They were written from individually inspected source CAD surfaces before ranking. Grades are not copied from category labels or the KG. The same AI assistant designed the method and wrote the references, so this is an illustrative, potentially biased study rather than an independent research benchmark.

## Reference data and protocol

Twenty previously selected test CADs were inspected in shaded three-view renders. Three training exemplars were also inspected. Each part observation is bound to a STEP SHA256 and neutral inspection ID. The complete reference set has **240 query–candidate judgments: 12 cases ×20 candidates**, in **three related intent families**. Nine cases form the primary exercise; three deliberately expose capability gaps. Eight methods with two representations produced **192 ranking runs**. No model was retrained and no human annotation files were changed. Scenarios and bounds were chosen after geometry inspection to illustrate specific cases, before looking at rankings; they are not a held-out sample of natural user queries. No method weights were tuned using these references.

Grade 2 means strong geometric evidence for an inspection shortlist, grade 1 a specific conditional candidate, and grade 0 mismatch or insufficient positive evidence under the written scope. No grade certifies torque, strength, material, tolerance or assembly compatibility. Conditional alternatives have conservative/permissive sensitivity grades; these are not confidence intervals or proven truth bounds.

Geometry references use a fresh 0.03 mm surface triangulation and an explicit 0.2 mm near-boundary guard. Disputed source boxes remain unknown. This shares OCCT with the implementation, so dimensional agreement is a cross-representation diagnostic. It is not independent CAD-system metrology. Reference cases are narrow hypothetical requirements, not measured real user needs.

The admitted generic function priors come from [SKF](https://evolution.skf.com/us/compromise-free/), [KTR](https://www.ktr.com/se/en/products/couplings/) and [Gates](https://go.gates.com/Power-Transmission-Solutions-by-Gates.html), with source-specific scope/caveats recorded separately. They do not supply TMCAD instance truth. The category baseline receives the same generic mapping, so the comparison does not reward the framework merely for knowing an intent/class association.

## Method schematic

![Method schematic](../../../docs/figures/method_schematic.png)

The upper branch constructs and queries the KG; the separate lower branch evaluates outputs. Prediction and generic knowledge are visibly distinct from computed geometry. Reference judgments never feed the graph. The example need is normalized to structured fields: JSON and registered aliases are the actual interface, not unrestricted language understanding. This schematic was generated with the built-in image tool and its text/arrows were checked. The prompt and provenance are in `docs/figures/`.

## Quantitative comparison

The following table uses **four matched-exemplar cases with nominal positive references**: shaft support and belt-wheel discovery, each with and without an envelope bound. These are related variants of only two families, not four independent trials. nDCG rewards earlier high-grade parts; P@5 counts grades 1 or 2; strong P@5 counts only grade 2. Dimensional counts concern the two constrained cases.

{header}{table}
![Quantitative results](quantitative.png)

The full framework's embedding nDCG is **{summary('embedding','kg_full')['mean_ndcg@5']:.3f}**, versus **{summary('embedding','geometry_class')['mean_ndcg@5']:.3f}** for geometry plus classification. All seven strong candidates across the four cases appear in the full top-five lists, but they occupy only **7/20 returned slots**: strong P@5 is 0.35. Filling every shortlist to five leaves irrelevant candidates. High nDCG therefore does not mean every returned part is useful.

The dimension filter removes confirmed oversized candidates: full embedding has **0/9** known violations compared with **4/10** for geometry plus classification. Near-boundary/unknown comparisons are excluded from this denominator. Filtering reduces geometric violations; it does not validate functions.

Removing the geometry ranking score produces **{summary('embedding','kg_no_geometry')['mean_ndcg@5']:.3f}** nDCG, slightly above the full method. Geometry is useful in some comparisons, but this run does **not** show that adding it always improves the functional shortlist. Descriptor geometry is particularly weak in this tiny train-normalized cohort. These observations cannot be generalized to the full dataset.

The ablation is now labeled **Remove exemplar cosine**. It removes only `(cosine(candidate, exemplar)+1)/2` from the weighted mean and its denominator. Classifier probabilities, exemplar class-distribution matching, generic function priors and physical dimension filtering remain available. It is not removal of all geometry-derived information. See the later [full-corpus paired report](../../full_scale/report/REPORT.md) for the larger experiment and explicitly labeled numerical plots.

Matched flat-table and graph rankings are identical. Structured evidence and filters are the useful mechanisms here; graph storage itself yields no measured advantage.

## Qualitative examples

**Shaft support under 220 mm:** T19 is a 327 mm carrier and is correctly removed. T05 is a 190 mm carrier with a concentric seat and mounting ears, and remains. T04 is only conditional because its pivot/bearing role is unresolved. Fastener-like filler still appears in the list.

![Support shortlist](qualitative_support_envelope.png)

**Shaft connection:** T02 and T07 are hollow diameter transitions with no visible keyway, clamp or opposed shaft hubs. Under the declared requirement they receive nominal grade 0, while a custom joining use remains a grade-1 permissive possibility. The nominal gallery has no positive complete shaft-connector candidate. The full method nevertheless answers all three shaft-connection cases: **100% empty-target answer rate**. A class name and a generic coupling prior fail to establish the required shaft attachment. A confidence cutoff alone may also fail because the classifier can confidently call a pipe-like coupling “coupling.”

![Shaft-connection failure](qualitative_connect_example.png)

**Compact belt wheel:** T17 has a continuous circumferential groove; a direct analytic-face check confirms opposed conical flanks. T06 has fine circumferential teeth and edge flanges. Both are strong broad belt-wheel candidates, but exact belt standard/pitch/fit is unspecified. Oversized mating flanges are removed by the 80 mm bound.

![Belt-wheel shortlist](qualitative_belt_envelope.png)

**V groove versus synchronous teeth:** the current structured query has only the broad belt function and an envelope. It produces identical rankings for these two profile requests, although the preferred candidate reverses (T17 versus T06). These are capability-gap tests, excluded from the matched-case headline table. Additional profile/functional-interface evidence and a typed query field are required before this distinction can be claimed.

![Profile limitation](qualitative_belt_v_profile_gap.png)

## How uncertain judgments affect the result

{sensitivities}

The permissive scenario admits some hollow transitions and loose mating elements as conditional shaft-connection candidates. It changes the answerability of the three shaft cases and raises the nine-case score substantially. This illustrates dependence on annotation scope, not a confidence interval. The nominal evidence requirement is intentionally stricter; both interpretations remain inspectable. A real expert review is necessary to decide which shortlist policy matches the intended application.

The numeric rule agrees with the same-kernel reference in {report['numeric_reference']['answered_cases']}/{report['numeric_reference']['known_cases']} known cases. There are also {report['numeric_reference']['unknown_reference_cases']} unknown reference comparisons, with {report['numeric_reference']['system_pass_on_unknown_reference_cases']} system passes on the near-boundary references (the nominal 80 mm bored plate). These are excluded from known-case agreement, not counted as confirmed successes. Near-boundary guards describe conservative reference uncertainty, while the method uses analytic dimensions. This is a software/numerical result, not certified engineering reasoning accuracy.

## Practical conclusion

The prototype can combine a geometry exemplar, predicted categories, traceable generic priors and explicit dimensional rejection. The current evidence supports the filter, source/status handling and runnable evaluation. It also exposes three necessary improvements before broader engineering use: scoped function/subtype evidence (especially pipe-like versus shaft couplings), a policy for weak or absent relevant candidates, and explicit belt-profile/interface fields. The present one-hop KG should not be described as a broad engineering reasoner.

All references are available in `annotations/ai_reference/`, including every part observation and rationale. `ai_reference_judgments.csv` includes source, AI reviewer, lower/upper sensitivity grades, source hashes and instance evidence. The original independent-expert evaluator still refuses these AI rows.

## Reproduce

From the repository root, using the existing environment:

```powershell
$cadkgPython = 'D:\\Postdoc\\research\\jepa_cad\\.venv\\Scripts\\python.exe'
& $cadkgPython -m cadkg reference-prepare
& $cadkgPython -m cadkg reference-run
& $cadkgPython -m cadkg reference-report
& $cadkgPython -m cadkg reference-run --output outputs/ai_reference/replay
```

Compare the `semantic_digest.json` values in `results/` and `replay/` for exact saved-reference reproducibility. Repreparation regenerates geometry evidence, not AI/human judgments. AI observations remain bound to the original source IDs and hashes; a different gallery requires new inspection. No API is needed for numeric/qualitative reproduction. The separately generated method schematic is an illustration; its prompt and generation provenance are stored with the figure.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(output / "report_manifest.json", receipt("reference-report", [root / "summary.json", root / "rankings.csv", evidence / "inspection_manifest.csv"], vars(args)))
    return {"report": str((output / "REPORT.md").resolve()), "figures": 5, "status": "AI_reference_not_expert_ground_truth"}
