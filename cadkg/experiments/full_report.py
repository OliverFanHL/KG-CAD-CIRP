"""Neutral pool inspection and labeled figures for the full-scale experiments."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import object_hash, read_csv, read_json, receipt, sha256, write_csv, write_json
from .full import LABELS


def prepare_pool(output):
    import subprocess
    import shutil
    import sys
    from PIL import Image, ImageDraw
    output = Path(output)
    evidence = output / "pool_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    catalog = read_json(output / "bundle/catalog.json")
    parts = {p["part_id"]: p for p in catalog["parts"]}
    manifest = read_csv(output / "functional_pool_manifest.csv")
    pre_dir=output/'unconstrained_evidence'
    pre={r['part_id']:r for r in read_csv(pre_dir/'manifest.csv')} if (pre_dir/'manifest.csv').exists() else {}
    failures = []
    for item in manifest:
        code, part = item["inspection_id"], parts[item["part_id"]]
        if not (evidence / f"{code}.json").exists() and part['part_id'] in pre:
            prior=pre[part['part_id']]
            oldcode=prior['inspection_id']
            data=read_json(pre_dir/f'{oldcode}.json')
            if data['source_sha256'] != item['source_sha256']:
                raise ValueError('Preinspection source changed')
            data.update(inspection_id=code,rendered_view_id=oldcode,
                aabb_quarantined=any(k.startswith('aabb_') and m.get('status')!='computed' for k,m in part['measurements'].items()))
            for suffix in ['png','npz']:
                shutil.copyfile(pre_dir/f'{oldcode}.{suffix}',evidence/f'{code}.{suffix}')
            write_json(evidence/f'{code}.json',data)
        if not (evidence / f"{code}.json").exists():
            request = evidence / f"request_{code}.json"
            write_json(request, {"part": part, "item": item})
            try:
                result = subprocess.run([sys.executable, "-m", "cadkg.geometry", "--request", str(request), "--output", str(evidence)],
                                        capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise ValueError(result.stderr[-1000:])
            except (subprocess.TimeoutExpired, ValueError) as exc:
                failures.append({"inspection_id": code, "error": str(exc)})
                continue
        data = read_json(evidence / f"{code}.json")
        if data["source_sha256"] != item["source_sha256"]:
            raise ValueError("Pool geometry source mismatch")
        print(f"Neutral pool evidence {code}/{len(manifest)}: {data['triangles']} facets", flush=True)
    # Contact sheets conceal category, scores, method, and previous AI grades.
    # Each row retains all three actual source-CAD views for geometry inspection.
    for start in range(0, len(manifest), 8):
        sheet = Image.new("RGB", (1500, 8*525), "white")
        for i,item in enumerate(manifest[start:start+8]):
            path = evidence / f"{item['inspection_id']}.png"
            if path.exists():
                with Image.open(path) as im:
                    sheet.paste(im.resize((1500,525)), (0,i*525))
                ImageDraw.Draw(sheet).text((8,i*525+8),item['inspection_id'],fill='#111111')
        sheet.save(evidence / f"contact_{start//8+1:02}.png")
    write_json(evidence / "failures.json", failures)
    blank = output / "human_pool_annotations.csv"
    if not blank.exists():
        write_csv(blank, [{**item, "evidence_view":f"pool_evidence/{item['inspection_id']}.png", "support_grade": "", "connect_grade": "", "belt_grade": "", "v_grade": "", "timing_grade": "",
                 "observation": "", "rationale": "", "reviewer": "", "source": "independent_expert"} for item in manifest])
    return {"requested_parts": len(manifest), "render_failures": len(failures), "human_annotations": "blank"}


def evaluate_pool(output, annotations):
    from ..evaluation import metrics
    from .reference import FUNCTION_SCOPES, numeric_reference
    output = Path(output)
    authoring = read_json(annotations)
    if authoring["source"] != "ai_reference" or authoring["human_expert_reviewed"] is not False:
        raise ValueError("Use independent-expert evaluator for reviewed expert truth; this study is explicitly AI reference")
    observations = {o["inspection_id"]: o for o in authoring["parts"]}
    manifest = read_csv(output / "functional_pool_manifest.csv")
    if set(observations) != {m["inspection_id"] for m in manifest}:
        raise ValueError("Every pooled/control part needs an actual inspected observation")
    protocol = read_json(output / "functional_protocol.json")
    if authoring.get('query_object_sha256') != protocol['query_sha256']:
        raise ValueError('Inspection grades belong to different functional scopes/queries')
    rankings = read_csv(output / "functional_rankings.csv")
    by_pid = {m["part_id"]: m for m in manifest}
    judgments, by_query = [], {}
    for query in protocol["queries"]:
        qrels = {}
        scope = FUNCTION_SCOPES[query["function_ids"][0]]
        if query["query_id"] == "belt_v_profile_gap":
            scope = "v"
        if query["query_id"] == "belt_timing_profile_gap":
            scope = "timing"
        for item in manifest:
            observation = observations[item["inspection_id"]]
            if observation.get('part_id') != item['part_id'] or observation["source_sha256"] != item["source_sha256"] or not observation["observation"] or not observation["rationale"]:
                raise ValueError("Grades need source hashes and inspected geometric evidence")
            data = read_json(output / f"pool_evidence/{item['inspection_id']}.json")
            verdict = numeric_reference(query, data)
            grade = observation["grades"][scope]
            if grade not in {0,1,2} or isinstance(grade,bool):
                raise ValueError("Invalid inspection grade")
            if verdict == "fail":
                grade = 0
            elif verdict == "unknown":
                grade = min(grade,1)
            row = {"query_id": query["query_id"], "part_id": item["part_id"], "inspection_id": item["inspection_id"],
                   "grade": grade, "constraint_verdict": verdict, "source": "ai_reference", "human_expert_reviewed": False,
                   "source_sha256": item["source_sha256"], "observation": observation["observation"], "rationale": observation["rationale"]}
            judgments.append(row)
            qrels[item["part_id"]] = row
        by_query[query["query_id"]] = qrels
    cases = []
    for kind in ["embedding","descriptor"]:
        for query in protocol["queries"]:
            qrels = by_query[query["query_id"]]
            for method in protocol["methods"]:
                selected = sorted([r for r in rankings if r["geometry_kind"] == kind and r["query_id"] == query["query_id"] and r["method"] == method], key=lambda r:int(r["rank"]))
                ids = [r["part_id"] for r in selected]
                m = metrics(ids, qrels, protocol["k"])
                # Recall/MAP denominator requires exhaustive full-gallery relevance;
                # omit rather than mislabel a pool denominator as full-gallery recall.
                cases.append({"geometry_kind": kind, "query_id": query["query_id"], "scenario_id": query["scenario_id"], "method": method,
                    "pool_ndcg@5": m["ndcg@5"], "precision@5": m["precision@5"],
                    "strong_precision@5": sum(qrels[pid]["grade"] == 2 for pid in ids)/5,
                    "pool_answerable": m["answerable"], "returned": len(ids),
                    "confirmed_dimension_violations": sum(qrels[pid]["constraint_verdict"] == "fail" for pid in ids),
                    "known_dimension_checks": sum(qrels[pid]["constraint_verdict"] in {"pass","fail"} for pid in ids),
                    "unknown_dimension_checks": sum(qrels[pid]["constraint_verdict"] == "unknown" for pid in ids)})
    summaries = []
    for kind in ["embedding","descriptor"]:
        for subset,qids in [("primary",[q["query_id"] for q in protocol["queries"]][:9]),
                            ("matched_exemplar",["support_example","support_envelope","connect_example","connect_envelope","belt_example","belt_envelope"]),
                            ("capability_gap",[q["query_id"] for q in protocol["queries"]][9:])]:
            for method in protocol["methods"]:
                chosen = [r for r in cases if r["geometry_kind"] == kind and r["method"] == method and r["query_id"] in qids]
                summaries.append({"geometry_kind": kind,"subset":subset,"method":method,"queries":len(chosen),
                    "mean_pool_ndcg@5": float(np.mean([r["pool_ndcg@5"] for r in chosen])),
                    "mean_precision@5": float(np.mean([r["precision@5"] for r in chosen])),
                    "mean_strong_precision@5": float(np.mean([r["strong_precision@5"] for r in chosen])),
                    "confirmed_dimension_violations":sum(r["confirmed_dimension_violations"] for r in chosen),
                    "known_dimension_checks":sum(r["known_dimension_checks"] for r in chosen),
                    "unknown_dimension_checks":sum(r["unknown_dimension_checks"] for r in chosen)})
    lookup = {(r["geometry_kind"],r["query_id"],r["method"]):r for r in cases}
    paired = []
    for r in cases:
        if r["method"] == "kg_full":
            for other in protocol["methods"]:
                if other != "kg_full":
                    paired.append({"geometry_kind":r["geometry_kind"],"query_id":r["query_id"],"scenario_id":r["scenario_id"],
                        "comparison":f"kg_full - {other}","delta_pool_ndcg@5":r["pool_ndcg@5"]-lookup[r["geometry_kind"],r["query_id"],other]["pool_ndcg@5"],
                        "delta_precision@5":r["precision@5"]-lookup[r["geometry_kind"],r["query_id"],other]["precision@5"]})
    write_csv(output / "functional_ai_judgments.csv", judgments)
    write_csv(output / "functional_per_query.csv", cases)
    write_csv(output / "functional_summary.csv", summaries)
    write_csv(output / "functional_paired_differences.csv", paired)
    write_json(output / "functional_evaluation.json", {"search_gallery":len(protocol["gallery"]),"inspected_pool":len(manifest),
        "judgment_pairs":len(judgments),"ranking_runs":len(cases),"human_expert_reviewed":False,"authoring_sha256":sha256(annotations),
        "pool_ndcg_scope":"IDCG over inspected union plus prior control pool. Full-gallery recall unknown.",
        "confidence_intervals":"Not estimated: only three related functional families; AI/model-design dependence remains."})
    write_json(output / "functional_semantic_digest.json", {"sha256":object_hash({"cases":cases,"summaries":summaries,"paired":paired,"grades":judgments}),
             "scope":"Frozen inspected pool and outputs; does not imply independent truth"})
    sensitivity=[]
    for variant in ['conditional_zero','nominal','conditional_two']:
        for kind in ['embedding','descriptor']:
            for query in protocol['queries'][:9]:
                qid=query['query_id']
                scope=FUNCTION_SCOPES[query['function_ids'][0]]
                qrels={pid:dict(r) for pid,r in by_query[qid].items()}
                for pid,row in qrels.items():
                    o=observations[by_pid[pid]['inspection_id']]
                    original=o['grades'][scope]
                    if original==1 and variant!='nominal':
                        value=0 if variant=='conditional_zero' else 2
                        row['grade']=0 if row['constraint_verdict']=='fail' else min(value,1) if row['constraint_verdict']=='unknown' else value
                for method in protocol['methods']:
                    selected=sorted([r for r in rankings if (r['geometry_kind'],r['query_id'],r['method'])==(kind,qid,method)],key=lambda r:int(r['rank']))
                    m=metrics([r['part_id'] for r in selected],qrels,5)
                    sensitivity.append({'grading_scenario':variant,'geometry_kind':kind,'query_id':qid,'method':method,
                         'pool_ndcg@5':m['ndcg@5'],'precision@5':m['precision@5'],
                         'interpretation':'Hypothetical reweighting of intrinsically conditional functional grades; not truth bounds or confidence intervals'})
    write_csv(output/'functional_annotation_sensitivity.csv',sensitivity)
    return {"ranking_runs":len(cases),"inspected_parts":len(manifest),"judgment_pairs":len(judgments)}


def make_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    report = output / "report"
    report.mkdir(parents=True,exist_ok=True)
    summary = read_csv(output / "class_summary.csv")
    methods = list(dict.fromkeys(r["method"] for r in summary))
    def val(kind,condition,method):
        return float(next(r for r in summary if (r["geometry_kind"],r["condition"],r["method"]) == (kind,condition,method))["class_balanced_ndcg@5"])
    fig, axes = plt.subplots(1,3,figsize=(18,7))
    for ax,condition,title in zip(axes,["unconstrained","q25_envelope","q75_envelope"],["No envelope","Tight envelope (validation Q25)","Loose envelope (validation Q75)"]):
        y=np.arange(len(methods))
        for kind,offset,color in [("embedding",-.18,"#286d9d"),("descriptor",.18,"#d48a34")]:
            bars=ax.barh(y+offset,[val(kind,condition,m) for m in methods],.33,label=kind,color=color)
            ax.bar_label(bars,fmt="%.4f",padding=3,fontsize=8)
        ax.set_yticks(y,[LABELS[m] for m in methods],fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0,1.14); ax.grid(axis="x",alpha=.15); ax.set_axisbelow(True)
        ax.set_title(title,fontsize=12); ax.set_xlabel("Class-balanced nDCG@5 — higher is better")
    fig.suptitle("Full-corpus auxiliary class + envelope retrieval (not functional ground truth)",fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.94),ncol=2)
    fig.tight_layout(rect=[0,.03,1,.89]); fig.savefig(report/"class_matrix.png",dpi=190); fig.savefig(report/"class_matrix.pdf"); plt.close(fig)
    paired=read_csv(output/"class_paired_differences.csv")
    diffs=[r for r in paired if r["comparison"]=="kg_full - kg_no_geometry"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    y=np.arange(len(diffs)); delta=np.array([float(r["class_balanced_delta"]) for r in diffs])
    lo=np.array([float(r["class_resampling_low"]) for r in diffs]); hi=np.array([float(r["class_resampling_high"]) for r in diffs])
    ax.errorbar(delta,y,xerr=np.array([np.maximum(0,delta-lo),np.maximum(0,hi-delta)]),fmt="o",color="#286d9d",capsize=4)
    for d,i in zip(delta,y): ax.annotate(f"{d:+.4f}",(d,i),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    ax.axvline(0,color="#888",lw=1); ax.set_yticks(y,[f"{r['geometry_kind']} | {r['condition']}" for r in diffs]); ax.invert_yaxis()
    ax.set_ylim(len(diffs)-.5,-.65)
    ax.set_xlabel("Full − remove exemplar cosine; positive favors Full")
    ax.set_title("Paired comparison; descriptive 95% class-resampling interval\n10 class clusters, no family-independent significance claim",fontsize=12)
    fig.tight_layout();fig.savefig(report/"paired_geometry.png",dpi=190);plt.close(fig)
    info=read_json(output/"class_matrix.json")
    table="| Method | Embedding Q25 nDCG | Descriptor Q25 nDCG | Embedding Q25 P@5 |\n|---|---:|---:|---:|\n"
    for m in methods:
        p=next(r for r in summary if (r["geometry_kind"],r["condition"],r["method"])==("embedding","q25_envelope",m))
        table+=f"|{LABELS[m]}|{val('embedding','q25_envelope',m):.4f}|{val('descriptor','q25_envelope',m):.4f}|{float(p['mean_precision@5']):.4f}|\n"
    functional="Functional evaluation awaits source-bound inspection of all pooled results."
    if (output/"functional_summary.csv").exists():
        fs=read_csv(output/"functional_summary.csv")
        fmethods=list(dict.fromkeys(r["method"] for r in fs)); yy=np.arange(len(fmethods))
        fig,axes=plt.subplots(1,2,figsize=(13,7))
        ftable="| Method | Embedding pool nDCG@5 | Descriptor pool nDCG@5 | Embedding P@5 | Strong P@5 |\n|---|---:|---:|---:|---:|\n"
        def fval(kind,m,key): return float(next(r for r in fs if (r['geometry_kind'],r['subset'],r['method'])==(kind,'matched_exemplar',m))[key])
        for ax,key,title in zip(axes,['mean_pool_ndcg@5','mean_precision@5'],['Pool nDCG@5','Precision@5']):
            for kind,offset,color in [('embedding',-.18,'#286d9d'),('descriptor',.18,'#d48a34')]:
                bars=ax.barh(yy+offset,[fval(kind,m,key) for m in fmethods],.33,color=color,label=kind);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
            ax.set_yticks(yy,[LABELS[m] for m in fmethods],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,1.14);ax.set_title(title+' — higher is better')
        fig.suptitle('Full test-gallery search; six paired exemplar cases; provisional AI relevance',fontsize=13)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2)
        fig.tight_layout(rect=[0,.03,1,.89]);fig.savefig(report/'functional_matrix.png',dpi=190);fig.savefig(report/'functional_matrix.pdf');plt.close(fig)
        for m in fmethods:
            ftable+=f"|{LABELS[m]}|{fval('embedding',m,'mean_pool_ndcg@5'):.4f}|{fval('descriptor',m,'mean_pool_ndcg@5'):.4f}|{fval('embedding',m,'mean_precision@5'):.4f}|{fval('embedding',m,'mean_strong_precision@5'):.4f}|\n"
        ev=read_json(output/'functional_evaluation.json')
        from .equality_audit import equality_audit
        equality=equality_audit(output)
        paired_functional_plot(output,report)
        functional=f"Search uses **{ev['search_gallery']} held-out candidates**, with **{ev['inspected_pool']} individually inspected pooled/control parts**, {ev['judgment_pairs']} provisional AI query–candidate grades and {ev['ranking_runs']} runs. The table below averages six exemplar cases in three families. All methods receive the same queries, gallery, frozen weights and grades. Pool nDCG's ideal denominator is the inspected pool, so it is not exhaustive full-gallery nDCG; full-gallery recall is unknown. P@5 covers every actually returned candidate. No independent human expert verification or functional significance claim.\n\n{ftable}\n![Functional comparison](functional_matrix.png)\n\nPaired case differences are in `../functional_paired_differences.csv`; do not count related variants as independent trials."
        functional += "\n\n" + qualitative_report(output, report)
        functional += f"\n\nThe equality audit found **{equality['exact_same_ndcg']}/{equality['pair_query_representation_comparisons']}** exactly equal nDCG comparisons: **{equality['identical_ordered_ids']}** have identical ordered part lists (including shared abstentions), while **{equality['same_ndcg_but_different_ids']}** retrieve different parts with the same grades at each rank. **{equality['rounding_only_at_3_decimals']}** additional equalities arise only from three-decimal rounding. These pair counts are descriptive and share queries; they are not independent sample sizes."
        fwin=[r for r in read_csv(output/'functional_paired_differences.csv') if r['comparison']=='kg_full - kg_no_geometry' and r['query_id'] in ['support_example','support_envelope','connect_example','connect_envelope','belt_example','belt_envelope']]
        fdeltas={kind:[float(r['delta_pool_ndcg@5']) for r in fwin if r['geometry_kind']==kind] for kind in ['embedding','descriptor']}
        functional += f"\n\nFull minus remove-cosine mean pool-nDCG change is **{np.mean(fdeltas['embedding']):+.4f}** for embeddings and **{np.mean(fdeltas['descriptor']):+.4f}** for descriptors over these six cases. Embeddings win/tie/lose {sum(d>1e-10 for d in fdeltas['embedding'])}/{sum(abs(d)<=1e-10 for d in fdeltas['embedding'])}/{sum(d< -1e-10 for d in fdeltas['embedding'])}; descriptors {sum(d>1e-10 for d in fdeltas['descriptor'])}/{sum(abs(d)<=1e-10 for d in fdeltas['descriptor'])}/{sum(d< -1e-10 for d in fdeltas['descriptor'])}. Both lose the compact broad belt-wheel case. Text-only cases have no available exemplar cosine, so their zero differences are expected. V-versus-timing queries have identical structured fields and rankings, despite reversed required profiles: scaling the KG does not repair the missing profile vocabulary. Flat-table and Full rankings remain equivalent.\n\n![Every paired functional difference](functional_paired.png)"
        sr=read_csv(output/'functional_annotation_sensitivity.csv')
        stable='| Conditional-grade scoring scenario | Embedding Full − remove cosine | Descriptor Full − remove cosine |\n|---|---:|---:|\n'
        qids={'support_example','support_envelope','connect_example','connect_envelope','belt_example','belt_envelope'}
        for variant in ['conditional_zero','nominal','conditional_two']:
            deltas=[]
            for kind in ['embedding','descriptor']:
                means={m:np.mean([float(r['pool_ndcg@5']) for r in sr if r['grading_scenario']==variant and r['geometry_kind']==kind and r['method']==m and r['query_id'] in qids]) for m in ['kg_full','kg_no_geometry']}
                deltas.append(means['kg_full']-means['kg_no_geometry'])
            stable+=f"|{variant}|{deltas[0]:+.4f}|{deltas[1]:+.4f}|\n"
        functional+='\n\nConditional functional judgments are also reweighted to zero or two as sensitivity scenarios, while numeric failures/unknowns retain their guards. These are hypothetical annotation interpretations, not confidence intervals or proven truth bounds.\n\n'+stable
    text=f"""# Full-scale CAD retrieval experiment

## Material Passport

- Origin: academic-research-suite validation workflow; local TMCAD source geometry and frozen legacy exports.
- Date/version: 2026-10-08 / full_scale_v1.
- Verification: computational matrix executed; independent human functional review absent.

The KG contains **{info['parts']:,} parts**, **{info['graph_nodes']:,} nodes** and **{info['graph_edges']:,} edges**. It has three source-supported generic function priors, not fabricated instance functions or compatibility edges. The ontology remains intentionally limited while instance coverage is full-corpus.

The auxiliary matrix uses **{info['train_exemplars']:,} training exemplars**, three query variants each, **{info['matched_test_gallery']:,} matched test candidates** and two representations/eight methods: **{info['ranking_runs']:,} ranking runs**. All original splits are retained. {info['trusted_dimension_gallery']:,} test candidates have trusted analytical extents; constrained variants use this common measurement-known gallery across every method. Unknown/failed measurements remain in the KG and unconstrained gallery. Exact-byte cross-split duplicates are excluded from held-out experiments; near-duplicate engineering families are not annotated.

## Meaning of remove exemplar cosine

`kg_no_geometry`, now labeled **Remove exemplar cosine**, sets aside only the score `g=(cosine(z_candidate,z_exemplar)+1)/2`. It retains predicted-category scores, the dot product of candidate/exemplar class probabilities, class-to-function priors and dimensional filtering. Both the classifier and dimensional measurements are geometry-derived. This is a ranking-signal ablation, not removal of every source of geometric information.

Full score is the weighted mean of available signals. With a functional exemplar query and no explicit category, Full averages `g`, `class_reference` and `semantic`; the ablation averages only `class_reference` and `semantic`. Equal weights are frozen before these results. Without an exemplar `g` is unavailable, so Full and the ablation are identical. Removing a signal also removes its denominator term. The class-only auxiliary tasks have no function signal. Thus Full and cosine+classifier differ there only when an envelope filter is requested.

The classifier/function ablation removes both class predictions and class-mediated function evidence, leaving exemplar cosine and the same filter. The no-filter ablation preserves all ranking signals. The flat-table control has identical information, equations and constraints to Full; equality is expected. No graph-storage effectiveness advantage is claimed.

## Auxiliary full-corpus results

These targets use existing class labels and computed envelopes, **not functional judgments**. Query categories come from the training exemplar label as explicit user-supplied categories; candidate labels are used only by evaluation. Bounds are the per-class 25th/75th percentiles of **validation** extents, frozen before test ranking. The constrained qrel is same class AND within the declared envelope. Filter success is expected by construction and does not prove engineering suitability.

{table}
![Numerically labeled full matrix](class_matrix.png)
![Paired geometry contribution](paired_geometry.png)

Each paired difference compares the same query and candidate set. Class-balanced scores weight ten classes equally. Intervals resample these ten observed class means; they describe class heterogeneity, not uncertainty from independent engineering families or natural user tasks. There are no p-values. Repeated variants share exemplars and must not be treated as independent evidence. Frozen class head test accuracy is {info['classifier_test']['accuracy']:.4f}; that does not establish functional accuracy. Larger runs may reduce sampling noise, but cannot force the geometry term to improve a task dominated by class predictions.

## Why many numbers are identical

The search gallery has 1,816 byte-unique matched test candidates; equality is largely structural rather than a consequence of the earlier 20-part gallery. **Not every matrix cell is a distinct active ablation.** Flat table and Full have identical information/formulas. Text-only functional queries have no exemplar, so removing exemplar cosine changes nothing. No-envelope queries make the filter ablation inactive. Category/no-cosine methods share the frozen classifier and measurements across representations, so changing the similarity vector does not affect those methods. The present source-supported prior links each of three functions to one class, meaning its propagated score is that class probability; it is not an independent instance-function predictor.

Class targets are binary and often reach a top-five ceiling even with imperfect overall classifier accuracy. Different correctly classified parts can produce exactly the same nDCG. The unconstrained category baseline has only ten distinct category requests, despite repeated training exemplar variants. P@5 changes in steps of 0.2 for each query; averaging six cases still gives coarse counts. Equal P@5 does not mean equal ordering, and equal judged grade sequences can yield equal nDCG for different parts. These repeated outcomes are not evidence of extra independent trials or graph-specific superiority.

Figures now use four decimals; raw CSVs retain full numerical precision. `../functional_equality_audit.csv` separates identical effective recipes, identical ordered parts, identical grade sequences, both-method abstentions and rounding-only equality. Functional query diversity is still only three related families, so a larger part gallery does not establish broad intent generalization.

## Functional full-gallery search

{functional}

## Reproduce and inspect

```powershell
$cadkgPython = 'D:\\Postdoc\\research\\jepa_cad\\.venv\\Scripts\\python.exe'
& $cadkgPython -m cadkg full-scale --stage all
```

`configs/full_scale.json` freezes the matrix. Stages `measure`, `build`, `class`, `functional-search`, `pool-evidence`, `functional-evaluate` and `report` can run separately. Measurement caches are bound to source SHA256; isolated workers have per-part timeouts and failures remain unknown. The numerical pipeline requires no API calls. AI observations are saved separately from the graph and human blank templates. See `../../../docs/EXPERIMENTS.md` for method equations and reproducibility details.

![Method schematic](../../../docs/figures/method_schematic.png)
"""
    (report/'REPORT.md').write_text(text,encoding='utf-8')
    write_json(report/'report_manifest.json',receipt('full-scale report',[output/'class_summary.csv',output/'class_paired_differences.csv',output/'functional_summary.csv']))
    return {'report':str((report/'REPORT.md').resolve())}


def qualitative_report(output, report):
    import matplotlib.pyplot as plt
    from .reference import draw_surface
    ranks=read_csv(output/'functional_rankings.csv')
    grades={(r['query_id'],r['part_id']):r for r in read_csv(output/'functional_ai_judgments.csv')}
    codes={m['part_id']:m['inspection_id'] for m in read_csv(output/'functional_pool_manifest.csv')}
    sections=[]
    for qid,title in [('support_example','Stationary mountable shaft support'),('connect_example','Shaft attachment evidence'),('belt_envelope','Belt wheel within 80 mm')]:
        methods=['geometry','kg_no_geometry','kg_full']
        fig=plt.figure(figsize=(12,10.5))
        explanations=[]
        for i,m in enumerate(methods):
            selected=sorted([r for r in ranks if (r['geometry_kind'],r['query_id'],r['method'])==('embedding',qid,m)],key=lambda r:int(r['rank']))[:3]
            for j,r in enumerate(selected):
                g=grades[qid,r['part_id']]
                code=codes[r['part_id']]
                with np.load(output/f'pool_evidence/{code}.npz',allow_pickle=False) as data:
                    ax=fig.add_subplot(3,3,i*3+j+1,projection='3d'); draw_surface(ax,data['triangles'])
                ax.set_title(f"{LABELS[m]} · rank {r['rank']}\n{code} · score {float(r['score']):.3f} · AI grade {g['grade']}\nEnvelope: {g['constraint_verdict']}",fontsize=9,color=['#ad473c','#a97925','#2d7a58'][int(g['grade'])])
                explanations.append({'method':m,'rank':int(r['rank']),'inspection_id':code,'grade':int(g['grade']),'observation':g['observation'],'rationale':g['rationale']})
        fig.suptitle(title+' — actual STEP geometry; provisional AI relevance',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(report/f'qualitative_{qid}.png',dpi=155);plt.close(fig)
        write_json(report/f'qualitative_{qid}.json',explanations)
        sections.append(f"**{title}.** Colored grade labels mean 0=mismatch/insufficient evidence, 1=conditional, 2=strong geometric shortlist evidence. Each score is a method-specific mean, not a calibrated probability. Evidence rationales are saved beside the figure.\n\n![{title}](qualitative_{qid}.png)")
    return '\n\n'.join(sections)


def paired_functional_plot(output,report):
    import matplotlib.pyplot as plt
    pairs=[r for r in read_csv(output/'functional_paired_differences.csv') if r['comparison']=='kg_full - kg_no_geometry']
    queries=list(dict.fromkeys(r['query_id'] for r in pairs))
    labels=['Support: text only','Support: exemplar','Support ≤220 mm','Shaft join: text only','Shaft join: exemplar','Shaft join ≤100 mm',
            'Belt wheel: text only','Belt wheel: exemplar','Belt wheel ≤80 mm','Support ≤100 mm (gap)','Single V groove ≤80 mm (gap)','Timing wheel ≤80 mm (gap)']
    fig,ax=plt.subplots(figsize=(12,8))
    y=np.arange(len(queries))
    for kind,offset,color in [('embedding',-.18,'#286d9d'),('descriptor',.18,'#d48a34')]:
        values=[float(next(r for r in pairs if r['geometry_kind']==kind and r['query_id']==qid)['delta_pool_ndcg@5']) for qid in queries]
        bars=ax.barh(y+offset,values,.32,label=kind,color=color)
        ax.bar_label(bars,labels=[f'{v:+.3f}' for v in values],padding=3,fontsize=8)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(-1.15,.62);ax.axvline(0,color='#777',lw=1);ax.axhline(8.5,color='#aaa',lw=1,ls='--')
    ax.set_xlabel('Full − remove exemplar cosine: pool nDCG@5; positive favors Full')
    ax.set_title('Paired full-gallery functional cases — provisional AI relevance\nRelated cases in three intent families; no independent-scenario confidence interval')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(report/'functional_paired.png',dpi=190);plt.close(fig)

