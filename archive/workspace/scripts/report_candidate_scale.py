"""Candidate-only evaluation report; all methods and negative results retained."""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from cadkg.io import read_csv,read_json,write_csv,write_json,sha256
from cadkg.experiments.verified_statistics import _flip_p,_bootstrap,_holm
from scripts.report_kg_constrained import mean,table,COMPARATORS

ROOT=Path('outputs/candidate_scale_v4')
CFG=Path('configs/candidate_scale_v4.json')
METRICS=['ndcg@5','precision@5','strong_precision@5','ndcg@10','ndcg@20','recall@5','recall@20','ap','ap@20']

def main():
    cfg=read_json(CFG);out=Path(cfg['output']);raw=read_csv(out/'per_query.csv');cohorts=read_json(out/'cohorts.json')
    audit=read_json(ROOT/'quality_gate.json');repro=read_json(out/'reproducibility.json');integrity=read_json(out/'integrity.json')
    if sha256(cfg['annotation_path'])!=audit['final_annotation_sha256']:raise ValueError('Annotation release changed')
    methods=integrity['methods'];families=sorted({r['family_id'] for r in raw})
    def select(kind,method,cohort='closed',subset='constrained_exemplar',grading='nominal',source=raw):
        def include(r):
            ex=r['has_exemplar']=='True';co=r['has_constraint']=='True'
            return dict(all=True,exemplar=ex,text=not ex,constrained_exemplar=ex and co,unconstrained_exemplar=ex and not co,constrained_text=not ex and co)[subset]
        return [r for r in source if (r['geometry_kind'],r['method'],r['cohort'],r['grading_scenario'])==(kind,method,cohort,grading) and include(r)]
    summary=[]
    for cohort in cohorts:
        for kind in ['embedding','descriptor']:
            for subset in ['all','exemplar','text','constrained_exemplar','unconstrained_exemplar','constrained_text']:
                for method in methods:
                    rows=select(kind,method,cohort,subset)
                    summary.append(dict(cohort=cohort,cads=len(cohorts[cohort]),geometry_kind=kind,subset=subset,method=method,queries=len(rows),families=len({r['family_id'] for r in rows}),**{m:mean(rows,m) for m in METRICS},dimension_failures=sum(int(r['dimension_failures']) for r in rows),dimension_unknowns=sum(int(r['dimension_unknowns']) for r in rows),requested_slots=5*len(rows),zero_relevance_queries=sum(int(r['relevant_total'])==0 for r in rows)))
    write_csv(ROOT/'candidate_summary.csv',summary)
    def s(kind,method='kg_full',cohort='closed',subset='constrained_exemplar'):
        return next(r for r in summary if (r['geometry_kind'],r['method'],r['cohort'],r['subset'])==(kind,method,cohort,subset))
    # Same focused 14 contrasts and eleven family clusters as the v3 report.
    rng=np.random.default_rng(cfg['seed']);comparisons=[];deltas=[]
    for kind in ['embedding','descriptor']:
        full=select(kind,'kg_full');assert len(full)==58
        springs=[i for i,f in enumerate(families) if f in {'compression_return','tension_return'}]
        groups=[[i] for i in range(len(families)) if i not in springs]+[springs]
        for method in COMPARATORS:
            base={r['query_id']:r for r in select(kind,method)};values=[]
            for fid in families:
                rows=[r for r in full if r['family_id']==fid]
                value=float(np.mean([float(r['ndcg@5'])-float(base[r['query_id']]['ndcg@5']) for r in rows]))
                deltas.append(dict(geometry_kind=kind,comparator=method,family_id=fid,delta=value));values.append(value)
            a=np.asarray(values);ci=_bootstrap(a,rng,20000);spring_ci=_bootstrap(a,rng,20000,groups)
            comparisons.append(dict(geometry_kind=kind,comparator=method,queries=58,families=11,delta=float(a.mean()),ci95_lower=ci[0],ci95_upper=ci[1],exact_p=_flip_p(a),spring_joint_p=_flip_p(a,groups),spring_ci95_lower=spring_ci[0],spring_ci95_upper=spring_ci[1],family_wins=int(sum(a>1e-12)),family_ties=int(sum(abs(a)<=1e-12)),family_losses=int(sum(a< -1e-12)),leave_one_family_out_min=float(min((a.sum()-v)/10 for v in a)),leave_one_family_out_max=float(max((a.sum()-v)/10 for v in a))))
    for r,p in zip(comparisons,_holm([r['exact_p'] for r in comparisons])):r.update(holm_p=p,holm_family_size=len(comparisons))
    write_csv(ROOT/'focused_comparisons.csv',comparisons);write_csv(ROOT/'focused_family_differences.csv',deltas)
    # Parent cohort rankings are label-independent; this directly isolates label effects.
    old=read_csv('outputs/kg_queries110_v3/per_query.csv');old_archive=np.load('outputs/kg_queries110_v3/complete_rankings.npz');archive=np.load(out/'complete_rankings.npz')
    rank_matches=0
    for kind in ['embedding','descriptor']:
        for q in read_json(cfg['expanded_query_path']):
            for method in methods:
                for field in ['order','score']:
                    a=archive[f"{kind}__{q['query_id']}__previous1350__{method}__{field}"]
                    b=old_archive[f"{kind}__{q['query_id']}__closed__{method}__{field}"]
                    if not np.array_equal(a,b):raise ValueError('Matched parent ranking changed')
                    rank_matches+=1
    correction=[]
    for kind in ['embedding','descriptor']:
        for method in methods:
            for subset in ['all','exemplar','constrained_exemplar','text']:
                before=select(kind,method,'closed',subset,source=old);after=select(kind,method,'previous1350',subset)
                correction.append(dict(geometry_kind=kind,method=method,subset=subset,**{f'old_{m}':mean(before,m) for m in METRICS},**{f'corrected_{m}':mean(after,m) for m in METRICS},**{f'delta_{m}':mean(after,m)-mean(before,m) for m in METRICS}))
    write_csv(ROOT/'label_correction_effect.csv',correction)
    grades=[]
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','intent_filter','geometry_class_filter','kg_no_class']:
            for scenario in ['conditional_zero','nominal','conditional_two']:
                rows=select(kind,method,grading=scenario)
                grades.append(dict(geometry_kind=kind,method=method,grading=scenario,ndcg5=mean(rows,'ndcg@5')))
    write_csv(ROOT/'grading_sensitivity.csv',grades)
    robust=[]
    rb=read_csv(out/'robustness_per_query.csv')
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','intent_filter','geometry_class_filter','kg_no_class','rrf_filter','geometry_intent_filter','kg_no_geometry']:
            for condition in sorted({(r['robustness'],r['severity']) for r in rb}):
                rows=[r for r in rb if (r['geometry_kind'],r['method'],r['has_exemplar'],r['has_constraint'],r['robustness'],r['severity'])==(kind,method,'True','True',*condition)]
                if rows:robust.append(dict(geometry_kind=kind,method=method,robustness=condition[0],severity=condition[1],ndcg5=mean(rows,'ndcg@5'),matched_nominal=s(kind,method)['ndcg@5']))
    write_csv(ROOT/'robustness_summary.csv',robust)
    # Full top-five evidence joins do not feed back into annotation production.
    from scripts.finalize_candidate_scale import resolved_records
    annotation=read_json(cfg['annotation_path']);obs={(r['family_id'],r['part_id']):r for r in resolved_records(cfg['annotation_path'])}
    queries={r['query_id']:r for r in read_json(out/'effective_protocol.json')['queries']}
    evidence=[]
    for r in read_csv(out/'rankings_top20.csv'):
        if r['cohort']!='closed' or int(r['rank'])>5:continue
        q=queries[r['query_id']];a=obs[q['scenario_id'],r['part_id']]
        evidence.append(dict(**r,family_id=q['scenario_id'],raw_geometry_grade=a['grade'],observation=a['observation'],rationale=a['rationale'],geometry_evidence=a['geometry_evidence'],source_sha256=a['source_sha256']))
    write_csv(ROOT/'top5_evidence.csv',evidence)
    write_json(ROOT/'analysis_manifest.json',dict(matched_parent_arrays_unchanged=rank_matches,query_matrix_unchanged=True,methods_unchanged=True,label_changes_frozen_before_scores=True,summary_sha256=sha256(ROOT/'candidate_summary.csv'),annotation_sha256=sha256(cfg['annotation_path']),raw_run_sha256=sha256(out/'run_manifest.json'),primary_tests=11,focused_tests=14,additional_cohort_tests=0))
    f=lambda v:f'{float(v):.4f}'
    main_table=table(['Method','Embedding nDCG@5','Descriptor nDCG@5','Embedding strong P@5','Descriptor strong P@5'],[[m,f(s('embedding',m)['ndcg@5']),f(s('descriptor',m)['ndcg@5']),f(s('embedding',m)['strong_precision@5']),f(s('descriptor',m)['strong_precision@5'])] for m in methods])
    cohort_table=table(['Cohort','CADs','Embedding Full','Descriptor Full','Intent + filter'],[[c,len(cohorts[c]),f(s('embedding',cohort=c)['ndcg@5']),f(s('descriptor',cohort=c)['ndcg@5']),f(s('descriptor','intent_filter',c)['ndcg@5'])] for c in cohorts])
    stats_table=table(['Representation','Full minus','Delta','95% interval','Holm p','Family W/T/L'],[[r['geometry_kind'],r['comparator'],f(r['delta']),f"[{f(r['ci95_lower'])}, {f(r['ci95_upper'])}]",f(r['holm_p']),f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in comparisons])
    grade_table=table(['Representation','Method','Conditional = 0','Nominal','Conditional = 2'],[[k,m,*[f(next(r['ndcg5'] for r in grades if (r['geometry_kind'],r['method'],r['grading'])==(k,m,g))) for g in ['conditional_zero','nominal','conditional_two']]] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']])
    depth_table=table(['Representation','Method','Recall@20','Full AP','nDCG@20'],[[k,m,*[f(s(k,m)[v]) for v in ['recall@20','ap','ndcg@20']]] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter','geometry_class_filter']])
    corr_table=table(['Representation','Old 1350 Full','Corrected 1350 Full','All 1816 Full'],[[k,*[f(v) for v in [next(r['old_ndcg@5'] for r in correction if (r['geometry_kind'],r['method'],r['subset'])==(k,'kg_full','constrained_exemplar')),s(k,cohort='previous1350')['ndcg@5'],s(k)['ndcg@5']]]] for k in ['embedding','descriptor']])
    rob_table=table(['Representation','Full nominal','10% permutation','30% permutation'],[[k,f(s(k)['ndcg@5']),*[f(next(r['ndcg5'] for r in robust if (r['geometry_kind'],r['method'],r['robustness'],float(r['severity']))==(k,'kg_full','classifier_permutation',fraction))) for fraction in [.1,.3]]] for k in ['embedding','descriptor']])
    primary=read_csv(out/'primary_comparisons.csv');ndup=read_json(out/'near_duplicate_audit.json')
    belt=next(r['delta'] for r in deltas if (r['geometry_kind'],r['comparator'],r['family_id'])==('descriptor','geometry_class_filter','belt_wheel'))
    no_belt=np.mean([r['delta'] for r in deltas if r['geometry_kind']=='descriptor' and r['comparator']=='geometry_class_filter' and r['family_id']!='belt_wheel'])
    added_zero=sorted({r['family_id'] for r in select('descriptor','kg_full','added466') if int(r['relevant_total'])==0})
    report=f'''# Annotation audit and candidate-scale evaluation

## Material Passport

- Date: 2026-10-21. Author/reviewer: Codex AI, shared context and same model family; no independent human adjudication.
- Verification: **{repro['status']} computational replay**, source integrity and separate metric implementation. Semantic relevance remains provisional AI assessment.
- Scope: 1,350 → **1,816 held-out CADs** (+466, +34.5%); 14,850 → **19,976 judgments**. Same 110 queries, 11 authored task families, 14 methods, two representations, equal weights, exemplars, models, constraints and robustness conditions.
- Design: [fixed candidate expansion plan](../../docs/CANDIDATE_SCALE_EVALUATION.md), [configuration](../../configs/candidate_scale_v4.json), initial `freeze.json`, and annotation-first `evaluation_freeze.json`. Sequential exploratory evaluation; historical results were known.

## What the annotation audit establishes

Every existing record passed completeness, grade/decision, rationale linkage, provenance and source binding checks; all 1,350 raw STEP files were rehashed before review. Integrity is not semantic correctness. A fresh seeded simple random sample of 100 CADs was visually rereviewed against all eleven unchanged rubrics (1,100 records). All 35 additional shared-rationale outlier CADs were inspected separately (385 records). Outlier selection used disagreements between catalog category and a repeated observation only to flag inspection; categories, previous grades and method ranks were not displayed in the source sheets and never supplied grades. This targeted selection was added after observed copied-rationale errors, before scoring, and is not a probability sample.

There were 57 proposed differences across both components. Explicit source-based adjudication accepted **32 corrections** and retained 25 parent grades where evidence was ambiguous or did not meet the existing rubric. The probability sample contributed **14/1,100 corrections (1.27%)**, with candidate-cluster bootstrap interval **0.55–2.18%**; these are same-model adjudicated changes, not independently measured annotation error. The targeted component contributed 18 corrections and is excluded from that rate. In the probability sample, changes were 3/1,000 rejects, 10/47 conditional grades (21.3%), and 1/53 shortlist grades. The earlier 166-CAD audit is historical evidence, not an independent replication to pool with this sample.

`audit_notes.txt`, `supplemental_notes.txt`, `audit_proposals.csv`, `adjudication.json`, `audit_review_pairs.csv` and [release corrections](../../annotations/intent_candidates1816_v4/corrections.csv) retain observations, previous/proposed/final grades, evidence and decisions. Unchanged parent records are preserved exactly; the other 1,215 parent CADs were not semantically revalidated this turn. No claim that all labels are correct follows from passing the audit.

The [evidence supplement](../../annotations/intent_candidates1816_v4/evidence_corrections.json) also applies the source-specific observations consistently to unchanged-grade roles on corrected CADs. It was packaged during report finalization from observations authored before scoring; its binding preserves every sealed numerical label. `resolved_records` applies that supplement for displayed evidence, including `top5_evidence.csv`. The original scored JSON remains unchanged for exact replay; old copied observations are retained as provenance, not endorsed as current evidence.

Concrete errors include a pulley-and-belt assembly described as a torsion spring; crankshaft and splined-shaft images carrying headed-fastener observations; threaded adapters carrying a crankshaft observation; bevel/worm mechanisms treated as cylindrical gears; a large passage flange treated as a standalone nut; and complete articulated/chain connections carrying a shaft or gear rationale. Some proposed upgrades were rejected: diagonal mesh lines do not establish helical thread, a bare bearing does not meet the mountable-carrier requirement, and generic carrier bores do not automatically establish a rotary seat. The fresh probability sample's raw proposals and retained ambiguities remain inspectable.

All 466 previously unjudged gallery members were rendered and individually inspected against all eleven rubrics, with source-linked notes. No retrieval-selected additions, category proxies, unjudged-as-reject fill, model retraining or performance-guided label changes were used. Grade 1 is a positive but unresolved preliminary shape assessment; it is not independent expert suitability. No null labels remain under these shape-only rubrics, which does not imply complete engineering evidence.

## Does annotation quality explain the old 0.67?

The reference 0.6767 was **family-balanced constrained-exemplar nDCG@5**, not classification accuracy. Its gains are exponential (0,1,3), so conditional versus strong labels matter; numeric reference unknowns cap grades at 1 and failures make them 0. An ideal score of one is benchmark ranking perfection, not a minimum acceptable engineering performance threshold.

{corr_table}

All **{rank_matches:,} parent-cohort order/score arrays** are unchanged. The label corrections leave constrained Full nDCG@5 unchanged on the original cohort for both representations, while other endpoints can change (complete per-method effects are in `label_correction_effect.csv`). Thus these supported annotation corrections do **not** explain or cure the old top-five score. This does not rule out additional systematic AI-label bias; unsampled semantic correctness remains unverified. The all-gallery descriptor change is +{f(s('descriptor')['ndcg@5']-s('descriptor',cohort='previous1350')['ndcg@5'])}, while embedding changes {f(s('embedding')['ndcg@5']-s('embedding',cohort='previous1350')['ndcg@5'])}; changes in candidate competition and relevance denominators are distinct from corrections.

## Larger-gallery effectiveness

The following retains every method on the same 58 constrained exemplar queries, equal-family averaging:

{main_table}

Same-filter controls distinguish screening from ranking. Full descriptor improves over unfiltered Full by {f(s('descriptor')['ndcg@5']-s('descriptor','kg_no_constraints')['ndcg@5'])}; its increment over geometry+classifier+filter is {f(s('descriptor')['ndcg@5']-s('descriptor','geometry_class_filter')['ndcg@5'])}. These are software comparisons on this benchmark. Full equals matched flat Full by construction; intent+filter equals category+filter for the admitted unit-weight priors. Neither equality can be fixed by more annotations. Text-only Full has no geometry signal and reduces to the class-function prior; both spring functions share the same spring prior, limiting role discrimination without exemplars. Broad language understanding or a graph-topology effectiveness benefit is not demonstrated.

## Candidate robustness and generalizability

{cohort_table}

`previous1350` isolates the corrected parent gallery; `added466` contains all previously unjudged test CADs; `new_only` contains all 1,392 candidates outside the original retrieval-pooled 424. Comparisons **within** each cohort are matched. Scores across cohorts have different candidate competition, prevalence and normalization and are descriptive. Eleven of 58 added-only constrained queries have no relevant candidate, in these families: {', '.join(added_zero) or 'none'}; existing metric convention assigns nDCG zero to zero-ideal cases, and `candidate_summary.csv` reports their counts. These zeros limit interpretation of the added-only absolute score; no query was removed from either method's comparison.

On the added-only cohort, descriptor Full is **{f(s('descriptor',cohort='added466')['ndcg@5'])}**, below same-filter intent **{f(s('descriptor','intent_filter','added466')['ndcg@5'])}**; embedding Full is **{f(s('embedding',cohort='added466')['ndcg@5'])}**. Full also trails intent on the expanded new-only cohort. This is evidence of limited within-benchmark transfer to candidates outside the earlier pool, not proof of external-domain failure. Candidate-category counts are preserved in `evaluation/sampling_distribution.csv`; adding the complement of a class-balanced cohort need not itself be class-balanced.

The numeric descriptor audit finds {ndup['train_matches']} train-matching test signatures; the deduplicated cohort retains {len(cohorts['deduplicated'])} CADs. These are descriptor-signature checks, not exhaustive design-family deduplication. The full eligible held-out gallery is now completely judged; it is not all 9,799 catalog parts, many of which are training/validation inputs. Promoting those to relevance-test cases would contaminate the existing protocol. Eleven authored families and same-source CADs still limit unseen-task and cross-domain claims.

## Statistical evidence and task concentration

The unchanged primary analysis has {sum(float(r['holm_p'])<.05 for r in primary)} of eleven Holm-adjusted comparisons below 0.05. The same existing focused analysis has {sum(r['holm_p']<.05 for r in comparisons)} of fourteen. Full does not establish superiority over the strong same-filter classifier/prior/rank-fusion controls after correction. Nonsignificance is not evidence of equivalence. Intervals below are unadjusted exploratory family-bootstrap intervals, whereas p-values use their declared Holm family; they need not agree at the same threshold.

{stats_table}

The descriptor prior increment over geometry+classifier+filter is concentrated in belt wheels: belt family delta {f(belt)}; mean across the other ten families {f(no_belt)}. `focused_family_differences.csv` retains every family loss. Joint-spring sign flips, clustered intervals and leave-one-family-out ranges are preserved. Neither 110 variants nor 19,976 correlated records are independent experimental units. Sign-flip exchangeability remains an exploratory assumption for this authored task set; more CADs do not increase the number of task clusters.

## Annotation interpretation, depth and existing perturbations

{grade_table}

These are alternative conditional-label policies with separately normalized ideals, not confidence intervals or new experimental conditions. Numeric unknowns remain capped at one even in the conditional-two scenario. Visible threads, internal bores, belt/gear profiles, bare-bearing versus carrier scope, and complete versus partial shaft attachment remain important ambiguity sources. Three views and same-kernel geometry cannot establish material, load rating, tolerance, mating fit or actual application.

{depth_table}

{rob_table}

The same five classifier-permutation seeds, tie orders and numerical guard values are retained; complete results include all seven previously perturbed methods in `robustness_summary.csv`. Synthetic permutation is a reliance diagnostic, not observed domain shift. Screening failures/unknowns and missing-result slots are in `candidate_summary.csv`; unknown is not a verified pass. The production filter and independent surface-bound reference share a geometry kernel, so their agreement is not external metrology. Exemplar choices remain fixed; historical large exemplar effects continue to limit robustness interpretation.

## Necessary improvements, without changing this experiment

1. Prioritize an independent human audit of positive/conditional cases and method disagreements, with views that can resolve internal thread, keyways and seats. Preserve prior/AI decisions, hide ranks and categories, collect per-feature evidence and adjudicate independently. The present review is shared-context AI consistency work, not inter-rater agreement.
2. Separate future rubric development from held-out testing. Clarify carrier-seat versus plain bore, short pin versus drive shaft, shaft hub versus passage flange, and sprocket/belt/gear profile boundaries in a development pilot. Do not retroactively rewrite ambiguous test labels to favor a method. Allow cannot-assess where decisive shape evidence truly cannot be read; keep geometric conditional suitability distinct from missing data.
3. Replace copied generic observations with source-specific feature evidence and use adequately enlarged views before grading. Retain automated hashes, split checks and rationale diagnostics. Fine mesh edges must never be taken as threads.
4. For future method work, address exemplar sensitivity, the coarse class-prior bottleneck and representation failure on subtle functional features. Use separate development labels for any weights, learned function model or exemplar selection; retuning on this completed benchmark would invalidate an untouched-test claim. This turn performs none of that tuning.
5. Stronger generalizability needs independent tasks and design/external-domain separation. Candidate count alone cannot establish it. Keep claims at preliminary CAD shortlisting and measured envelope screening.

## Reproducibility and fairness

- {repro['csv_tables_exactly_replayed']} core CSV tables exactly replayed; {repro['complete_ranking_arrays_exactly_replayed']:,} complete order/score arrays exactly matched; {repro['metric_cases_independently_checked']:,} metric cutoff cases checked using the separate existing metric implementation; {repro['raw_step_hashes_verified']:,} raw STEP hashes verified.
- A separately implemented grade constructor checked 599,280 query/candidate/conditional-policy cases. Full has no surface-reference failures among constrained top-five returns, but 10 embedding and 14 descriptor slots remain reference-unknown out of 290 requested slots each. All 202 exact-byte cross-split contamination exclusions remain unchanged; the descriptor signature sensitivity separately removes train-matched and repeated gallery signatures.
- Frozen annotation hashes, unchanged query/model/prior/exemplar bindings, all complete rankings, family statistics, corrections, negative results and source-view evidence are saved. Labels never enter serving graphs or classifier inputs.
- Parent arrays exactly unchanged; no test-fitted descriptor normalization, train exemplar leakage or method-specific candidate access introduced. Exact-byte contamination exclusions remain frozen. Same-filter methods receive the same candidate, constraint and unknown policies; unfiltered controls remain explicit ablations. Geometry-only text abstentions must not be read as fair semantic superiority evidence.
- Follow the commands below. Mechanically assembling reviewed notes is reproducible; visual judgment is not guaranteed to repeat identically.

```powershell
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.finalize_candidate_scale compare
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.finalize_candidate_scale publish
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.finalize_candidate_scale evidence-overlay
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.run_verified_evaluation --config configs/candidate_scale_v4.json
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.run_verified_evaluation --config configs/candidate_scale_v4.json --output outputs/candidate_scale_v4/evaluation/replay
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.validate_verified_evaluation --config configs/candidate_scale_v4.json
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.validate_candidate_scale
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.report_candidate_scale
```

## Statistical fallacy review: 11/11 checked

| Check | Disposition |
|---|---|
| Simpson reversal | Cohort/family tables retain reversals and task-concentrated gains. |
| Ecological inference | Means describe authored benchmark tasks, not individual engineering suitability. |
| Selection/Berkson bias | Original retrieval pool, class-balanced parent and full held-out complement are explicit. |
| Collider conditioning | Returned-only scores and candidate selection are not causal effects. |
| Base rates | Full cohort prevalence, zero-positive cases, strong grades and random feasible expectations retained. |
| Regression to mean | Annotation corrections are not a performance intervention or optimization target. |
| Survivorship | Every existing query/method, empty/zero-ideal case and negative result retained. |
| Look-elsewhere | Same eleven primary and fourteen focused Holm families retained; no tests on added-cohort diagnostics. |
| Forking paths | Initial scope frozen, adaptive rationale audit disclosed, labels sealed before scoring; exploratory sequential design. |
| Causal overclaim | Software contrasts do not establish industrial or physical causal benefits. |
| Reverse causality | No real-world directional causal inference is estimated. |

Methodological grounding: [NIST's pooled-versus-sampled judgment comparison](https://www.nist.gov/publications/comparison-pooled-and-sampled-relevance-judgments) identifies completeness and system-selection bias concerns in pooled test collections; completing the eligible gallery addresses missing judgments here, not assessor correctness. [Smucker, Allan and Carterette](https://ciir-publications.cs.umass.edu/getpdf.php?id=744) compare paired IR significance procedures and support tests of the reported statistic; their TREC findings do not validate this benchmark's small, dependent family set.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(dict(report=str(ROOT/'REPORT.md'),primary_significant=sum(float(r['holm_p'])<.05 for r in primary),focused_significant=sum(r['holm_p']<.05 for r in comparisons),matched_parent_arrays=rank_matches))

if __name__=='__main__':main()
