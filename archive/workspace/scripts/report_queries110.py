"""Expanded effectiveness, provenance, difficulty and family-paired comparisons."""
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256,object_hash
from cadkg.reasoning import check_constraints
from cadkg.experiments.verified_statistics import _flip_p,_bootstrap,_holm
from scripts.run_verified_evaluation import grade_vector
from scripts.report_kg_constrained import NAMES,COMPARATORS,METRICS,family_values,mean,table

CFG=Path('configs/kg_queries110_v3.json')


def main():
    cfg=read_json(CFG);out=Path(cfg['output']);raw=read_csv(out/'per_query.csv')
    protocol=read_json(out/'effective_protocol.json');queries={q['query_id']:q for q in protocol['queries']}
    old_protocol=read_json(Path(cfg['source_run'])/'protocol.json');old_ids={q['query_id'] for q in old_protocol['queries']}
    annotation=read_json(cfg['annotation_path']);audit=read_json(cfg['annotation_audit_path'])
    assert audit['audit_complete'] and sha256(cfg['annotation_path'])==audit['final_annotation_sha256']
    # Compare every historic per-query row, not rounded aggregate numbers.
    old=read_csv('outputs/kg_constrained_audited_v2/per_query.csv')
    key=lambda r:tuple(r[k] for k in ['cohort','grading_scenario','geometry_kind','query_id','method'])
    previous={key(r):r for r in old};current={key(r):r for r in raw if r['query_id'] in old_ids}
    assert previous==current
    archive=np.load(out/'complete_rankings.npz');old_archive=np.load('outputs/kg_constrained_audited_v2/complete_rankings.npz')
    assert all(np.array_equal(old_archive[k],archive[k]) for k in old_archive.files)
    continuity=dict(original_per_query_rows_exactly_preserved=len(previous),original_complete_ranking_arrays_exactly_preserved=len(old_archive.files))
    write_json(out/'original49_continuity.json',continuity)
    def select(kind,method,subset='constrained_exemplar',cohort='closed',origin='all',grading='nominal'):
        def include(r):
            q=queries[r['query_id']];ex=bool(q['reference_part_id']);co=bool(q['constraints'])
            return {'all':True,'exemplar':ex,'text':not ex,'constrained_exemplar':co and ex,
                'constrained_text':co and not ex,'unconstrained_exemplar':ex and not co}[subset]
        return [r for r in raw if (r['cohort'],r['geometry_kind'],r['method'],r['grading_scenario'])==(cohort,kind,method,grading)
            and include(r) and (origin=='all' or (r['query_id'] in old_ids)==(origin=='original49'))]
    summaries=[]
    for cohort in ['closed','new_only','original_pool','deduplicated']:
        for kind in ['embedding','descriptor']:
            for subset in ['all','exemplar','text','constrained_exemplar','constrained_text','unconstrained_exemplar']:
                for origin in ['all','original49','added61']:
                    for method in NAMES:
                        rows=select(kind,method,subset,cohort,origin)
                        if not rows:continue
                        summaries.append(dict(cohort=cohort,geometry_kind=kind,subset=subset,origin=origin,method=method,
                            queries=len(rows),families=len({r['family_id'] for r in rows}),
                            **{m:mean(rows,m) for m in METRICS},returned_slots=sum(int(r['returned']) for r in rows),
                            top5_slots=5*len(rows),dimension_failures=sum(int(r['dimension_failures']) for r in rows),
                            dimension_unknowns=sum(int(r['dimension_unknowns']) for r in rows)))
    write_csv(out/'expanded_summary.csv',summaries)
    def s(kind,method,subset='constrained_exemplar',cohort='closed',origin='all'):
        return next(r for r in summaries if tuple(r[k] for k in ['cohort','geometry_kind','subset','origin','method'])==(cohort,kind,subset,origin,method))
    comparisons=[];deltas=[];rng=np.random.default_rng(cfg['seed'])
    for kind in ['embedding','descriptor']:
        full=select(kind,'kg_full');families=sorted({r['family_id'] for r in full})
        springs=[i for i,f in enumerate(families) if f in {'compression_return','tension_return'}]
        groups=[[i] for i in range(len(families)) if i not in springs]+[springs]
        assert len(full)==58 and len(families)==11
        for method in COMPARATORS:
            base={r['query_id']:r for r in select(kind,method)}
            diff=[]
            for fid in families:
                rows=[r for r in full if r['family_id']==fid]
                values={m:float(np.mean([float(r[m])-float(base[r['query_id']][m]) for r in rows])) for m in METRICS}
                deltas.append(dict(geometry_kind=kind,comparator=method,family_id=fid,queries=len(rows),**{'delta_'+m:v for m,v in values.items()}))
                diff.append(values['ndcg@5'])
            a=np.asarray(diff);ci=_bootstrap(a,rng,20000);spring_ci=_bootstrap(a,rng,20000,groups)
            comparisons.append(dict(geometry_kind=kind,comparator=method,queries=58,families=11,delta=float(a.mean()),
                ci95_lower=ci[0],ci95_upper=ci[1],exact_p=_flip_p(a),spring_joint_p=_flip_p(a,groups),
                spring_ci95_lower=spring_ci[0],spring_ci95_upper=spring_ci[1],family_wins=int(sum(a>1e-12)),
                family_ties=int(sum(abs(a)<=1e-12)),family_losses=int(sum(a< -1e-12)),
                leave_one_family_out_min=float(min((a.sum()-v)/10 for v in a)),leave_one_family_out_max=float(max((a.sum()-v)/10 for v in a))))
    for r,p in zip(comparisons,_holm([r['exact_p'] for r in comparisons])):r.update(holm_p=p,holm_family_size=len(comparisons))
    write_csv(out/'expanded_constrained_comparisons.csv',comparisons);write_csv(out/'expanded_family_differences.csv',deltas)
    cohorts=read_json(out/'cohorts.json');ids=cohorts['closed'];parts={p['part_id']:p for p in read_json(Path(cfg['source_run'])/'bundle/catalog.json')['parts']}
    obs={(r['family_id'],r['part_id']):r for r in annotation['judgments']};evidence={}
    for r in annotation['judgments']:
        if r['part_id'] not in evidence:evidence[r['part_id']]=read_json(Path(cfg['evidence_root'])/f"{r['inspection_id']}.json")
    difficulty=[];qrel_signatures=set();structured_signatures=set()
    for qid,q in queries.items():
        grades=grade_vector(q,ids,obs,evidence);feasible=sum(check_constraints(parts[p],q['constraints'])['status'] in {'pass','not_requested'} for p in ids)
        sig=object_hash([q['scenario_id'],grades.tolist()]);qrel_signatures.add(sig)
        rank_sig=object_hash([q['function_ids'],q['reference_part_id'],q['constraints']]);structured_signatures.add(rank_sig)
        fraction=feasible/len(ids)
        difficulty.append(dict(query_id=qid,family_id=q['scenario_id'],origin='original49' if qid in old_ids else 'added61',
            has_exemplar=bool(q['reference_part_id']),has_constraint=bool(q['constraints']),
            bound_mm=q['constraints'][0]['value'] if q['constraints'] else '',feasible=feasible,feasible_fraction=fraction,
            size_selectivity='tight' if fraction<=1/3 else 'medium' if fraction<=2/3 else 'loose',
            relevant=int(sum(grades>0)),strong=int(sum(grades==2)),qrel_signature=sig,structured_signature=rank_sig))
    write_csv(out/'query_difficulty.csv',difficulty);byq={r['query_id']:r for r in difficulty}
    strata=[]
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','kg_no_constraints','geometry_class_filter','intent_filter','rrf_filter']:
            for stratum in ['tight','medium','loose']:
                rows=[r for r in select(kind,method) if byq[r['query_id']]['size_selectivity']==stratum]
                if rows:strata.append(dict(geometry_kind=kind,method=method,size_selectivity=stratum,queries=len(rows),families=len({r['family_id'] for r in rows}),ndcg5=mean(rows,'ndcg@5')))
    write_csv(out/'selectivity_summary.csv',strata)
    sensitivities=[]
    for kind in ['embedding','descriptor']:
        for grading in ['conditional_zero','nominal','conditional_two']:
            for method in ['kg_full','geometry_class_filter','intent_filter','kg_no_class']:
                rows=select(kind,method,grading=grading)
                sensitivities.append(dict(geometry_kind=kind,method=method,grading=grading,ndcg5=mean(rows,'ndcg@5'),strong_precision5=mean(rows,'strong_precision@5')))
    write_csv(out/'expanded_grading_sensitivity.csv',sensitivities)
    robustness=[];rob=read_csv(out/'robustness_per_query.csv')
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','geometry_class_filter','intent_filter','kg_no_class']:
            for condition in sorted({(r['robustness'],r['severity']) for r in rob}):
                rows=[r for r in rob if r['geometry_kind']==kind and r['method']==method and r['has_constraint']=='True' and r['has_exemplar']=='True' and (r['robustness'],r['severity'])==condition]
                if rows:robustness.append(dict(geometry_kind=kind,method=method,robustness=condition[0],severity=condition[1],ndcg5=mean(rows,'ndcg@5'),nominal_ndcg5=s(kind,method)['ndcg@5']))
    write_csv(out/'expanded_robustness_summary.csv',robustness)
    # Conservative full-gallery bounds retain unjudged-return uncertainty.
    bound_rows=read_csv(out/'full_gallery_bounds.csv');bounds=[]
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','geometry_class_filter','intent_filter']:
            rows=[r for r in bound_rows if r['geometry_kind']==kind and r['method']==method and queries[r['query_id']]['constraints'] and queries[r['query_id']]['reference_part_id']]
            bounds.append(dict(geometry_kind=kind,method=method,queries=len(rows),unjudged_at5=sum(int(r['unjudged_at5']) for r in rows),
                ndcg_lower=mean(rows,'ndcg_lower'),ndcg_upper=mean(rows,'ndcg_upper')))
    write_csv(out/'expanded_full_gallery_bounds.csv',bounds)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    methods=['kg_full','kg_no_constraints','geometry_class_filter','intent_filter','rrf_filter']
    labels=['KG-constrained Full','Full without constraints','Geometry + classifier + filter','Intent prior + filter','Rank fusion + filter']
    fig,ax=plt.subplots(figsize=(9,4));y=np.arange(len(methods))
    for off,kind,color in [(-.18,'embedding','#446b91'),(.18,'descriptor','#d28441')]:
        ax.barh(y+off,[s(kind,m)['ndcg@5'] for m in methods],height=.36,color=color,label=kind.capitalize())
    ax.set(yticks=y,yticklabels=labels,xlim=(0,1),xlabel='Family-balanced nDCG@5; 58 constrained exemplar queries')
    ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'effectiveness_110.{ext}',dpi=200)
    plt.close(fig)
    f=lambda x:f'{float(x):.4f}'
    main_table=table(['Method','Embedding nDCG@5','Descriptor nDCG@5','Embedding strong P@5','Descriptor strong P@5'],
        [[NAMES[m],*[f(s(k,m)[metric]) for metric in ['ndcg@5','strong_precision@5'] for k in ['embedding','descriptor']]] for m in ['kg_full',*COMPARATORS]])
    comp_table=table(['Representation','Full minus','Delta nDCG@5','95% interval','Holm p','Family W/T/L'],
        [[r['geometry_kind'],r['comparator'],f(r['delta']),f"[{f(r['ci95_lower'])}, {f(r['ci95_upper'])}]",f(r['holm_p']),f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in comparisons])
    origin_table=table(['Query origin','Queries','Embedding Full','Embedding intent+filter','Descriptor Full','Descriptor intent+filter'],
        [[origin,s('descriptor','kg_full',origin=origin)['queries'],*[f(s(k,m,origin=origin)['ndcg@5']) for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']]] for origin in ['original49','added61','all']])
    cohort_table=table(['Cohort','Candidates','Embedding Full','Embedding intent+filter','Descriptor Full','Descriptor intent+filter'],
        [[c,len(cohorts[c]),*[f(s(k,m,cohort=c)['ndcg@5']) for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']]] for c in cohorts])
    subset_table=table(['Inputs','Queries','Embedding Full','Descriptor Full','Intent+filter'],
        [[sub,s('descriptor','kg_full',subset=sub)['queries'],f(s('embedding','kg_full',subset=sub)['ndcg@5']),f(s('descriptor','kg_full',subset=sub)['ndcg@5']),f(s('descriptor','intent_filter',subset=sub)['ndcg@5'])] for sub in ['all','exemplar','text','constrained_exemplar','constrained_text','unconstrained_exemplar']])
    compliance_table=table(['Method','Embedding fail/unknown','Descriptor fail/unknown','Slots each'],
        [[NAMES[m],*[f"{s(k,m)['dimension_failures']}/{s(k,m)['dimension_unknowns']}" for k in ['embedding','descriptor']],s('embedding',m)['top5_slots']] for m in ['kg_full','kg_no_constraints','geometry_class_filter','intent_filter']])
    robust_table=table(['Representation','Method','Nominal','10% permutation','30% permutation'],
        [[k,NAMES[m],f(s(k,m)['ndcg@5']),*[f(next(r['ndcg5'] for r in robustness if (r['geometry_kind'],r['method'],r['robustness'],float(r['severity']))==(k,m,'classifier_permutation',fraction))) for fraction in [.1,.3]]]
            for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']])
    grade_table=table(['Representation','Method','Conditional = 0','Nominal','Conditional = 2'],
        [[k,NAMES[m],*[f(next(r['ndcg5'] for r in sensitivities if (r['geometry_kind'],r['method'],r['grading'])==(k,m,g))) for g in ['conditional_zero','nominal','conditional_two']]]
            for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']])
    deeper_table=table(['Representation','Method','Recall@20','AP','nDCG@20'],
        [[k,NAMES[m],*[f(s(k,m)[metric]) for metric in ['recall@20','ap','ndcg@20']]] for k in ['embedding','descriptor'] for m in ['kg_full','geometry_class_filter','intent_filter','rrf_filter']])
    run=read_json(out/'run_manifest.json');replay=read_json(out/'replay/run_manifest.json');check=read_json(out/'reproducibility.json');serving=read_json(out/'new_query_serving_checks.json')
    primary=read_csv(out/'primary_comparisons.csv');primary_rejections=[r for r in primary if float(r['holm_p'])<.05]
    focused_rejections=[r for r in comparisons if r['holm_p']<.05]
    constrained=[d for d in difficulty if d['has_constraint']]
    strata_count=Counter(d['size_selectivity'] for d in constrained)
    noanswer=sum(r['relevant']==0 for r in difficulty)
    e,d=s('embedding','kg_full'),s('descriptor','kg_full')
    text=f'''# Expanded evaluation of the KG-constrained method: 110 queries

## Material and design

The frozen design contains **110 structured variants, ten per each of eleven authored task families**. The original 49 remain unchanged; 61 additions use only validation-derived size percentiles. There are **77 exemplar queries (58 constrained, 19 unconstrained)** and **33 text-only queries (22 constrained, 11 unconstrained)**. The previous focus of 19 was the constrained exemplar subset of 49, not a smaller replacement for the entire benchmark.

Ten per family is a practical balanced alternative to the suggested 108. The additions exercise tight/loose size requests with the existing nineteen inspected training exemplars and text-only requests. There are {len(structured_signatures)} distinct structured requests and {len(qrel_signatures)} distinct family/query relevance vectors. Multiple exemplars can share relevance vectors; this is meaningful ranking variation but not independent relevance evidence. Among the 80 constrained variants, closed-gallery feasible proportions span {min(x['feasible_fraction'] for x in constrained):.1%}–{max(x['feasible_fraction'] for x in constrained):.1%}; fixed thirds classify {strata_count['tight']} tight, {strata_count['medium']} medium and {strata_count['loose']} loose cases. There are {noanswer} zero-relevance queries; no queries were discarded or replaced after scoring.

Labels are the corrected **audited-v2 14,850 task–part assessments over 1,350 CADs**, SHA256 `{sha256(cfg['annotation_path'])}`. The completed random audit reviewed 1,826 pairs over 166 CADs and corrected 36 grades; the primary 100-CAD random sample changed 22/1,100 grades. It was a same-model visual rereview, not independent human validation. Unsampled labels retain uncertainty; conditional grades receive explicit sensitivity analysis. New variants reuse the unchanged shape rubric and combine it with a per-query numeric verdict. No new annotation records or independently validated task families are claimed.

The [prespecified extension](../../docs/QUERY_EXPANSION_110.md), validation source IDs/sizes, quantiles, structured queries and hashes are saved. Selection uses no test annotations, test ranking outcomes or test sizes. The benchmark covers the existing eleven roles and class-conditioned validation sizes; it is not a random sample of industrial requests. Only largest stored-axis extent upper bounds are evaluated, not actual assembly fit, load, material or interface compatibility.

## Fairness and verification

All fourteen methods, both representations, trained artifacts, scoring weights, cutoffs, constraint handling, seeds, robustness settings and candidate galleries remain unchanged. The full gallery has 1,816 candidates; the completely judged cohort has 1,350. The original **{continuity['original_per_query_rows_exactly_preserved']:,} metric rows and {continuity['original_complete_ranking_arrays_exactly_preserved']:,} ranking/score arrays match exactly**. All {serving['matched']:,} new-query × original-method × representation serving checks match both top-five IDs and scores to 1e-12. The independently implemented metric calculation checks {check['metric_cases_independently_checked']:,} cutoff cases; all {check['complete_ranking_arrays_exactly_replayed']:,} arrays and {check['csv_tables_exactly_replayed']} primary CSV tables replay exactly. Raw STEP hashes match for all 1,350 candidates.

This extension changes query composition, so differences between old and expanded aggregate scores are not model improvements. Family means precede all benchmark averages. Inference retains eleven families, with a joint spring-family dependence sensitivity. The original eleven primary comparisons remain Holm corrected; the focused constrained comparisons below form a separate exploratory fourteen-test family. Sequential dataset expansion after previous results does not provide a new confirmatory test. The 0.02 practical nDCG threshold is descriptive, not an engineering-validated utility threshold.

## Direct performance: 58 constrained exemplar queries

{main_table}

![Expanded KG-constrained retrieval evaluation](figures/effectiveness_110.png)

KG-constrained Full achieves **{e['ndcg@5']:.4f} embedding / {d['ndcg@5']:.4f} descriptor nDCG@5**. The same-filter intent baseline achieves {s('embedding','intent_filter')['ndcg@5']:.4f} in either representation. Strong precision counts only grade-2 geometry suitability. Full descriptor minus same-filter intent is {d['ndcg@5']-s('descriptor','intent_filter')['ndcg@5']:+.4f}; embedding is {e['ndcg@5']-s('embedding','intent_filter')['ndcg@5']:+.4f}. The isolated function-prior contribution is Full minus geometry+classifier+filter: {e['ndcg@5']-s('embedding','geometry_class_filter')['ndcg@5']:+.4f} embedding and {d['ndcg@5']-s('descriptor','geometry_class_filter')['ndcg@5']:+.4f} descriptor. These controls share the numeric screening mechanism.

{compliance_table}

Failures and unknowns use the separate surface reference with the unchanged 0.2 mm guard; unknowns do not demonstrate satisfaction. There are 290 requested top-five slots per representation. The no-filter comparison measures screening benefit. It cannot establish a graph-specific benefit because the matched flat Full and KG Full have identical complete rankings, and the current class-function mappings can be expressed as a flat class prior. Embedding and descriptor without exemplar geometry also remain identical by construction: no representation-dependent signal is used in that ablation.

## Paired statistical analysis

{comp_table}

The focused analysis has {len(focused_rejections)} Holm-adjusted p-values below 0.05; the original eleven-comparison primary family on all 77 exemplar queries has {len(primary_rejections)}. Exact p-values, unadjusted exploratory intervals, joint-spring results, leave-one-family-out extrema and all negative family differences are saved. The larger variant count does not create 110 independent tasks. Inspect the same-filter controls and pool sensitivities before making any superiority claim.

In the focused analysis, both representations improve over Full without constraints (Holm p=0.0137 each), and descriptor Full improves over geometry with the same filter (Holm p=0.0234). None of the comparisons against geometry+classifier+filter, intent+filter, rank fusion+filter or geometry+intent+filter survives this adjustment. The original primary analysis also finds descriptor Full above geometry+filter (Holm p=0.0215). These are exploratory evidence for the combined retrieval and screening mechanism; they do not isolate a KG-topology benefit or establish superiority over the strongest matched controls.

## Continuity, broader inputs and candidate pools

Constrained exemplar queries only, original versus added variants:

{origin_table}

Input subsets (family-balanced means; query counts are per representation/method):

{subset_table}

Constrained exemplar candidate-pool sensitivity:

{cohort_table}

The apparent descriptor advantage is not stable across pools: on the newer 926-CAD pool, descriptor Full scores {s('descriptor','kg_full',cohort='new_only')['ndcg@5']:.4f}, below intent+filter at {s('descriptor','intent_filter',cohort='new_only')['ndcg@5']:.4f}; embedding Full is {s('embedding','kg_full',cohort='new_only')['ndcg@5']:.4f}. This weakens a generalization claim despite the higher closed-cohort descriptor mean. The additional queries expose this limitation and must not be selected or dropped according to it.

The full 1,816-part gallery is incompletely judged. `expanded_full_gallery_bounds.csv` retains conservative nDCG intervals and unjudged-return counts, rather than coding missing annotations as negatives. The new-only pool contains 926 additional audited-cohort CADs; the deduplicated pool excludes train-matched numerical descriptor signatures, not every possible semantic near-duplicate. Neither pool is a new independent task set.

`selectivity_summary.csv` stratifies constrained exemplar performance using fixed thirds of feasible-gallery proportion; these are descriptive and can have unequal family composition. `expanded_grading_sensitivity.csv` changes conditional grades to zero/two. `expanded_robustness_summary.csv` averages the existing five-seed tie-order and classifier-permutation conditions within family and retains guard sensitivities. No additional baselines or robustness conditions were introduced for this expansion.

## Depth, annotation sensitivity and robustness

Deeper retrieval on the same 58 constrained exemplar queries:

{deeper_table}

Conditional-grade sensitivity (nDCG@5, separate normalization for each grading scenario):

{grade_table}

Classifier perturbation, mean across the five existing seeds and eleven families:

{robust_table}

Geometry-containing Full is less sensitive than intent-only filtering to these synthetic category-probability permutations. Descriptor Full nevertheless declines as corruption increases. Permutations are stress tests, not calibrated estimates of real deployment classifier error. The guard sensitivity changes reference judgments near the measurement boundary; it does not change serving thresholds or establish dimensional metrology accuracy. Exact tie-order results, per-seed outcomes and all guard cases remain available. Robustness to these perturbations does not resolve the newer-pool performance loss.

## Cost, sufficiency and interpretation

The complete expanded evaluation took **{run['elapsed_seconds']:.1f} seconds**, including **{serving['total_serving_seconds']:.1f} seconds** checking real serving results. Deterministic replay took {replay['elapsed_seconds']:.1f} seconds. No model retraining or new CAD rendering was required. This is a modest cost for wider constraint coverage. Query count measures evaluation coverage, not serving scalability; the separate audited systems study varies gallery size and concurrency.

110 is sufficient for a more informative within-benchmark sensitivity evaluation at this stage. It is not sufficient to establish unseen-task generalization or a distinct KG-topology advantage. The expanded results must be described as performance of an intent-guided, exemplar-based, dimension-screened shortlisting method. Same-filter comparisons isolate the proposed combination, while exact KG/flat equality prevents a graph-specific ranking-superiority claim. Prioritize independent annotation review and genuinely new task/constraint families before adding hundreds of correlated variants. Preserve negative findings and avoid using these test outcomes to tune weights or select favorable queries.

## Reproduce

```powershell
python -m scripts.prepare_expanded_queries
python -m scripts.run_verified_evaluation --config configs/kg_queries110_v3.json
python -m scripts.run_verified_evaluation --config configs/kg_queries110_v3.json --output outputs/kg_queries110_v3/replay
python -m scripts.validate_verified_evaluation --config configs/kg_queries110_v3.json
python -m scripts.report_queries110
```
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    write_json(out/'expanded_analysis.json',dict(queries=110,families=11,original49=49,added61=61,constrained_exemplar=58,
        distinct_structured_requests=len(structured_signatures),distinct_family_relevance_vectors=len(qrel_signatures),zero_relevance_queries=noanswer,
        focused_holm_rejections=len(focused_rejections),primary_holm_rejections=len(primary_rejections),
        original49_continuity=continuity,source='ai_reference',human_expert_reviewed=False,
        input_sha256={str(CFG):sha256(CFG),cfg['annotation_path']:sha256(cfg['annotation_path']),
            str(out/'run_manifest.json'):sha256(out/'run_manifest.json'),str(out/'reproducibility.json'):sha256(out/'reproducibility.json')},
        script_sha256=sha256(__file__),report_sha256=sha256(out/'REPORT.md')))
    print(main_table);print(comp_table.replace('Delta','Difference'));print(origin_table)


if __name__=='__main__':main()
