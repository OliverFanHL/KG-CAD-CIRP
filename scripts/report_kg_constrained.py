"""Direct, paired effectiveness analysis of the KG-constrained method on audited labels."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from cadkg.io import read_csv,read_json,write_csv,write_json,sha256
from cadkg.experiments.verified_statistics import _flip_p,_bootstrap,_holm

OUT=Path('outputs/kg_constrained_audited_v2')
COMPARATORS=['kg_no_constraints','kg_no_class','geometry_class_filter','intent_filter','rrf_filter','geometry_intent_filter','kg_no_geometry']
NAMES={'kg_full':'KG-constrained Full','kg_no_constraints':'Full without constraints','kg_no_class':'Geometry + same constraints',
       'geometry_class_filter':'Geometry + classifier + same constraints','intent_filter':'Intent/class prior + same constraints',
       'rrf_filter':'Rank fusion + same constraints','geometry_intent_filter':'Geometry + intent + same constraints',
       'kg_no_geometry':'Full without exemplar geometry','flat_full':'Matched flat Full'}
METRICS=['ndcg@5','precision@5','strong_precision@5','recall@20','ap','ndcg@20']

def family_values(rows,metric):
    grouped=defaultdict(list)
    for r in rows:grouped[r['family_id']].append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in grouped.items()}

def mean(rows,metric):return float(np.mean(list(family_values(rows,metric).values())))

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    config=read_json('configs/kg_constrained_audited_v2.json');raw=read_csv(OUT/'per_query.csv')
    annotation=read_json(config['annotation_path']);audit=read_json(config['annotation_audit_path'])
    assert sha256(config['annotation_path'])==audit['final_annotation_sha256']
    def select(kind,method,subset='constrained',cohort='closed',grading='nominal'):
        return [r for r in raw if r['cohort']==cohort and r['grading_scenario']==grading and r['geometry_kind']==kind and r['method']==method
          and (r['has_exemplar']=='True' if subset!='text' else r['has_exemplar']=='False')
          and (subset not in {'constrained','unconstrained'} or (r['has_constraint']=='True')==(subset=='constrained'))]
    summaries=[]
    for cohort in ['closed','new_only','original_pool','deduplicated']:
        for kind in ['embedding','descriptor']:
            for subset in ['constrained','unconstrained','exemplar','text']:
                for method in NAMES:
                    rows=select(kind,method,subset,cohort)
                    result=dict(cohort=cohort,geometry_kind=kind,subset=subset,method=method,queries=len(rows),families=len({r['family_id'] for r in rows}))
                    result.update({metric:mean(rows,metric) for metric in METRICS})
                    result.update(returned_slots=sum(int(r['returned']) for r in rows),top5_slots=5*len(rows),
                        dimension_failures=sum(int(r['dimension_failures']) for r in rows),dimension_unknowns=sum(int(r['dimension_unknowns']) for r in rows))
                    summaries.append(result)
    write_csv(OUT/'focused_summary.csv',summaries)
    def s(kind,method,subset='constrained',cohort='closed'):
        return next(r for r in summaries if (r['cohort'],r['geometry_kind'],r['subset'],r['method'])==(cohort,kind,subset,method))
    rng=np.random.default_rng(config['seed']);comparisons=[];family_deltas=[]
    for kind in ['embedding','descriptor']:
        full=select(kind,'kg_full');lookup={r['query_id']:r for r in full};families=sorted({r['family_id'] for r in full})
        assert len(full)==19 and len(families)==11
        springs=[i for i,f in enumerate(families) if f in {'compression_return','tension_return'}]
        groups=[[i] for i in range(len(families)) if i not in springs]+[springs]
        for other in COMPARATORS:
            baseline={r['query_id']:r for r in select(kind,other)}
            assert lookup.keys()==baseline.keys()
            diff={metric:[] for metric in METRICS}
            for fid in families:
                a=[r for r in full if r['family_id']==fid]
                values={metric:float(np.mean([float(r[metric])-float(baseline[r['query_id']][metric]) for r in a])) for metric in METRICS}
                family_deltas.append(dict(geometry_kind=kind,comparison=f'kg_full - {other}',family_id=fid,**{f'delta_{k}':v for k,v in values.items()}))
                for metric in METRICS:diff[metric].append(values[metric])
            values=np.asarray(diff['ndcg@5']);ci=_bootstrap(values,rng,20000);spring_ci=_bootstrap(values,rng,20000,groups)
            comparisons.append(dict(geometry_kind=kind,comparator=other,subset='constrained',families=len(families),queries=len(full),
              **{f'delta_{metric}':float(np.mean(v)) for metric,v in diff.items()},ci95_lower=ci[0],ci95_upper=ci[1],
              exact_p=_flip_p(values),spring_joint_p=_flip_p(values,groups),spring_ci95_lower=spring_ci[0],spring_ci95_upper=spring_ci[1],
              family_wins=int(sum(values>1e-12)),family_ties=int(sum(abs(values)<=1e-12)),family_losses=int(sum(values< -1e-12)),
              leave_one_family_out_min=float(min((values.sum()-v)/10 for v in values)),leave_one_family_out_max=float(max((values.sum()-v)/10 for v in values))))
    for r,p in zip(comparisons,_holm([r['exact_p'] for r in comparisons])):r['holm_p']=p;r['holm_family_size']=len(comparisons)
    write_csv(OUT/'constrained_comparisons.csv',comparisons);write_csv(OUT/'constrained_family_differences.csv',family_deltas)
    # Capture concrete retrieval outcomes; every returned example has source evidence and an audited label.
    annotations={(r['family_id'],r['part_id']):r for r in annotation['judgments']}
    queries={q['query_id']:q for q in read_json(Path(config['source_run'])/'protocol.json')['queries']}
    detail=[]
    for r in read_csv(OUT/'rankings_top20.csv'):
        q=queries[r['query_id']]
        if r['cohort']!='closed' or not q['constraints'] or int(r['rank'])>5 or r['method'] not in ['kg_full','kg_no_constraints','geometry_class_filter','intent_filter']:continue
        a=annotations[q['scenario_id'],r['part_id']]
        detail.append(dict(**r,family_id=q['scenario_id'],geometry_grade=a['grade'],inspection_id=a['inspection_id'],
            observation=a['observation'],rationale=a['rationale'],evidence=a['geometry_evidence']))
    write_csv(OUT/'constrained_top5_evidence.csv',detail)
    sensitivity=[]
    for kind in ['embedding','descriptor']:
        for grading in ['conditional_zero','nominal','conditional_two']:
            for method in ['kg_full','geometry_class_filter','intent_filter','kg_no_class']:
                rows=select(kind,method,grading=grading)
                sensitivity.append(dict(geometry_kind=kind,grading=grading,method=method,ndcg5=mean(rows,'ndcg@5'),strong_precision5=mean(rows,'strong_precision@5')))
    write_csv(OUT/'focused_grading_sensitivity.csv',sensitivity)
    # Existing robustness is scored against audited labels; report matched constrained subset only.
    robustness=[];rob=read_csv(OUT/'robustness_per_query.csv')
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','geometry_class_filter','intent_filter','kg_no_class']:
            for condition in sorted({(r['robustness'],r['severity']) for r in rob}):
                rows=[r for r in rob if r['geometry_kind']==kind and r['method']==method and r['has_constraint']=='True' and r['has_exemplar']=='True' and (r['robustness'],r['severity'])==condition]
                if rows:robustness.append(dict(geometry_kind=kind,method=method,robustness=condition[0],severity=condition[1],ndcg5=mean(rows,'ndcg@5'),matched_nominal_ndcg5=s(kind,method)['ndcg@5']))
    write_csv(OUT/'focused_robustness.csv',robustness)
    v1=read_csv('outputs/verified_evaluation_v1/per_query.csv');label_impact=[]
    for kind in ['embedding','descriptor']:
        a=select(kind,'kg_full');b=[r for r in v1 if r['cohort']=='closed' and r['grading_scenario']=='nominal' and r['method']=='kg_full' and r['geometry_kind']==kind and r['has_exemplar']=='True' and r['has_constraint']=='True']
        label_impact.append(dict(geometry_kind=kind,constrained_v1=mean(b,'ndcg@5'),constrained_audited_v2=mean(a,'ndcg@5'),delta=mean(a,'ndcg@5')-mean(b,'ndcg@5')))
    write_csv(OUT/'annotation_revision_effect.csv',label_impact)
    f=lambda x:f'{float(x):.4f}'
    main_table=table(['Method','Embedding nDCG@5','Descriptor nDCG@5','Embedding strong P@5','Descriptor strong P@5'],[[NAMES[m],f(s('embedding',m)['ndcg@5']),f(s('descriptor',m)['ndcg@5']),f(s('embedding',m)['strong_precision@5']),f(s('descriptor',m)['strong_precision@5'])] for m in ['kg_full',*COMPARATORS]])
    screen_table=table(['Method','Embedding failures/unknowns','Descriptor failures/unknowns','Slots per representation'],[[NAMES[m],f"{s('embedding',m)['dimension_failures']}/{s('embedding',m)['dimension_unknowns']}",f"{s('descriptor',m)['dimension_failures']}/{s('descriptor',m)['dimension_unknowns']}",s('descriptor',m)['top5_slots']] for m in ['kg_full','kg_no_constraints','kg_no_class','geometry_class_filter','intent_filter']])
    stat_table=table(['Representation','Full minus comparator','Δ nDCG@5','95% interval','Exact p','Holm p','Family W/T/L'],[[r['geometry_kind'],r['comparator'],f(r['delta_ndcg@5']),f"[{f(r['ci95_lower'])}, {f(r['ci95_upper'])}]",f(r['exact_p']),f(r['holm_p']),f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in comparisons])
    cohorts_table=table(['Cohort','Embedding Full','Embedding intent+filter','Descriptor Full','Descriptor intent+filter'],[[c,*[f(s(k,m,cohort=c)['ndcg@5']) for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']]] for c in ['closed','new_only','original_pool','deduplicated']])
    input_table=table(['Input subset','Embedding Full','Descriptor Full','Intent + filter'],[[subset,f(s('embedding','kg_full',subset)['ndcg@5']),f(s('descriptor','kg_full',subset)['ndcg@5']),f(s('descriptor','intent_filter',subset)['ndcg@5'])] for subset in ['constrained','unconstrained','exemplar','text']])
    deeper=table(['Representation','Method','Recall@20','Full AP'],[[k,NAMES[m],f(s(k,m)['recall@20']),f(s(k,m)['ap'])] for k in ['embedding','descriptor'] for m in ['kg_full','geometry_class_filter','intent_filter']])
    # Explicitly attribute gains to screening versus prior beyond matched filter.
    drivers={k:dict(screening_gain=s(k,'kg_full')['ndcg@5']-s(k,'kg_no_constraints')['ndcg@5'],
        function_prior_gain=s(k,'kg_full')['ndcg@5']-s(k,'geometry_class_filter')['ndcg@5'],
        geometry_gain=s(k,'kg_full')['ndcg@5']-s(k,'kg_no_geometry')['ndcg@5'],
        class_reference_gain=s(k,'kg_full')['ndcg@5']-s(k,'geometry_intent_filter')['ndcg@5']) for k in ['embedding','descriptor']}
    alltests=read_csv(OUT/'primary_comparisons.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    methods=['kg_full','kg_no_constraints','kg_no_class','geometry_class_filter','intent_filter','rrf_filter','geometry_intent_filter']
    labels=['KG-constrained Full','Full without constraints','Geometry + filter','Geometry + classifier + filter','Intent prior + filter','Rank fusion + filter','Geometry + intent + filter']
    fig,ax=plt.subplots(figsize=(9,4.7));y=np.arange(len(methods))
    for off,kind,color in [(-.18,'embedding','#446b91'),(.18,'descriptor','#d28441')]:ax.barh(y+off,[s(kind,m)['ndcg@5'] for m in methods],height=.36,color=color,label=kind.capitalize())
    ax.set(yticks=y,yticklabels=labels,xlim=(0,1),xlabel='Family-balanced nDCG@5; 19 dimension-constrained queries');ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'kg_constrained_effectiveness.{ext}',dpi=220)
    plt.close(fig)
    report=f'''# Direct performance evaluation of the KG-constrained method

## Material Passport

- Version: kg-constrained-audited-v2, exploratory effectiveness study.
- Labels: `annotations/intent_batch2_audited_v2/judgments.json`; SHA256 `{sha256(config['annotation_path'])}`.
- Data: 1,350 held-out CADs, 14,850 task–part grades, eleven existing authored families, 49 frozen variants. Nineteen explicitly dimension-constrained exemplar queries are the focus.
- Models, priors, weights, queries and methods remain frozen. The random audit was completed before this re-evaluation. Labels remain AI-authored; unsampled annotations are not independently validated.
- Plan: [KG-constrained evaluation design](../../docs/KG_CONSTRAINED_EVALUATION.md). The earlier scalability study is a separate systems result, not the method's retrieval evaluation.

## 1. What is actually being evaluated?

Full combines exemplar geometry, classifier-reference agreement and class–function priors, then filters by verified numeric measurements. In these experiments the requested constraints are **largest stored-axis extent upper bounds**. They are not automatic thread/hook/gear-interface constraints, verified assembly compatibility, load limits or material constraints. Functional suitability comes from separate visual qrels. Every method shares the same candidates and corrected labels.

Three questions are distinguished: (i) does enforcing an explicit dimension bound improve constrained retrieval? (ii) does the KG function prior improve ranking beyond the same dimension filter and classifier/geometry access? (iii) does the complete proposed method outperform practical matched-input alternatives? This prevents attributing the entire screening benefit to graph structure.

## 2. Main results on the nineteen constrained queries

{main_table}

![KG-constrained effectiveness](figures/kg_constrained_effectiveness.png)

Full descriptor nDCG@5 is **{s('descriptor','kg_full')['ndcg@5']:.4f}**, versus {s('descriptor','kg_no_constraints')['ndcg@5']:.4f} without the constraint filter and {s('descriptor','intent_filter')['ndcg@5']:.4f} for intent with the same filter. Full embedding is **{s('embedding','kg_full')['ndcg@5']:.4f}**, versus {s('embedding','kg_no_constraints')['ndcg@5']:.4f} and {s('embedding','intent_filter')['ndcg@5']:.4f}, respectively.

The strong P@5 column counts only grade-2 suitability; grade-1 candidates do not count as strong. nDCG uses exponential gains 0/1/3. Metrics average within each family before across eleven families, avoiding treating query variants or CAD records as independent tasks.

## 3. Constraint compliance and gain attribution

{screen_table}

Counts are surface-reference fail/unknown among 95 top5 slots per representation. Unknown near the provisional 0.2 mm boundary is not a verified pass. The source surface reference and serving measurements use the same CAD kernel; compliance establishes consistency with this reference, not independent metrology or manufacturable fit.

- **Screening contribution:** Full minus Full-without-constraints is {drivers['embedding']['screening_gain']:.4f} embedding / {drivers['descriptor']['screening_gain']:.4f} descriptor nDCG@5.
- **Function-prior contribution beyond geometry+classifier+the same filter:** {drivers['embedding']['function_prior_gain']:.4f} / {drivers['descriptor']['function_prior_gain']:.4f}.
- **Exemplar-geometry contribution beyond classifier+prior+the same filter:** {drivers['embedding']['geometry_gain']:.4f} / {drivers['descriptor']['geometry_gain']:.4f}.
- **Classifier-reference contribution beyond geometry+prior+the same filter:** {drivers['embedding']['class_reference_gain']:.4f} / {drivers['descriptor']['class_reference_gain']:.4f}.

Numeric constraints also define the query's relevance: a source-reference failure receives zero qrel gain. The strong screening improvement is therefore an expected useful mechanism, but cannot by itself establish novel KG-specific reasoning. The matched flat method returns identical complete rankings; current priors are one unit-weight class mapping per function. Graph storage alone does not supply incremental predictive information.

## 4. Paired statistical assessment

{stat_table}

Fourteen focused comparisons use exact two-sided sign flips on eleven family means, with Holm adjustment across the fourteen tests. Intervals are unadjusted 95% percentile family bootstraps (20,000 draws). Family dependence, authored task selection and AI judgments make all inference exploratory. The spring pair's joint-flip sensitivity and leave-one-family-out ranges are saved. Nonsignificance is not equivalence.

Number of focused Holm-adjusted tests below 0.05: **{sum(r['holm_p']<.05 for r in comparisons)} / 14**. The original all-exemplar eleven-test family is retained separately in `primary_comparisons.csv`; it has {sum(float(r['holm_p'])<.05 for r in alltests)} adjusted rejections. Do not combine selected raw p-values from the two families into a confirmatory claim.

## 5. Performance outside the original retrieval-selected pool

{cohorts_table}

These are constrained-query scores. Original-pool and new-only galleries differ in size/composition, so comparisons across cohorts are descriptive. Within each cohort every method sees the identical gallery. `constrained_family_differences.csv` retains all losses and task-concentrated gains. `focused_grading_sensitivity.csv` reports strict and permissive interpretations of conditional grades; `focused_robustness.csv` reports the existing synthetic classifier, exact-tie and guard sensitivity on the matched constrained subset.

On the newly added 926 candidates, both Full variants are below the same-filter intent baseline: descriptor {s('descriptor','kg_full',cohort='new_only')['ndcg@5']:.4f} and embedding {s('embedding','kg_full',cohort='new_only')['ndcg@5']:.4f}, versus {s('descriptor','intent_filter',cohort='new_only')['ndcg@5']:.4f}. The aggregate descriptor gain therefore does not establish robust superiority outside the original retrieval-selected pool. The simpler geometry+intent+filter control is also slightly above descriptor Full on the complete gallery, so including classifier-reference agreement is not consistently beneficial.

## 6. Unconstrained, text-only and deeper retrieval results

{input_table}

Text-only Full reduces to the admitted category/function prior; geometry methods without an exemplar abstain. Their zero utility is not evidence of semantic superiority under matched inputs. All-exemplar scores include the 19 constrained and 19 unconstrained examples; this subset separation prevents broad aggregates from obscuring constraint-specific performance.

{deeper}

Recall and full AP use complete qrel denominators in the annotated 1,350-CAD gallery. Full-gallery 1,816-CAD retrieval is separately checked against legacy rankings with explicit bounds for unjudged returns; full-gallery recall remains unknown. `constrained_top5_evidence.csv` links every focused Full/baseline return to its actual visual assessment for error analysis.

## 7. Annotation revision and reproducibility

The v2 random audit changed 36 assessments. Relative to the previous v1 constrained Full nDCG, the annotation-only changes are {label_impact[0]['delta']:.4f} embedding and {label_impact[1]['delta']:.4f} descriptor. Models and rankings were not retuned to improve the revised grades. Larger query counts do not compensate for remaining annotation uncertainty or the eleven-family scope.

Complete rankings, per-query/family metrics, all seeds, input/code hashes, negative deltas, sensitivity rows and old-gallery equality checks are saved. The repeat run and independent metric/source checks are in `reproducibility.json`. They verify computation, not independent relevance truth.

Reproduction:

```powershell
python -m scripts.run_verified_evaluation --config configs/kg_constrained_audited_v2.json
python -m scripts.run_verified_evaluation --config configs/kg_constrained_audited_v2.json --output outputs/kg_constrained_audited_v2/replay
python -m scripts.validate_verified_evaluation --config configs/kg_constrained_audited_v2.json
python -m scripts.report_kg_constrained
```

## 8. What the paper can claim

The proposed method is directly evaluated here as a retrieval system. Its dimension filter reduces returned dimension violations and substantially improves dimension-constrained ranking compared with the same scoring method without filtering. The descriptor combination has stronger observed constrained retrieval than the embedding combination. Gains over strong same-filter controls, and specifically the incremental KG prior contribution, must be judged using the paired comparisons and task losses above; screening success alone does not establish a graph-specific effectiveness gain.

The one focused adjusted result is descriptor Full versus the same method without constraints (Holm p=0.0273). Full descriptor versus the same-filter intent control has Holm p=1.0, as does the isolated function-prior comparison versus geometry+classifier+filter. Thus, there is evidence for the constrained retrieval mechanism on this benchmark, while the current evidence is insufficient to claim an independently demonstrated KG-specific advantage. This distinction should be explicit in the paper's contribution statement.

The evidence supports preliminary geometry-aware, intent-guided, dimension-screened CAD shortlisting within this authored AI-graded benchmark. It does not establish verified engineering constraint reasoning, graph-topology superiority, universal improvement, independent human ground truth or unseen-task/industrial generalization. New effectiveness tasks should wait for independent review and clearly specified task/constraint semantics; this report makes the existing method's actual performance explicit without manufacturing a larger validated task count.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    write_json(OUT/'focused_analysis.json',dict(config_sha256=sha256('configs/kg_constrained_audited_v2.json'),report_sha256=sha256(OUT/'REPORT.md'),
        annotation_sha256=sha256(config['annotation_path']),script_sha256=sha256(__file__),gain_attribution=drivers,
        constrained_holm_rejections=sum(r['holm_p']<.05 for r in comparisons),same_filter_controls=True,new_task_families=False))
    print(main_table);print(screen_table);print(stat_table.replace('Δ','Delta'))

if __name__=='__main__':main()
