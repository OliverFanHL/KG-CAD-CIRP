"""Report the named proposed method, matched controls and limits of evidence."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from cadkg.io import read_csv,read_json,write_csv,write_json,sha256
from cadkg.experiments.verified_statistics import _bootstrap,_flip_p,_holm
from scripts.report_kg_constrained import COMPARATORS,mean,table,NAMES

ROOT=Path('outputs/feature_review_v5')


def main():
    cfg=read_json('configs/feature_review_v5.json');out=Path(cfg['output'])
    if read_json(ROOT/'quality_gate.json')['null_pairs']:
        from scripts.report_partial_feature_review import main as partial_main
        return partial_main()
    validation=read_json(ROOT/'validation.json');assert validation['status']=='VERIFIED'
    audit=read_json(ROOT/'quality_gate.json');assert sha256(cfg['annotation_path'])==audit['annotation_sha256']
    raw=read_csv(out/'per_query.csv');old=read_csv('outputs/candidate_scale_v4/evaluation/per_query.csv')
    methods=read_json(out/'integrity.json')['methods'];cohorts=read_json(out/'cohorts.json')
    families=sorted({r['family_id'] for r in raw})
    def select(kind,method,cohort='closed',subset='constrained_exemplar',grading='nominal',source=raw):
        def include(r):
            ex=r['has_exemplar']=='True';co=r['has_constraint']=='True'
            return dict(all=True,exemplar=ex,text=not ex,constrained_exemplar=ex and co,
                        unconstrained_exemplar=ex and not co,constrained_text=not ex and co)[subset]
        return [r for r in source if (r['geometry_kind'],r['method'],r['cohort'],r['grading_scenario'])==
                (kind,method,cohort,grading) and include(r)]
    metric_names=['ndcg@5','precision@5','strong_precision@5','ndcg@10','ndcg@20','recall@5','recall@20','ap','ap@20']
    summary=[];effects=[]
    for cohort in cohorts:
        for kind in ['embedding','descriptor']:
            for subset in ['all','exemplar','text','constrained_exemplar','unconstrained_exemplar','constrained_text']:
                for method in methods:
                    rows=select(kind,method,cohort,subset)
                    record=dict(cohort=cohort,cads=len(cohorts[cohort]),geometry_kind=kind,subset=subset,
                        method=method,is_proposed_method=method=='kg_full',queries=len(rows),
                        families=len({r['family_id'] for r in rows}),
                        zero_relevance_queries=sum(int(r['relevant_total'])==0 for r in rows),
                        dimension_failures=sum(int(r['dimension_failures']) for r in rows),
                        dimension_unknowns=sum(int(r['dimension_unknowns']) for r in rows),
                        **{m:mean(rows,m) for m in metric_names})
                    summary.append(record)
                    before=select(kind,method,cohort,subset,source=old)
                    effects.append(dict(geometry_kind=kind,method=method,cohort=cohort,subset=subset,
                        **{f'v4_{m}':mean(before,m) for m in metric_names},
                        **{f'v5_{m}':record[m] for m in metric_names},
                        **{f'delta_{m}':record[m]-mean(before,m) for m in metric_names}))
    write_csv(ROOT/'candidate_summary.csv',summary);write_csv(ROOT/'annotation_revision_effect.csv',effects)
    def s(kind,method='kg_full',cohort='closed',subset='constrained_exemplar'):
        return next(r for r in summary if (r['geometry_kind'],r['method'],r['cohort'],r['subset'])==(kind,method,cohort,subset))
    rng=np.random.default_rng(cfg['seed']);comparisons=[];deltas=[]
    springs=[i for i,f in enumerate(families) if f in {'compression_return','tension_return'}]
    groups=[[i] for i in range(len(families)) if i not in springs]+[springs]
    for kind in ['embedding','descriptor']:
        full=select(kind,'kg_full');assert len(full)==58
        for method in COMPARATORS:
            base={r['query_id']:r for r in select(kind,method)};values=[]
            for fid in families:
                rows=[r for r in full if r['family_id']==fid]
                delta=float(np.mean([float(r['ndcg@5'])-float(base[r['query_id']]['ndcg@5']) for r in rows]))
                deltas.append(dict(geometry_kind=kind,comparator=method,family_id=fid,delta=delta));values.append(delta)
            a=np.asarray(values);ci=_bootstrap(a,rng,20000);sci=_bootstrap(a,rng,20000,groups)
            comparisons.append(dict(geometry_kind=kind,proposed_method='kg_full',comparator=method,queries=58,families=11,
                delta=float(a.mean()),ci95_lower=ci[0],ci95_upper=ci[1],exact_p=_flip_p(a),spring_joint_p=_flip_p(a,groups),
                spring_ci95_lower=sci[0],spring_ci95_upper=sci[1],family_wins=int(sum(a>1e-12)),
                family_ties=int(sum(abs(a)<=1e-12)),family_losses=int(sum(a< -1e-12)),
                leave_one_family_out_min=float(min((a.sum()-v)/10 for v in a)),
                leave_one_family_out_max=float(max((a.sum()-v)/10 for v in a))))
    for r,p in zip(comparisons,_holm([r['exact_p'] for r in comparisons])):r.update(holm_p=p,holm_family_size=len(comparisons))
    write_csv(ROOT/'focused_comparisons.csv',comparisons);write_csv(ROOT/'focused_family_differences.csv',deltas)
    grading=[]
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','intent_filter','geometry_class_filter','geometry_intent_filter','kg_no_class']:
            for scenario in ['conditional_zero','nominal','conditional_two']:
                grading.append(dict(geometry_kind=kind,method=method,grading=scenario,
                    ndcg5=mean(select(kind,method,grading=scenario),'ndcg@5')))
    write_csv(ROOT/'grading_sensitivity.csv',grading)
    robust=[];rb=read_csv(out/'robustness_per_query.csv')
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','intent_filter','geometry_class_filter','kg_no_class','rrf_filter','geometry_intent_filter','kg_no_geometry']:
            for condition in sorted({(r['robustness'],r['severity']) for r in rb}):
                rows=[r for r in rb if (r['geometry_kind'],r['method'],r['has_exemplar'],r['has_constraint'],r['robustness'],r['severity'])==
                      (kind,method,'True','True',*condition)]
                if rows:robust.append(dict(geometry_kind=kind,method=method,robustness=condition[0],severity=condition[1],
                    ndcg5=mean(rows,'ndcg@5'),matched_nominal=s(kind,method)['ndcg@5']))
    write_csv(ROOT/'robustness_summary.csv',robust)
    records=read_json(cfg['annotation_path'])['judgments'];obs={(r['family_id'],r['part_id']):r for r in records}
    queries={r['query_id']:r for r in read_json(out/'effective_protocol.json')['queries']};top=[];errors=Counter()
    for r in read_csv(out/'rankings_top20.csv'):
        if r['cohort']!='closed' or int(r['rank'])>5:continue
        q=queries[r['query_id']];a=obs[q['scenario_id'],r['part_id']]
        top.append(dict(**r,is_proposed_method=r['method']=='kg_full',family_id=q['scenario_id'],raw_geometry_grade=a['grade'],
            observation=a['observation'],rationale=a['rationale'],geometry_evidence=a['geometry_evidence'],source_sha256=a['source_sha256']))
        if r['method']=='kg_full' and r['geometry_kind']=='descriptor' and q['reference_part_id'] and q['constraints']:
            errors[q['scenario_id'],a['grade']]+=1
    write_csv(ROOT/'top5_evidence.csv',top)
    write_csv(ROOT/'proposed_top5_geometry_grades.csv',[dict(family_id=f,raw_geometry_grade=g,slots=n) for (f,g),n in sorted(errors.items())])
    changes=read_csv(Path(cfg['annotation_path']).parent/'changes.csv');change_counts=Counter(r['family_id'] for r in changes)
    grade_counts=Counter(str(r['grade']) for r in records)
    f=lambda v:f'{float(v):.4f}'
    main_table=table(['Method','Role','Embedding nDCG@5','Descriptor nDCG@5','Descriptor strong P@5'],
        [[m,'**Proposed**' if m=='kg_full' else 'Matched flat control' if m=='flat_full' else 'Control / ablation',
          f(s('embedding',m)['ndcg@5']),f(s('descriptor',m)['ndcg@5']),f(s('descriptor',m)['strong_precision@5'])] for m in methods])
    cohort_table=table(['Candidate cohort','CADs','Proposed embedding','Proposed descriptor','Intent + same filter'],
        [[c,len(cohorts[c]),f(s('embedding',cohort=c)['ndcg@5']),f(s('descriptor',cohort=c)['ndcg@5']),
          f(s('descriptor','intent_filter',c)['ndcg@5'])] for c in cohorts])
    stats_table=table(['Representation','Full minus control','Delta','95% family interval','Holm p','Family W/T/L'],
        [[r['geometry_kind'],r['comparator'],f(r['delta']),f"[{f(r['ci95_lower'])}, {f(r['ci95_upper'])}]",f(r['holm_p']),
          f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in comparisons])
    revision_table=table(['Proposed method','v4 same1816','v5 same1816','Difference'],
        [[k,*[f(next(r[v] for r in effects if (r['geometry_kind'],r['method'],r['cohort'],r['subset'])==
                     (k,'kg_full','closed','constrained_exemplar'))) for v in ['v4_ndcg@5','v5_ndcg@5','delta_ndcg@5']]] for k in ['embedding','descriptor']])
    family_table=table(['Task family','Changed grades','Full embedding','Full descriptor','Intent + same filter'],
        [[fid,change_counts[fid],*[f(mean([r for r in select(k,m) if r['family_id']==fid],'ndcg@5'))
           for k,m in [('embedding','kg_full'),('descriptor','kg_full'),('descriptor','intent_filter')]]] for fid in families])
    grading_table=table(['Representation','Method','Conditional=0','Nominal','Conditional=2'],
        [[k,m,*[f(next(r['ndcg5'] for r in grading if (r['geometry_kind'],r['method'],r['grading'])==(k,m,g)))
                for g in ['conditional_zero','nominal','conditional_two']]] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']])
    depth_table=table(['Representation','Method','P@5','Strong P@5','Recall@20','AP'],
        [[k,m,*[f(s(k,m)[v]) for v in ['precision@5','strong_precision@5','recall@20','ap']]]
         for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter','geometry_intent_filter']])
    rob_table=table(['Representation','Full nominal','10% classifier permutation','30% classifier permutation'],
        [[k,f(s(k)['ndcg@5']),*[f(next(r['ndcg5'] for r in robust if (r['geometry_kind'],r['method'],r['robustness'],float(r['severity']))==
                  (k,'kg_full','classifier_permutation',fraction))) for fraction in [.1,.3]]] for k in ['embedding','descriptor']])
    sample=read_json(ROOT/'second_review_comparison.json')
    second_text=(f"The targeted 80 second pass covered 880 role pairs: exact grade agreement {sample['exact_grade_agreement']:.3%}; "
        f"agreement among pairs where either review assigned 1/2/null {sample['nontrivial_pair_agreement']:.3%}. "
        f"There were {sample['disagreements']} disagreements on {sample['disagreement_cads']} CADs. "
        'This sample is enriched for difficult cases and AI judgments share a model family; neither overall agreement nor kappa estimates human annotation accuracy.')
    ndup=read_json(out/'near_duplicate_audit.json');primary=read_csv(out/'primary_comparisons.csv')
    sig=[r['comparison'] for r in primary if float(r['holm_p'])<.05]
    focused_sig=[r['geometry_kind']+': Full - '+r['comparator'] for r in comparisons if r['holm_p']<.05]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted=['kg_full','geometry_class_filter','geometry_intent_filter','intent_filter','rrf_filter','kg_no_class','kg_no_constraints']
    labels=['Proposed Full','Geometry + class + filter','Geometry + intent + filter','Intent + filter','Rank fusion + filter','Geometry + filter','Full without filter']
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
    y=np.arange(len(plotted))
    for ax,metric,title in zip(axes,['ndcg@5','strong_precision@5'],['Ranking utility (nDCG@5)','Confirmed geometric matches (strong P@5)']):
        for offset,kind,color in [(-.18,'embedding','#8998a6'),(.18,'descriptor','#146c94')]:
            values=[s(kind,m)[metric] for m in plotted]
            ax.barh(y+offset,values,height=.32,label=kind.capitalize(),color=color)
        ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2));ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].get_yticklabels()[0].set_fontweight('bold')
    axes[1].legend(loc='lower right',frameon=False)
    fig.suptitle('Same 1,816 candidates · 58 constrained exemplar queries · 11 equally weighted task families',fontsize=12)
    fig.text(.5,.015,'AI-reviewed shape judgments; conditional matches are excluded from strong precision. No independent human certification.',ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(ROOT/'proposed_method_comparison.png',dpi=160);fig.savefig(ROOT/'proposed_method_comparison.pdf');plt.close(fig)
    report=f'''# Full source-feature audit and fixed-framework evaluation, v5

## Provenance and scope

This release completes source-view AI review of **1,816 held-out CADs × 11 roles = 19,976 judgments**, with a validation-only 44-CAD development pilot and an 80-CAD blind second review. It is an engineering-shape assessment by AI, **not independent human validation**. Historical test results were known to the coordinator. Separate first-pass contexts saw neutral source views and a frozen rubric, with no old grades, ranks, method scores or source categories. No retrieval model, prior, weight, query or experimental condition was tuned.

The same 110 queries, 11 task families, 14 methods, 2 representations and 1,816 candidate gallery are rerun. This is the full eligible test gallery (versus earlier 1,350 and original 424); expanding to training/validation candidates would violate the existing held-out design. Seventy-seven exemplar queries form the existing primary comparison subset; 58 are dimension-constrained and support the existing focused comparison family. The 33 text queries remain included. More candidates and query variants do not create more independent task families.

## Revision completed before scoring

The [frozen rubric](FROZEN_RUBRIC.md) was developed on 44 validation CADs, with no exact STEP overlap with test. Its case notes explicitly resolve carrier seat versus bore, short pin/coupling versus drive shaft, shaft-retaining hub versus open flange, gear/belt/sprocket tooth boundaries, and compression/tension/torsion coil distinctions. Grade1 requires plausible form and a named limitation; missing decisive evidence permits null/cannot-assess. Visible smooth simplified fasteners remain conditional. Mesh tessellation is never thread evidence.

All 1,860 pilot/test CADs have six new source-bound enlarged views, orthographic surfaces with edges hidden and two uncapped visualization cutaways. Individual images were opened where fine features mattered. Cutaways are aids to interpretation, not independent measurements. Every positive/conditional role has named per-feature states and view locators. Incompatible roles use actual source-form exclusions. Generic historical observations are replaced across all roles while the previous release remains immutable.

{second_text}

Explicit source adjudication produced **{audit['changes']} changed task–CAD grades on {audit['changed_cads']} CADs**. Counts after adjudication: {grade_counts.get('2',0)} shortlist, {grade_counts.get('1',0)} conditional, {grade_counts.get('0',0)} reject, {audit['null_pairs']} cannot-assess. All old decisions and new evidence/dispositions remain in [changes](../../annotations/intent_feature_review_v5/changes.csv), [adjudication](adjudication.json), frozen first/second pass notes and prior annotation hashes. Some changes reflect stronger evidence, others operational boundary refinement. **The score difference is not a pure error-correction effect**, and this revised benchmark is not an untouched confirmatory test.

## Which results are the proposed method?

**`kg_full` is the proposed complete method.** Embedding and descriptor are its two pre-existing geometry representations. Every other row is a control/ablation. `flat_full` is the matched flat implementation, and `geometry_intent_filter` must not be relabeled as the proposed method if it performs better. The following nDCG@5 scores average queries within each of 11 families, then families equally, on the 58 constrained exemplar queries. nDCG is ranking utility against these graded judgments, not classification accuracy or percent engineering compatibility.

{main_table}

{depth_table}

![Proposed Full and matched controls](proposed_method_comparison.png)

## Does the evidence establish that the method works?

Interpret four separate claims: useful benchmark shortlisting, incremental full-method advantage, graph contribution, and wider generalization. Strong P@5 measures grade 2 plus the existing independent size check; a conditional candidate does not count as a confirmed match. Retrieved parts still lack independently validated loads, fits, materials and production suitability.

The existing focused 14-comparison family yields the following paired task-family effects. Intervals are exploratory95% family bootstrap intervals; Holm adjustment uses all 14 comparisons, not a selected favorable pair. The separate primary 11-comparison family uses all 77 exemplar queries and is preserved in `evaluation/primary_comparisons.csv`.

{stats_table}

Primary Holm-significant comparisons: {', '.join(sig) or 'none'}. Focused Holm-significant comparisons: {', '.join(focused_sig) or 'none'}. Positive effect sizes without reliable separation from a strong matched control do not establish full-method superiority. Exactly equal `kg_full`/`flat_full` rankings provide **no retrieval-quality evidence for a graph-representation advantage**. Statistics are conditional on eleven authored AI-graded families, not randomized evidence or independent population confirmation.

## Annotation contribution and methodological limits

All **{validation['all_historical_ranking_arrays_unchanged']:,} complete v4/v5 order and score arrays are identical**. Thus the following changes isolate revised grading/evidence at identical candidate counts and retrieval behavior; they do not represent a better-trained method.

{revision_table}

{family_table}

Annotations can change the measured score and reveal false positives, but they cannot repair retrieval order. Remaining limits include coarse class-to-function priors, shape similarity that does not directly encode per-feature requirements, equal fusion weights, and dimension filtering restricted to a measured envelope bound. Text-only Full lacks exemplar geometry and reduces to the existing class/function prior; two spring roles share a coarse class prior despite different attachment geometry. No engineering success threshold for an nDCG value such as 0.67 was validated; assess its utility against matched controls and strong precision, not the number alone. See [top-five source evidence](top5_evidence.csv) for concrete failure inspection rather than inferring error causes from an aggregate score.

The 19 fixed training exemplars were also checked in new source views, before scoring, without replacing any exemplar. Sixteen have grade 2 evidence for their requested role; three are conditional: E004 is a smooth socket-headed fastener with no modeled thread, E007 is a smooth-bore hex nut, and E011 is a flat bolted disk without a register. These are legitimate conditional examples under the rubric, but shape similarity to them need not rank every required feature highly. This is a plausible exemplar limitation, not a measured causal attribution or permission to tune on test outcomes. [Source-specific exemplar checks](exemplar_role_checks.csv) and their frozen notes retain the evidence.

## Robustness and candidate composition

{grading_table}

{rob_table}

The classifier-permutation condition is the existing synthetic sensitivity, not a realistic measured deployment error rate. Tie-order and dimension-guard summaries remain in [robustness_summary.csv](robustness_summary.csv). Conditional-grade sensitivity reveals dependence on simplified geometry; it does not justify choosing a favorable grade convention after seeing scores.

{cohort_table}

These are candidate-cohort diagnostics using the same queries, not additional configurations or independent data domains. Zero-relevance queries stay in each denominator. Descriptor signature deduplication retains {ndup['remaining']} CADs, with {len(ndup['discordant_groups'])} groups now having inconsistent role grades; such groups are reported rather than resolved by copying labels. Descriptor signatures do not prove exact geometric or design identity.

## Reproduction and safeguards

Raw STEP/image hashes and split checks passed for all 1,816 test and 44 pilot CADs: no exact train–test or pilot–test overlap. Exact-byte contaminated candidates remain excluded under the existing protocol. The source hash table, review/rubric freezes, full annotation history, unchanged-ranking check, independent metric replay, independent task/size transform check and manifest/code hashes are retained. Labels were frozen before new scoring. This prevents new score-directed annotation selection; it cannot erase prior test exposure or create human independence.

Validation: **{validation['status']}**; {validation['independent_metric_cases']:,} separately computed metric-cutoff cases, {validation['independently_constructed_task_constraint_cases']:,} independently constructed task/constraint grade cases, and 1,816 raw STEP hashes. The replay reproduces deterministic core tables and all ranking arrays. See [validation.json](validation.json), [quality_gate.json](quality_gate.json), [fixed plan](../../docs/FEATURE_REVIEW_EVALUATION.md) and its [count clarification](PLAN_CLARIFICATION.md).

Reproduce from the repository root with the existing Python environment:

```powershell
& $cadkgPython -m scripts.run_verified_evaluation --config configs/feature_review_v5.json
& $cadkgPython -m scripts.run_verified_evaluation --config configs/feature_review_v5.json --output outputs/feature_review_v5/evaluation/replay
& $cadkgPython -m scripts.validate_verified_evaluation --config configs/feature_review_v5.json
& $cadkgPython -m scripts.validate_feature_review evaluation
& $cadkgPython -m scripts.report_feature_review
```

Further work should obtain genuine independent human adjudication using the frozen evidence packet, evaluate unseen task families/source distributions, and model task-defining interfaces. Such future work must keep rubric development and model selection outside its held-out test. The present run supports only the claims justified by the matched comparisons above.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'analysis_manifest.json',dict(annotation_sha256=sha256(cfg['annotation_path']),
        proposed_method='kg_full',primary_queries=77,focused_queries=58,primary_tests=11,focused_tests=14,
        matrix_unchanged=True,source='AI reference',human_expert_reviewed=False,
        report_sha256=sha256(ROOT/'REPORT.md'),summary_sha256=sha256(ROOT/'candidate_summary.csv')))
    print(dict(report=str(ROOT/'REPORT.md'),proposed_embedding=s('embedding')['ndcg@5'],proposed_descriptor=s('descriptor')['ndcg@5']))


if __name__=='__main__':main()
