"""Report fixed-method results with unresolved reference labels left explicit."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256
from scripts.report_kg_constrained import table,mean
from scripts.partial_feature_review import transformed_set,METRICS
from cadkg.experiments.reference import numeric_reference

ROOT=Path('outputs/feature_review_v5')

def main():
    cfg=read_json('configs/feature_review_v5.json');audit=read_json(ROOT/'quality_gate.json')
    validation=read_json(ROOT/'partial_validation.json');assert validation['status']=='VERIFIED'
    out=ROOT/'partial_evaluation';run=read_json(out/'run_manifest.json')
    assert sha256(out/'run_manifest.json')==validation['partial_run_manifest_sha256']
    for p,h in run['input_sha256'].items():assert sha256(p)==h,p
    for name,h in run['output_sha256'].items():assert sha256(out/name)==h,name
    records=read_json(cfg['annotation_path'])['judgments'];assert sha256(cfg['annotation_path'])==audit['annotation_sha256']
    summary=read_csv(out/'summary_bounds.csv');families=read_csv(out/'family_bounds.csv');comparisons=read_csv(out/'comparison_bounds.csv')
    methods=run['methods'];cohorts=read_json(out/'cohorts.json');robust=read_csv(out/'robustness_summary_bounds.csv')
    def s(kind,m='kg_full',cohort='closed',subset='constrained_exemplar',grading='nominal'):
        return next(r for r in summary if (r['geometry_kind'],r['method'],r['cohort'],r['subset'],r['grading_scenario'])==(kind,m,cohort,subset,grading))
    def bounds(r,metric='ndcg@5'):
        a,b=float(r[metric+'_lower']),float(r[metric+'_upper'])
        return f'{a:.4f}' if abs(a-b)<1e-12 else f'[{a:.4f}, {b:.4f}]'
    def pair(kind,m,subset='constrained_exemplar'):
        return next(r for r in read_csv(out/'full_minus_control_bounds.csv') if (r['cohort'],r['geometry_kind'],r['comparator'],r['grading_scenario'],r['subset'])==('closed',kind,m,'nominal',subset))
    changes=read_csv(Path(cfg['annotation_path']).parent/'changes.csv');counts=Counter(str(r['grade']) for r in records)
    source={r['review_id']:r for r in read_csv(ROOT/'render_manifest.csv')}
    nulls=[dict(review_id=r['inspection_id'],family_id=r['family_id'],reason=r['observation'],geometry_evidence=r['geometry_evidence']) for r in records if r['grade'] is None]
    write_csv(ROOT/'cannot_assess.csv',nulls)
    candidate=[]
    for r in summary:candidate.append(dict(**r,is_proposed_method=r['method']=='kg_full',candidate_cads=len(cohorts[r['cohort']])))
    write_csv(ROOT/'candidate_summary.csv',candidate)
    old=read_csv('outputs/candidate_scale_v4/evaluation/per_query.csv');effects=[]
    def chosen(r,kind,m,subset):
        ex=r['has_exemplar']=='True';co=r['has_constraint']=='True'
        keep=dict(all=True,exemplar=ex,text=not ex,constrained_exemplar=ex and co,unconstrained_exemplar=ex and not co,constrained_text=not ex and co)[subset]
        return keep and (r['geometry_kind'],r['method'],r['cohort'],r['grading_scenario'])==(kind,m,'closed','nominal')
    for kind in ['embedding','descriptor']:
        for m in methods:
            for subset in ['all','exemplar','text','constrained_exemplar']:
                rows=[r for r in old if chosen(r,kind,m,subset)];a=s(kind,m,subset=subset)
                effects.append(dict(geometry_kind=kind,method=m,subset=subset,v4_ndcg5=mean(rows,'ndcg@5'),v5_lower=a['ndcg@5_lower'],v5_upper=a['ndcg@5_upper'],difference_lower=float(a['ndcg@5_lower'])-mean(rows,'ndcg@5'),difference_upper=float(a['ndcg@5_upper'])-mean(rows,'ndcg@5')))
    write_csv(ROOT/'annotation_revision_effect.csv',effects)
    protocol=read_json(out/'effective_protocol.json');queries={q['query_id']:q for q in protocol['queries']};obs={(r['family_id'],r['part_id']):r for r in records};top=[];errors=Counter()
    evidence={r['part_id']:read_json(Path(cfg['evidence_root'])/(r['inspection_id']+'.json')) for r in records}
    for r in read_csv(ROOT/'evaluation/rankings_top20.csv'):
        if r['cohort']!='closed' or int(r['rank'])>5:continue
        q=queries[r['query_id']];a=obs[q['scenario_id'],r['part_id']];verdict=numeric_reference(q,evidence[r['part_id']]);possible=transformed_set(a['grade'],verdict)
        top.append(dict(**r,is_proposed_method=r['method']=='kg_full',family_id=q['scenario_id'],raw_geometry_grade=a['grade'],admissible_evaluation_grades='|'.join(map(str,possible)),size_verdict=verdict,observation=a['observation'],rationale=a['rationale'],geometry_evidence=a['geometry_evidence'],source_sha256=a['source_sha256']))
        if r['method']=='kg_full' and r['geometry_kind']=='descriptor' and q['reference_part_id'] and q['constraints']:errors[q['scenario_id'],str(a['grade'])]+=1
    write_csv(ROOT/'top5_evidence.csv',top);write_csv(ROOT/'proposed_top5_geometry_grades.csv',[dict(family_id=f,raw_geometry_grade=g,slots=n) for (f,g),n in sorted(errors.items())])
    main_table=table(['Method','Identity','Embedding nDCG@5','Descriptor nDCG@5','Descriptor strong P@5'],[[m,'**Proposed**' if m=='kg_full' else 'Matched flat control' if m=='flat_full' else 'Control / ablation',bounds(s('embedding',m)),bounds(s('descriptor',m)),bounds(s('descriptor',m),'strong_precision@5')] for m in methods])
    paired_table=table(['Family','Representation','Full minus control','nDCG@5 difference envelope'],[[r['comparison_family'],r['geometry_kind'],r['comparator'],bounds(r)] for r in comparisons])
    depth_table=table(['Representation','Method','P@5','Strong P@5','Recall@20','AP'],[[k,m,*[bounds(s(k,m),v) for v in ['precision@5','strong_precision@5','recall@20','ap']]] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter','geometry_intent_filter']])
    revision_table=table(['Proposed method','v4 same 1816','v5 same 1816','Difference'],[[k,*[f"{next(r[v] for r in effects if (r['geometry_kind'],r['method'],r['subset'])==(k,'kg_full','constrained_exemplar')):.4f}" for v in ['v4_ndcg5']],bounds(s(k)),f"[{next(r['difference_lower'] for r in effects if (r['geometry_kind'],r['method'],r['subset'])==(k,'kg_full','constrained_exemplar')):.4f}, {next(r['difference_upper'] for r in effects if (r['geometry_kind'],r['method'],r['subset'])==(k,'kg_full','constrained_exemplar')):.4f}]"] for k in ['embedding','descriptor']])
    grade_table=table(['Representation','Method','Conditional=0','Nominal','Conditional=2'],[[k,m,*[bounds(s(k,m,grading=g)) for g in ['conditional_zero','nominal','conditional_two']]] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter']])
    robust_table=table(['Representation','Full nominal','10% classifier permutation','30% classifier permutation'],[[k,bounds(s(k)),*[bounds(next(r for r in robust if (r['geometry_kind'],r['method'],r['robustness'],float(r['severity']),r['subset'])==(k,'kg_full','classifier_permutation',v,'constrained_exemplar'))) for v in [.1,.3]]] for k in ['embedding','descriptor']])
    cohort_table=table(['Cohort','CADs','Full embedding','Full descriptor','Intent + same filter'],[[c,len(cohorts[c]),bounds(s('embedding',cohort=c)),bounds(s('descriptor',cohort=c)),bounds(s('descriptor','intent_filter',cohort=c))] for c in cohorts])
    second=read_json(ROOT/'second_review_comparison.json');dup=read_json(ROOT/'evaluation/near_duplicate_audit.json')
    family_table=table(['Task','Changed grades','Full descriptor nDCG@5','Intent + same filter'],[[fid,sum(r['family_id']==fid for r in changes),*[bounds(next(r for r in families if (r['cohort'],r['geometry_kind'],r['method'],r['grading_scenario'],r['subset'],r['family_id'])==('closed','descriptor',m,'nominal','constrained_exemplar',fid))) for m in ['kg_full','intent_filter']]] for fid in sorted({r['family_id'] for r in records})])
    conclusions=[]
    for kind in ['embedding','descriptor']:
        p=pair(kind,'intent_filter');g=pair(kind,'geometry_intent_filter')
        relation='is positive for every admissible completion' if float(p['ndcg@5_lower'])>0 else 'is negative for every admissible completion' if float(p['ndcg@5_upper'])<0 else 'includes zero'
        conclusions.append(f"For **proposed Full/{kind}**, nDCG@5 is **{bounds(s(kind))}**, strong P@5 is **{bounds(s(kind),'strong_precision@5')}**, and Full minus intent-with-the-same-filter is **{bounds(p)}** ({relation}). Full minus geometry+intent+filter is **{bounds(g)}**. These are benchmark effects conditional on the AI rubric, not statistical superiority or human-validated engineering success.")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted=['kg_full','geometry_class_filter','geometry_intent_filter','intent_filter','rrf_filter','kg_no_class','kg_no_constraints'];labels=['Proposed Full','Geometry + class + filter','Geometry + intent + filter','Intent + filter','Rank fusion + filter','Geometry + filter','Full without filter']
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True);y=np.arange(len(plotted))
    for ax,metric,title in zip(axes,['ndcg@5','strong_precision@5'],['Ranking utility','Strong precision at 5']):
        for offset,kind,color in [(-.18,'embedding','#8998a6'),(.18,'descriptor','#146c94')]:
            lo=np.array([float(s(kind,m)[metric+'_lower']) for m in plotted]);hi=np.array([float(s(kind,m)[metric+'_upper']) for m in plotted]);mid=(lo+hi)/2
            ax.barh(y+offset,lo,height=.32,color=color,label=kind.capitalize());ax.errorbar(mid,y+offset,xerr=(hi-lo)/2,fmt='none',ecolor='#222',capsize=2)
        ax.set_xlim(0,1);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].get_yticklabels()[0].set_fontweight('bold');axes[1].legend(loc='lower right',frameon=False)
    fig.suptitle('1,816 candidates · 58 constrained exemplar queries · 11 equally weighted families',fontsize=12)
    fig.text(.5,.015,'Bars: conservative lower bounds; whiskers: annotation uncertainty envelopes, not confidence intervals. AI reference judgments.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(ROOT/'proposed_method_comparison.png',dpi=160);fig.savefig(ROOT/'proposed_method_comparison.pdf');plt.close(fig)
    report=f'''# Source-feature audit and fixed-framework rerun, v5

## Result and claim

**`kg_full` is the proposed method.** Embedding and descriptor are its two existing representations. Other methods below are controls or ablations; a better control is not renamed as the proposed method.

{chr(10).join(conclusions)}

The results measure retrieval shortlisting against this revised benchmark. **They do not prove incremental superiority of the complete method, graph contribution, engineering fitness, or broad generalizability.** `kg_full` and `flat_full` have exactly identical orders and scores, so this experiment supplies no retrieval-quality evidence for a graph-representation advantage. Formal point significance is withheld because two reference labels remain unreadable. Even an effect whose entire annotation envelope is positive is not a sampling confidence statement.

## What was reviewed and corrected

All **1,816 held-out CADs × 11 roles = 19,976 judgments** received new source review. A separate **44-CAD validation pilot** established the frozen operational boundaries; the test rubric was not chosen from new scores. Three separate first-pass AI contexts saw neutral six-view images with old labels, categories, ranks, methods and scores hidden. They considered every role and supplied source-specific feature states for positives and conditionals. All 303 navigation sheets were actually inspected and fine features were enlarged.

The targeted blind second review covered 80 CADs / 880 role pairs. Exact agreement was {second['exact_grade_agreement']:.2%}, but agreement among nonzero/uncertain pairs was {second['nontrivial_pair_agreement']:.2%}; {second['disagreements']} disagreements occurred on {second['disagreement_cads']} CADs. This difficult-case sample is not a population error estimate. **All reviews were AI; this is not independent human inter-rater validation.**

All **518 flagged CADs** were freshly adjudicated before scoring: 171 and 179 by another first-pass AI reviewer using other-shard sources, and 168 by the coordinator. These packets contained neutral sources and the frozen rubric, without previous answers. Source decisions govern; proposed changes were not automatically accepted. Full notes, per-feature evidence, adjudicator identity, hashes and [old/new changes](../../annotations/intent_feature_review_v5/changes.csv) preserve the history. The coordinator knew historical aggregate performance, and the operational rubric is a retrospective refinement. **Changes are supported source/rubric revisions, not all demonstrable historical errors, and this is not a newly untouched confirmatory test.**

Final release: **{audit['changes']} changed grades on {audit['changed_cads']} CADs**; {counts['2']} shortlist, {counts['1']} conditional, {counts['0']} reject and {audit['null_pairs']} cannot-assess. Carrier-seat versus plain bore, short pin versus drive shaft, retained shaft hub versus passage, gear/belt/sprocket profiles, and spring attachments have explicit boundaries. Smooth simplified nuts/fasteners remain conditional where appropriate. Fine mesh/cut-edge serrations never establish threads. Cutaways are uncapped visualization aids, not physical section faces or metrology.

Two cannot-assess pairs remain: Z0590/compression_return has overlapping annular layers whose terminal ends cannot be read; Z0605/tension_return lacks the connecting coil body between terminal fragments. They stay in the gallery and qrels, with admissible raw grades 0..2. They are not imputed as rejects, turned into geometric conditionals or dropped. See [cannot-assess evidence](cannot_assess.csv).

## Fixed experimental results

The framework remains **110 queries, 11 task families, 14 methods, 2 representations**, with identical model, exemplars, weights, constraints, seeds, candidate eligibility and robustness conditions. The full 1,816 eligible test gallery is larger than the earlier 1,350 and original 424. Adding training or validation CADs would compromise the held-out design; no such expansion occurred.

Tables use query means within each family, then equal means across eleven families. Main results use the existing 58 constrained exemplar queries. The existing primary subset has 77 exemplar queries; all 33 text queries remain evaluated. nDCG is graded ranking utility, not percent accuracy or engineering compatibility. All methods and all-query/text/cohort subsets are in [candidate_summary.csv](candidate_summary.csv).

Values in brackets are **conservative annotation uncertainty envelopes**, not confidence intervals. Exact values are printed when envelopes collapse. DCG and ideal DCG are bounded separately; AP/recall numerator and relevance denominator are bounded separately. Extrema need not be jointly attainable. Every admissible raw grade is passed through the unchanged conditional and dimension transforms, including the existing size guard. Zero-relevance cases remain zero and remain in averages. No favorable completion or point p-value is selected.

{main_table}

{depth_table}

![Proposed Full versus controls](proposed_method_comparison.png)

## Matched comparisons and whether the method works

{paired_table}

The eleven existing primary comparisons use 77 exemplar queries; the fourteen focused comparisons use 58 constrained exemplars. The comparison families are preserved, but point permutation p-values, Holm claims and sampling confidence intervals are withheld under incomplete labels. Difference envelopes use paired queries and the same labels; identical rankings imply exact zero. These envelopes describe possible benchmark effects, not statistical evidence from new independent task families.

Useful shortlisting is supported only to the degree shown by strong precision and matched controls. A conditional match does not count in strong P@5. Full-method incremental advantage must be judged against intent+filter, geometry+intent+filter, geometry+class+filter and rank fusion with the same filtering. Graph contribution is unsupported by the exact flat equality. Generalization beyond these eleven authored tasks and this source distribution remains untested. Increasing candidate count does not increase the number of independent task families.

## Annotation effects and other limitations

Every **{validation['all_historical_ranking_arrays_unchanged']:,} complete order/score array** equals v4 exactly. Scores changed because the reference grading changed; retrieval order did not improve.

{revision_table}

{family_table}

The limited performance cannot be attributed to annotations alone. Remaining methodological limits include coarse class-to-function priors, shape similarity without direct task-interface verification, equal component fusion, and a dimension check limited to an envelope bound. Text-only Full has no exemplar geometry; spring roles share a coarse class despite different attachment requirements. The audit found both earlier ambiguity and genuine visible distinctions; no nDCG success threshold such as 0.67 has been independently validated. These observations identify plausible limitations, not measured causal fractions of error. [Top-five source evidence](top5_evidence.csv) preserves actual retrieved shapes and effective admissible grades for inspection.

All 19 fixed training exemplars were separately rendered and checked before scoring, without replacement. Sixteen have strong intended-role evidence; three remain conditional: E004 smooth socket-headed fastener, E007 smooth-bore hex nut, E011 flat bolted disk without register. Similarity to simplified exemplars is a plausible limitation, not a causal proof or grounds for tuning on test results. See [exemplar checks](exemplar_role_checks.csv).

## Robustness and candidate composition

{grade_table}

{robust_table}

Classifier permutation is the existing synthetic sensitivity condition, not an observed deployment error rate. Full tie-order, constraint-guard and classifier-condition results and comparisons remain in [robustness intervals](partial_evaluation/robustness_summary_bounds.csv). Conditional=0/2 are unchanged grading sensitivities, not alternative chosen ground truth.

{cohort_table}

These are candidate-composition diagnostics, not new experimental configurations or independent source domains. Descriptor signature deduplication retains {dup['remaining']} CADs and has {len(dup['discordant_groups'])} groups with differing role grades. Signature similarity does not prove exact geometric identity; labels were not copied to force agreement.

## Reproducibility and next evidence needed

Final annotations, rubric, source images, numeric evidence, analysis plan and code were hashed before new scoring. Retrieval was actually executed twice with the existing method implementations; all {validation['all_fresh_replay_arrays_equal']:,} nominal arrays and {validation['all_fresh_robustness_replay_orders_equal']:,} complete robustness orders agree. All historical robustness top-20 lists agree. Conservative interval evaluation was also executed twice with matching deterministic outputs. Separately expressed task/size transforms verified {validation['independent_task_constraint_grade_sets_verified']:,} grade sets; 1,816 raw STEP hashes were rechecked. The initial 1,860 pilot/test source checks found no exact train–test or pilot–test overlap. Existing exact-byte contamination exclusions remain in place.

Validation: **{validation['status']}**. Synthetic exhaustive checks validate interval containment, including uncertain ideal rankings and relevance denominators; these checks validate computation, not annotation truth. See [partial validation](partial_validation.json), [ranking replay validation](evaluation/ranking_validation.json), [synthetic checks](partial_evaluation/synthetic_validation.json), [frozen plan](../../docs/FEATURE_REVIEW_EVALUATION.md) and its [query-count clarification](PLAN_CLARIFICATION.md).

Reproduce using the existing Python environment, fresh output folders, and the frozen v5 release:

```powershell
& $cadkgPython -m scripts.rerun_feature_rankings --output outputs/feature_review_v5/evaluation
& $cadkgPython -m scripts.rerun_feature_rankings --output outputs/feature_review_v5/ranking_replay
& $cadkgPython -m scripts.validate_feature_review ranking-replay
& $cadkgPython -m scripts.partial_feature_review evaluate
& $cadkgPython -m scripts.partial_feature_review evaluate --output outputs/feature_review_v5/partial_replay
& $cadkgPython -m scripts.validate_feature_review partial
& $cadkgPython -m scripts.report_feature_review
```

The frozen human handoff packet remains ready, but genuine independent human adjudication has not occurred. It is the next requirement for human reliability claims. An independently sourced, newly held-out test with rubric/model selection separated from testing is needed for stronger generalization claims. The present larger candidate evaluation supplies reproducible AI-reference evidence and the specific limitations above, without claiming proof beyond it.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'analysis_manifest.json',dict(proposed_method='kg_full',annotation_sha256=audit['annotation_sha256'],null_pairs=audit['null_pairs'],metrics='conservative outer intervals',primary_queries=77,focused_queries=58,primary_comparisons=11,focused_comparisons=14,point_significance_withheld=True,human_expert_reviewed=False,report_sha256=sha256(ROOT/'REPORT.md')))
    print({'report':str(ROOT/'REPORT.md'),'proposed_embedding':bounds(s('embedding')),'proposed_descriptor':bounds(s('descriptor'))})

if __name__=='__main__':main()
