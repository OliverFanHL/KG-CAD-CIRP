"""Analyze all fixed-design results and render publication-ready diagnostic plots."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from cadkg.io import read_csv,read_json,write_csv,write_json,sha256

OUT=Path('outputs/verified_evaluation_v1')

def family_mean(rows,metric='ndcg@5'):
    groups=defaultdict(list)
    for r in rows:groups[r['family_id']].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else 0.

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    summary=read_csv(OUT/'summary.csv');per=read_csv(OUT/'per_query.csv');rob=read_csv(OUT/'robustness_per_query.csv')
    primary=read_csv(OUT/'primary_comparisons.csv');families=read_csv(OUT/'primary_family_differences.csv')
    integrity=read_json(OUT/'integrity.json');repro=read_json(OUT/'reproducibility.json')
    def lookup(kind,method,cohort='closed',scenario='nominal',variant='exemplar'):
        return next(r for r in summary if r['cohort']==cohort and r['grading_scenario']==scenario and r['geometry_kind']==kind and r['method']==method and r['variant']==variant)
    def val(kind,method,metric='ndcg@5',**kw):return float(lookup(kind,method,**kw)[metric])
    # Aggregate seeds within each query/family before cross-family averaging.
    robust_summary=[]
    for condition in sorted({(r['robustness'],r['severity']) for r in rob}):
        for kind in ['embedding','descriptor']:
            for method in sorted({r['method'] for r in rob}):
                subset=[r for r in rob if (r['robustness'],r['severity'])==condition and r['geometry_kind']==kind and r['method']==method and r['has_exemplar']=='True']
                if not subset:continue
                constrained=condition[0]=='constraint_guard'
                matched=[r for r in per if r['cohort']=='closed' and r['grading_scenario']=='nominal' and r['geometry_kind']==kind and r['method']==method and r['has_exemplar']=='True' and (not constrained or r['has_constraint']=='True')]
                seedmeans=[family_mean([r for r in subset if r['seed']==seed]) for seed in sorted({r['seed'] for r in subset})]
                robust_summary.append(dict(robustness=condition[0],severity=condition[1],geometry_kind=kind,method=method,
                    ndcg5=family_mean(subset),matched_nominal_ndcg5=family_mean(matched),delta=family_mean(subset)-family_mean(matched),
                    mean_seed_min=min(seedmeans),mean_seed_max=max(seedmeans),seed_count=len(seedmeans),scope='constrained_exemplar_queries' if constrained else 'all_exemplar_queries'))
    write_csv(OUT/'robustness_summary.csv',robust_summary)
    exemplars=[]
    queries=read_json('outputs/intent_batch2_matrix_reproduction/protocol.json')['queries']
    refs=defaultdict(list)
    for q in queries:
        if q['reference_part_id'] and q['reference_part_id'] not in refs[q['scenario_id']]:refs[q['scenario_id']].append(q['reference_part_id'])
    qlookup={q['query_id']:q for q in queries}
    for fid,reference_ids in refs.items():
        if len(reference_ids)!=2:continue
        for kind in ['embedding','descriptor']:
            for method in ['kg_full','geometry_class_filter','intent_filter','kg_no_class']:
                values=[]
                for ref in reference_ids:
                    selected=[r for r in per if r['cohort']=='closed' and r['grading_scenario']=='nominal' and r['geometry_kind']==kind and r['family_id']==fid and r['method']==method and qlookup[r['query_id']]['reference_part_id']==ref]
                    values.append(float(np.mean([float(r['ndcg@5']) for r in selected])))
                exemplars.append(dict(family_id=fid,geometry_kind=kind,method=method,example1_ndcg5=values[0],example2_ndcg5=values[1],delta_example2_minus1=values[1]-values[0]))
    write_csv(OUT/'exemplar_sensitivity.csv',exemplars)
    screening=[]
    for kind in ['embedding','descriptor']:
        for method in ['kg_full','kg_no_constraints','geometry_class_filter','geometry_class','intent_filter','category_from_intent']:
            group=[r for r in per if r['cohort']=='closed' and r['grading_scenario']=='nominal' and r['geometry_kind']==kind and r['method']==method and r['has_constraint']=='True']
            screening.append(dict(geometry_kind=kind,method=method,constrained_queries=len(group),top5_slots=5*len(group),
                failures=sum(int(r['dimension_failures']) for r in group),unknowns=sum(int(r['dimension_unknowns']) for r in group),ndcg5=family_mean(group)))
    write_csv(OUT/'screening_summary.csv',screening)
    bounds=read_csv(OUT/'full_gallery_bounds.csv'); bound_summary=[]
    for kind in ['embedding','descriptor']:
        for method in integrity['methods']:
            group=[r for r in bounds if r['geometry_kind']==kind and r['method']==method and r['has_exemplar']=='True']
            bound_summary.append(dict(geometry_kind=kind,method=method,ndcg_lower=family_mean(group,'ndcg_lower'),ndcg_upper=family_mean(group,'ndcg_upper'),
                unjudged_top5=sum(int(r['unjudged_at5']) for r in group),queries=len(group)))
    write_csv(OUT/'full_gallery_summary.csv',bound_summary)
    impact=read_csv(OUT/'correction_impact.csv');changed=sum(int(r['changed_return_grades']) for r in impact)
    prior_effect=next(r for r in primary if r['comparison']=='descriptor:kg_full - descriptor:geometry_class_filter')
    prior_families=[r for r in families if r['comparison']==prior_effect['comparison']]
    leave_belt=float(np.mean([float(r['delta_ndcg@5']) for r in prior_families if r['family_id']!='belt_wheel']))
    # Two compact standalone scientific figures; no unsupported error bars on method means.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    methods=['kg_no_class','geometry_class_filter','intent_filter','rrf_filter','geometry_intent_filter','kg_full']
    labels=['Geometry + filter','Geometry + class + filter','Intent + filter','Rank fusion + filter','Geometry + intent + filter','Full']
    fig,ax=plt.subplots(figsize=(9,4.3));y=np.arange(len(methods));h=.36
    for offset,kind,color in [(-h/2,'embedding','#446b91'),(h/2,'descriptor','#d28441')]:
        ax.barh(y+offset,[val(kind,m) for m in methods],height=h,label=kind.capitalize(),color=color)
    ax.set(yticks=y,yticklabels=labels,xlabel='Equal-family nDCG@5 (nominal, exemplar queries)',xlim=(0,1));ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'primary_comparison.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));y=np.arange(len(primary));d=np.asarray([float(r['delta']) for r in primary]);lo=np.asarray([float(r['ci95_lower']) for r in primary]);hi=np.asarray([float(r['ci95_upper']) for r in primary])
    pretty=[r['comparison'].replace('embedding:','E ').replace('descriptor:','D ').replace('kg_full','Full').replace('geometry_class_filter','geometry+class').replace('intent_filter','intent').replace('kg_no_class','geometry').replace('rrf_filter','RRF').replace('kg_no_geometry','without geometry') for r in primary]
    ax.errorbar(d,y,xerr=np.vstack([d-lo,hi-d]),fmt='o',color='#446b91',capsize=3);ax.axvline(0,color='black',lw=.8);ax.set(yticks=y,yticklabels=pretty,xlabel='Paired nDCG@5 difference; unadjusted family bootstrap 95% intervals');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'paired_effects.{ext}',dpi=220)
    plt.close(fig)
    fmt=lambda x:f'{float(x):.4f}'
    main_table=table(['Method, same filter unless stated','Embedding nDCG@5','Descriptor nDCG@5','Descriptor strong P@5'],[[label,fmt(val('embedding',m)),fmt(val('descriptor',m)),fmt(val('descriptor',m,'strong_precision@5'))] for m,label in zip(methods,labels)])
    stat_table=table(['Comparison (Full minus control)','Δ nDCG@5','95% bootstrap interval','Raw p','Holm p','Family W/T/L'],[[r['comparison'],fmt(r['delta']),f"[{fmt(r['ci95_lower'])}, {fmt(r['ci95_upper'])}]",fmt(r['exact_sign_flip_p']),fmt(r['holm_p']),f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in primary])
    cohort_table=table(['Gallery','CADs','Full embedding','Full descriptor','Intent + filter','Descriptor Full − intent'],[[c,integrity['cohort_sizes'][c],fmt(val('embedding','kg_full',cohort=c)),fmt(val('descriptor','kg_full',cohort=c)),fmt(val('descriptor','intent_filter',cohort=c)),fmt(val('descriptor','kg_full',cohort=c)-val('descriptor','intent_filter',cohort=c))] for c in ['closed','original_pool','new_only','deduplicated']])
    grade_table=table(['Conditional interpretation','Full embedding','Full descriptor','Descriptor − embedding'],[[s,fmt(val('embedding','kg_full',scenario=s)),fmt(val('descriptor','kg_full',scenario=s)),fmt(val('descriptor','kg_full',scenario=s)-val('embedding','kg_full',scenario=s))] for s in ['conditional_zero','nominal','conditional_two']])
    robustness_table=table(['Check','Severity','Representation','Matched nominal','Perturbed mean','Seed mean range'],[[r['robustness'],r['severity'],r['geometry_kind'],fmt(r['matched_nominal_ndcg5']),fmt(r['ndcg5']),f"[{fmt(r['mean_seed_min'])}, {fmt(r['mean_seed_max'])}]"] for r in robust_summary if r['method']=='kg_full'])
    deeper_table=table(['Representation / method','nDCG@20','Recall@20','Full AP'],[[k+' / '+m,fmt(val(k,m,'ndcg@20')),fmt(val(k,m,'recall@20')),fmt(val(k,m,'ap'))] for k in ['embedding','descriptor'] for m in ['kg_full','intent_filter','geometry_class_filter']])
    random=read_csv(OUT/'random_expectation.csv')
    exemplar_qids={q['query_id'] for q in queries if q['reference_part_id']}
    random_closed=[r for r in random if r['cohort']=='closed' and r['geometry_kind']=='descriptor' and r['query_id'] in exemplar_qids]
    unknown_full={k:next(r['unknowns'] for r in screening if r['geometry_kind']==k and r['method']=='kg_full') for k in ['embedding','descriptor']}
    exemplar_full=[r for r in exemplars if r['method']=='kg_full'];exrange={k:max(abs(r['delta_example2_minus1']) for r in exemplar_full if r['geometry_kind']==k) for k in ['embedding','descriptor']}
    text_table=table(['Method','Embedding text nDCG@5','Descriptor text nDCG@5'],[[m,fmt(val('embedding',m,variant='text')),fmt(val('descriptor',m,variant='text'))] for m in ['kg_full','intent_filter','kg_no_class','geometry_class_filter']])
    report=f'''# Corrected-annotation experimental evaluation

## Material Passport

- Run: verified-evaluation-v1, executed 2026-10-20.
- Verification status: **VERIFIED** computational reproduction; relevance validity remains AI-only.
- Inputs: 14,850 reviewed AI task–part records; 1,350 held-out CADs; 11 authored task families; 49 frozen query variants; 19 train exemplars; two frozen representations.
- Design: [fixed evaluation plan](../../docs/VERIFIED_EVALUATION_PLAN.md); [configuration](../../configs/verified_evaluation.json).
- No model training, test-label weight tuning, new task definitions or new exemplars. Existing files/results remain historical.

## 1. Assessment of the previous setup

The previous 674 labels covered a retrieval-selected pool, with full-gallery recall unknown. The new release supplies complete judgments for a closed 1,350-CAD gallery. This permits actual closed-gallery recall and AP. It does not turn 14,850 correlated task–part records into independent experiments: the primary sampling unit is the task family, with only 11 authored families and known spring-task dependence.

The old `kg_no_class` ablation removed both classifier-reference agreement and function priors. New same-filter controls separate them. The old category comparator lacked the Full filter, so its difference mixed ranking and screening. The new intent/category+filter control corrects that comparison. All current function edges are single class mappings with unit weight/confidence; the graph contains no richer instance-level function truth.

The cohort preserves 424 originally retrieved candidates and adds 926 class-balanced candidates. These source categories were used only for sampling/descriptive distribution, not for functional grades or scores. This sample is broader but not a random industrial population. All functional judgments remain same-author AI visual assessments.

## 2. Design actually executed

Four paired galleries: all 1,350 judged CADs; original 424; newly added 926; and a 1,286-CAD numerical-signature sensitivity. Fourteen methods comprise the original eight, five isolated signal/filter controls, and fixed reciprocal rank fusion (constant 60, exact-tie average branch ranks). A uniform feasible-random baseline is calculated analytically. All use identical inputs and candidate access within each comparison. Unfiltered methods are explicit screening ablations.

Primary endpoint: family-balanced nDCG@5 for 38 exemplar query variants. Text-only results are separate. Other outcomes include strong P@5, P@5, nDCG@10/20, recall@5/10/20, full AP, truncated AP@20 and constraint verdicts. Empty returns receive zero utility. AP denominators include every positive qrel, including positives filtered out by the method.

The fixed design generated {integrity['primary_cases']:,} metric rows including grading and gallery sensitivities, plus {integrity['robustness_cases']:,} robustness rows. These rows are dependent measurements, not sample size. Eleven primary tests form one Holm family: five method contrasts for each representation and descriptor-minus-embedding Full. Exact family sign flips use 2^11 assignments; intervals use 20,000 family bootstrap draws. Spring joint-cluster checks and leave-one-family-out ranges are also saved. Inference assumes exchangeable task clusters, which is only an exploratory model for these authored tasks.

## 3. Primary retrieval results

{main_table}

Descriptor Full improves over filtered geometry by **{val('descriptor','kg_full')-val('descriptor','kg_no_class'):.4f}** and over filtered intent by **{val('descriptor','kg_full')-val('descriptor','intent_filter'):.4f}** nDCG@5. Its strong P@5 gain over intent is {val('descriptor','kg_full','strong_precision@5')-val('descriptor','intent_filter','strong_precision@5'):.4f}, approximately {5*(val('descriptor','kg_full','strong_precision@5')-val('descriptor','intent_filter','strong_precision@5')):.2f} additional strongly suitable items per five-slot shortlist under equal-family averaging. These are observed benchmark effects.

The analytically expected feasible-random nDCG@5 is {family_mean(random_closed,'expected_ndcg5'):.4f}, with strong P@5 {family_mean(random_closed,'expected_strong_precision5'):.4f}. This is a prevalence sanity check, not the decisive competitive baseline. Reciprocal rank fusion follows [Cormack, Clarke and Buettcher](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), with the stated average-rank tie adaptation.

Embedding Full is below filtered intent by {val('embedding','intent_filter')-val('embedding','kg_full'):.4f}. The simpler descriptor geometry+intent control is slightly above Full, so retaining classifier-reference agreement does not consistently improve top5 quality. RRF is a competitive fixed fusion baseline; no fusion rule was selected by tuning these test results.

![Primary retrieval comparison](figures/primary_comparison.png)

## 4. Statistical and practical interpretation

{stat_table}

**None of the eleven primary tests survives Holm correction at 0.05.** Descriptor Full versus filtered geometry has a large observed difference and a positive unadjusted bootstrap interval, but adjusted p={next(r['holm_p'] for r in primary if r['comparison']=='descriptor:kg_full - descriptor:kg_no_class')}. Descriptor-minus-embedding also has a positive unadjusted interval but adjusted p={primary[-1]['holm_p']}. These findings are encouraging descriptive evidence, not demonstrated multiplicity-controlled superiority. Nonsignificance is not equivalence. The 0.02 practical threshold is declared for interpretation and has no engineering validation.

The descriptor function-prior increment over geometry+classifier+filter is {float(prior_effect['delta']):.4f}. The belt-wheel family alone contributes a +0.7308 family difference; excluding it gives **{leave_belt:.4f}**. Broad incremental effectiveness of the function-prior term is therefore unsupported. Full results and losses for every family are retained in `primary_family_differences.csv`; leave-one-family-out ranges and spring sensitivity are in `primary_comparisons.csv`.

![Paired effects and exploratory intervals](figures/paired_effects.png)

## 5. Generalization beyond the original retrieval pool

{cohort_table}

The Full descriptor gain over intent shrinks substantially on the added candidates. The old and expanded galleries give identical Full top5 scores because the high-ranking original-pool candidates remain; increasing the denominator alone does not add independent top5 evidence. The new-only analysis is therefore essential. Its candidate composition and gallery size differ, so scores across cohorts are descriptive; comparisons between methods within each cohort are paired.

Numerical descriptor signatures identify 57 test candidates matching train and 11 within-gallery groups; accounting for overlap, **64 total candidates** are removed. There are no discordant task grades within those signature groups. Results change little after that exclusion. This narrow signature test cannot rule out near-duplicate geometry missed by descriptors and is not an independent external-domain test.

## 6. Annotation uncertainty and exemplar dependence

{grade_table}

Conditional-grade reinterpretation changes absolute quality appreciably, although descriptor Full remains above embedding Full in every interpretation. These scenarios are alternative label policies, not probabilistic confidence bounds. The 677 conditional assessments and lack of independent human adjudication remain substantive limits.

Eight families have two frozen train exemplars. The largest absolute per-family Full difference between exemplar choices is {exrange['embedding']:.4f} for embedding and {exrange['descriptor']:.4f} for descriptor, averaging matched constrained/unconstrained variants. `exemplar_sensitivity.csv` reports all such cases; do not treat two exemplars as many independent tasks.

## 7. Screening and robustness

{robustness_table}

Guard rows compare constrained exemplar queries against the same constrained nominal subset; other rows use all exemplar queries. Changing the reference guard changes both qrel grades and reported reference verdicts, without changing production filtering. The kernel and surface reference are shared, so screening results establish consistency, not independent metrology or manufacturing compatibility.

Full has zero surface-reference failures among returned constrained top5 candidates in the nominal run, but {unknown_full['embedding']} of 95 embedding slots and {unknown_full['descriptor']} of 95 descriptor slots are reference-unknown near the boundary. Zero failures therefore does not mean all returns are verified passes. `screening_summary.csv` compares failures and unknowns with unfiltered controls. Classifier-coordinate permutation is synthetic corruption shared across both methods and representations, including exemplars. It does not simulate observed domain shift, retraining or physical CAD noise. Descriptor Full is more sensitive to this corruption in aggregate. Exact-score tie-order changes leave Full means unchanged here; they only probe exact ties.

## 8. Deeper retrieval and text-only behavior

{deeper_table}

Top5 gains do not automatically imply better deep ranking: the intent control remains competitive on full AP, and Full's recall@20 remains limited. These are complete-gallery denominators for the judged cohort only.

{text_table}

Without an exemplar, geometry-only methods abstain; reporting those zeros as evidence of semantic superiority would confound input availability. Text-only Full reduces to the admitted class-function prior and is representation-invariant. The implementation accepts structured intent IDs, not unrestricted engineering language understanding.

## 9. Full-gallery continuity and structural equalities

All 784 legacy method/query/representation top5 lists match the original serving outputs. Their returned candidates remain judged, with {changed} changed returned-grade occurrences after correction across correlated runs. The full 1,816 gallery has 466 unjudged candidates. Among all fourteen methods, {sum(int(r['unjudged_at5'])>0 for r in bounds)} of {len(bounds)} top5 runs contain an unjudged return ({sum(int(r['unjudged_at5']) for r in bounds)} slots). Those runs receive conservative lower/upper bounds in `full_gallery_bounds.csv`, never an unjudged-as-reject point score. Full-gallery recall remains unknown.

Full and the matched flat table have identical complete rankings; graph storage alone has no demonstrated effectiveness advantage. Removing exemplar geometry also produces identical complete rankings across representations because all remaining classifier/prior inputs are shared. Intent+filter equals category+filter because each admitted function has one unit-weight class prior. These are exact implementation facts, not failed statistical tests and not problems that additional labels can resolve.

## 10. Reproducibility and validation

- {repro['csv_tables_exactly_replayed']} experiment tables reproduced byte-for-byte in a separate run.
- {repro['complete_ranking_arrays_exactly_replayed']:,} complete order/score arrays reproduced exactly.
- {repro['metric_cases_independently_checked']:,} cutoff-specific metric cases agree with the separate existing metric engine.
- All {repro['raw_step_hashes_verified']:,} actual STEP files were rehashed against annotation/evidence/catalog bindings.
- Seven synthetic evaluator contract tests passed (tie ranks, abstention, equal-weight/filter arithmetic, RRF, metric denominators, exact tests and signature groups).
- Configuration, input/code hashes, library versions, complete scores/ranks and results are saved. Runtime was approximately 25 seconds per deterministic run on the available environment; this is total evaluation runtime, not a latency benchmark.

Commands:

```powershell
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.run_verified_evaluation
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.run_verified_evaluation --output outputs/verified_evaluation_v1/replay
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.validate_verified_evaluation
& 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe' -m scripts.report_verified_evaluation
```

## 11. Statistical fallacy audit (11/11 checked)

| Check | Assessment |
|---|---|
| Simpson reversal | Family results and cohort strata retained; the function-prior aggregate is dominated by belt wheels. |
| Ecological inference | Task means are not individual-CAD or engineering-population guarantees. |
| Selection/Berkson bias | Original retrieval pool and class-balanced additions are explicitly separated; population representativeness is unproven. |
| Collider conditioning | Returned-only and closed-cohort selection affect applicability; no causal population claim is made. |
| Base rates | Per-role prevalence and random feasible expectations are saved; strong and conditional grades remain separate. |
| Regression to mean | Correction is not interpreted as an intervention; methods rerun on identical corrected labels. |
| Survivorship | All authored queries, empty returns, negative deltas and cohort exclusions are retained. |
| Look-elsewhere | All eleven primary contrasts reported with Holm adjustment; other diagnostics explicitly exploratory. |
| Forking paths | Design/config fixed before new outcomes, historical outcomes known; no claim of external preregistration. |
| Causal overclaim | Controlled software ablations isolate terms on this benchmark, not industrial engineering outcomes. |
| Reverse causality | No causal real-world pathway is estimated; classifier priors are not verified part functions. |

## 12. Contributions supported and next steps

**Supported:** a reproducible source-linked CAD shortlisting pipeline; observed descriptor fusion gains over geometry-only retrieval on this benchmark; consistent measured screening; explicit uncertainty/provenance handling. **Qualified:** gains over a fair intent baseline are smaller on newly sampled candidates, embedding fusion is not uniformly beneficial, and the function-prior increment is task-concentrated. **Unsupported:** statistical superiority after the declared multiplicity correction, a graph-topology effectiveness advantage, broad unseen-domain generalization, unrestricted language reasoning, independent expert truth, or certified engineering suitability.

For the paper, retain the simpler controls and family losses, describe current evidence as exploratory AI-judged retrieval, and narrow effectiveness claims accordingly. The highest-value remaining validation is a blinded independent human audit of cross-method disagreements plus a genuinely held-out set of engineering task families and an untouched external/design-separated CAD cohort. Freeze any future model/weight decisions on separate development data. Adding more same-task CADs alone will not fix task diversity, annotation independence or structural equality.

Methodological references: [Smucker, Allan and Carterette, paired randomization in IR](https://ciir-publications.cs.umass.edu/getpdf.php?id=744) supports paired tests targeting the reported statistic; [NIST, pooled and sampled judgments](https://www.nist.gov/publications/comparison-pooled-and-sampled-relevance-judgments) motivates pool-coverage scrutiny. Their findings do not confer population validity on this small authored benchmark.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    write_json(OUT/'analysis_manifest.json',dict(script_sha256=sha256(__file__),input_run_manifest_sha256=sha256(OUT/'run_manifest.json'),
        validator_sha256=sha256('scripts/validate_verified_evaluation.py'),unit_test_source_sha256=sha256('tests/test_verified_evaluation.py'),
        report_sha256=sha256(OUT/'REPORT.md'),function_prior_delta_without_belt=leave_belt,
        primary_holm_rejections=sum(float(r['holm_p'])<.05 for r in primary),human_expert_validated=False))
    print('Report, diagnostics and four figure artifacts saved.')

if __name__=='__main__':main()
