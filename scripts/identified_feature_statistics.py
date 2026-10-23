"""Restore predeclared nDCG tests only when every targeted value is identified.

No null is imputed. The gate covers all14 methods, both representations and all
77 exemplar queries, before any comparison is tested. Other endpoints retain
their partial-qrel bounds. Same original11 primary and14 focused hypotheses.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256
from cadkg.experiments.verified_statistics import analyze,_bootstrap,_flip_p,_holm
from scripts.report_kg_constrained import COMPARATORS,table

ROOT=Path('outputs/feature_review_v5');SEAL=ROOT/'identified_analysis_freeze.json'

def main(action,output=None):
    if action=='freeze':
        assert not SEAL.exists()
        paths=[Path(__file__),Path('cadkg/experiments/verified_statistics.py'),Path('scripts/report_kg_constrained.py'),Path('configs/feature_review_v5.json'),ROOT/'evaluation_freeze.json',ROOT/'partial_evaluation/run_manifest.json',ROOT/'partial_evaluation/per_query_bounds.csv']
        write_json(SEAL,dict(reason='Execute unchanged predeclared tests only if all targeted per-query nDCG5 bounds collapse; no label completion or favorable pair selection',
            timing='After first interval evaluation showed endpoint identification; before computing any new p-values',
            input_sha256={str(p):sha256(p) for p in paths},primary_tests=11,focused_tests=14,endpoint='ndcg@5',all_method_identification_gate=True))
        print('Identified-endpoint analysis gate frozen');return
    frozen=read_json(SEAL)
    for p,h in frozen['input_sha256'].items():assert sha256(p)==h,p
    if action=='merge':return merge()
    out=Path(output) if output else ROOT/'identified_statistics';assert not out.exists()
    cfg=read_json('configs/feature_review_v5.json');raw=read_csv(ROOT/'partial_evaluation/per_query_bounds.csv')
    selected=[r for r in raw if (r['cohort'],r['grading_scenario'],r['has_exemplar'])==('closed','nominal','True')]
    assert len(selected)==77*14*2 and len({r['family_id'] for r in selected})==11
    assert all(float(r['ndcg@5_lower'])==float(r['ndcg@5_upper']) for r in selected),'Withhold point tests: endpoint not fully identified'
    rows=[dict(r,**{'ndcg@5':float(r['ndcg@5_lower'])}) for r in selected]
    out.mkdir();primary=analyze(rows,cfg,out)
    rng=np.random.default_rng(cfg['seed']);families=sorted({r['family_id'] for r in rows})
    springs=[i for i,f in enumerate(families) if f in {'compression_return','tension_return'}]
    groups=[[i] for i in range(11) if i not in springs]+[springs];comparisons=[];deltas=[]
    for kind in ['embedding','descriptor']:
        full=[r for r in rows if (r['geometry_kind'],r['method'],r['has_constraint'])==(kind,'kg_full','True')];assert len(full)==58
        for method in COMPARATORS:
            base={r['query_id']:r for r in rows if (r['geometry_kind'],r['method'],r['has_constraint'])==(kind,method,'True')};values=[]
            assert len(base)==58
            for fid in families:
                cases=[r for r in full if r['family_id']==fid]
                delta=float(np.mean([r['ndcg@5']-base[r['query_id']]['ndcg@5'] for r in cases]));values.append(delta)
                deltas.append(dict(geometry_kind=kind,comparator=method,family_id=fid,delta=delta))
            a=np.asarray(values);ci=_bootstrap(a,rng,20000);sci=_bootstrap(a,rng,20000,groups)
            comparisons.append(dict(geometry_kind=kind,proposed_method='kg_full',comparator=method,queries=58,families=11,
                delta=float(a.mean()),ci95_lower=ci[0],ci95_upper=ci[1],exact_p=_flip_p(a),spring_joint_p=_flip_p(a,groups),
                spring_ci95_lower=sci[0],spring_ci95_upper=sci[1],family_wins=int(sum(a>1e-12)),family_ties=int(sum(abs(a)<=1e-12)),family_losses=int(sum(a< -1e-12)),
                leave_one_family_out_min=float(min((a.sum()-v)/10 for v in a)),leave_one_family_out_max=float(max((a.sum()-v)/10 for v in a))))
    for r,p in zip(comparisons,_holm([r['exact_p'] for r in comparisons])):r.update(holm_p=p,holm_family_size=14)
    write_csv(out/'focused_comparisons.csv',comparisons);write_csv(out/'focused_family_differences.csv',deltas)
    write_json(out/'run_manifest.json',dict(status='VERIFIED_IDENTIFIED_ENDPOINT',nulls_imputed=False,identified_query_method_cases=len(rows),
        endpoint='ndcg@5',primary_comparisons=len(primary),focused_comparisons=len(comparisons),
        original_statistics_code_used=True,seed=cfg['seed'],bootstrap_samples=20000,
        significance_scope='Exploratory conditional on eleven authored AI-graded families; no population or human confirmation',
        input_sha256={**frozen['input_sha256'],str(SEAL):sha256(SEAL)},output_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print('All2156 exemplar query/method values identified; unchanged11+14 tests executed')

def merge():
    first=ROOT/'identified_statistics';replay=ROOT/'identified_statistics_replay'
    a=read_json(first/'run_manifest.json');b=read_json(replay/'run_manifest.json');assert a==b
    for name,h in a['output_sha256'].items():assert sha256(first/name)==sha256(replay/name)==h
    base=ROOT/'REPORT.md';original=base.read_text(encoding='utf-8');text=original
    text=text.replace('Formal point significance is withheld because two reference labels remain unreadable. Even an effect whose entire annotation envelope is positive is not a sampling confidence statement.',
        'Every one of the2,156 primary exemplar query/method nDCG@5 values is exactly identified despite the two nulls. The original eleven primary and fourteen focused comparisons can therefore be computed without imputing any label. Other uncertain endpoints retain bounds. An annotation envelope is not a sampling confidence interval.')
    text=text.replace('The eleven existing primary comparisons use 77 exemplar queries; the fourteen focused comparisons use 58 constrained exemplars. The comparison families are preserved, but point permutation p-values, Holm claims and sampling confidence intervals are withheld under incomplete labels. Difference envelopes use paired queries and the same labels; identical rankings imply exact zero. These envelopes describe possible benchmark effects, not statistical evidence from new independent task families.',
        'The eleven original primary comparisons use77 exemplar queries and the fourteen focused comparisons use58 constrained exemplars. Every targeted per-query nDCG@5 envelope collapses exactly; the two nulls do not alter these values. The original statistical implementation, seed,20,000 family bootstrap replicates, exact family sign flips and separate Holm families are retained. No favorable completion, hypothesis or comparator was selected. Difference envelopes use paired queries and the same labels; identical rankings imply exact zero. Confidence intervals below describe exploratory family sampling variation conditional on this authored benchmark.')
    primary=read_csv(first/'primary_comparisons.csv');focused=read_csv(first/'focused_comparisons.csv')
    stat_table=table(['Representation','Full minus control','Delta nDCG@5','95% family interval','Holm p','Family W/T/L'],[[r['geometry_kind'],r['comparator'],f"{float(r['delta']):.4f}",f"[{float(r['ci95_lower']):.4f}, {float(r['ci95_upper']):.4f}]",f"{float(r['holm_p']):.4f}",f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"] for r in focused])
    sigp=[r['comparison'] for r in primary if float(r['holm_p'])<.05];sigf=[r['geometry_kind']+': Full - '+r['comparator'] for r in focused if float(r['holm_p'])<.05]
    section=f'''### Original tests on the identified nDCG endpoint

{stat_table}

Primary Holm-significant comparisons: {', '.join(sigp) or 'none'}. Focused Holm-significant comparisons: {', '.join(sigf) or 'none'}. The complete primary11 table and spring-dependence/leave-one-family-out sensitivities are preserved in [identified statistics](identified_statistics/primary_comparisons.csv) and [focused statistics](identified_statistics/focused_comparisons.csv).

Identification was checked across all14 methods and both representations, not selected favorable pairs. The same predeclared tests were restored after that gate, with the decision and code sealed before calculating p-values. This is an execution of the original analysis on an exactly known endpoint, not a new hypothesis family. Retrospective annotation refinement and AI reference limitations still prevent a confirmatory population claim.

'''
    text=text.replace('## Annotation effects and other limitations',section+'## Annotation effects and other limitations')
    text=text.replace('point p-values, Holm claims and sampling confidence intervals are withheld','point inference is withheld only for nonidentified endpoints')
    text+='\n\n## Supplementary calculation and execution records\n\nAll original cutoff5/10/20, AP@20 and uniform-feasible random expectation diagnostics are retained under the same matrix; see [supplement validation](supplement_validation.json) and [supplement guide](SUPPLEMENT_README.md). [Execution notes](EXECUTION_NOTES.md) document the separately frozen path cache used to accelerate file identity checks; it changes no numerical calculation. [Critical audit findings](AUDIT_FINDINGS.md) give grade-transition counts and proposed-change dispositions. The original conservative report is preserved as REPORT.md; this integrated report restores only tests whose endpoint is exactly identified. No annotation or retrieval setting was changed.\n'
    final=ROOT/'FINAL_REPORT.md';assert not final.exists();final.write_text(text,encoding='utf-8')
    write_json(ROOT/'identified_validation.json',dict(status='VERIFIED',endpoint='ndcg@5',identified_cases=2156,primary_tests=11,focused_tests=14,statistics_replay_equal=True,nulls_imputed=False,original_conservative_report_sha256=sha256(base),final_report_sha256=sha256(final),identified_manifest_sha256=sha256(first/'run_manifest.json')))
    print('Integrated report written; identified-endpoint statistics reproduce')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate','merge']);p.add_argument('--output');a=p.parse_args();main(a.action,a.output)
