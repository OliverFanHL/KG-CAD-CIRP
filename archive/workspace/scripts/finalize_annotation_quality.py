"""Compare explicit rereviews with frozen annotations, then publish an audited version."""
from collections import defaultdict,Counter
from pathlib import Path
import argparse
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256
from scripts.visual_annotation_review import ROLE

OUT=Path('outputs/annotation_quality_audit_v2')

def main(publish=False):
    source=Path('annotations/intent_batch2_verified/judgments.json');data=read_json(source)
    plan=read_json(OUT/'sampling_plan.json')
    if sha256(source)!=plan['annotation_sha256']:raise ValueError('Audit source changed')
    old={(r['part_id'],r['family_id']):r for r in data['judgments']}
    manifest={r['audit_id']:r for r in read_csv(OUT/'sample_manifest.csv')}
    notes={}
    for line in (OUT/'review_notes.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        code,marks,observation=line.split('|',2)
        if code in notes:raise ValueError('Duplicate review')
        notes[code]=dict(grades={ROLE[x[0]]:int(x[1]) for x in marks.split(',') if x},observation=observation)
    if set(notes)!=set(manifest):raise ValueError('Incomplete sampled rereview')
    pairs=[];changes=[]
    for code,note in notes.items():
        item=manifest[code]
        for fid in ROLE.values():
            previous=old[item['part_id'],fid];g=note['grades'].get(fid,0)
            row=dict(audit_id=code,family_id=fid,part_id=item['part_id'],source_id=item['source_inspection_id'],
                old=previous['grade'],proposed=g,component=item['sample_component'],observation=note['observation'])
            pairs.append(row)
            if g!=previous['grade']:changes.append(row)
    write_csv(OUT/'proposed_changes.csv',changes)
    if not publish:
        print('Reviewed CADs',len(notes),'proposed differences',len(changes))
        for r in changes:print(r['audit_id'],r['source_id'],r['family_id'],r['old'],r['proposed'],r['component'])
        return
    destination=Path('annotations/intent_batch2_audited_v2');destination.mkdir(exist_ok=True)
    pairlookup={(r['part_id'],r['family_id']):r for r in pairs}
    tasks={r['family_id']:r for r in read_json('annotations/review_tasks.json')['tasks']}
    for row in data['judgments']:
        review=pairlookup.get((row['part_id'],row['family_id']))
        if not review:continue
        grade=review['proposed'];row['grade']=grade;row['decision']={0:'reject',1:'conditional',2:'shortlist'}[grade]
        row['observation']=review['observation'];term={0:'reject',1:'conditional',2:'shortlist'}[grade]
        row['rationale']=term.capitalize()+': '+tasks[row['family_id']]['rules'][term]+' Evidence: '+review['observation']
        row['quality_audit_id']=review['audit_id'];row['quality_audit_source']=str(OUT/'review_notes.txt')
    data['parent_annotation_sha256']=sha256(source);data['quality_audit']='Random 100-CAD cluster sample plus 66 grade-stratified pairs; all 166 selected CADs rereviewed against all 11 existing scopes. AI-only review.'
    write_json(destination/'judgments.json',data);write_csv(destination/'judgments.csv',data['judgments'])
    write_csv(destination/'corrections.csv',changes);write_csv(OUT/'all_reviewed_pairs.csv',pairs)
    primary=[r for r in pairs if r['component']=='primary_srs']
    bycad=defaultdict(list)
    for r in primary:bycad[r['part_id']].append(int(r['old']!=r['proposed']))
    errors=np.asarray([np.mean(x) for x in bycad.values()]);rng=np.random.default_rng(plan['seed'])
    boot=rng.choice(errors,(20000,len(errors))).mean(1)
    targeting={(r['part_id'],r['family_id']) for r in plan['supplementary_pairs']}
    supplemental=[r for r in pairs if (r['part_id'],r['family_id']) in targeting]
    bygrade={str(g):dict(reviewed=sum(r['old']==g for r in primary),changed=sum(r['old']==g and r['old']!=r['proposed'] for r in primary)) for g in [0,1,2]}
    report=dict(seed=plan['seed'],primary_cads=len(bycad),primary_pairs=len(primary),reviewed_cads=len(notes),reviewed_pairs=len(pairs),
        supplementary_target_pairs=len(supplemental),supplementary_target_changes=sum(r['old']!=r['proposed'] for r in supplemental),
        primary_changed_pairs=sum(r['old']!=r['proposed'] for r in primary),primary_pair_disagreement_rate=float(errors.mean()),
        primary_cluster_bootstrap_ci95=[float(x) for x in np.quantile(boot,[.025,.975])],
        primary_cads_with_changes=int(sum(any(x) for x in bycad.values())),all_changed_pairs=len(changes),
        by_original_grade=bygrade,confusion={f'{a}->{b}':sum(r['old']==a and r['proposed']==b for r in primary) for a in [0,1,2] for b in [0,1,2]},
        final_annotation_sha256=sha256(destination/'judgments.json'),human_expert_reviewed=False,
        uncertainty_scope='Same-model rereview disagreement, not an independent ground-truth error estimate; intervals resample 100 CAD clusters, not 1100 independent records.',
        audit_complete=True,unresolved_sample_records=0,
        allowed_next_stage='Label-independent serving efficiency and scalability only; no new relevance superiority claims or additional effectiveness query families.',
        annotation_quality_claim='Sampled visual consistency assessed and corrections published; global or independent human correctness is not certified.')
    write_json(OUT/'quality_gate.json',report);print(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args();main(args.publish)
