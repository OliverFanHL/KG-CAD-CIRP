"""Package explicit source-view assessments without category or score access."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from cadkg.io import read_csv,read_json,write_csv,write_json,sha256
from scripts.prepare_candidate_scale import ROOT,PARENT,ANN
from scripts.visual_annotation_review import ROLE

def read_notes(path):
    result={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):continue
        code,marks,observation=line.split('|',2)
        if code in result:raise ValueError(f'Duplicate note {code}')
        grades={}
        for mark in filter(None,marks.split(',')):
            g=None if mark[1:]=='?' else int(mark[1:])
            if mark[0] not in ROLE or g not in (None,0,1,2):raise ValueError(mark)
            if ROLE[mark[0]] in grades:raise ValueError('Duplicate role')
            grades[ROLE[mark[0]]]=g
        if not observation.strip():raise ValueError('Missing observation')
        result[code]=dict(grades=grades,observation=observation)
    return result

def compare():
    data=read_json(PARENT);old={(r['part_id'],r['family_id']):r for r in data['judgments']}
    notes={**read_notes(ROOT/'audit_notes.txt'),**read_notes(ROOT/'supplemental_notes.txt')};manifest=read_csv(ROOT/'audit_manifest.csv')+read_csv(ROOT/'supplemental_manifest.csv')
    if set(notes)!={r['review_id'] for r in manifest}:raise ValueError('Incomplete audit')
    differences=[]
    for item in manifest:
        note=notes[item['review_id']]
        for fid in ROLE.values():
            previous=old[item['part_id'],fid];new=note['grades'].get(fid,0)
            if new!=previous['grade']:
                differences.append(dict(review_id=item['review_id'],part_id=item['part_id'],inspection_id=item['inspection_id'],family_id=fid,previous_grade=previous['grade'],proposed_grade=new,previous_observation=previous['observation'],observation=note['observation']))
    write_csv(ROOT/'audit_proposals.csv',differences)
    for r in differences:print(r['review_id'],r['inspection_id'],r['family_id'],r['previous_grade'],'->',r['proposed_grade'],'OLD:',r['previous_observation'])
    print('proposed differences',len(differences),flush=True)

def publish():
    freeze=read_json(ROOT/'freeze.json')
    for path,digest in freeze['inputs'].items():
        if sha256(path)!=digest:raise ValueError(f'Frozen input changed: {path}')
    parent=read_json(PARENT);data=deepcopy(parent);old={(r['part_id'],r['family_id']):deepcopy(r) for r in data['judgments']}
    tasks={r['family_id']:r for r in read_json('annotations/review_tasks.json')['tasks']}
    audit={**read_notes(ROOT/'audit_notes.txt'),**read_notes(ROOT/'supplemental_notes.txt')};added=read_notes(ANN/'visual_notes.txt')
    am=read_csv(ROOT/'audit_manifest.csv')+read_csv(ROOT/'supplemental_manifest.csv');nm=read_csv(ROOT/'added_manifest.csv')
    if set(audit)!={r['review_id'] for r in am} or set(added)!={r['review_id'] for r in nm}:raise ValueError('Incomplete visual review')
    # Adjudication is an explicit reviewed decision, not an automatic grade overwrite.
    decisions=read_json(ROOT/'adjudication.json');decision_map={(r['review_id'],r['family_id']):r for r in decisions}
    proposed=read_csv(ROOT/'audit_proposals.csv')
    if set(decision_map)!={(r['review_id'],r['family_id']) for r in proposed}:raise ValueError('Every disagreement needs an explicit disposition')
    observations={(r['part_id'],r['family_id']):r for r in data['judgments']};changes=[];review_pairs=[]
    def row_for(item,note,fid,g):
        term={0:'reject',1:'conditional',2:'shortlist',None:'cannot_assess'}[g]
        reason=tasks[fid]['rules'][term] if g is not None else 'Cannot assess decisive evidence from the available views.'
        return dict(family_id=fid,part_id=item['part_id'],inspection_id=item['inspection_id'],source_sha256=item['source_sha256'],grade=g,decision=term,observation=note['observation'],rationale=term.capitalize()+': '+reason+' Evidence: '+note['observation'],geometry_evidence=item['geometry_evidence'],source='ai_reference',human_expert_reviewed=False,reviewer='Codex AI source-view reviewer',review_status='visually_reviewed',inspection_origin='Neutral six-CAD enlarged views, candidate-scale-v4 '+item['review_id'])
    for item in am:
        code=item['review_id'];note=audit[code]
        for fid in ROLE.values():
            previous=old[item['part_id'],fid];g=note['grades'].get(fid,0);decision=decision_map.get((code,fid))
            final=decision['final_grade'] if decision else g
            review_pairs.append(dict(review_id=code,component=item['component'],part_id=item['part_id'],family_id=fid,previous_grade=previous['grade'],blind_proposed_grade=g,final_grade=final,changed=previous['grade']!=final))
            # Preserve parent bytes at record level except evidence-supported grade corrections.
            if final!=previous['grade']:
                r=row_for(item,note,fid,final);r['adjudication_reason']=decision['reason'];r['previous_grade']=previous['grade']
                observations[item['part_id'],fid]=r
                changes.append(dict(review_id=code,part_id=item['part_id'],family_id=fid,inspection_id=item['inspection_id'],source_sha256=item['source_sha256'],previous_grade=previous['grade'],revised_grade=final,previous_observation=previous['observation'],observation=note['observation'],reason=decision['reason'],geometry_evidence=item['geometry_evidence']))
    for item in nm:
        note=added[item['review_id']]
        for fid in ROLE.values():
            observations[item['part_id'],fid]=row_for(item,note,fid,note['grades'].get(fid,0))
    data['judgments']=[observations[k] for k in sorted(observations)]
    if any(r['grade'] is None for r in data['judgments']):raise ValueError('Unresolved qrels: cannot publish complete closed-gallery evaluation')
    data.update(parent_annotation_sha256=sha256(PARENT),quality_audit='Full integrity audit; new 100-CAD probability rereview plus 35 shared-rationale outliers with explicit disagreement adjudication; all remaining 466 eligible CADs visually assessed. AI-only, shared context.',release='intent-candidates1816-v4')
    write_json(ANN/'judgments.json',data);write_csv(ANN/'corrections.csv',changes);write_csv(ROOT/'audit_review_pairs.csv',review_pairs)
    # Matched cohort counterfactual changes labels only, with no new experiment configurations.
    corrected=deepcopy(data);corrected['judgments']=[r for r in data['judgments'] if (r['part_id'],r['family_id']) in old]
    write_json(ANN/'corrected_parent1350.json',corrected)
    bycad=defaultdict(list)
    for r in review_pairs:
        if r['component']=='candidate_srs':bycad[r['part_id']].append(int(r['changed']))
    values=np.asarray([np.mean(v) for v in bycad.values()]);rng=np.random.default_rng(freeze['seed'])
    ci=np.quantile(rng.choice(values,(20000,len(values))).mean(1),[.025,.975])
    primary=[r for r in review_pairs if r['component']=='candidate_srs']
    report=dict(audit_complete=True,final_annotation_sha256=sha256(ANN/'judgments.json'),parent_annotation_sha256=sha256(PARENT),existing_records=len(old),existing_cads=1350,new_cads=466,new_records=5126,final_cads=1816,final_records=len(data['judgments']),fresh_audit_cads=100,fresh_audit_pairs=len(primary),supplemental_cads=35,total_rereviewed_pairs=len(review_pairs),blind_disagreements=len(proposed),corrections=len(changes),primary_corrections=sum(r['changed'] for r in primary),adjudicated_change_fraction=float(values.mean()),cad_cluster_bootstrap_ci95=ci.tolist(),by_previous_grade={g:dict(reviewed=sum(r['previous_grade']==g for r in primary),changed=sum(r['previous_grade']==g and r['changed'] for r in primary)) for g in [0,1,2]},human_expert_reviewed=False,unresolved_records=0,limitations='Shared-model rereview and benchmark-conditional assessment; neither independent annotation accuracy nor unseen-task generalization.',frozen_before_scoring=True,freeze_sha256=sha256(ROOT/'freeze.json'),notes_sha256={str(p):sha256(p) for p in [ROOT/'audit_notes.txt',ROOT/'supplemental_notes.txt',ROOT/'adjudication.json',ANN/'visual_notes.txt']})
    write_json(ROOT/'quality_gate.json',report);print(report)

def evidence_overlay():
    """Correct source observations across all roles without changing frozen grades.

    Authored source observations predate scoring. This supplement was packaged
    during report finalization, after numeric evaluation; never claim otherwise.
    """
    path=ANN/'judgments.json';data=read_json(path)
    notes={**read_notes(ROOT/'audit_notes.txt'),**read_notes(ROOT/'supplemental_notes.txt')}
    manifest={r['part_id']:r for r in read_csv(ROOT/'audit_manifest.csv')+read_csv(ROOT/'supplemental_manifest.csv')}
    corrected_ids={r['part_id'] for r in read_csv(ANN/'corrections.csv')}
    tasks={r['family_id']:r for r in read_json('annotations/review_tasks.json')['tasks']}
    repairs=[]
    for r in data['judgments']:
        if r['part_id'] not in corrected_ids:continue
        item=manifest[r['part_id']];note=notes[item['review_id']]
        if r['observation']==note['observation']:continue
        term={0:'reject',1:'conditional',2:'shortlist'}[r['grade']]
        repairs.append(dict(family_id=r['family_id'],part_id=r['part_id'],inspection_id=r['inspection_id'],source_sha256=r['source_sha256'],frozen_grade=r['grade'],previous_observation=r['observation'],revised_observation=note['observation'],previous_rationale=r['rationale'],revised_rationale=term.capitalize()+': '+tasks[r['family_id']]['rules'][term]+' Evidence: '+note['observation'],review_id=item['review_id'],reason='Apply the pre-scoring source-specific adjudicated observation consistently to every role on this CAD; no grade change.'))
    write_json(ANN/'evidence_corrections.json',dict(applies_to_annotation_sha256=sha256(path),scope='Observation/rationale metadata only; every frozen numerical grade is unchanged.',packaging_stage='After numeric evaluation, from source-view notes authored before scoring.',source_notes_sha256={str(p):sha256(p) for p in [ROOT/'audit_notes.txt',ROOT/'supplemental_notes.txt']},corrections=repairs))
    print(dict(provenance_only_repairs=len(repairs),cads=len(corrected_ids),numeric_labels_unchanged=True))

def resolved_records(path):
    """Apply the explicit evidence supplement for display, keeping scored qrels."""
    path=Path(path);data=read_json(path);sidecar=path.parent/'evidence_corrections.json'
    if not sidecar.exists():return data['judgments']
    overlay=read_json(sidecar)
    if sha256(path)!=overlay['applies_to_annotation_sha256']:raise ValueError('Evidence supplement binding mismatch')
    rows={(r['family_id'],r['part_id']):r for r in data['judgments']}
    for fix in overlay['corrections']:
        row=rows[fix['family_id'],fix['part_id']]
        if row['grade']!=fix['frozen_grade'] or row['source_sha256']!=fix['source_sha256'] or row['observation']!=fix['previous_observation']:raise ValueError('Evidence supplement changes grade/source or has stale provenance')
        row.update(observation=fix['revised_observation'],rationale=fix['revised_rationale'],evidence_supplement=str(sidecar))
    return list(rows.values())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compare','publish','evidence-overlay']);args=p.parse_args()
    if args.action=='compare':compare()
    elif args.action=='publish':publish()
    else:evidence_overlay()
