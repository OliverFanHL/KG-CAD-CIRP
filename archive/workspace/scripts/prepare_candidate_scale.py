"""Freeze candidate-only expansion and prepare neutral evidence; never infer labels."""
from __future__ import annotations
import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cadkg.io import read_json, read_csv, write_json, write_csv, sha256

ROOT = Path('outputs/candidate_scale_v4')
PARENT = Path('annotations/intent_batch2_audited_v2/judgments.json')
ANN = Path('annotations/intent_candidates1816_v4')

def sheets(items, prefix):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    for start in range(0, len(items), 6):
        page = Image.new('RGB', (1800,1200), 'white')
        for j, item in enumerate(items[start:start+6]):
            with Image.open(item['geometry_evidence']) as src:
                src = src.convert('RGB')
                x, y = j%2*900, j//2*400
                ImageDraw.Draw(page).text((x+8,y), item['review_id'], font=font, fill='black')
                for k in range(3):
                    crop = src.crop((k*500,55,(k+1)*500,src.height))
                    yy,xx = np.where(np.any(np.asarray(crop)<230, axis=2))
                    if len(xx):
                        crop = crop.crop((max(0,xx.min()-5),max(0,yy.min()-5),min(crop.width,xx.max()+6),min(crop.height,yy.max()+6)))
                    crop.thumbnail((285,340))
                    page.paste(crop,(x+k*300+(300-crop.width)//2,y+40+(350-crop.height)//2))
        page.save(ROOT/f'{prefix}_{start//6+1:02}.png')

def prepare():
    ROOT.mkdir(exist_ok=True); ANN.mkdir(exist_ok=True); (ROOT/'evidence').mkdir(exist_ok=True)
    source = Path('outputs/intent_batch2_matrix_reproduction')
    protocol = read_json(source/'protocol.json'); parent=read_json(PARENT)
    catalog=read_json(source/'bundle/catalog.json'); parts={r['part_id']:r for r in catalog['parts']}
    old={}; pairs=set(); errors=[]; flags=[]
    tasks={r['family_id']:r for r in read_json('annotations/review_tasks.json')['tasks']}
    for row in parent['judgments']:
        pid,fid=row['part_id'],row['family_id']; key=(fid,pid)
        if key in pairs:errors.append(f'duplicate {key}')
        pairs.add(key);old.setdefault(pid,row)
        if fid not in tasks or type(row['grade']) is not int or row['grade'] not in (0,1,2):errors.append(f'invalid {key}')
        if row['decision']!={0:'reject',1:'conditional',2:'shortlist'}[row['grade']]:errors.append(f'decision {key}')
        if row['review_status']!='visually_reviewed' or not row['observation'] or not row['rationale']:errors.append(f'incomplete {key}')
        if row['source_sha256']!=parts[pid]['file_sha256'] or parts[pid]['split']!='test':errors.append(f'source/split {key}')
        if row['source']!='ai_reference' or row['human_expert_reviewed'] is not False:errors.append(f'provenance {key}')
        if row['observation'] not in row['rationale']:flags.append(dict(part_id=pid,family_id=fid,flag='observation_not_in_rationale'))
        if not Path(row['geometry_evidence']).is_file():errors.append(f'missing image {key}')
    if len(pairs)!=len(old)*len(tasks):errors.append('incomplete task matrix')
    for pid,row in old.items():
        if sha256(parts[pid]['absolute_path'])!=row['source_sha256']:errors.append(f'raw bytes {pid}')
        path=Path('outputs/intent_batch2_expanded_2x/evidence')/f"{row['inspection_id']}.json"
        e=read_json(path)
        if e['part_id']!=pid or e['source_sha256']!=row['source_sha256']:errors.append(f'evidence {pid}')
        shutil.copyfile(path,ROOT/'evidence'/path.name)
    additions=sorted(set(protocol['gallery'])-set(old))
    rng=np.random.default_rng(20261021+4)
    sampled=sorted(rng.choice(sorted(old),100,replace=False).tolist()); rng.shuffle(sampled)
    audit=[dict(review_id=f'R{i:03d}',part_id=pid,inspection_id=old[pid]['inspection_id'],source_sha256=old[pid]['source_sha256'],geometry_evidence=old[pid]['geometry_evidence'],component='candidate_srs') for i,pid in enumerate(sampled,1)]
    rng.shuffle(additions)
    new=[dict(review_id=f'V{i:04d}',inspection_id=f'V{i:04d}',part_id=pid,source_sha256=parts[pid]['file_sha256'],geometry_evidence=str(ROOT/'evidence'/f'V{i:04d}.png'),component='added_all_remaining') for i,pid in enumerate(additions,1)]
    if errors:raise ValueError(errors[:20])
    write_csv(ROOT/'audit_manifest.csv',audit);write_csv(ROOT/'added_manifest.csv',new)
    write_csv(ROOT/'mechanical_flags.csv',flags,fields=['part_id','family_id','flag'])
    write_json(ROOT/'mechanical_audit.json',dict(records=len(pairs),cads=len(old),source_files_rehashed=len(old),complete_task_matrix=True,errors=errors,flags=len(flags),grade_counts=dict(Counter(r['grade'] for r in parent['judgments'])),scope='Integrity and rationale linkage; not semantic correctness'))
    inputs=[PARENT,'configs/kg_queries110_v3.json','annotations/intent_queries110_v3/queries.json','annotations/intent_queries110_v3/protocol_extension.json',source/'protocol.json',source/'bundle/catalog.json',source/'bundle/graph.json',source/'bundle/knowledge.json','annotations/review_tasks.json','docs/CANDIDATE_SCALE_EVALUATION.md']
    write_json(ROOT/'freeze.json',dict(created_date='2026-10-21',stage='before_new_annotation_and_effectiveness',seed=20261025,parent_cads=len(old),added_cads=len(new),target_cads=len(protocol['gallery']),queries=110,methods=14,representations=2,audit_srs=100,audit_inclusion_probability=100/len(old),inputs={str(p):sha256(p) for p in inputs},added_manifest_sha256=sha256(ROOT/'added_manifest.csv'),audit_manifest_sha256=sha256(ROOT/'audit_manifest.csv'),limitations='Historical performance known; same-model/shared-context AI review, not independent expert validation.'))
    sheets(audit,'audit')
    print(dict(existing=len(old),added=len(new),audit=len(audit),flags=len(flags)),flush=True)

def render():
    from cadkg.experiments.intent import evidence
    catalog=read_json('outputs/intent_batch2_matrix_reproduction/bundle/catalog.json')
    items=read_csv(ROOT/'added_manifest.csv')
    evidence(ROOT/'evidence',items,catalog,reuse='outputs/full_scale',workers=4)
    sheets(items,'added')

def supplemental():
    """Replay the pre-scoring rationale diagnostic; categories flag, never grade."""
    rows=read_json(PARENT)['judgments'];by={r['part_id']:r for r in rows}
    parts={p['part_id']:p for p in read_json('outputs/intent_batch2_matrix_reproduction/bundle/catalog.json')['parts']}
    groups=defaultdict(list)
    for r in by.values():groups[r['observation']].append(r)
    flags=[]
    for group in groups.values():
        if len(group)<3:continue
        majority=Counter(parts[r['part_id']]['label'] for r in group).most_common(1)[0][0]
        for r in group:
            if parts[r['part_id']]['label']!=majority:
                flags.append(dict(part_id=r['part_id'],inspection_id=r['inspection_id'],geometry_evidence=r['geometry_evidence'],observation=r['observation'],category=parts[r['part_id']]['label']))
    sampled={r['part_id'] for r in read_csv(ROOT/'audit_manifest.csv')}
    items=[dict(review_id=f'Q{i:03d}',part_id=r['part_id'],inspection_id=r['inspection_id'],source_sha256=by[r['part_id']]['source_sha256'],geometry_evidence=r['geometry_evidence'],component='rationale_outlier') for i,r in enumerate([r for r in flags if r['part_id'] not in sampled],1)]
    write_csv(ROOT/'rationale_outlier_diagnostics.csv',flags);write_csv(ROOT/'supplemental_manifest.csv',items)
    sheets(items,'supplemental');print(dict(flags=len(flags),additional_cads=len(items)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render','sheets','supplemental']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='render':render()
    elif args.action=='supplemental':supplemental()
    else:sheets(read_csv(ROOT/'added_manifest.csv'),'added')
