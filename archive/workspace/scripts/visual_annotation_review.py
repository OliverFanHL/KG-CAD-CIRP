"""Neutral contact sheets and packaging for explicitly authored CAD reviews.

No category, classifier, ranking or method is used to assign a grade. Review
notes are supplied by the reviewer after viewing each neutral contact sheet.
An omitted role in a reviewed note means an explicitly reviewed wrong form;
an uncertainty must be supplied as a null grade, never silently made zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from cadkg.io import read_csv, read_json, write_csv, write_json, sha256

ROOT = Path('outputs/intent_batch2_expanded_2x')
OUT = Path('outputs/annotation_verification')
NOTES = Path('annotations/intent_batch2_verified/visual_notes.txt')
ROLE = dict(S='support_body',C='shaft_connector',B='belt_wheel',T='threaded_clamp',
            N='nut_retention',M='bracket_mount',F='flanged_passage',G='gear_mesh',
            D='shaft_drive',P='compression_return',E='tension_return')


def sheets():
    OUT.mkdir(exist_ok=True)
    items = read_csv(ROOT/'pool_manifest.csv')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    for start in range(0,len(items),30):
        page = Image.new('RGB',(1800,2300),'white')
        for index,item in enumerate(items[start:start+30]):
            col,row = index%3,index//3
            x,y = col*600,row*230
            path = ROOT/'evidence'/f"{item['inspection_id']}.png"
            with Image.open(path) as source:
                crop = source.crop((0,48,source.width,source.height))
                page.paste(crop.resize((600,205)),(x,y+25))
            ImageDraw.Draw(page).text((x+8,y),item['inspection_id'],font=font,fill='black')
        page.save(OUT/f'page_{start//30+1:02}.png')
    print({'pages':45,'candidates':len(items)})


def assemble():
    notes = {}
    for line in NOTES.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        codes,marks,observation = line.split('|',2)
        if observation.startswith('@'):
            observation = notes[observation[1:]]['observation']
        grades = {ROLE[mark[0]]:(None if mark[1]=='?' else int(mark[1])) for mark in marks.split(',') if mark}
        for code in codes.split(','):
            if code in notes:
                raise ValueError(f'Duplicate visual note: {code}')
            notes[code] = dict(grades=grades,observation=observation)
    override_path = NOTES.parent/'enlarged_review.json'
    if override_path.exists():
        for item in read_json(override_path):
            if item['inspection_id'] not in notes:
                raise ValueError('Enlarged review requires an existing individual review')
            notes[item['inspection_id']] = dict(
                grades={ROLE[m[0]]:int(m[1]) for m in item['marks'].split(',') if m},
                observation=item['observation'], enlarged=True)
    manifest = {r['inspection_id']:r for r in read_csv(ROOT/'pool_manifest.csv')}
    tasks = {r['family_id']:r for r in read_json('annotations/review_tasks.json')['tasks']}
    protocol = read_json(ROOT/'protocol.json')
    previous = read_json('annotations/intent_batch2/judgments.json')
    prior = {(r['family_id'],r['part_id']):r for r in previous['judgments']}
    reviewed, changes = {}, []
    page_by_code = {code:f'{i//30+1:02}' for i,code in enumerate(manifest)}
    for code,note in notes.items():
        if code not in manifest:
            raise ValueError(f'Unknown CAD {code}')
        item = manifest[code]
        evidence_path = ROOT/'evidence'/f'{code}.json'
        evidence = read_json(evidence_path)
        if evidence['part_id'] != item['part_id'] or evidence['source_sha256'] != item['source_sha256']:
            raise ValueError(f'Source binding mismatch {code}')
        if set(note['grades'])-set(tasks):
            raise ValueError(f'Unknown task in {code}')
        for fid,task in tasks.items():
            grade = note['grades'].get(fid,0)
            if grade is not None and (type(grade) is not int or grade not in (0,1,2)):
                raise ValueError(f'Invalid grade {code}/{fid}')
            decision = {0:'reject',1:'conditional',2:'shortlist',None:'cannot_assess'}[grade]
            if grade == 2:
                reason = f"Shortlist: {task['rules']['shortlist']}"
            elif grade == 1:
                reason = f"Conditional: {task['rules']['conditional']}"
            elif grade is None:
                reason = 'Cannot assess from these source views; the decisive shape or interface is concealed or indistinguishable.'
            else:
                reason = f"Reject: the observed form conflicts with this task's required geometry. Required form: {task['rules']['shortlist']}"
            reason += ' Evidence: '+note['observation']
            if fid in note.get('reasons',{}):
                reason = note['reasons'][fid]+' Evidence: '+note['observation']
            row = dict(family_id=fid,part_id=item['part_id'],inspection_id=code,
                       source_sha256=item['source_sha256'],grade=grade,decision=decision,
                       observation=note['observation'],rationale=reason,
                       geometry_evidence=str(ROOT/'evidence'/f'{code}.png'),
                       source='ai_reference',human_expert_reviewed=False,
                       reviewer='Codex AI source-view reviewer',review_status='visually_reviewed',
                       inspection_origin='Individual three-view source-CAD review; neutral page '+page_by_code[code],
                       enlarged_review=note.get('enlarged',False))
            old = prior.get((fid,item['part_id']))
            if old and old['grade'] != grade:
                changes.append(dict(family_id=fid,part_id=item['part_id'],inspection_id=code,
                                    previous_grade=old['grade'],revised_grade=grade,rationale=reason))
            reviewed[fid,item['part_id']] = row
    rows = []
    for item in manifest.values():
        for fid in tasks:
            key = fid,item['part_id']
            if key in reviewed:
                rows.append(reviewed[key])
            elif key in prior:
                rows.append(dict(prior[key],review_status='prior_ai_inspection_not_yet_reverified'))
            else:
                rows.append(dict(family_id=fid,part_id=item['part_id'],inspection_id=item['inspection_id'],
                                 source_sha256=item['source_sha256'],grade=None,decision=None,
                                 observation='',rationale='',review_status='unreviewed',human_expert_reviewed=False))
    data = dict(source='ai_reference',human_expert_reviewed=False,
                query_sha256=protocol['query_sha256'],scope_sha256=protocol['scope_sha256'],
                reviewer='Codex AI source-view reviewer',judgments=rows,
                disclosure='AI visual source-CAD assessment under the unchanged 11 rubrics. Historical AI judgments remain separately identifiable. No grades are assigned from catalog labels or retrieval scores.')
    destination = NOTES.parent/'judgments.json'
    write_json(destination,data)
    write_json(NOTES.parent/'corrections.json',changes)
    counts = dict(target_records=len(rows),visually_reviewed_records=len(reviewed),
                  visually_reviewed_cads=len(notes),corrections_to_prior=len(changes),
                  complete_records=sum(r['review_status']!='unreviewed' for r in rows),
                  unreviewed_records=sum(r['review_status']=='unreviewed' for r in rows),
                  cannot_assess_records=sum(r['review_status']=='visually_reviewed' and r['grade'] is None for r in rows),
                  numeric_grades=sum(r['grade'] is not None for r in rows),
                  matrix_changed=False)
    write_json(NOTES.parent/'coverage.json',counts)
    print(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['sheets','assemble'])
    args = parser.parse_args()
    sheets() if args.action=='sheets' else assemble()
