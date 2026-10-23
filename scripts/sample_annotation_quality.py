"""Probability audit sample and neutral enlarged source-view sheets."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cadkg.io import read_json,write_json,write_csv,sha256

OUT=Path('outputs/annotation_quality_audit_v2')
SOURCE=Path('annotations/intent_batch2_verified/judgments.json')

def main():
    OUT.mkdir(exist_ok=True)
    rows=read_json(SOURCE)['judgments'];byid=defaultdict(list)
    for r in rows:byid[r['part_id']].append(r)
    rng=np.random.default_rng(20261021)
    primary=sorted(rng.choice(sorted(byid),100,replace=False).tolist())
    strata=defaultdict(list)
    for r in rows:
        if r['part_id'] not in primary:strata[r['family_id'],r['grade']].append(r)
    targeted=[]
    for (fid,grade),group in sorted(strata.items()):
        selected=rng.choice(len(group),min(2,len(group)),replace=False)
        for i in selected:targeted.append(dict(family_id=fid,grade_stratum=grade,part_id=group[i]['part_id'],stratum_population=len(group)))
    ids=sorted(set(primary)|{r['part_id'] for r in targeted});rng.shuffle(ids)
    manifest=[]
    for i,pid in enumerate(ids,1):
        source=byid[pid][0]
        manifest.append(dict(audit_id=f'A{i:03}',part_id=pid,source_inspection_id=source['inspection_id'],
          source_sha256=source['source_sha256'],geometry_evidence=source['geometry_evidence'],
          sample_component='primary_srs' if pid in primary else 'supplemental',
          targeted_families='|'.join(r['family_id'] for r in targeted if r['part_id']==pid)))
    write_json(OUT/'sampling_plan.json',dict(seed=20261021,annotation_sha256=sha256(SOURCE),population_cads=len(byid),
        primary_srs_cads=primary,primary_cad_inclusion_probability=100/len(byid),primary_record_count=1100,
        supplementary_pairs=targeted,unique_review_cads=len(ids),
        inference='Use candidate-level SRS for population error estimates; supplementary grade-stratified cases reported separately.',
        gate='Do not expand experiments until sampled reviews, corrections and unresolved risks are reported. Same-model review does not establish independent human validity.'))
    write_csv(OUT/'sample_manifest.csv',manifest)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    for start in range(0,len(manifest),6):
        page=Image.new('RGB',(1800,1200),'white')
        for j,item in enumerate(manifest[start:start+6]):
            src=Image.open(item['geometry_evidence']).convert('RGB');x,y=(j%2)*900,(j//2)*400
            ImageDraw.Draw(page).text((x+8,y),item['audit_id'],font=font,fill='black')
            for k in range(3):
                crop=src.crop((k*500,55,(k+1)*500,src.height))
                pixels=np.asarray(crop);mask=np.any(pixels<230,axis=2)
                yy,xx=np.where(mask)
                if len(xx):crop=crop.crop((max(0,xx.min()-5),max(0,yy.min()-5),min(crop.width,xx.max()+6),min(crop.height,yy.max()+6)))
                crop.thumbnail((285,340));page.paste(crop,(x+k*300+(300-crop.width)//2,y+40+(350-crop.height)//2))
        page.save(OUT/f'blind_{start//6+1:02}.png')
    print({'primary_cads':len(primary),'supplementary_pairs':len(targeted),'unique_cads':len(ids),'pages':(len(ids)+5)//6})

if __name__=='__main__':main()
