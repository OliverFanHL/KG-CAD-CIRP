"""Prepare enlarged evidence for reviewing annotation disagreements."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from cadkg.io import read_json

out=Path('outputs/annotation_verification')
codes=list(dict.fromkeys(r['inspection_id'] for r in read_json('annotations/intent_batch2_verified/corrections.json')))
codes+=['X0653','X0356']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
for start in range(0,len(codes),6):
    page=Image.new('RGB',(1800,1200),'white')
    for j,code in enumerate(codes[start:start+6]):
        src=Image.open(f'outputs/intent_batch2_expanded_2x/evidence/{code}.png').convert('RGB')
        regions=[]
        for k in range(3):
            crop=src.crop((k*500,55,(k+1)*500,src.height))
            mask=crop.point(lambda p:255 if p<245 else 0).convert('L')
            box=mask.getbbox()
            if box: crop=crop.crop(box)
            crop.thumbnail((280,340))
            regions.append(crop)
        x,y=(j%2)*900,(j//2)*400
        ImageDraw.Draw(page).text((x+8,y),code,font=font,fill='black')
        for k,crop in enumerate(regions):page.paste(crop,(x+k*300+(300-crop.width)//2,y+40+(350-crop.height)//2))
    page.save(out/f'disagreement_{start//6+1:02}.png')
print(len(codes))
