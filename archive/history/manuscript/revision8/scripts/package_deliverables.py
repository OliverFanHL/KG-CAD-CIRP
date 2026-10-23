"""Package the visually reviewed Word documents and scientific figure assets."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
report=json.loads((ROOT/'provenance/final_validation.json').read_text())
assert report['status']=='PASS'
review=json.loads((ROOT/'provenance/visual_review.json').read_text())
assert review['status']=='VISUALLY_REVIEWED'
for r in report['documents']:assert sha(ROOT/(r['document']+'.docx'))==r['docx_sha256']

def archive(path,files):
    records=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    with ZipFile(path,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p,r in zip(files,records):z.write(p,r['path'])
        z.writestr('archive_manifest.json',json.dumps(records,indent=2))
    with ZipFile(path) as z:
        assert z.testzip() is None
        for r in records:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    return dict(path=path.name,bytes=path.stat().st_size,files=len(records)+1,sha256=sha(path))

figfiles=sorted(list((ROOT/'figures').glob('figure[23]*'))+
    [ROOT/'scripts/generate_figures.py',ROOT/'scripts/generate_revision7_assets.py',
     ROOT/'requirements.txt',ROOT/'README.md',ROOT/'provenance/figure_generation.json',
     ROOT/'data/summary.csv',ROOT/'data/family_metrics.csv',ROOT/'data/figure_annotations.json'])
figzip=archive(ROOT/'updated_figures.zip',figfiles)
exclude={'.qa','__pycache__'}
allfiles=sorted(p for p in ROOT.rglob('*') if p.is_file() and not(set(p.relative_to(ROOT).parts)&exclude)
    and p.suffix.lower() in {'.docx','.pptx','.png','.svg','.pdf','.tex','.bib','.bbl','.py','.ps1','.csv','.json','.md','.txt'}
    and p.name!='deliverables.json')
complete=archive(ROOT/'complete_revision8_package.zip',allfiles)
final=dict(revision=8,manuscript='manuscript.docx',supplement='supplement.docx',
    main_pages=next(d['pages'] for d in report['documents'] if d['document']=='manuscript'),
    supplement_pages=next(d['pages'] for d in report['documents'] if d['document']=='supplement'),
    formulas='Editable LaTeX text; zero rendered equation objects',table_style='Three-line academic style',
    source_results_unchanged=True,
    figure_assets=figzip,complete_source_package=complete)
write(ROOT/'deliverables.json',final)
print(json.dumps(final,indent=2))
