"""Package revision 5 and the complete newly collected GPT study assets.

No inference or metric evaluation. Original CAD-KG source data dependencies
remain identified by the fixed protocol; the experiment archive contains all
new API inputs/outputs and analysis scripts, not another copy of TMCAD.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
STUDY=REPO/'outputs/llm_geometry_matched_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def clean_file(path):
    return not set(path.relative_to(PAPER).parts)&{'.tools','.qa','.dependencies','__pycache__'} and path.suffix.lower() in {
      '.tex','.pdf','.bib','.bbl','.md','.py','.json','.jsonl','.csv','.npz','.png','.svg','.txt'} and path.name not in {'revise_visual_llm_manuscript.py','package_manifest.json'}

def zip_verified(target, entries):
    names=[name for _,name in entries]
    assert len(names)==len(set(names))
    secret=re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}')
    def text_has_secret(value):
        if isinstance(value,dict):
            # Opaque API-encrypted model content can randomly contain key-like
            # base64 substrings. Preserve it byte-for-byte; never decode it.
            return any(text_has_secret(v) for k,v in value.items() if k!='encrypted_content')
        if isinstance(value,list):return any(text_has_secret(v) for v in value)
        return isinstance(value,str) and bool(secret.search(value.encode()))
    for path,_ in entries:
        assert path.is_file() and path.suffix.lower()!='.env'
        if path.suffix.lower()=='.jsonl':
            assert not any(text_has_secret(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()),f'Credential-like plaintext requires review: {path.name}'
        elif path.suffix.lower()=='.json':
            assert not text_has_secret(read(path)),f'Credential-like plaintext requires review: {path.name}'
        elif path.suffix.lower() in {'.py','.md','.txt','.tex','.csv'}:
            assert not secret.search(path.read_bytes()),f'Credential-like text requires review: {path.name}'
    assert target.resolve().is_relative_to((REPO/'manuscript').resolve())
    temporary=target.with_suffix('.zip.inprogress')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in entries:
            z.write(path,name)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for path,name in entries:
            assert hashlib.sha256(z.read(name)).hexdigest()==sha(path),name
    temporary.replace(target)
    return dict(path=target.relative_to(REPO).as_posix(),files=len(entries),bytes=target.stat().st_size,sha256=sha(target))

def main():
    result=read(STUDY/'results.json');assert result['gate']['passed']
    assert read(STUDY/'independent_replay_audit.json')['status']=='PASS'
    # Refresh interpretation-only audit snapshots after the missing-face cause
    # was corrected; no response, ranking, metric, test or frozen input changes.
    data=PAPER/'data/llm_geometry';snap=read(data/'snapshot_manifest.json')
    for name in ('post_execution_audit.md','post_execution_audit.json'):
        shutil.copyfile(STUDY/name,data/name)
        snap['copied_top_level_files'][name]['sha256']=sha(data/name)
        snap['copied_top_level_files'][name]['bytes']=(data/name).stat().st_size
    write(data/'snapshot_manifest.json',snap)
    pub=read(PAPER/'provenance/llm_geometry_publication.json')
    pub['snapshot_manifest_sha256']=sha(data/'snapshot_manifest.json')
    write(PAPER/'provenance/llm_geometry_publication.json',pub)
    subprocess.run([sys.executable,str(PAPER/'scripts/validate_package.py')],check=True)
    source=(PAPER/'main.tex').read_text(encoding='utf-8');sup=(PAPER/'supplement.tex').read_text(encoding='utf-8')
    files=sorted(p for p in PAPER.rglob('*') if p.is_file() and clean_file(p))
    manifest=dict(title='Knowledge-guided visual evidence for CAD retrieval with large language models',revision=5,
        prepared_date='2026-10-22',main_pdf_pages=6,supplement_pdf_pages=7,
        main_figures=len(re.findall(r'\\begin\{figure\*?\}',source)),
        main_tables=len(re.findall(r'\\begin\{table\*?\}',source)),
        supplement_figures=len(re.findall(r'\\begin\{figure\*?\}',sup)),
        supplement_tables=len(re.findall(r'\\begin\{table\*?\}',sup)),
        cited_references=len(re.findall(r'\\bibitem\{',(PAPER/'references.bbl').read_text())),
        model='gpt-6-luna',completed_role_cad_jobs=2084,
        estimated_api_cost_usd=read(STUDY/'evidence_completion.json')['estimated_cost_usd'],
        publication_gate_passed=True,primary_queries=264,gallery=1816,representations=2,methods=3,
        qrels_unchanged=True,expert_reviewed=False,
        prior_revision_archive='manuscript/revision_archive/revision_4_before_visual_llm',
        files=[dict(path=p.relative_to(PAPER).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(PAPER/'package_manifest.json',manifest)
    paper_entries=[(p,'cirp_draft/'+p.relative_to(PAPER).as_posix()) for p in files]
    paper_entries.append((PAPER/'package_manifest.json','cirp_draft/package_manifest.json'))
    paper=zip_verified(REPO/'manuscript/cirp_annals_manuscript_package.zip',paper_entries)
    experiment_entries=[]
    for p in sorted(STUDY.rglob('*')):
        if p.is_file() and 'independent_replay' not in p.relative_to(STUDY).parts and not p.name.startswith('.'):
            experiment_entries.append((p,p.relative_to(REPO).as_posix()))
    wanted=[p for p in (REPO/'scripts').glob('*.py') if 'llm_geometry' in p.name or p.name=='neutralize_llm_cad_views.py']
    wanted+=[REPO/'docs/LLM_GEOMETRY_MATCHED_V1.md']
    for p in wanted:experiment_entries.append((p,p.relative_to(REPO).as_posix()))
    study=zip_verified(REPO/'manuscript/llm_geometry_matched_experiment_package.zip',experiment_entries)
    output=dict(manuscript=paper,experiment=study,credentials_included=False,
       experiment_dependency_scope='All new CAD-view inputs, API outputs, fixed protocols, metrics and new scripts; full metric recomputation uses the original CAD-KG source workspace named by the protocol hashes.',
       portable_plotting='Manuscript package redraws quantitative figures from its own snapshots without original dataset, network or API.')
    write(REPO/'manuscript/revision_5_deliverables.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
