"""Package the completed, audited three-collection paper and reproducible data."""
from pathlib import Path
import argparse,importlib.util,json,re,subprocess,sys
REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
STUDY=REPO/'outputs/llm_robustness_v7'
spec=importlib.util.spec_from_file_location('package_helpers',REPO/'scripts/package_llm_geometry_deliverables.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation-python',default=sys.executable,
        help='Python environment containing the figure requirements; defaults to this interpreter.')
    args=parser.parse_args()
    result=h.read(STUDY/'results.json');audit=h.read(STUDY/'post_execution_audit.json')
    assert result['primary_ndcg_identified']
    assert str(audit.get('status','')).upper().startswith('PASS')
    assert h.read(STUDY/'independent_replay_audit.json')['status']=='PASS'
    visual=h.read(PAPER/'provenance/revision7_pdf_review.json')
    assert visual['review_status']=='VISUALLY_REVIEWED', 'Inspect every rendered manuscript page before packaging'
    for name in ('main','supplement'):
        entry=visual['documents'][name]
        assert entry['pdf_sha256']==h.sha(PAPER/(name+'.pdf'))
        assert entry['tex_sha256']==h.sha(PAPER/(name+'.tex'))
        assert not entry['layout_warnings'], entry['layout_warnings']
    subprocess.run([args.validation_python,str(PAPER/'scripts/validate_package.py')],check=True)
    from pypdf import PdfReader
    mainpages=len(PdfReader(PAPER/'main.pdf').pages);suppages=len(PdfReader(PAPER/'supplement.pdf').pages)
    allowed={'.tex','.pdf','.bib','.bbl','.md','.py','.json','.jsonl','.csv','.npz','.png','.svg','.txt','.pptx','.mjs'}
    exclude={'.tools','.qa','.dependencies','__pycache__','cad_reproduction','publication_v7'}
    files=sorted(p for p in PAPER.rglob('*') if p.is_file() and p.suffix.lower() in allowed and
        not set(p.relative_to(PAPER).parts)&exclude and p.name not in {'revise_visual_llm_manuscript.py','package_manifest.json'})
    text=(PAPER/'main.tex').read_text(encoding='utf8');sup=(PAPER/'supplement.tex').read_text(encoding='utf8')
    manifest=dict(revision=7,title='Knowledge-guided relational evidence for functional CAD retrieval',
        prepared_date='2026-10-22',main_pdf_pages=mainpages,supplement_pdf_pages=suppages,
        main_figures=len(re.findall(r'\\begin\{figure\*?\}',text)),main_tables=len(re.findall(r'\\begin\{table\*?\}',text)),
        supplement_tables=len(re.findall(r'\\begin\{table\*?\}',sup)),cited_references=19,
        models=['gpt-6-luna','gpt-6.1-sol'],response_collections_per_arm=3,
        new_calls=14588,historical_KG_calls_reused=4168,API_sampling_seed_supported=False,
        local_order_seeds=[17,29,43],encoder_checkpoint_fixed=True,
        updated_primary_source='data/robustness_v7/summary.csv',
        files=[dict(path=p.relative_to(PAPER).as_posix(),bytes=p.stat().st_size,sha256=h.sha(p)) for p in files])
    h.write(PAPER/'package_manifest.json',manifest)
    paper=h.zip_verified(REPO/'manuscript/cirp_annals_manuscript_package.zip',
        [(p,'cirp_draft/'+p.relative_to(PAPER).as_posix()) for p in files]+[(PAPER/'package_manifest.json','cirp_draft/package_manifest.json')])
    entries=[]
    for name in ('llm_geometry_matched_v1','llm_geometry_sol_matched_v1','llm_robustness_v7',
                 'robustness_local_v7','robustness_remaining_local_v7'):
        root=REPO/'outputs'/name
        for p in sorted(root.rglob('*')):
            parts=p.relative_to(root).parts
            if p.is_file() and 'independent_replay' not in parts and '__pycache__' not in parts and not p.name.startswith('.'):
                entries.append((p,p.relative_to(REPO).as_posix()))
    names={p for p in (REPO/'scripts').glob('*.py') if 'llm_geometry' in p.name or 'llm_robustness' in p.name or p.name in {
        'neutralize_llm_cad_views.py','repeat_remaining_local_candidates_v7.py',
        'finish_revision7_pipeline.py','integrate_revision7_assets.py',
        'revise_manuscript_v7.py','render_revision7_manuscript.py','package_revision7_deliverables.py'}}
    for p in sorted(names):entries.append((p,p.relative_to(REPO).as_posix()))
    for p in sorted((REPO/'docs').glob('revision7*.md')):entries.append((p,p.relative_to(REPO).as_posix()))
    complete=h.zip_verified(REPO/'manuscript/repeated_observation_experiment_package.zip',entries)
    final=dict(manuscript=paper,repeated_observation_study=complete,credentials_included=False,
        API_new_requests=14588,API_historical_requests_reused=4168,collection_repeats_per_arm=3,
        scope='Three repeated API acquisitions conditional on fixed encoder. Local seeds control query/job order only; no API sampling seeds or repeated encoder training.',
        experiments='Same 264 queries, 11 roles, 1,816-CAD gallery, 2,084 role-CAD pairs per collection and 528 candidate pools.',
        runtime_dependency='Complete saved responses/inputimages and analytical artifacts included; metric replay additionally uses the unchanged source CADKG gallery/query/reference workspace identified by hashes.',
        portable_visualization='Rebuild publication figures/tables directly from manuscript snapshots; no API or original dataset needed.')
    h.write(REPO/'manuscript/revision_7_deliverables.json',final)
    print(json.dumps(final,indent=2))
if __name__=='__main__':main()
