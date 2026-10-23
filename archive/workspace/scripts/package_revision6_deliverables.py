"""Package checked Figure 1 revision and the separate matched model comparison."""
from pathlib import Path
import hashlib,importlib.util,json,re,subprocess,sys,zipfile
REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
FIG=REPO/'manuscript/figure1_redraw'
LUNA=REPO/'outputs/llm_geometry_matched_v1'
SOL=REPO/'outputs/llm_geometry_sol_matched_v1'
spec=importlib.util.spec_from_file_location('package_helpers',REPO/'scripts/package_llm_geometry_deliverables.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    subprocess.run([sys.executable,str(PAPER/'scripts/validate_package.py')],check=True)
    arc=REPO/'manuscript/revision_archive/revision_5_before_icon_schematic'
    old=(arc/'main.tex').read_text(encoding='utf8');new=(PAPER/'main.tex').read_text(encoding='utf8')
    def normalize(s):
        start=s.index('\\begin{figure*}[t]');end=s.index('\\end{figure*}',start)+len('\\end{figure*}')
        s=s[:start]+'FRAMEWORK_FIGURE'+s[end:]
        s=re.sub(r'The overall schematic (?:reuses|was redesigned)[^\n]+','SCHEMATIC_DISCLOSURE',s)
        return s
    assert normalize(old)==normalize(new),'Unexpected manuscript change beyond Figure1/disclosure'
    assert sha(PAPER/'supplement.tex')==sha(arc/'supplement.tex')
    assert sha(PAPER/'supplement.pdf')==sha(arc/'supplement.pdf')
    final=json.loads((FIG/'.build/figure1_framework_final.validation.json').read_text())
    assert final['presentationLayout']['warningCount']==0 if 'warningCount' in final['presentationLayout'] else final['presentationLayout']['warning_count']==0
    qa=dict(revision=6,main_pdf_pages=6,supplement_pdf_pages=7,
        manuscript_changes='Figure1 artwork/caption and generative-image disclosure only',
        numerical_claims_unchanged=True,supplement_unchanged=True,
        all_six_main_pages_visually_inspected=True,final_native_PPTX_reimport_render_inspected=True,
        PPTX_native_text_runs=37,PPTX_native_connectors=44,PPTX_picture_objects=24,
        layout_warnings=0,source_CAD_media_hashes_verified=True,
        input_view_sha256=sha(FIG/'assets/Z0972_evaluated_six_views.png'),
        final_pptx_sha256=sha(FIG/'output/figure1_framework_final.pptx'),
        main_pdf_sha256=sha(PAPER/'main.pdf'),supplement_pdf_sha256=sha(PAPER/'supplement.pdf'))
    write(PAPER/'provenance/revision_6_pdf_qa.json',qa)
    allowed={'.tex','.pdf','.bib','.bbl','.md','.py','.json','.jsonl','.csv','.npz','.png','.svg','.txt','.pptx','.mjs'}
    excluded={'.tools','.qa','.dependencies','__pycache__','cad_reproduction'}
    files=sorted(p for p in PAPER.rglob('*') if p.is_file() and
        not set(p.relative_to(PAPER).parts)&excluded and p.suffix.lower() in allowed and
        p.name not in {'revise_visual_llm_manuscript.py','package_manifest.json'})
    manifest=dict(revision=6,prepared_date='2026-10-22',title='Knowledge-guided visual evidence for CAD retrieval with large language models',
        main_pdf_pages=6,supplement_pdf_pages=7,main_figures=4,main_tables=2,cited_references=19,
        main_model='gpt-6-luna',main_numerical_results_unchanged=True,
        separate_model_comparison='gpt-6.1-sol',
        editable_schematic='figures/editable_framework/figure1_framework_editable.pptx',
        prior_revision_archive='manuscript/revision_archive/revision_5_before_icon_schematic',
        files=[dict(path=p.relative_to(PAPER).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(PAPER/'package_manifest.json',manifest)
    paper=h.zip_verified(REPO/'manuscript/cirp_annals_manuscript_package.zip',
        [(p,'cirp_draft/'+p.relative_to(PAPER).as_posix()) for p in files]+[(PAPER/'package_manifest.json','cirp_draft/package_manifest.json')])
    # Full new comparison records and reused Luna views/records, once each.
    entries=[]
    for root in (LUNA,SOL):
        assert json.loads((root/'independent_replay_audit.json').read_text())['status']=='PASS'
        for p in sorted(root.rglob('*')):
            if p.is_file() and 'independent_replay' not in p.relative_to(root).parts and not p.name.startswith('.'):
                entries.append((p,p.relative_to(REPO).as_posix()))
    scripts={p for p in (REPO/'scripts').glob('*.py') if 'llm_geometry' in p.name or p.name=='neutralize_llm_cad_views.py'}
    # Preserve import helpers; original dataset/queries/rankings remain source-workspace dependencies.
    scripts.update(p for p in (REPO/'scripts').glob('*.py') if 'query_diversity' in p.name or 'feature_review' in p.name)
    for p in sorted(scripts):entries.append((p,p.relative_to(REPO).as_posix()))
    entries.append((REPO/'docs/LLM_GEOMETRY_MATCHED_V1.md','docs/LLM_GEOMETRY_MATCHED_V1.md'))
    comparison=h.zip_verified(REPO/'manuscript/llm_model_comparison_package.zip',entries)
    edit=PAPER/'figures/editable_framework'
    figure_entries=[(p,'figure1/'+p.relative_to(edit).as_posix()) for p in sorted(edit.rglob('*')) if p.is_file() and 'cad_reproduction' not in p.parts]
    figure_entries.append((PAPER/'figures/framework_redrawn.png','figure1/figure1_framework.png'))
    figure=h.zip_verified(REPO/'manuscript/figure1_editable_assets.zip',figure_entries)
    output=dict(manuscript=paper,editable_figure=figure,matched_model_comparison=comparison,
       credentials_included=False,
       comparison_dependency_scope='Both models retained API records, reused image inputs, protocols, rankings, metrics, tests, audits and scripts. Full metric replay uses the fixed original CAD-KG gallery/query/reference workspace named by protocol hashes; no new API calls are needed.',
       image_generation='Built-in image-generation concept; requested GPT-Image2 identity not exposed. Final native diagram uses original source CAD renders and licensed SVG icons.',
       manuscript_update_scope='Figure1 and image-generation disclosure only. Sol comparison reported separately; Luna primary results unchanged.')
    write(REPO/'manuscript/revision_6_deliverables.json',output)
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
