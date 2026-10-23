"""Integrate only the authorized Figure 1 redraw; keep Luna numerical claims.

The separate Sol study is deliberately not substituted into the manuscript.
"""
from pathlib import Path
import hashlib,json,shutil

REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
FIG=REPO/'manuscript/figure1_redraw'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    archive=REPO/'manuscript/revision_archive/revision_5_before_icon_schematic'
    if not archive.exists():
        archive.mkdir(parents=True)
        for name in ('main.tex','main.pdf','supplement.tex','supplement.pdf','README.md','package_manifest.json'):
            shutil.copyfile(PAPER/name,archive/name)
        shutil.copyfile(PAPER/'figures/framework_overview.tex',archive/'framework_overview.tex')
        shutil.copyfile(REPO/'manuscript/cirp_annals_manuscript_package.zip',archive/'cirp_annals_manuscript_package.zip')
    editable=PAPER/'figures/editable_framework'
    editable.mkdir(parents=True,exist_ok=True)
    for name in ('build_figure1.mjs','imagegen_prompt_v2.txt','render_cad_assets.py'):
        shutil.copyfile(FIG/name,editable/name)
    shutil.copytree(FIG/'assets',editable/'assets',dirs_exist_ok=True)
    shutil.copyfile(FIG/'figure1_concept_v2.png',editable/'design_reference_gpt_image2.png')
    shutil.copyfile(FIG/'output/figure1_framework_final.pptx',editable/'figure1_framework_editable.pptx')
    shutil.copyfile(FIG/'output/figure1_framework.png',PAPER/'figures/framework_redrawn.png')
    (PAPER/'figures/framework_overview.tex').write_text(
        '% Source-bound CAD inserts; native editable diagram in editable_framework/.\n'
        '\\includegraphics[width=17.74cm]{framework_redrawn.png}\n',encoding='utf8')
    text=(PAPER/'main.tex').read_text(encoding='utf8')
    start=text.index('\\caption{Overall framework and visual-evidence refinement.') if '\\caption{Overall framework and visual-evidence refinement.' in text else text.index('\\caption{Knowledge-guided CAD retrieval.')
    end=text.index('\n\\label{fig:framework}',start)
    caption=r'''\caption{Knowledge-guided CAD retrieval. Source geometry and predicted categories supply $G,A,F$, while measured extents determine size eligibility $C$. The KG binds the requested role to four geometric predicates and their component relationships. Candidate views and these clauses enter the visual LLM, which records states and cited views. Deterministic coverage $H$ refines the eligible pool by $(3s+H)/4$. CAD inserts show actual source geometries at independent display scales with illustrative ceramic materials. The view montage reproduces an archived input, including two display clips.}'''
    text=text[:start]+caption+text[end:]
    text=text.replace('The overall schematic reuses a generated illustration with editable terminology and workflow corrections; CAD and observation-graph visualizations derive from the recorded sources.',
        'The overall schematic was redesigned with built-in OpenAI image generation and reconstructed as an editable diagram using source CAD renders and licensed icons; CAD and observation-graph visualizations derive from the recorded sources.')
    (PAPER/'main.tex').write_text(text,encoding='utf8')
    readme=(PAPER/'README.md').read_text(encoding='utf8').replace('Revision 5,','Revision 6,')
    readme=readme.replace('Figure 1 reuses the original generated schematic with an editable refinement\nworkflow.',
        'Figure 1 uses an icon-led open KG and process diagram with actual ceramic CAD\nrenders and the exact evaluated six-view input. Its editable PowerPoint, vector\nicons, CAD assets, design prompt and source builder are in\n`figures/editable_framework/`. The generated design reference is inspiration;\nits scientific connections and inserted geometry were corrected in the final\nnative diagram.')
    if 'Sol model candidate' not in readme:
        readme+='\n## Separate Sol model candidate\n\nThe additional GPT-6.1 Sol matched comparison is retained separately under\n`outputs/llm_geometry_sol_matched_v1` in the source workspace and in a separate\nresults archive. It showed no clear advantage over Luna. This revision changes\nFigure 1 and its disclosure; all Luna outcomes and manuscript claims are retained.\n'
    (PAPER/'README.md').write_text(readme,encoding='utf8')
    (editable/'README.md').write_text('''# Editable Figure 1

Open `figure1_framework_editable.pptx` to edit the labels, formulas, graph nodes
and connectors. CAD renders and licensed vector icons are individual picture
objects and can be moved or replaced. The slide is not a flattened screenshot.
`assets/` contains the original image and SVG inserts. Ceramic shaders illustrate
geometry and are not measured materials. `Z0972_evaluated_six_views.png` is the
exact archived model input, not the new ceramic six-camera rendering.

The layout was developed using built-in OpenAI image generation (GPT-Image2
requested; resolved model identity is not exposed by that tool). The imagegen
concept is retained only as `design_reference_gpt_image2.png`; its scientific
connections and CAD inserts were corrected/replaced in the final native diagram.
No generated CAD geometry is used in the final slide. The final paper artwork
is `../framework_redrawn.png`, 3200×1800 pixels (about456dpi at178mm width).
Captions belong to the LaTeX manuscript, not the slide.

`build_figure1.mjs` rebuilds the native objects through @oai/artifact-tool. Call
`buildFigure(presentation)` inside `artifactSession.run`, or from a Node
Presentation.create({slideSize:{width:1600,height:900}}) file workflow. The
bundled SVG icons allow rebuilding without installing Lucide. Lucide1.8.0
sources are distributed under ISC; the full license is in `assets/icons/`.

The unchanged `render_cad_assets.py` is the source renderer named by the CAD
provenance. For portable mesh-only regeneration, run
`python reproduce_cad_inserts.py` with NumPy, Pillow and VTK available.
It verifies the delivered frozen triangle meshes and writes beauty views to a
new `cad_reproduction` folder without changing the packaged inserts. Original
STEP identities remain in the provenance. The evaluated six-view image is
reused byte-for-byte and is not regenerated by this display renderer.
''',encoding='utf8')
    source_view=REPO/'outputs/llm_geometry_matched_v1/images/9e159dd21fd267ccf103.png'
    assert sha(source_view)==sha(editable/'assets/Z0972_evaluated_six_views.png')
    proof=dict(revision=6,mode='built-in image generation concept followed by native editable reconstruction',
        requested_image_model='GPT-Image 2',resolved_image_model='not exposed by built-in tool',
        study_results_unchanged=True,caption_external=True,actual_CAD_inserts=True,
        exact_evaluated_view_sha256=sha(source_view),
        finalization_sha256=sha(FIG/'.build/figure1_framework_final.validation.json'),
        outputs={str(p.relative_to(PAPER)).replace('\\','/'):sha(p) for p in [
            PAPER/'figures/framework_redrawn.png',PAPER/'figures/framework_overview.tex',
            editable/'figure1_framework_editable.pptx',editable/'build_figure1.mjs',
            editable/'imagegen_prompt_v2.txt',editable/'assets/cad_render_provenance.json']})
    (PAPER/'provenance/figure1_redraw_v6.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
