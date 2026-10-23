"""Record completed document QA and check portable artwork reproduction."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'manuscript/cirp_draft'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

publication=read(PAPER/'provenance/repeated_observation_publication_v7.json')
assert sha(PAPER/publication['generator'])==publication['generator_sha256']
matches={}
for name,digest in publication['outputs'].items():
    original=PAPER/name
    regenerated=PAPER/'.qa/repro_v7'/Path(name).name
    matches[name]=sha(original)==sha(regenerated)==digest
assert all(matches.values()),matches
audit=dict(status='PASS',scope='Portable figure/table regeneration from packaged snapshots only; no original dataset or API.',
           exact_output_hashes=matches,
           source_sha256={p.name:sha(p) for p in sorted((PAPER/'data/robustness_v7').iterdir()) if p.is_file()},
           generator_sha256=publication['generator_sha256'])
write(PAPER/'provenance/portable_reproduction_v7.json',audit)

visual=read(PAPER/'provenance/revision7_pdf_review.json')
for name,document in visual['documents'].items():
    assert sha(PAPER/(name+'.pdf'))==document['pdf_sha256']
    assert sha(PAPER/(name+'.tex'))==document['tex_sha256']
    assert not document['layout_warnings']
    for page in document['rendered_pages']:assert sha(PAPER/page['path'])==page['sha256']
visual.update(review_status='VISUALLY_REVIEWED',
    review_scope='All 7 manuscript pages and all 8 supplementary pages visually inspected at final rendered versions. Figure/table placement, caption readability, panel alignment, symbols, references and page boundaries checked; no clipping or overlap found.',
    reviewed_utc=datetime.now(timezone.utc).isoformat())
write(PAPER/'provenance/revision7_pdf_review.json',visual)

state=read(ROOT/'manuscript/revision7_working_state.json')
state.update(status='SCIENTIFIC_AND_VISUAL_REVIEW_COMPLETE_PACKAGING',completion_pipeline_session=None,
    completion_pipeline='Completed; do not rerun acquisition or automatic manuscript integration.',
    main_editor='Final revised TeX sources in manuscript/cirp_draft; do not overwrite with integration template.',
    publication_generator='manuscript/cirp_draft/scripts/generate_revision7_assets.py',
    remaining=['Create and verify final manuscript and complete study archives.'],
    local_repeat_scope='Three actual order-seed executions at one frozen encoder/classifier checkpoint; no repeated surface sampling or retraining.',
    table2_display='Fixed deterministic scores; mean ± sample SD only for three GPT acquisitions.',
    last_review_utc=datetime.now(timezone.utc).isoformat())
write(ROOT/'manuscript/revision7_working_state.json',state)
write(ROOT/'outputs/llm_robustness_v7/pipeline_state.json',dict(stage='SCIENTIFIC_AND_VISUAL_REVIEW_COMPLETE',utc=datetime.now(timezone.utc).isoformat()))
print(json.dumps(dict(portable_artifacts=len(matches),exact=all(matches.values()),review_status=visual['review_status']),indent=2))
