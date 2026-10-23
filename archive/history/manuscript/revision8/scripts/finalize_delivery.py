"""Bind final visual review and retain compact scientific support assets."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'cirp_draft'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
support={
    'knowledge_rules.json':OLD/'data/llm_geometry/knowledge_rules.json',
    'role_only_contract.json':OLD/'data/robustness_v7/role_only_contract.json',
    'qualitative_diagnostics.json':OLD/'data/robustness_v7/qualitative_diagnostics.json',
    'frozen_feature_rubric.md':OLD/'data/frozen_feature_rubric.md',
    'prior_relations.csv':OLD/'data/prior_relations.csv'}
assets={}
for name,source in support.items():
    target=ROOT/'data'/name;shutil.copyfile(source,target)
    assets['data/'+name]=dict(sha256=sha(target),source=str(source.relative_to(ROOT.parent)))
ppt=ROOT/'figures/figure1_editable.pptx'
shutil.copyfile(OLD/'figures/editable_framework/figure1_framework_editable.pptx',ppt)
assets['figures/figure1_editable.pptx']=dict(sha256=sha(ppt),source='cirp_draft/figures/editable_framework/figure1_framework_editable.pptx')
license_source=OLD/'figures/editable_framework/assets/icons/LUCIDE_LICENSE.txt'
license_target=ROOT/'figures/LUCIDE_LICENSE.txt';shutil.copyfile(license_source,license_target)
assets['figures/LUCIDE_LICENSE.txt']=dict(sha256=sha(license_target),source='cirp_draft/figures/editable_framework/assets/icons/LUCIDE_LICENSE.txt')
write(ROOT/'provenance/scientific_support_assets.json',assets)
# Limit this record to original plotting source CSVs, not derived annotations.
edit=json.loads((ROOT/'provenance/editorial_revision.json').read_text())
edit['data_sha256']={n:v for n,v in edit['data_sha256'].items() if n.endswith('.csv')}
write(ROOT/'provenance/editorial_revision.json',edit)
valid=json.loads((ROOT/'provenance/final_validation.json').read_text())
assert valid['status']=='PASS'
for d in valid['documents']:assert sha(ROOT/(d['document']+'.docx'))==d['docx_sha256']
review=dict(status='VISUALLY_REVIEWED',reviewed_utc=datetime.now(timezone.utc).isoformat(),
    scope='Every manuscript and supplementary Word page inspected at final 120-dpi renders. Figures 2 and 3 checked at embedded display size; numeric labels, legends, paired error bars and captions are readable without overlaps. Three-line tables, seven numbered LaTeX source formulas, references and page boundaries checked.',
    documents=[dict(document=d['document'],pages=d['pages'],docx_sha256=d['docx_sha256'],
        rendered_pdf_sha256=d['rendered_pdf_sha256'],page_sha256=d['page_sha256']) for d in valid['documents']],
    display_rule='Three-decimal numeric labels; SD numerals below 0.001 omitted. All actual error bars preserved; no experimental result changed.')
write(ROOT/'provenance/visual_review.json',review)
write(ROOT.parent/'current_manuscript.json',dict(revision=8,status='COMPLETE',
    manuscript='manuscript/revision8/manuscript.docx',supplement='manuscript/revision8/supplement.docx',
    figures='manuscript/revision8/figures',previous_research_package='manuscript/cirp_annals_manuscript_package.zip',
    experiments_unchanged=True))
print(json.dumps(dict(review_status=review['status'],reviewed_pages=sum(d['pages'] for d in valid['documents']),support_assets=len(assets)),indent=2))
