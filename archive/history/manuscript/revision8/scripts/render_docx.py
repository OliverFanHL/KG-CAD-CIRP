"""Rasterize native-Word PDF exports through the bundled document renderer.

Windows runtime has Microsoft Word and no bundled LibreOffice. The document
skill renderer is used for rasterization with its conversion hook bound to the
already exported Word PDF. No system LibreOffice or external service is used.
"""
from pathlib import Path
import importlib.util,json,os,tempfile
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/fhlie/.cache/codex-runtimes/codex-primary-runtime/plugins/openai-primary-runtime/plugins/documents/skills/documents')
spec=importlib.util.spec_from_file_location('document_skill_renderer',SKILL/'render_docx.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
tmp=ROOT/'.qa/temp';tmp.mkdir(exist_ok=True,parents=True)
tempfile.tempdir=str(tmp);os.environ['TEMP']=str(tmp);os.environ['TMP']=str(tmp)
records=[]
for stem in ('manuscript','supplement'):
    folder=ROOT/'.qa'/stem;pdf=folder/(stem+'.pdf')
    assert pdf.is_file() and pdf.stat().st_size>0
    renderer.convert_to_pdf=lambda *a,**k:(str(pdf),'Native Microsoft Word fixed-format export')
    pages=renderer.rasterize(str(ROOT/(stem+'.docx')),str(folder),120,False,False)
    records.append(dict(document=stem,pages=len(pages),dpi=120,renderer='Microsoft Word + document-skill rasterizer'))
(ROOT/'.qa/rasterization.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
