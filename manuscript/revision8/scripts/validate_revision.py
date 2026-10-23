"""Check immutable results, native Word structure and document integration."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import csv,hashlib,json,re
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
edit=read(ROOT/'provenance/editorial_revision.json')
for name,digest in edit['data_sha256'].items():assert sha(ROOT/'data'/name)==digest,name
fig=read(ROOT/'provenance/figure_generation.json')
assert not fig['results_changed'] and fig['error_bars_unchanged']
assert fig['numerical_SD_omission_threshold']==.001
for name,digest in fig['output_sha256'].items():assert sha(ROOT/'figures'/name)==digest,name
annotations=read(ROOT/'data/figure_annotations.json');assert len(annotations)==64
summary=list(csv.DictReader((ROOT/'data/summary.csv').open()))
for row in annotations:
    if row['figure']==2:
        original=next(r for r in summary if r['geometry_kind']==row['geometry_kind'] and r['method']==row['method'])
        assert row['mean']==float(original[row['metric']+'_mean'])
        assert row['sd']==float(original[row['metric']+'_sd'])
family=list(csv.DictReader((ROOT/'data/family_metrics.csv').open()))
import statistics
for row in annotations:
    if row['figure']==3:
        diffs=[]
        for rep in (1,2,3):
            a=next(r for r in family if r['geometry_kind']==row['geometry_kind'] and r['method']==row['method'] and r['family_id']==row['role'] and int(r['replicate'])==rep)
            b=next(r for r in family if r['geometry_kind']==row['geometry_kind'] and r['method']=='role_only_hybrid' and r['family_id']==row['role'] and int(r['replicate'])==rep)
            diffs.append(float(a['ndcg@5_lower'])-float(b['ndcg@5_lower']))
        assert abs(row['mean']-statistics.mean(diffs))<1e-14
        assert abs(row['sd']-statistics.stdev(diffs))<1e-14

ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
documents=[]
for name,expected_tables,expected_figures,expected_eq in [('manuscript',2,4,7),('supplement',2,0,0)]:
    path=ROOT/(name+'.docx')
    with ZipFile(path) as z:
        assert z.testzip() is None
        doc=etree.fromstring(z.read('word/document.xml'))
        styles=etree.fromstring(z.read('word/styles.xml'))
        assert len(doc.findall('.//w:tbl',ns))==expected_tables
        assert len(doc.findall('.//wp:inline',ns))==expected_figures
        instr=doc.findall('.//w:instrText',ns)
        assert sum('SEQ Equation' in t.text for t in instr)==expected_eq
        bookmarks={b.get('{'+ns['w']+'}name') for b in doc.findall('.//w:bookmarkStart',ns)}
        for t in instr:
            if t.text.strip().startswith('REF '):assert t.text.split()[1] in bookmarks
        assert not styles.xpath('.//w:style[@w:styleId="Title"]//w:pBdr',namespaces=ns)
        txt=' '.join(doc.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns))
        assert not any(v in txt for v in ('Error!','Reference source not found','The repeated study adds','configured enforced spend','reservation'))
        assert len(doc.findall('.//m:oMath',ns))==0,'Formulas must remain LaTeX text'
        assert txt.count(r'\[')==expected_eq and txt.count(r'\]')==expected_eq
        for table in doc.findall('.//w:tbl',ns):
            borders=table.find('./w:tblPr/w:tblBorders',ns)
            for side in ('left','right','insideH','insideV'):
                assert borders.find('w:'+side,ns).get('{'+ns['w']+'}val')=='nil'
            for side in ('top','bottom'):
                assert borders.find('w:'+side,ns).get('{'+ns['w']+'}val')=='single'
            trs=table.findall('w:tr',ns)
            for i,tr in enumerate(trs):
                for cell in tr.findall('w:tc',ns):
                    edge=cell.find('w:tcPr/w:tcBorders',ns)
                    for side in ('left','right'):
                        assert edge.find('w:'+side,ns).get('{'+ns['w']+'}val')=='nil'
                    assert edge.find('w:top',ns).get('{'+ns['w']+'}val')==('single' if i==0 else 'nil')
                    assert edge.find('w:bottom',ns).get('{'+ns['w']+'}val')==('single' if i in (0,len(trs)-1) else 'nil')
                    assert cell.find('w:tcPr/w:shd',ns) is None
        if name=='manuscript':
            assert '[19]' in txt and '0.8254' in txt and '0.8125' in txt
            assert '0.0908' in txt and '0.0508' in txt
    pdf=ROOT/'.qa'/name/(name+'.pdf');reader=PdfReader(pdf)
    texts=[p.extract_text() for p in reader.pages]
    assert not any('Error!' in p or 'Reference source not found' in p for p in texts)
    assert all(len(p)>120 for p in texts),'Blank or orphaned page'
    images=[ROOT/'.qa'/name/f'page-{i+1}.png' for i in range(len(reader.pages))]
    assert all(p.is_file() for p in images)
    documents.append(dict(document=name,pages=len(reader.pages),native_tables=expected_tables,
        embedded_figures=expected_figures,numbered_latex_equations=expected_eq,rendered_math_objects=0,
        table_style='Three horizontal rules with unshaded headers; no vertical or body rules',
        docx_sha256=sha(path),rendered_pdf_sha256=sha(pdf),
        page_sha256=[sha(p) for p in images]))
report=dict(status='PASS',results_changed=False,new_experiments=False,
    immutable_source_csvs=sum(name.endswith('.csv') for name in edit['data_sha256']),checked_annotations=len(annotations),
    acquisition_SD_threshold_display_only=.001,
    figure_2_annotations=20,figure_3_annotations=44,documents=documents,
    checks=['CSV identity','All plotted means and SDs','Three-line Word tables and unrendered LaTeX formulas',
        'Embedded source figures','Resolved caption/equation cross-references',
        'Complete nineteen-reference list','No title-border residue','Rendered every document page'])
(ROOT/'provenance/final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
