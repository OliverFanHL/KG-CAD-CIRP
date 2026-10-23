"""Compile and render the revised PDFs for page-by-page visual inspection.

Use the bundled Python runtime (pypdf + pypdfium2). This does not certify visual
quality; a reviewer must inspect the saved pages before changing review_status.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess

REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
QA=PAPER/'.qa/revision_7'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    assert (PAPER/'provenance/repeated_observation_publication_v7.json').exists(), 'Current data integration required'
    QA.mkdir(parents=True,exist_ok=True)
    report=dict(review_status='RENDERED_AWAITING_VISUAL_INSPECTION',documents={})
    from pypdf import PdfReader
    import pypdfium2
    for name in ('main','supplement'):
        if args.compile:
            env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(PAPER/'.tools/texcache')
            with (QA/(name+'_compiler.txt')).open('w',encoding='utf8') as log:
                done=subprocess.run([str(PAPER/'.tools/tectonic/tectonic.exe'),'-X','compile',
                    name+'.tex','--only-cached','--keep-logs','--keep-intermediates'],
                    cwd=PAPER,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert done.returncode==0, name+' compilation failed; see QA compiler log'
        pdf=PAPER/(name+'.pdf');reader=PdfReader(pdf)
        contents='\n'.join(p.extract_text() or '' for p in reader.pages)
        assert '??' not in contents, name+' has unresolved reference markers'
        log=(PAPER/(name+'.log')).read_text(encoding='utf8',errors='replace')
        warnings=[line for line in log.splitlines() if re.search(r'Overfull|undefined|Missing character|multiply defined',line,re.I)]
        doc=pypdfium2.PdfDocument(pdf)
        renders=[]
        for i in range(len(doc)):
            path=QA/f'{name}_{i+1:02d}.png'
            page=doc[i];bitmap=page.render(scale=1.6);im=bitmap.to_pil().convert('RGB');im.save(path)
            renders.append(dict(path=path.relative_to(PAPER).as_posix(),sha256=sha(path),pixels=list(im.size)))
            bitmap.close();page.close()
        doc.close()
        report['documents'][name]=dict(pages=len(reader.pages),pdf_sha256=sha(pdf),
            tex_sha256=sha(PAPER/(name+'.tex')),layout_warnings=warnings,rendered_pages=renders)
    path=PAPER/'provenance/revision7_pdf_review.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{'pages':v['pages'],'layout_warnings':v['layout_warnings']} for k,v in report['documents'].items()},indent=2))

if __name__=='__main__':main()
