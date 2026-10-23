"""Check the numerical result files and documentation links offline."""
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,re

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'results/manifest.json').read_text())
    problems=[]
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            problems.append('Result file differs: '+row['path'])
    documents=[ROOT/'README.md']
    for folder in ['docs','results','data']:
        documents.extend(p for p in (ROOT/folder).rglob('*.md') if not any(x.startswith('reproduced') for x in p.relative_to(ROOT).parts) and 'geometry' not in p.relative_to(ROOT).parts)
    for p in documents:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if '://' in link or link.startswith('#'):continue
            target=unquote(link.split('#')[0]).strip('<>')
            if target and not (p.parent/target).exists():problems.append('Broken link in '+str(p.relative_to(ROOT))+': '+link)
    result=dict(status='FAIL' if problems else 'PASS',result_files=len(manifest['files']),navigation_documents=len(documents),problems=problems)
    print(json.dumps(result,indent=2))
    if problems:raise SystemExit(1)

if __name__=='__main__':main()
