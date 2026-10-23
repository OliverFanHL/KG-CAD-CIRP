"""Verify and unpack the final study datasets (standard library only)."""
from pathlib import Path
import hashlib,json,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf8'))
    expected=[row for row in manifest['files'] if row['storage']=='package']
    with tempfile.TemporaryFile() as combined:
        for row in manifest['parts']:
            p=(ROOT/row['path']).resolve()
            if not p.is_relative_to(ROOT):raise ValueError('Unsafe dataset part path')
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Damaged dataset part: '+row['path'])
            with p.open('rb') as f:shutil.copyfileobj(f,combined)
        combined.seek(0)
        with zipfile.ZipFile(combined) as dataset:
            members=set(dataset.namelist())
            for row in expected:
                name=row['path'];member=row.get('source_member',name)
                if member not in members:raise ValueError('Missing dataset member: '+name)
                target=(ROOT/name).resolve()
                if not target.is_relative_to(ROOT/'data'):raise ValueError('Unsafe dataset path')
                if target.exists():
                    if sha(target)!=row['sha256']:raise ValueError('Refusing to overwrite changed input: '+name)
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with dataset.open(member) as source,target.open('wb') as destination:shutil.copyfileobj(source,destination)
                if sha(target)!=row['sha256']:raise ValueError('Restored file differs: '+name)
    for row in manifest['files']:
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT/'data'):raise ValueError('Unsafe input path')
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Input differs: '+row['path'])
    (ROOT/'data/prepared_inputs.json').write_text(json.dumps(dict(status='PASS',files=manifest['files']),indent=2)+'\n',encoding='utf8')
    print('Verified',len(manifest['files']),'study inputs; unpacked dataset files are available under data/.')

if __name__=='__main__':main()
