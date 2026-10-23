"""Verify and restore the lossless, split research archive (standard library only)."""
from pathlib import Path
import hashlib,json,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((ROOT/'data/artifacts_manifest.json').read_text())
    expected={x['path']:x for x in manifest['files'] if x['storage']=='archive'}
    with tempfile.TemporaryFile() as combined:
        for row in manifest['parts']:
            p=ROOT/row['path']
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Damaged archive part: '+row['path'])
            with p.open('rb') as f:shutil.copyfileobj(f,combined)
        combined.seek(0)
        with zipfile.ZipFile(combined) as archive:
            if set(archive.namelist())!=set(expected):raise ValueError('Unexpected archive members')
            for name in archive.namelist():
                target=(ROOT/name).resolve()
                if not target.is_relative_to(ROOT):raise ValueError('Unsafe archive path')
                row=expected[name]
                if target.exists():
                    if sha(target)!=row['sha256']:raise ValueError('Refusing to overwrite changed file: '+name)
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                if sha(target)!=row['sha256']:raise ValueError('Restored file differs: '+name)
    print('Verified and restored',len(expected),'research artifacts.')

if __name__=='__main__':main()
