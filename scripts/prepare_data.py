"""Verify the research artifacts and expose inputs in the public data layout."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'archive/workspace'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def expose(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('Refusing to replace changed input: '+str(target))
        return
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)

def main():
    manifest=json.loads((ROOT/'archive/artifacts/manifest.json').read_text())
    expected={x['path']:x for x in manifest['files'] if x['storage']=='archive'}
    with tempfile.TemporaryFile() as combined:
        for row in manifest['parts']:
            p=ROOT/'archive/artifacts'/Path(row['path']).name
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Damaged archive part: '+p.name)
            with p.open('rb') as f:shutil.copyfileobj(f,combined)
        combined.seek(0)
        with zipfile.ZipFile(combined) as archive:
            if set(archive.namelist())!=set(expected):raise ValueError('Unexpected archive members')
            for name in archive.namelist():
                base=ROOT if name.startswith('data/encoder/') else WORK
                target=(base/name).resolve()
                if not target.is_relative_to(base.resolve()):raise ValueError('Unsafe archive path')
                row=expected[name]
                if target.exists():
                    if sha(target)!=row['sha256']:raise ValueError('Changed archived evidence: '+name)
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                if sha(target)!=row['sha256']:raise ValueError('Restored file differs: '+name)
    index=[]
    def publish(source,destination):
        expose(WORK/source,ROOT/destination)
        index.append(dict(path=destination,sha256=sha(ROOT/destination)))
    publish('annotations/intent_feature_review_v5/judgments.json','data/annotations/judgments.json')
    for n in ['catalog.json','bundle_manifest.json']:
        publish('outputs/intent_batch2_matrix_reproduction/bundle/'+n,'data/catalog/'+n)
    for n in ['knowledge.json','graph.json']:
        publish('outputs/intent_batch2_matrix_reproduction/bundle/'+n,'data/knowledge/'+n)
    for source,name in [('llm_geometry_matched_v1','directed_luna'),('llm_geometry_sol_matched_v1','directed_sol'),('llm_robustness_v7','repeated_observations')]:
        for n in ['study.json','pool_manifest.json','protocol_freeze.json','api_responses.jsonl','graph_support.json','graph_evidence.jsonl']:
            publish('outputs/'+source+'/'+n,'data/observations/'+name+'/'+n)
    for source,destination in [('outputs/llm_geometry_matched_v1/images','data/observations/images'),('outputs/feature_review_v5/evidence','data/geometry/evidence')]:
        for p in (WORK/source).rglob('*'):
            if p.is_file():publish(str(p.relative_to(WORK)).replace('\\','/'),destination+'/'+str(p.relative_to(WORK/source)).replace('\\','/'))
    for n in ['complete_rankings.npz','cohorts.json','effective_protocol.json']:
        publish('outputs/query_diversity_v6/evaluation/'+n,'data/rankings/'+n)
    (ROOT/'data/prepared_inputs.json').write_text(json.dumps(dict(status='PASS',files=index),indent=2)+'\n')
    print('Verified',len(expected),'archived files; prepared',len(index),'study inputs under data/.')

if __name__=='__main__':main()
