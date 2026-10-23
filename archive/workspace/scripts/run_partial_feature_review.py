"""Run frozen partial scoring with an equivalent cached path-membership check.

Only filesystem path membership is cached. Hash verification, annotations,
retrieval orders, transforms, metrics and aggregation remain the frozen code.
"""
import argparse
from pathlib import Path
from cadkg.io import read_json,write_json,sha256
import scripts.partial_feature_review as scoring

ROOT=Path('outputs/feature_review_v5')

def main(action,output=None):
    seal=ROOT/'execution_adapter_freeze.json'
    if action=='freeze':
        assert not seal.exists(),'Adapter already frozen'
        paths=[Path(__file__),Path(scoring.__file__),ROOT/'evaluation_freeze.json',ROOT/'evaluation/ranking_validation.json']
        # Equivalent path semantics include relative aliases and missing paths.
        mapping={'outputs/feature_review_v5/FROZEN_RUBRIC.md':'not-read',
                 'configs/../configs/candidate_scale_v4.json':'not-read'}
        resolved={Path(p).resolve() for p in mapping}
        for path in ['outputs/feature_review_v5/FROZEN_RUBRIC.md','configs/candidate_scale_v4.json','missing_membership_fixture.json']:
            expected=any(Path(p).resolve()==Path(path).resolve() for p in mapping)
            assert expected==(Path(path).resolve() in resolved)
        write_json(seal,dict(stage='Before scoring with adapter; before any completed revised metric result',
            reason='Replace repeated filesystem path-resolution scans with cached set membership; no numerical or experiment changes',
            interrupted_initial_attempt='Uncached process interrupted before output commit; original log retained at partial_evaluation.log',
            input_sha256={str(p):sha256(p) for p in paths},membership_fixture_cases=3))
        print('Execution adapter frozen');return
    adapter=read_json(seal)
    for p,h in adapter['input_sha256'].items():assert sha256(p)==h,p
    cache={}
    def cached_membership(path,mapping):
        key=id(mapping)
        if key not in cache:cache[key]=(mapping,{Path(p).resolve() for p in mapping})
        assert cache[key][0] is mapping
        scoring.require(Path(path).resolve() in cache[key][1],f'Required input absent from evaluation freeze: {path}')
    scoring._frozen_path=cached_membership
    out=Path(output) if output else ROOT/'partial_evaluation'
    result=scoring.evaluate('configs/feature_review_v5.json',output=out)
    manifest=read_json(out/'run_manifest.json')
    manifest['input_sha256'].update(adapter['input_sha256'])
    manifest['input_sha256'][str(seal)]=sha256(seal)
    manifest['code_sha256'][str(Path(__file__))]=sha256(__file__)
    manifest['execution_adapter']='Frozen equivalent cached Path.resolve membership; all original hash/metric checks retained'
    write_json(out/'run_manifest.json',manifest)
    print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate']);p.add_argument('--output');a=p.parse_args();main(a.action,a.output)
