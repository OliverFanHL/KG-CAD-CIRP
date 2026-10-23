"""Three ordered re-executions of the two remaining historical local controls.

Reuse the sealed serving-reconstruction implementation, without editing it.
Full/flat are execution anchors; No-G and RRF complete the historical panel.
"""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys
REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'outputs/robustness_local_v7/replay_local.py'
OUT=REPO/'outputs/robustness_remaining_local_v7'
spec=importlib.util.spec_from_file_location('saved_local_replay',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=OUT;m.METHODS=('kg_no_geometry','rrf_filter','kg_full','flat_full')
m.EXISTING=('kg_no_geometry','rrf_filter','kg_full')
base_inputs=m.frozen_inputs
def inputs():
    return {**base_inputs(),str(Path(__file__).relative_to(REPO)).replace('\\','/'):m.sha(Path(__file__))}
m.frozen_inputs=inputs
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=int);a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.run is not None:m.run(a.run);return
    m.prepare()
    for seed in m.SEEDS:
        subprocess.run([sys.executable,str(Path(__file__)),'--run',str(seed)],check=True)
    sums={seed:m.csvrows(OUT/f'seed{seed}/summary.csv') for seed in m.SEEDS}
    data=[]
    import numpy as np
    for kind in m.KINDS:
        for method in m.METHODS:
            local=[next(r for r in sums[s] if r['geometry_kind']==kind and r['method']==method) for s in m.SEEDS]
            row=dict(geometry_kind=kind,method=method,runs=3,seeds='17|29|43')
            for metric in m.METRICS:
                for end in ('lower','upper'):
                    vals=[float(r[metric+'_'+end]) for r in local]
                    assert len(set(vals))==1
                    row[metric+'_'+end+'_mean']=vals[0];row[metric+'_'+end+'_sd']=0.
            data.append(row)
    m.writecsv(OUT/'three_pass_summary.csv',data)
    with np.load(OUT/'seed17/rankings.npz') as first:
        names=set(first.files)
        for seed in (29,43):
            with np.load(OUT/f'seed{seed}/rankings.npz') as other:
                assert names==set(other.files)
                assert all(np.array_equal(first[n],other[n]) for n in names)
    m.dump(OUT/'verification.json',dict(status='PASS',methods=list(m.METHODS),seeds=list(m.SEEDS),
        additional_historical_candidates=['kg_no_geometry','rrf_filter'],
        source_implementation_sha256=m.sha(SOURCE),ranking_arrays_identical=True,
        total_ranking_cases=4*528*3,existing_full_array_checks=3*528*3,training_repeated=False,
        interpretation='Three deterministic serving executions with one frozen representation checkpoint.'))
    print('All historical local candidate configurations have three execution passes.')
if __name__=='__main__':main()
