"""Annotation-gated real serving benchmark, bounded to the available CAD corpus."""
from __future__ import annotations
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import ctypes
import gc
import hashlib
import json
import os
import platform
import sys
import time
import tracemalloc
import numpy as np
from cadkg.bundle import load_bundle
from cadkg.experiments.full import retrieval_config
from cadkg.graph import build_graph
from cadkg.io import read_json,write_json,write_csv,sha256,object_hash
from cadkg.query import validate_query
from cadkg.retrieval import Retriever

def memory():
    if os.name!='nt':return {}
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
    p=Counters();p.cb=ctypes.sizeof(p)
    current=ctypes.windll.kernel32.GetCurrentProcess
    current.restype=wintypes.HANDLE
    handle=current()
    getter=ctypes.windll.psapi.GetProcessMemoryInfo
    getter.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    getter.restype=wintypes.BOOL
    if not getter(handle,ctypes.byref(p),p.cb):return {}
    return {n:int(getattr(p,n)) for n in ['WorkingSetSize','PeakWorkingSetSize','PrivateUsage']}

def make_workload(protocol,knowledge,catalog,count,seed):
    rng=np.random.default_rng(seed);families=protocol['families'];refs=defaultdict(list)
    for q in protocol['queries']:
        ref=q['reference_part_id']
        if ref and ref not in refs[q['scenario_id']]:refs[q['scenario_id']].append(ref)
    attributes=['aabb_large_mm','aabb_mid_mm','aabb_small_mm']
    validation={a:np.asarray([p['measurements'][a]['value'] for p in catalog['parts'] if p['split']=='val' and p['measurements'].get(a,{}).get('status')=='computed']) for a in attributes}
    workload=[];signatures=set()
    def add(q):
        q=validate_query(q,knowledge)
        signature=object_hash({k:q[k] for k in ['function_ids','reference_part_id','constraints','category_ids']})
        if signature in signatures:return False
        signatures.add(signature);q['query_id']=f'W{len(workload)+1:04}';workload.append(q);return True
    probes=[]
    for f in families:
        fid=f['family_id'];base=dict(scenario_id=fid,function_ids=[f['function_id']],category_ids=[],constraints=[],reference_part_id=None,split='test',text='Structured systems workload; no relevance claim.')
        add(base);probes.append(dict(base))
        for ref in refs[fid]:add(dict(base,reference_part_id=ref))
        probes.append(dict(base,reference_part_id=refs[fid][0]))
        constraints=[dict(attribute=a,op='<=',value=float(np.quantile(validation[a],.5)),unit='mm') for a in attributes]
        probes.append(dict(base,reference_part_id=refs[fid][0],constraints=constraints))
    for i,q in enumerate(probes):q['query_id']=f'S{i+1:03}'
    # Balance generated query families, while numerical bounds use validation only.
    cursor=0
    while len(workload)<count:
        f=families[cursor%len(families)];fid=f['family_id'];cursor+=1
        ref=None if rng.random()<.2 else refs[fid][int(rng.integers(len(refs[fid])))]
        band=[(.05,.2),(.4,.6),(.8,.95)][int(rng.integers(3))]
        dimensions=attributes if rng.random()<.5 else attributes[:1]
        constraints=[]
        for a in dimensions:
            lo,hi=np.quantile(validation[a],band)
            bound=float(rng.uniform(lo,hi)) if hi>lo else float(lo)+float(rng.uniform(1e-7,1e-5))
            constraints.append(dict(attribute=a,op='<=',value=bound,unit='mm'))
        functions=[f['function_id']]
        if rng.random()<.1:
            others=[x['function_id'] for x in families if x['function_id']!=f['function_id']]
            functions.append(others[int(rng.integers(len(others)))])
        add(dict(scenario_id=fid,function_ids=functions,category_ids=[],constraints=constraints,reference_part_id=ref,split='test',text='Parameterized systems query, validation-derived bounds; no functional qrels.'))
    return workload,probes

def stratified_order(parts,ids,seed):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for pid in sorted(ids):groups[parts[pid]['label']].append(pid)
    for values in groups.values():rng.shuffle(values)
    order=[]
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:order.append(groups[key].pop())
    return order

def main():
    cfg=read_json('configs/scalability_audited_v2.json');gate=read_json(cfg['quality_gate'])
    if not gate['audit_complete'] or gate['unresolved_sample_records'] or sha256(cfg['annotations'])!=gate['final_annotation_sha256']:raise ValueError('Annotation audit gate not satisfied')
    out=Path(cfg['output']);out.mkdir(exist_ok=True);started=time.perf_counter()
    load_start=time.perf_counter();catalog,knowledge,graph=load_bundle(cfg['bundle']);load_seconds=time.perf_counter()-load_start
    protocol=read_json(cfg['protocol']);parts={p['part_id']:p for p in catalog['parts']}
    retrievers={};init=[]
    for kind in cfg['geometry_kinds']:
        a=time.perf_counter();retrievers[kind]=Retriever(catalog,knowledge,graph,retrieval_config(protocol['settings'],kind))
        init.append(dict(kind=kind,seconds=time.perf_counter()-a,memory=memory()))
    usable=set.intersection(*(set(r.vectors) for r in retrievers.values()))
    usable={p for p in usable if parts[p]['prediction'] is not None and all(np.linalg.norm(r.vectors[p])>1e-12 for r in retrievers.values())}
    order=stratified_order(parts,usable,cfg['seed']);sizes=[n for n in cfg['candidate_sizes'] if n<=len(order)]
    if sizes[-1]!=len(order):sizes.append(len(order))
    galleries={n:sorted(order[:n]) for n in sizes};fixed=protocol['gallery']
    workload,probes=make_workload(protocol,knowledge,catalog,cfg['requests'],cfg['seed'])
    write_json(out/'workload.json',workload);write_json(out/'scale_probes.json',probes);write_json(out/'galleries.json',galleries)
    metadata=dict(config=cfg,bundle_load_hash_check_seconds=load_seconds,ranker_initialization=init,
      python=sys.version,numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),
      thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
      gc_enabled=gc.isenabled(),process_memory_after_load=memory(),resident_catalog_parts=len(catalog['parts']),resident_graph_nodes=len(graph['nodes']),resident_graph_edges=len(graph['edges']),
      requested_scale=sizes,independent_semantic_tasks=len(protocol['families']),distinct_workload_signatures=len(workload),
      query_families=dict(Counter(q['scenario_id'] for q in workload)),query_constraint_counts=dict(Counter(str(len(q['constraints'])) for q in workload)),
      query_function_counts=dict(Counter(str(len(q['function_ids'])) for q in workload)),query_exemplar_counts=dict(Counter(str(bool(q['reference_part_id'])) for q in workload)),
      temperature='Loaded/warm process; no OS cache flushing, CPU isolation or clock control',network='Not measured; in-process API plus JSON serialization')
    write_json(out/'metadata.json',metadata)
    rows=[];warmups=[];batch=[];signatures={};exact_controls=[]
    def call(kind,q,method,selected,k=5,**tags):
        if time.perf_counter()-started>cfg['hard_timeout_seconds']:raise TimeoutError('Configured total benchmark timeout')
        wall=time.perf_counter();cpu=time.thread_time()
        result=retrievers[kind].rank(q,method,k,selected)
        ranked_done=time.perf_counter();body=json.dumps(result,separators=(',',':'),ensure_ascii=False)
        end=time.perf_counter();cpu_seconds=time.thread_time()-cpu
        ids=[r['part_id'] for r in result['results']];scores=[r['score'] for r in result['results']]
        signature=object_hash(dict(ids=ids,scores=scores))
        row=dict(geometry_kind=kind,query_id=q['query_id'],family_id=q['scenario_id'],method=method,candidate_count=len(selected),top_k=k,
          start_perf=wall,wall_ms=(end-wall)*1000,ranking_ms=(ranked_done-wall)*1000,serialization_ms=(end-ranked_done)*1000,
          thread_cpu_ms=cpu_seconds*1000,response_bytes=len(body.encode('utf-8')),ranked_count=result['ranked_count'],returned=len(ids),
          constraints=len(q['constraints']),functions=len(q['function_ids']),has_exemplar=bool(q['reference_part_id']),result_sha256=signature,**tags)
        return row
    for kind in retrievers:
        for q in workload[:cfg['warmups_per_kind']]:warmups.append(call(kind,q,'kg_full',fixed,stage='warmup'))
    write_csv(out/'warmups.csv',warmups)
    rng=np.random.default_rng(cfg['seed'])
    for trial in range(cfg['workload_passes']):
        indices=rng.permutation(len(workload))
        # Alternate representation order for each pass to reduce ordering confounding.
        kinds=cfg['geometry_kinds'] if trial%2==0 else cfg['geometry_kinds'][::-1]
        for kind in kinds:
            block_start=time.perf_counter();checkpoints={100,500,len(workload)}
            for j,index in enumerate(indices,1):
                q=workload[int(index)];row=call(kind,q,'kg_full',fixed,stage='workload',trial=trial)
                key=(kind,q['query_id']);old=signatures.setdefault(key,row['result_sha256']);assert old==row['result_sha256']
                rows.append(row)
                if j in checkpoints:
                    elapsed=time.perf_counter()-block_start;batch.append(dict(stage='serial_workload',geometry_kind=kind,trial=trial,completed=j,seconds=elapsed,queries_per_second=j/elapsed))
                    print(f'Workload pass {trial+1} {kind}: {j}/{len(workload)}',flush=True)
    write_csv(out/'measurements.csv',rows);write_csv(out/'batch_throughput.csv',batch)
    # Interleave all methods and representations within each gallery/repetition.
    for n in sizes:
        for trial in range(cfg['scale_repeats']):
            blocks=[(kind,method) for kind in retrievers for method in cfg['methods']];rng.shuffle(blocks)
            for kind,method in blocks:
                for index in rng.permutation(len(probes)):
                    q=probes[int(index)];row=call(kind,q,method,galleries[n],stage='candidate_scale',trial=trial);rows.append(row)
                    if method in {'kg_full','flat_full'}:
                        key=(kind,q['query_id'],n,trial);pair=signatures.setdefault(key,{})
                        pair[method]=row['result_sha256']
                        if len(pair)==2:
                            if pair['kg_full']!=pair['flat_full']:raise ValueError('Graph/flat mismatch')
                            exact_controls.append(dict(geometry_kind=kind,query_id=q['query_id'],candidate_count=n,trial=trial,exact_order_and_score=True))
                print(f'Candidate scale n={n}, pass={trial+1}, {kind}/{method}',flush=True)
            write_csv(out/'measurements.csv',rows)
    for kind in retrievers:
        for k in cfg['top_k']:
            for trial in range(2):
                for index in rng.permutation(len(probes)):
                    rows.append(call(kind,probes[int(index)],'kg_full',fixed,k,stage='return_size',trial=trial))
            print(f'Return-size complete: {kind}, k={k}',flush=True)
    concurrency=[]
    for kind in retrievers:
        chosen=[workload[int(i)] for i in rng.permutation(len(workload))[:cfg['thread_requests']]]
        for workers in cfg['thread_workers']:
            start=time.perf_counter();completed=[]
            def worker(q,submitted):
                row=call(kind,q,'kg_full',fixed,stage='thread_batch',trial=workers);row['response_with_queue_ms']=(time.perf_counter()-submitted)*1000
                return row
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures=[]
                for q in chosen:
                    submitted=time.perf_counter();futures.append(pool.submit(worker,q,submitted))
                for future in as_completed(futures):completed.append(future.result())
            elapsed=time.perf_counter()-start;rows.extend(completed)
            concurrency.append(dict(geometry_kind=kind,workers=workers,requests=len(completed),batch_seconds=elapsed,queries_per_second=len(completed)/elapsed,
                service_p50_ms=float(np.median([r['wall_ms'] for r in completed])),service_p95_ms=float(np.quantile([r['wall_ms'] for r in completed],.95)),
                response_with_queue_p95_ms=float(np.quantile([r['response_with_queue_ms'] for r in completed],.95))))
            print(f'Thread batch complete: {kind}, workers={workers}',flush=True)
    write_csv(out/'measurements.csv',rows);write_csv(out/'thread_throughput.csv',concurrency);write_csv(out/'graph_flat_agreement.csv',exact_controls)
    # Allocation tracing is separate from uninstrumented timed calls.
    allocations=[];graph_build=[]
    for n in sizes:
        subset={**catalog,'parts':[parts[p] for p in galleries[n]]}
        for trial in range(2):
            gc.collect();a=time.perf_counter();g=build_graph(subset,knowledge);built=time.perf_counter();encoded=json.dumps(g,separators=(',',':'))
            graph_build.append(dict(parts=n,trial=trial,build_ms=(built-a)*1000,serialization_ms=(time.perf_counter()-built)*1000,serialized_bytes=len(encoded.encode()),nodes=len(g['nodes']),edges=len(g['edges'])))
            del g,encoded
    write_csv(out/'graph_build.csv',graph_build)
    for n in [sizes[0],sizes[-1]]:
        for method in cfg['methods']:
            gc.collect();tracemalloc.start();before=memory();row=call('descriptor',probes[1],method,galleries[n],stage='allocation_only')
            current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            allocations.append(dict(parts=n,method=method,python_peak_bytes=peak,python_retained_bytes=current,rss_before=before,rss_after=memory()))
    write_json(out/'allocations.json',allocations)
    metadata['process_memory_final']=memory();metadata['elapsed_seconds']=time.perf_counter()-started;write_json(out/'metadata.json',metadata)
    paths=['configs/scalability_audited_v2.json',cfg['protocol'],cfg['quality_gate'],cfg['annotations'],'docs/SCALABILITY_AUDIT_PLAN.md']+[str(Path(cfg['bundle'])/p) for p in ['catalog.json','knowledge.json','graph.json','bundle_manifest.json']]
    write_json(out/'run_manifest.json',dict(input_sha256={p:sha256(p) for p in paths},code_sha256={p:sha256(p) for p in [__file__,'cadkg/retrieval.py','cadkg/query.py','cadkg/reasoning.py','cadkg/graph.py']},
       artifacts_sha256={str(p.name):sha256(p) for p in out.glob('*') if p.is_file()},measurements=len(rows),graph_flat_exact_pairs=len(exact_controls),workload_exact_replay_cases=2*len(workload),relevance_metrics_computed=False))
    print(f'COMPLETE: {len(rows)} serving measurements; {time.perf_counter()-started:.1f}s',flush=True)

if __name__=='__main__':main()
