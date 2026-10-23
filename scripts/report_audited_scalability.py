"""Summarize actual serving timings and validate deterministic workloads."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256,object_hash

OUT=Path('outputs/scalability_audited_v2')

def summary(group):
    def q(name,p):return float(np.quantile([float(r[name]) for r in group],p))
    return dict(samples=len(group),mean_ms=float(np.mean([float(r['wall_ms']) for r in group])),
        p50_ms=q('wall_ms',.5),p95_ms=q('wall_ms',.95),p99_ms=q('wall_ms',.99),
        median_ranking_ms=q('ranking_ms',.5),median_json_ms=q('serialization_ms',.5),
        median_response_bytes=q('response_bytes',.5),median_ranked_candidates=q('ranked_count',.5))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    meta=read_json(OUT/'metadata.json');manifest=read_json(OUT/'run_manifest.json');rows=read_csv(OUT/'measurements.csv');gate=read_json('outputs/annotation_quality_audit_v2/quality_gate.json')
    assert gate['audit_complete'] and gate['unresolved_sample_records']==0
    for name,digest in manifest['artifacts_sha256'].items():assert sha256(OUT/name)==digest
    workload=read_json(OUT/'workload.json');signatures={object_hash({k:q[k] for k in ['function_ids','reference_part_id','constraints','category_ids']}) for q in workload}
    assert len(signatures)==1000
    # Validate two timing passes and concurrent read-only retrieval against serial signatures.
    serial={};paired=defaultdict(list)
    for r in rows:
        if r['stage']=='workload':paired[r['geometry_kind'],r['query_id']].append(r['result_sha256'])
    assert len(paired)==2000 and all(len(v)==2 and len(set(v))==1 for v in paired.values())
    serial={k:v[0] for k,v in paired.items()}
    threaded=[r for r in rows if r['stage']=='thread_batch']
    assert all(serial[r['geometry_kind'],r['query_id']]==r['result_sha256'] for r in threaded)
    controls=read_csv(OUT/'graph_flat_agreement.csv');assert len(controls)==660 and all(r['exact_order_and_score']=='True' for r in controls)
    grouped=defaultdict(list)
    for r in rows:
        if r['stage']=='thread_batch':continue
        key=(r['stage'],r['geometry_kind'],r['method'],int(r['candidate_count']),int(r['top_k']))
        grouped[key].append(r)
    summaries=[]
    for key,group in sorted(grouped.items()):summaries.append(dict(zip(['stage','geometry_kind','method','candidate_count','top_k'],key),**summary(group)))
    write_csv(OUT/'latency_summary.csv',summaries)
    pass_summaries=[]
    for kind in ['embedding','descriptor']:
        for trial in ['0','1']:
            group=[r for r in rows if r['stage']=='workload' and r['geometry_kind']==kind and r['trial']==trial]
            pass_summaries.append(dict(geometry_kind=kind,trial=trial,**summary(group)))
    write_csv(OUT/'workload_passes.csv',pass_summaries)
    paths=defaultdict(list)
    for r in rows:
        if r['stage']=='workload':paths[r['geometry_kind'],r['has_exemplar'],r['constraints'],r['functions']].append(r)
    write_csv(OUT/'workload_path_summary.csv',[dict(geometry_kind=k[0],has_exemplar=k[1],constraints=k[2],functions=k[3],**summary(v)) for k,v in sorted(paths.items())])
    scale=[r for r in summaries if r['stage']=='candidate_scale'];slopes=[]
    for kind in ['embedding','descriptor']:
        for method in meta['config']['methods']:
            selected=[r for r in scale if r['geometry_kind']==kind and r['method']==method]
            x=np.asarray([r['candidate_count'] for r in selected]);y=np.asarray([r['p50_ms'] for r in selected]);slope,intercept=np.polyfit(np.log(x),np.log(y),1)
            slopes.append(dict(geometry_kind=kind,method=method,loglog_median_slope=float(slope),n_sizes=len(x),min_candidates=int(x.min()),max_candidates=int(x.max()),
                scope='Descriptive measured-range candidate-scan growth; not asymptotic proof or extrapolation'))
    write_csv(OUT/'scaling_slopes.csv',slopes)
    ratios=[];lookup={(r['geometry_kind'],r['query_id'],r['candidate_count'],r['trial'],r['method']):r for r in rows if r['stage']=='candidate_scale'}
    for kind in ['embedding','descriptor']:
        for n in meta['requested_scale']:
            values=[]
            for r in rows:
                if r['stage']=='candidate_scale' and r['geometry_kind']==kind and int(r['candidate_count'])==n and r['method']=='kg_full':
                    flat=lookup[kind,r['query_id'],r['candidate_count'],r['trial'],'flat_full'];values.append(float(r['wall_ms'])/float(flat['wall_ms']))
            ratios.append(dict(geometry_kind=kind,candidates=n,paired_cases=len(values),median_graph_flat_ratio=float(np.median(values)),p10_ratio=float(np.quantile(values,.1)),p90_ratio=float(np.quantile(values,.9))))
    write_csv(OUT/'graph_flat_time_ratios.csv',ratios)
    def lookup_summary(stage,kind,method,n,k=5):return next(r for r in summaries if (r['stage'],r['geometry_kind'],r['method'],r['candidate_count'],r['top_k'])==(stage,kind,method,n,k))
    f=lambda x:f'{float(x):.2f}'
    workload_table=table(['Representation','Pass','Median ms','p95 ms','p99 ms (descriptive)'],[[r['geometry_kind'],int(r['trial'])+1,f(r['p50_ms']),f(r['p95_ms']),f(r['p99_ms'])] for r in pass_summaries])
    scale_table=table(['Candidates','Embedding Full median/p95 ms','Descriptor Full median/p95 ms','Embedding flat median ms','Descriptor flat median ms'],[[n,
        f(lookup_summary('candidate_scale','embedding','kg_full',n)['p50_ms'])+'/'+f(lookup_summary('candidate_scale','embedding','kg_full',n)['p95_ms']),
        f(lookup_summary('candidate_scale','descriptor','kg_full',n)['p50_ms'])+'/'+f(lookup_summary('candidate_scale','descriptor','kg_full',n)['p95_ms']),
        f(lookup_summary('candidate_scale','embedding','flat_full',n)['p50_ms']),f(lookup_summary('candidate_scale','descriptor','flat_full',n)['p50_ms'])] for n in meta['requested_scale']])
    throughput=read_csv(OUT/'batch_throughput.csv');full_batches=[r for r in throughput if int(r['completed'])==1000]
    throughput_table=table(['Representation','Pass','1,000-request elapsed s','Observed serial requests/s'],[[r['geometry_kind'],int(r['trial'])+1,f(r['seconds']),f(r['queries_per_second'])] for r in full_batches])
    threads=read_csv(OUT/'thread_throughput.csv');thread_table=table(['Representation','Workers','Batch requests/s','Service p95 ms','Queued response p95 ms'],[[r['geometry_kind'],r['workers'],f(r['queries_per_second']),f(r['service_p95_ms']),f(r['response_with_queue_p95_ms'])] for r in threads])
    graph_build=read_csv(OUT/'graph_build.csv');build_table=table(['Parts','Nodes','Edges','Mean build ms','Mean serialize ms','Compact JSON MiB'],[[n,
        next(r['nodes'] for r in graph_build if int(r['parts'])==n),next(r['edges'] for r in graph_build if int(r['parts'])==n),
        f(np.mean([float(r['build_ms']) for r in graph_build if int(r['parts'])==n])),f(np.mean([float(r['serialization_ms']) for r in graph_build if int(r['parts'])==n])),
        f(np.mean([float(r['serialized_bytes'])/2**20 for r in graph_build if int(r['parts'])==n]))] for n in meta['requested_scale']])
    allocation=read_json(OUT/'allocations.json');allocation_table=table(['Candidates','Method','Traced transient Python peak MiB'],[[r['parts'],r['method'],f(r['python_peak_bytes']/2**20)] for r in allocation])
    return_table=table(['Representation','k','Median total ms','Median JSON ms','Median response KiB'],[[r['geometry_kind'],r['top_k'],f(r['p50_ms']),f(r['median_json_ms']),f(r['median_response_bytes']/1024)] for r in summaries if r['stage']=='return_size'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    styles={'kg_full':('Full graph','#446b91'),'flat_full':('Matched flat','#d28441'),'kg_no_class':('Geometry + filter','#58946a')}
    for ax,kind in zip(axes,['embedding','descriptor']):
        for method,(label,color) in styles.items():
            selected=sorted([r for r in scale if r['geometry_kind']==kind and r['method']==method],key=lambda r:r['candidate_count'])
            ax.plot([r['candidate_count'] for r in selected],[r['p50_ms'] for r in selected],'o-',label=label,color=color)
        ax.set(xlabel='Actual candidate gallery size',title=kind.capitalize(),xscale='log',yscale='log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Median serving + JSON time (ms)');axes[1].legend();fig.tight_layout();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figdir/f'candidate_scaling.{ext}',dpi=220)
    plt.close(fig)
    slope_text='; '.join(f"{r['geometry_kind']} {r['method']}: {r['loglog_median_slope']:.2f}" for r in slopes)
    report=f'''# Annotation-gated efficiency and scalability evaluation

## Material Passport

- Completed: 2026-10-20; systems-only, exploratory single-machine study.
- Annotation prerequisite: [random audit report](../annotation_quality_audit_v2/REPORT.md), corrected v2 release, 36 corrections. No relevance labels are consumed by timing or scaling.
- Protocol: [fixed systems plan](../../docs/SCALABILITY_AUDIT_PLAN.md); configuration and complete measurement logs are retained.
- Hardware: {meta['processor']}, {meta['logical_cpus']} logical CPUs; {meta['platform']}. Python/NumPy and thread environment are recorded in `metadata.json`.
- Measurements time the actual `Retriever.rank` implementation plus complete JSON response construction. They exclude UI/network transport, STEP extraction and model training.

## 1. Are the original 49 queries sufficient?

**No, for efficiency/scalability claims.** They represent eleven authored task families and nineteen exemplars, with correlated variants. Their relevance scores cannot establish latency, throughput or catalog growth. The earlier approximately 25-second vectorized experiment runner is not the serving implementation and must not be cited as serving performance.

The relevant expansion is controlled computational coverage: candidate count, constraint count/selectivity, exemplar availability, conjunctions, return count, graph size and request volume. Repeated paraphrases do not exercise new code paths. The present 1,000 unique structured signatures are a workload expansion, not 1,000 independent engineering intents. Broader effectiveness generalization still requires genuinely new task families and independent labels.

## 2. Annotation quality prerequisite

The primary probability audit reviewed all eleven assessments on 100 uniformly sampled CADs: 22/1,100 changed (2.0%, cluster-bootstrap interval 1.1–3.0%). Conditional decisions were less stable: 10/51 changed. Supplemental targeted sampling and complete cross-role review yielded 36 total corrections across 166 CADs. Source images exposed thread, hook, drive-feature and rationale-binding problems; criteria clarifications and a versioned correction log are published.

This is same-model rereview, not independent human correctness. The dataset is not globally certified. The gate permits label-independent systems evaluation only; no new effectiveness query family or relevance-superiority experiment was introduced. Historical effectiveness scores retain their prior annotation hash.

## 3. Workload and fairness

Two differently randomized passes over 1,000 distinct structured requests for each representation produce 4,000 main serving measurements at the original 1,816-CAD gallery. The workload is balanced at 90–91 requests per existing family: 201 without an exemplar, 799 with one, 30 unconstrained, 498 with one constraint, 472 with three, and 104 with two requested functions. Numeric bounds derive from validation measurements, not test relevance. This is synthetic coverage rather than a measured deployment traffic distribution.

Candidate scaling uses 33 matched probes (text-only, exemplar, exemplar+three constraints per family), five nested stratified real galleries and two repetitions per representation/method. Full graph, matched flat and filtered geometry receive the same candidates, model, scores, constraints, fields and serializer. These galleries mix dataset splits only for systems scaling and establish no retrieval generalization. The full 9,798 usable candidates are real catalog entries; no duplicated synthetic 100k corpus is used.

All eight-per-representation warmups are stored separately. Garbage collection remains enabled; the machine is not isolated and CPU frequency/OS caches are not controlled. Both complete query passes and all exact graph/flat pairs agree computationally. Timings naturally vary and are not expected to replay bit-for-bit.

## 4. Warm latency and query-volume behavior

{workload_table}

{throughput_table}

The 100/500/1,000 nested request makespans are in `batch_throughput.csv`. Batch throughput includes logging/signature work between calls; per-call latency covers rank and JSON construction. The p95 is the main tail summary; p99 is descriptive because each pass contains only about ten observations beyond that quantile. Repeated passes measure scheduling variation, not independent task samples. There is no production SLA or external-system latency comparison.

## 5. Candidate scan scaling

{scale_table}

![Candidate scaling](figures/candidate_scaling.png)

Measured median log–log slopes across the five sizes: {slope_text}. These are descriptive fits within the available range, not asymptotic proof. The code computes signals for every candidate, then applies its filter and sorts feasible candidates. Its approximate work is O(Nd + N log N) per request, plus explanation construction and output serialization; total serial request work grows approximately with request count. It is an exhaustive scan, not sublinear indexed retrieval. Three-constraint queries do not avoid computing the preceding scores for filtered candidates.

`graph_flat_time_ratios.csv` retains matched query/repetition ratios. Full and matched flat return identical ordered IDs and scores in all 660 scale comparisons. Any timing difference is implementation overhead, not an effectiveness benefit of graph storage or a comparison against an external graph database. The measured catalog size is the upper limit of the claim; no million-part/distributed scalability is demonstrated.

The matched flat implementation is faster at every measured gallery size in the aggregate median. At 9,798 candidates, Full requires about 195–198 ms versus 160–164 ms for matched flat. This evidence does **not** support a graph-specific efficiency advantage. The measured Full slope is approximately 1.08–1.09, consistent with slightly superlinear cost over this range. `workload_path_summary.csv` separates exemplar, constraint-count and conjunction paths so mixed-workload tails are not mistaken for repeated-call noise.

## 6. Return size, concurrency and throughput limits

{return_table}

The same exhaustive candidate scan is performed for each k. Larger k mainly increases returned explanation/JSON size; k=5/20/100 does not create a new relevance benchmark.

{thread_table}

Each row is a fixed 66-request batch. Service time excludes executor waiting; queued response includes it. This is not open-loop arrival-rate saturation testing and not HTTP throughput. Threads share a read-only Python ranker; they do not establish distributed parallel scalability. Report observed throughput changes rather than assuming four workers yield fourfold capacity. All 396 threaded results match their serial result hashes.

Observed throughput **decreases** as thread workers rise from one to four (embedding 30.15→21.81 requests/s; descriptor 27.70→18.72 requests/s), while service tails increase. This implementation does not demonstrate thread-level throughput scaling. Python-level computation/coordination is a plausible contributor, but no profiling experiment here isolates the cause.

## 7. Graph construction, residency and memory

{build_table}

Graph construction uses actual source parts and existing relations; STEP extraction was reused and excluded. Class/function vocabulary stays fixed, so this probes growth in part/prediction/measurement nodes and edges, not richer arbitrary multi-hop reasoning. JSON serialization is measured separately from graph building.

The verified bundle load plus integrity hashes took {meta['bundle_load_hash_check_seconds']:.2f}s. Ranker initialization is separately recorded and excludes that load. The loaded process held {meta['resident_catalog_parts']:,} catalog parts, {meta['resident_graph_nodes']:,} graph nodes and {meta['resident_graph_edges']:,} edges, with working set {meta['process_memory_after_load']['WorkingSetSize']/2**20:.1f} MiB and load-time peak {meta['process_memory_after_load']['PeakWorkingSetSize']/2**20:.1f} MiB. These are whole-process numbers, not graph-only allocation. Full catalog residency stays fixed while candidate subsets vary; scan-size timing therefore does not measure resident-memory scaling.

Untimed instrumented allocation checks:

{allocation_table}

`tracemalloc` observes Python allocations, not every native allocation. Its instrumented calls are excluded from serving latency/throughput tables. Graph JSON bytes and node/edge counts provide a separate storage-growth measurement. Both original representations and the full loaded graph occupy memory even when one is timed.

## 8. Integrity, reproducibility and scope

- 1,000 distinct validated computational query signatures; all workload definitions and gallery IDs saved.
- 2,000 representation/query results agree across two timing passes.
- 660 matched graph/flat order-and-score checks pass across five gallery sizes and both repetitions.
- 396 threaded outputs agree with serial outputs.
- Input/code/artifact SHA256 hashes, individual times, CPU times, response bytes, candidate counts, initialization and memory metadata are preserved.
- The first launch stopped before timing began because of a Windows handle typing issue in memory instrumentation; the corrected launch completed. This is recorded rather than silently treating the failed launch as a timing replicate.
- No relevance metrics, new functional labels, model retraining, score tuning or serving optimization were used.

The benchmark shows this implementation's measured costs up to ~10k actual CADs on this machine. It does not by itself establish that the cost is acceptable for an unspecified industrial SLA. A substantially larger independent catalog, varied graph connectivity and deployment-specific open-loop concurrency/hardware tests are required for larger-scale claims.

## 9. Recommended scale and claim revisions

For the current paper, 1,000 parameterized requests × two passes is a practical diagnostic workload, with 33 paired structural probes across five real gallery sizes. That gives much stronger evidence than multiplying the original 49 variants and keeps redundant computation bounded. Report p50/p95, throughput, memory, graph size and precise timing boundaries. Keep p99 exploratory and state the synthetic workload mix.

For an effectiveness extension, use genuinely new task families rather than renamed variants: an initial target of 50 independently authored task families, with 3–5 meaningful cases each (150–250 cases), is a practical development goal, not a power guarantee. Final size should follow a pilot of family-level variance and a prespecified meaningful effect. Independently audit positive/conditional judgments first and keep development/test tasks separate. Do not equate 1,000 timing parameters with 1,000 validated tasks.

For scaling beyond the measured ~10k parts, acquire a real 50k–100k catalog and report its connectivity, representation/index construction and hardware; do not extrapolate these curves or duplicate existing CAD records to manufacture semantic scale. The current exhaustive scan may need indexing or early filtering, but an optimization would require a separate exactness/quality comparison and is not silently introduced here.

Benchmark methodology references: [pyperf documentation](https://pyperf.readthedocs.io/en/latest/) for explicit warmups, metadata and memory reporting, and [SPEC methodology](https://www.spec.org/power/docs/SPEC-Power_and_Performance_Methodology.pdf) for separating throughput and response-time questions. This custom benchmark is not certified by either project.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    validation=dict(status='VERIFIED_COMPUTATIONAL_AGREEMENT',distinct_valid_query_signatures=1000,workload_pass_result_pairs=2000,
        graph_flat_exact_pairs=660,thread_serial_exact_pairs=len(threaded),measurements=len(rows),annotation_quality_gate='sample_review_complete_systems_only',relevance_validity='not evaluated by this benchmark')
    write_json(OUT/'validation.json',validation)
    write_json(OUT/'analysis_manifest.json',dict(code_sha256=sha256(__file__),benchmark_manifest_sha256=sha256(OUT/'run_manifest.json'),report_sha256=sha256(OUT/'REPORT.md')))
    print(validation)
    print(workload_table);print(scale_table);print(thread_table)

if __name__=='__main__':main()
