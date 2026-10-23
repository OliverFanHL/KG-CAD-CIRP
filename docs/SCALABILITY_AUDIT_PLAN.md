# Annotation-gated efficiency and scalability study

## Material Passport

Created 2026-10-20 after completing the random annotation audit. User authorized a gated annotation review followed by justified scale experiments. This is a fixed exploratory systems design, not external preregistration or independent human assessment.

## Audit gate and limits

The probability sample is 100 CADs selected uniformly without replacement (seed 20261021), with all eleven task records rereviewed. Sixty-six supplemental task–grade-stratified pairs improve coverage of rare positive/conditional cases; all tasks on their 66 CADs were also examined. Neutral sheets hide existing grades, source classes and method ranks. The 1,826 visual rereviews and adjudication produce a separate audited annotation version and an immutable correction log.

There are 22 disagreements in 1,100 probability-sampled records, concentrated in conditional labels (10/51). Thirty-six corrections across all selected CADs are justified by enlarged views and clarified applications of unchanged task scopes. This does not validate independent expert ground truth or all unsampled records. Broader effectiveness claims remain gated on independent positive/conditional review and genuinely new task families. The allowed expansion below consumes no functional labels: timing, memory, throughput and exact ranking consistency. Previous effectiveness numbers retain their original annotation-version provenance.

## Why 49 variants are insufficient for systems claims

They cover only 11 families, 19 exemplars and a narrow set of scoring paths; repeated phrases would not create workload diversity. The previous 25-second vectorized experimental run is not serving latency. Efficiency requires timing the actual public retrieval path and response serialization. Scalability requires changing candidate count and graph size; multiplying query IDs alone only increases total work. More semantic tasks are needed for effectiveness generalization, not for characterizing the existing serving algorithm's cost.

## Bounded workload and scale

- Freeze 1,000 distinct **structured parameterized workloads**, not 1,000 independent engineering intents. Include all existing unconstrained intent/reference combinations; remaining queries vary train-inspected exemplars, one or three numeric constraints, low/mid/high validation-derived bounds, and supported two-function conjunctions. Do not tune to annotated relevance.
- Two randomized passes through all 1,000 requests per representation at the original 1,816 test gallery; primary warm latency p50/p95, throughput and nested 100/500/1,000 request makespans. p99 is descriptive only. Report actual distinct query and computational signatures.
- Candidate scaling at 250, 1,000, 1,816, 4,000 and up to 9,798 actual usable catalog parts. Use nested seeded category-stratified subsets and the same 33 probes (three computational cases per family: text, exemplar, exemplar with three constraints). Two repeats, both representations, Full graph, matched flat and filtered geometry. The all-corpus scale points deliberately mix train/val/test and are **systems-only**. Reference and exact-source duplicates are excluded by the existing ranker. Report examined and ranked counts separately.
- Return-size sensitivity at k=5/20/100 on 33 probes and 1,816 candidates. Include JSON response time and byte size; do not silently omit explanations or charge them only to graph.
- Batch throughput for 66 requests at 1/2/4 worker threads using the same loaded ranker. Report service and submit-to-completion times separately; this is a batch thread test, not open-loop network capacity or a production SLA.
- Graph build/serialization at the five real catalog sizes, plus untimed allocation tracing on small/large cases. Separate import/extraction (reused), bundle load with hash verification, ranker initialization, graph construction, warm ranking/serialization and allocation instrumentation. Keep full-catalog residency explicit during candidate-scan scaling.

## Fairness, reproducibility and cost

No new model, approximation, cache or scoring implementation is introduced. All timed queries use `Retriever.rank`; the earlier vectorized experiment runner is excluded. Full and matched flat receive identical information, output fields and JSON encoding. Interleave methods and randomize request order; retain garbage collection/default OS scheduling. Warmups are logged separately. Record wall/CPU time, machine/OS/Python/NumPy/thread environment, input/code hashes, all queries, candidate IDs, per-call timestamps/latencies, process memory, result hashes and failures. Do not run other experiment processes concurrently.

A preliminary one-query timing on the real 9,798-part catalog was ~0.25 seconds, motivating the bounded two-pass workload and 33-query scale panel. It was a runtime sizing check, not a result selected for reporting. A 30-minute hard timeout bounds the whole run. Avoid fabricated 100k/1M duplicated corpora or hardware extrapolation. Fit descriptive log–log latency slopes within the measured range; these are empirical scaling diagnostics, not asymptotic proof. Compare graph/flat paired time ratios without claiming significant speed differences from correlated request repetitions.

The proposed scale is sufficient to characterize this implementation up to the available ~10k catalog and diagnose cost growth. It cannot support million-part, distributed, industrial-throughput or unseen-task performance claims. Those require genuinely larger collections and deployment-specific traces, hardware and concurrency tests.

Benchmark reporting follows the principles of explicit warmups, metadata and memory measurement described in [pyperf's official documentation](https://pyperf.readthedocs.io/en/latest/) and separates response time from throughput as in [SPEC's performance methodology](https://www.spec.org/power/docs/SPEC-Power_and_Performance_Methodology.pdf). This implementation does not claim pyperf or SPEC certification.
