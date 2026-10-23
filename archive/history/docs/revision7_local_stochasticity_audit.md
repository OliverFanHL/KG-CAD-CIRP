# Revision 7: local stochasticity and repeatability audit

The three local passes test repeatability of serving and evaluation conditional on one fixed representation checkpoint, one frozen gallery, and one frozen query panel. They do not estimate variability from training or CAD sampling. No new training, geometry extraction, query selection, or API call was performed in this audit.

## What each seed changes

| Stage | Audited seed / operation | Consequence of changing seed |
|---|---|---|
| Original surface sampling | seed 42; part-specific offset from the hexadecimal sample ID | Changes sampled faces/barycentric points and therefore the cached 512-point, 16-neighbour representation. The present audit reuses the existing cache. |
| Original supervised encoder training | seed 42 | Changes initialization, minibatch order, and dropout draws. A new training run would be required to measure this source of variability. |
| Embedding inference | best validation checkpoint, epoch 17; inference/evaluation mode | Fixed input meshes and checkpoint produce fixed exported 128-dimensional vectors; dropout is disabled. |
| Descriptor normalization | training-only deterministic statistics | No random operation in serving. |
| Query generation | frozen descriptor-FPS panel, pool seed 20261021 | Changing this selection seed would create different query tasks, rather than repeat the existing tasks. |
| Local rank serving | execution seeds 17, 29, 43 | Initializes Python/NumPy RNG and changes query/representation execution order. The score calculation and part-ID tie break do not use randomness. |
| Bootstrap evaluation | existing bootstrap seed 20261020 | Controls numerical resampling of uncertainty estimates; does not alter rankings. No new bootstrap was run here. |
| LLM acquisition | outside this local audit | Fresh model responses can vary; queue-order seeds do not constitute an API sampling-seed control. The separate three-collection LLM study estimates response-collection variability conditional on the same checkpoint and candidate pool. |

## Original training and available execution environment

The original metadata records supervised class training with a surface-neighbour encoder, 128-dimensional output, ten-class head, cross-entropy loss, AdamW, batch size 16, maximum 80 epochs, and early-stopping patience 12. The saved run completed 29 epochs and selected epoch 17 with validation macro-F1 0.8860212671531593. Only training and validation data select this checkpoint. The checkpoint SHA-256 is `332a9661b782c7a343a6ff0e709818e0e5a77a8d5166b0d6b9267fa62d38ef62`.

The historical CUDA run took 1,080.16 s (18.00 min), with a 609,984,512-byte mesh cache and 25.36 s preload. Three comparable runs would be approximately 54 min on comparable hardware if early stopping were comparable; this is a historical cost projection, not a measured present-runtime cost. The present Python environment reports Torch 2.14.0+cpu, `cuda_available=False`, and zero accessible CUDA devices. This establishes CUDA unavailability in this runtime, not physical GPU absence. No CPU training runtime has been measured.

The existing surface cache records 512 points, 16 neighbours, tolerance 0.003 and seed 42. Its eight-worker construction took 9,204.67 s (2.56 h), computing 9,703 objects and reusing 95 cached objects without failure. Reusing the cache during a training-seed study would isolate training randomness; changing the mesh seed would add a separate sampling study. Neither is part of these serving repeats.

Source paths, exact hashes and original training metadata are recorded in `outputs/robustness_local_v7/training_audit.json`. The original project is `D:/Postdoc/research/jepa_cad/jev_tmcad_v0`.

## Three fresh local execution passes

Each pass reconstructed descriptor and embedding retrievers from the frozen source bundle. It shuffled all 528 query/representation jobs using its designated seed and recomputed seven ranking configurations on the same 1,816 candidates and 264 queries (11 families, 24 queries per family): geometry alone; G+C; G+A+C; G+F+C; G+A+F without C; Full G+A+F+C; and the matched flat Full control. Here A denotes predicted-category agreement with the reference and F the category-derived functional prior.

For every query and representation, the audit also called the actual `Retriever.rank` Full serving path, checked the complete candidate order, and compared all serving scores with the independently reconstructed component recipe. The five existing primary methods were compared with the complete saved v6 ranking and score arrays and with their numerical-reference-adjusted nominal metric envelopes.

All three passes completed successfully:

- 1,584 actual Full serving calls;
- 11,088 recomputed method/query/representation cases;
- 7,920 exact comparisons of existing v6 complete ranking arrays and corresponding metric envelopes;
- 7,392 saved ranking arrays per pass, all byte-identical by member name across the three passes;
- identical Full/flat orders and scores in every matched case;
- identical orders between serving and the independent component recipe; maximum score difference 2.220446049250313e-16, attributable to floating-point summation order;
- identical family-averaged metric values in all three passes, hence sample SD exactly zero.

The original aggregation step mistakenly required byte-identical ZIP archives, although shuffled query execution intentionally changes ZIP member order. It stopped at this packaging check after all three successful passes. A separate `finalize_verification.py` compares exact uncompressed NPY bytes in sorted member order, then aggregates the unchanged outputs. The frozen replay source and pass files were preserved. This correction changes neither rankings nor statistics. The sorted-member SHA-256 is `b668071d323d60ed537a65f053ec9c234fd7796e560ab24d398bc17cbf0fe761`.

The audit reports family averages with equal weight for each of the eleven families. The repeat table preserves lower/upper metric envelopes for the two unresolved source judgments and uses sample SD with denominator n−1 across the three actual pass means. SD=0 is a deterministic execution result; it must not be described as robustness to encoder training, new CAD sampling, independently selected queries, or stochastic LLM generation.

Results and exact source hashes are in `outputs/robustness_local_v7/verification.json`, `protocol.json`, `compact_ablation_three_pass.csv`, and the three `seed*/` directories. No frozen study or manuscript source was modified.
