# Extend the experiments

Extensions use separate plans, annotations, and output directories. Keep the supplied `data/`, `results/`, and their manifests unchanged so the reported study remains reproducible. Run the following commands from the repository root after installation and `python -m scripts.prepare_data`.

## Add an LLM observer baseline

Export the same 2,084 unique role–part tasks for a model of your choice, with three independent response collections:

```text
python -m scripts.export_experiment export-baseline --model YOUR_MODEL_ID --design directed --collections 3 --output experiments/my_model_directed/plan.json
python -m scripts.export_experiment export-baseline --model YOUR_MODEL_ID --design holistic --collections 3 --output experiments/my_model_holistic/plan.json
```

Each plan contains 6,252 jobs. `directed` uses the study's four atomic geometric requirements and bindings; `holistic` uses the matched whole-role assessment. Both retain the original neutral images, image hashes, instructions, payloads, and output schemas. The model name is a user-supplied identifier, not a recommended provider or an API connection. The exporter writes plans only and makes no requests.

Use your own model collector to submit each job's `instructions`, `instruction_payload`, `schema`, and neutral `image_path`. Resolve image paths from this repository's root. Do not include relevance grades, source categories, original filenames, or study outcomes in model requests. Record the provider, requested and resolved model versions, collection date, decoding/reasoning settings, seed support, token limits, usage, latency, and response identifiers. Fix these settings and the comparison plan before examining extension outcomes. A single shared model can also be compared under both observer designs.

Save one JSONL response record per planned `job_id`, including failed requests. Required fields are:

| Field | Contract |
|---|---|
| `job_id` | Exact ID from this exported plan |
| `status` | `completed` for success; a failure state otherwise |
| `output_schema_valid` | JSON Boolean; true only for a validated structured response |
| `output_json` | Parsed object matching that job's schema; null on failure |
| Additional provenance | Preserve resolved model, settings, timestamps, response ID, raw response, and usage alongside these fields |

A directed object maps the four job-specific predicate IDs to `{status, views, observation}`. `status` is `yes`, `no`, or `unknown`. A holistic object maps `role_support` to `{score, views, observation}`, with score in `{0, 0.125, ..., 1}`. Use the exact schemas in the plan. Do not replace failed jobs with another candidate or silently drop them. The evaluator requires complete job coverage and applies the original whole-job neutral fallback of 0.5 for failed or invalid observations.

Evaluate the independently collected responses offline:

```text
python -m scripts.evaluate_baseline --plan experiments/my_model_directed/plan.json --responses experiments/my_model_directed/responses.jsonl --output experiments/my_model_directed/evaluation
```

The same command supports holistic plans. It verifies the plan against the source tasks, source input hashes, and image bytes; uses the unchanged 264 queries, 1,816 candidates, annotations, dimensional rules, and fixed pools; and evaluates both `(3 × Full + H_or_J) / 4` hybrid refinement and score-only refinement. Candidates outside the refinement pool retain the initial Full order. Ties use the original gallery index.

Outputs include `support.json`, per-query metrics, role/collection metrics, summary means and sample SDs, descriptive `family_comparisons.csv` against Full and the geometry–intent filter, response-validity records, and an evaluation receipt binding the plan and response hashes. The receipt records the requested model and any resolved-model identifiers supplied by the collector; offline scoring cannot authenticate a provider's model identity. One collection has no estimable collection SD, represented by an empty CSV value. This helper reports descriptive comparisons; it does not issue significance claims. For inferential extensions, prespecify contrasts and one multiple-testing family, use roles as paired units, retain uncertainty bounds, and document the additional tests. The original `scripts.reproduce` command remains fixed to the reported observers and cannot evaluate new model arms.

## Review or add CAD relevance annotations

Export blank records for all eleven roles for a chosen CAD, or a small batch:

```text
python -m scripts.export_experiment export-annotations --part-id 00286ee90ab7896d8d9c --output experiments/review/packet.json
python -m scripts.export_experiment export-annotations --limit 10 --output experiments/review/batch.json
```

Use `data/catalog/parts.csv` to choose stable part IDs; repeat `--part-id` to select several. The default batch selects the first ten sorted gallery IDs. Any catalog part can be exported, including parts outside the evaluated gallery. Every selected part receives eleven blank role records. The packet binds the parent annotation, catalog, and rubric hashes, retains source hashes and existing inspection IDs where available, and leaves review fields blank. It never modifies or invents a completed judgment. It does not expose existing grades in the review packet.

Review geometry using the [relevance rubric](../data/study/relevance_rubric.md). Neutral rendered images are linked when available; other catalog parts require acquisition of their original STEP geometry and generation of suitable views. The small representative STEP set does not cover every catalog part. Annotators should be blinded to model scores, rankings, and previous grades. Record:

| Field | Required review information |
|---|---|
| `family_id`, `part_id`, `source_sha256` | Preserve the packet's role, stable identity, and geometry binding |
| `grade` | 0, 1, or 2 under the rubric; null if unresolved |
| `decision`, `observation`, `feature_evidence`, `rationale` | Observed evidence, uncertainty, and reason for the grade |
| `reviewer`, `review_date`, `review_method` | Actual reviewer identity or governed pseudonym, date, and human/model-assisted review method |
| `human_expert_reviewed`, `adjudication` | Actual expert involvement and conflict resolution, if performed |

Retain individual reviews and disagreement records, then produce an explicitly versioned adjudicated release. Do not infer human or expert authorship from a model-generated record. Keep unresolved labels as null; the supplied metric code represents their admissible bounds. For re-review of existing candidates, retain the parent release and publish the changed records, review evidence, reviewer provenance, and new annotation digest separately.

The export helper does **not** merge reviewed packets or recompute the publication results with changed labels. A new label release requires an explicitly configured evaluation dataset and fresh input manifest. Validate uniqueness of `(family_id, part_id)`, all eleven role assignments, grade domains, source hashes, and evidence bindings before running an adapted evaluator. Do not overwrite the supplied judgment file merely to obtain a new score.

## Expand the candidate gallery or query panel

Appending annotation rows alone does not extend retrieval. New CADs require verified STEP sources and hashes, the same geometry preprocessing, descriptors and embeddings from the fixed selected encoder, classifier probabilities, dimensional evidence, and complete eleven-role relevance assessments. Assign stable part and inspection IDs; rebuild catalog/knowledge records, sorted gallery indices, complete eligible ranking arrays, and matched candidate pools. Preserve the split policy and avoid fitting on test judgments. See [preprocessing](../preprocessing/encoder/README.md) and [methodology](METHODOLOGY.md).

Additional queries must specify their reference CAD, role, constraints, and origin; balance conditions and role-level aggregation deliberately. Gallery or panel changes alter the benchmark, so create a new dataset release and evaluation configuration. The supplied baseline evaluator deliberately retains the original panel and gallery. Revalidate eligibility, score/tie rules, null-label bounds, role-level aggregation, and source hashes in the new pipeline; report results separately from the original study.
