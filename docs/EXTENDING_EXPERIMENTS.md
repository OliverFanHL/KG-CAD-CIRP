# Extend the experiments

Use separate plans, annotation releases, and output directories for new experiments to preserve the reported study. Run the commands below from the repository root after [installation and data preparation](REPRODUCIBILITY.md).

## Add an LLM observer baseline

Export the same 2,084 unique role–part tasks for a model of your choice, with three independent response collections:

```text
python -m scripts.export_experiment export-baseline --model YOUR_MODEL_ID --design directed --collections 3 --output experiments/my_model_directed/plan.json
python -m scripts.export_experiment export-baseline --model YOUR_MODEL_ID --design holistic --collections 3 --output experiments/my_model_holistic/plan.json
```

Each plan contains 6,252 jobs. `directed` uses the study's four atomic geometric requirements and bindings; `holistic` uses the matched whole-role assessment. Both include the original neutral images, image hashes, instructions, payloads, and output schemas. Replace `YOUR_MODEL_ID` with the identifier for your model.

### Collect responses

Submit the exported jobs through your own model collector, using each job's `instructions`, `instruction_payload`, `schema`, and neutral `image_path`. Resolve image paths relative to the repository root. To maintain blinding, exclude relevance grades, source categories, original filenames, and study outcomes from requests.

Record the provider, requested and resolved model versions, collection date, decoding and reasoning settings, seed support, token limits, usage, latency, and response identifiers. Fix the settings and comparison plan before examining outcomes. Using the same model for both designs provides a matched observer comparison.

Save one JSONL response record per planned `job_id`, including failed requests. Required fields are:

| Field | Required value |
|---|---|
| `job_id` | Exact ID from this exported plan |
| `status` | `completed` for success; a failure state otherwise |
| `output_schema_valid` | JSON Boolean; true only for a validated structured response |
| `output_json` | Parsed object matching that job's schema; null on failure |
| Additional provenance | Preserve resolved model, settings, timestamps, response ID, raw response, and usage alongside these fields |

A directed response maps the four job-specific predicate IDs to `{status, views, observation}`, with `status` set to `yes`, `no`, or `unknown`. A holistic response maps `role_support` to `{score, views, observation}`, with score in `{0, 0.125, ..., 1}`. Follow the exact schemas in the plan. Include failed jobs under their original IDs: the evaluator requires complete coverage and assigns failed or invalid observations a neutral whole-job score of 0.5.

### Evaluate responses

Evaluate the independently collected responses offline:

```text
python -m scripts.evaluate_baseline --plan experiments/my_model_directed/plan.json --responses experiments/my_model_directed/responses.jsonl --output experiments/my_model_directed/evaluation
```

The command also supports holistic plans. It verifies source tasks, input hashes, and images, then evaluates the new responses on the original 264 queries, 1,816 candidates, annotations, dimensional rules, and refinement pools. Both hybrid refinement, `(3 × Full + H_or_J) / 4`, and score-only refinement are evaluated. Candidates outside the pool retain the initial Full order; ties use the original gallery index.

Outputs include:

- `support.json` and response-validity records.
- Per-query metrics, role and collection metrics, summary means, and sample SDs. A single collection leaves collection SD empty.
- `family_comparisons.csv`, with descriptive comparisons against Full and the geometry–intent filter.
- An evaluation receipt containing plan and response hashes, the requested model, and collector-supplied resolved-model identifiers. Model provenance depends on the collector's records.

For inferential analysis, prespecify contrasts and a multiple-testing family, use roles as paired units, retain uncertainty bounds, and document the additional tests. Use `scripts.evaluate_baseline` for new observers; `scripts.reproduce` reproduces the reported study.

## Review or add CAD relevance annotations

Export blank records for all eleven roles for a chosen CAD, or a small batch:

```text
python -m scripts.export_experiment export-annotations --part-id 00286ee90ab7896d8d9c --output experiments/review/packet.json
python -m scripts.export_experiment export-annotations --limit 10 --output experiments/review/batch.json
```

Choose stable part IDs from `data/catalog/parts.csv`; repeat `--part-id` to select several. The default batch selects the first ten sorted gallery IDs. Any catalog part can be exported, including parts outside the evaluated gallery.

Each selected part receives eleven blank role records with source hashes and existing inspection IDs where available. The packet records the parent annotation, catalog, and rubric hashes and omits existing grades for blinded review.

Review geometry using the [relevance rubric](../data/study/relevance_rubric.md), with annotators blinded to model scores, rankings, and previous grades. Packets link neutral rendered images where available. For other parts, obtain the original STEP geometry and generate suitable views; the representative STEP set covers only a small subset of the catalog. Record:

| Field | Required review information |
|---|---|
| `family_id`, `part_id`, `source_sha256` | Preserve the packet's role, stable identity, and geometry binding |
| `grade` | 0, 1, or 2 under the rubric; null if unresolved |
| `decision`, `observation`, `feature_evidence`, `rationale` | Observed evidence, uncertainty, and reason for the grade |
| `reviewer`, `review_date`, `review_method` | Reviewer identity or governed pseudonym, date, and human or model-assisted review method |
| `human_expert_reviewed`, `adjudication` | Expert involvement and conflict resolution, if performed |

Retain individual reviews and disagreement records in a versioned, adjudicated release. Record human and model-assisted contributions explicitly. Keep unresolved labels as null so the metric code can calculate admissible bounds. For re-review of existing candidates, publish the changed records, evidence, reviewer provenance, and new annotation digest alongside the parent release.

To evaluate a new label release, merge the reviewed packets into a separate evaluation dataset and create a new input manifest. Validate uniqueness of `(family_id, part_id)`, all eleven role assignments, grade domains, source hashes, and evidence bindings before running an adapted evaluator. Packet export covers review preparation; merging and evaluation configuration are separate steps.

## Expand the candidate gallery or query panel

New CADs require verified STEP sources and hashes, the same geometry preprocessing, descriptors and embeddings from the selected encoder, classifier probabilities, dimensional evidence, and complete eleven-role relevance assessments. Assign stable part and inspection IDs, then rebuild catalog and knowledge records, sorted gallery indices, complete eligible ranking arrays, and matched candidate pools. Preserve the split policy and keep test judgments out of model fitting. See [preprocessing](../preprocessing/encoder/README.md) and [methodology](METHODOLOGY.md).

Additional queries must specify their reference CAD, role, constraints, and origin. Define the balance of query conditions and role-level aggregation in the evaluation plan.

Gallery or panel changes require a new dataset release and evaluation configuration; the supplied baseline evaluator uses the original panel and gallery. Validate eligibility, scoring and tie rules, null-label bounds, role-level aggregation, and source hashes in the adapted pipeline, and report the new benchmark separately.
