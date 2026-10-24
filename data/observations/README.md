# Visual observations

Run `python -m scripts.prepare_data` to populate this directory.

## Where the LLM outputs are stored

Paths below are relative to the repository root. Large files become available locally after data preparation.

| Output | Location |
|---|---|
| Luna's first KG-directed raw responses | `data/observations/directed_luna/api_responses.jsonl` |
| Sol's first KG-directed raw responses | `data/observations/directed_sol/api_responses.jsonl` |
| Additional KG responses and all holistic responses | `data/observations/repeated_observations/api_responses.jsonl` |
| Combined evidence for all nine collections | `data/observations/repeated_observations/graph_evidence.jsonl` |
| Combined H/J support scores | `data/observations/repeated_observations/graph_support.json` |
| Per-role requirements and bindings | `data/study/knowledge_rules.json` |
| Holistic assessment contract | `data/study/role_only_contract.json` |
| CAD input images | `data/observations/images/<part_id>.png` |

Each raw response row includes `job_id`, `part_id`, `family_id`, requested and resolved model identifiers, `output_json`, `output_text`, `raw_response`, usage, and validation status. `raw_response` preserves the API response; `output_json` contains the parsed assessment. Directed evidence records contain clause observations, view citations, source-image hashes, and coverage H. Holistic records contain score J. These visual assessments support preliminary geometric shortlisting and require engineering review before use.

## Join observations to source records

The combined evidence identifies each observation by `arm`, `replicate`, `family_id`, and `part_id`. `response_id` links the observation to its raw response. For the first KG-directed collections, `source_job_id` identifies the raw row in the corresponding model file; other collections use `job_id` in the repeated-observation file. Inspection IDs used in the paper, such as Z0430, map to `part_id` through `data/annotations/judgments.json`.

## Usage examples

From the repository root, inspect Z0430's shaft-role evidence and its parsed model response:

```text
python -m scripts.inspect_llm --inspection-id Z0430 --role shaft_drive --raw
```

This selects KG-directed Luna collection 1 by default and prints the evidence, input-image location, raw-response file location, parsed model output, and usage. The reported coverage is H = 0 for this example. Use `--arm kg_sol --collection 2` to select another KG collection, or `--arm role_only_luna` for the holistic design.

Compare saved support for the shaft candidate Z0972 across all nine collections:

```text
python -m scripts.inspect_llm --inspection-id Z0972 --role shaft_drive --compare
```

The output lists the observation design, collection number, H/J support, and validity for each record. Use `--part-id <stable_id>` instead of `--inspection-id` when joining directly to ranking or catalog records. Candidates outside a role's refinement pool have no observation. Both examples read saved data offline.

Small saved examples are directly browsable on GitHub: [rejected shaft candidate Z0430](examples/shaft_rejected.json) and [supported shaft candidate Z0972](examples/shaft_supported.json). Each includes the source-linked evidence and parsed Luna response. The complete raw API records remain in the files listed above.

## Collection contents

- `images/`: neutral CAD montages used for visual observation.
- `directed_luna/`, `directed_sol/`: each model's first KG-directed collection.
- `repeated_observations/`: the additional KG collections and three holistic Luna collections.

Each collection includes a study specification (`study.json`, containing instructions, response schema, and job payloads), candidate pools, raw responses, source-linked geometric evidence, and support scores. There are three collections per observer arm.

Coverage H averages four clause states (yes = 1, no = 0, unknown = 0.5); holistic J uses the same nine-level range. Failed or invalid responses receive a whole-record neutral score of 0.5. Raw records include source identities and acquisition metadata.

The main comparison evaluates requirement decomposition, component-binding prompts, and coverage aggregation together. Graph storage is not isolated as an experimental factor. Collection SD measures acquisition variability with fixed representations and a fixed query panel. See the [methodology](../../docs/METHODOLOGY.md) for the evaluation design.
