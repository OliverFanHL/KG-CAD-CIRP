# Knowledge-guided relational evidence for functional CAD retrieval

Research code and datasets for functional CAD retrieval using geometric similarity, category-derived priors, dimensional screening, and source-linked visual evidence.

## Explore the repository

| Material | Location |
|---|---|
| Numerical results | [Results guide and tables](results/README.md) |
| Supporting datasets | [Data guide](data/README.md) |
| Saved LLM outputs and examples | [Response files, evidence records, and usage examples](data/observations/README.md) |

## Method and findings

An initial ranker combines geometry, exemplar-category agreement, and category–function priors, subject to measured size constraints. A visual observer assesses four geometric requirements and their component bindings for each role. Their deterministic coverage score refines the initial ranking within a fixed candidate pool. The matched holistic control uses the same images and score range.

Evaluation uses **1,816 candidate CADs, 264 queries, eleven roles, and three response collections per observer**, with embedding and descriptor representations.

| Observer | Embedding nDCG@5 | Descriptor nDCG@5 |
|---|---:|---:|
| KG-directed Luna | 0.8254 | 0.8125 |
| KG-directed Sol | 0.8275 | 0.8112 |

Luna's mean improvement over the matched holistic control is 0.0447 / 0.0210. Corrected role-level comparisons remain inconclusive (Holm p = 0.090820 / 0.050781). Annotation provenance is recorded in the supplied judgments file. Collection variability is conditional on a fixed encoder; materials, loads, tolerances, and mating compatibility are outside the evaluated claims. See the [methodology](docs/METHODOLOGY.md) and [result interpretation](results/README.md).

## Install and reproduce

Use Python 3.12. Create and activate an environment (`.venv\Scripts\Activate.ps1` on PowerShell; `source .venv/bin/activate` on Unix), then run from the repository root:

```text
python -m venv .venv
```

After activation:

```text
python -m pip install -e ".[technical,dev]"
python -m scripts.prepare_data
python -m scripts.reproduce --metrics-only
python -m scripts.reproduce
python -m pytest -q
```

Data preparation verifies the supplied artifacts and exposes them under `data/`. Reproduction rebuilds observations, rankings, metrics, and paired tests using saved model responses, with **no API calls or credentials**. Outputs go to `results/reproduced_metrics/` and `results/reproduced/`; choose a fresh directory with `--output` for another execution. Allow about 3 GB of disk space for the packaged data and reproduction outputs. See [reproduction instructions](docs/REPRODUCIBILITY.md).

## Repository layout

| Directory | Purpose |
|---|---|
| `results/` | Final numerical tables, query-level results, and validation evidence |
| `data/` | Queries, relevance annotations, role requirements, CAD examples, encoder assets, and prepared source evidence |
| `cadkg/` | Geometry, knowledge graph, retrieval, annotation, and the study evaluation pipeline in `cadkg/study/` |
| `scripts/` | Prepare data, reproduce results, inspect LLM outputs, and validate the repository |
| `configs/` | Retrieval settings and the study specification |
| `preprocessing/` | Encoder and geometry preprocessing implementation |
| `tests/` | Core software and geometry checks |
| `docs/` | Methodology, reproduction, and validation guides |

Representative STEP models, labels, source hashes, the selected encoder, saved observations, and all inputs for the primary offline reproduction are supplied. See the [data guide](data/README.md) for the complete raw TMCAD corpus source. Dataset and upstream materials retain their original rights; no new blanket license is assigned.

## Example usage

After preparing the data, inspect the saved shaft-role evidence and model output for candidate Z0430:

```text
python -m scripts.inspect_llm --inspection-id Z0430 --role shaft_drive --raw
```

Compare candidate Z0972 across the three observation designs and their collections:

```text
python -m scripts.inspect_llm --inspection-id Z0972 --role shaft_drive --compare
```

These examples read saved observations, print structured results, and make no API calls. [The LLM output guide](data/observations/README.md) explains file locations and join keys.

## Extend the study

See [extension workflows](docs/EXTENDING_EXPERIMENTS.md) for exporting matched LLM baseline plans, evaluating separately collected responses, and preparing blank CAD annotation packets. New experiments use separate outputs and preserve the reported study.
