# Functional CAD retrieval with knowledge-guided large language models

Research code and datasets for functional CAD retrieval using geometric similarity, category-derived priors, dimensional screening, and source-linked visual evidence.

## Start here

| Material | Location |
|---|---|
| Numerical results | [Results guide and tables](results/README.md) |
| Supporting datasets | [Data guide](data/README.md) |
| Saved LLM outputs and examples | [Response files, evidence records, and usage examples](data/observations/README.md) |
| Study design | [Methodology](docs/METHODOLOGY.md) |
| Reproduction | [Setup and reproduction guide](docs/REPRODUCIBILITY.md) |
| New experiments | [Extension guide](docs/EXTENDING_EXPERIMENTS.md) |

## Method and findings

The initial ranker combines geometric similarity, exemplar-category agreement, and category–function priors, subject to measured size constraints. A visual observer assesses four geometric requirements and their component bindings for each role. A coverage score aggregates these assessments and refines the ranking within a fixed candidate pool. A matched holistic observer assesses the whole role using the same images and score range.

Evaluation uses **1,816 candidate CADs, 264 queries, eleven roles, and three response collections per observer**, with embedding and descriptor representations.

| Observer | Embedding nDCG@5 | Descriptor nDCG@5 |
|---|---:|---:|
| KG-directed Luna | 0.8254 | 0.8125 |
| KG-directed Sol | 0.8275 | 0.8112 |

Luna's mean improvement over the matched holistic control is 0.0447 for embeddings and 0.0210 for descriptors. The role-level comparisons are inconclusive after multiple-testing correction (Holm p = 0.090820 and 0.050781, respectively). The study measures response-collection variability with a fixed encoder and evaluates geometric suitability for preliminary shortlisting. Materials, loads, tolerances, and mating compatibility require further engineering assessment. See the [methodology](docs/METHODOLOGY.md), [annotation provenance](data/README.md), and [result interpretation](results/README.md).

## Install and reproduce

Use Python 3.12. From the repository root, create an environment:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Unix, then run:

```text
python -m pip install -e ".[technical,dev]"
python -m scripts.prepare_data
python -m scripts.reproduce --metrics-only
python -m scripts.reproduce
python -m pytest -q
```

Data preparation verifies the supplied files and unpacks study inputs under `data/`. Reproduction rebuilds observations, rankings, metrics, and paired tests offline from saved model responses. Outputs go to `results/reproduced_metrics/` and `results/reproduced/`; use `--output` to select a fresh directory for subsequent runs. Allow about 3 GB of disk space for the packaged data and reproduction outputs. See the [reproduction guide](docs/REPRODUCIBILITY.md) for workflow details and raw-data requirements.

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
| `docs/` | Methodology, reproduction, and extension guides |

The repository supplies representative STEP models, labels, source hashes, the selected encoder, saved observations, and all inputs for the primary offline reproduction. See the [data guide](data/README.md) for the complete raw TMCAD corpus source and attribution. Dataset and upstream materials remain subject to their original license terms.

## Example usage

After preparing the data, inspect the saved shaft-role evidence and model output for candidate Z0430:

```text
python -m scripts.inspect_llm --inspection-id Z0430 --role shaft_drive --raw
```

Compare candidate Z0972 across the three observation designs and their collections:

```text
python -m scripts.inspect_llm --inspection-id Z0972 --role shaft_drive --compare
```

The [LLM output guide](data/observations/README.md) explains the saved observations, file locations, and join keys used by these commands.

## Extend the study

The [extension guide](docs/EXTENDING_EXPERIMENTS.md) covers matched LLM baselines, evaluation of new response collections, CAD annotation packets, and requirements for expanding the benchmark.
