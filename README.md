# KG-CAD-CIRP

Research code and archived evidence for **Knowledge-guided relational evidence for functional CAD retrieval**. The publication baseline is [manuscript revision 8](manuscript/revision8/README.md), identified by `manuscript/current_manuscript.json`.

The method combines geometric similarity, predicted category agreement, category–function priors and measured dimensional screening. Knowledge-guided visual observation then assesses four authored geometric clauses per role and their component bindings. Within a frozen candidate pool, coverage refines the initial score as `(3 × Full + H) / 4`. Source views, prompts, model responses, clause states and rankings remain traceable.

The primary panel contains **1,816 candidate CADs, 264 queries, eleven roles and three response collections per observer**. KG-directed Luna achieves mean nDCG@5 **0.8254 / 0.8125** (embedding / descriptor); Sol achieves **0.8275 / 0.8112**. Luna improves the matched holistic control by **0.0447 / 0.0210**, but corrected role-level tests remain inconclusive (Holm p **0.090820 / 0.050781**). Judgments are AI-derived, without independent expert adjudication. Three collections measure acquisition variability conditional on one fixed encoder; they do not measure retraining variability or establish engineering compatibility.

## Install and reproduce

Use Python 3.12 (Python 3.10+ is supported by the core library). Run all commands from the repository root. PowerShell activation uses `.venv\Scripts\Activate.ps1`; on Unix use `source .venv/bin/activate`.

```text
python -m venv .venv
python -m pip install -e ".[technical,dev]"
python -m scripts.restore_artifacts
python -m pytest -q
python -m scripts.reproduce_release --metrics-only
python -m scripts.reproduce_release
```

Activate the new environment before installation. `requirements-validated.txt` records the versions used in the original environment; broad supported ranges are in `pyproject.toml`. The artifact restore command uses only Python's standard library. It verifies archive parts and individual file hashes, and refuses to overwrite changed evidence. Allow roughly 3 GB of free space for unpacking and replay.

The metrics-only command regenerates the five aggregate tables from archived per-query metrics. Full replay reconstructs observations from saved responses, reranks frozen initial arrays and recalculates per-query metrics, role means, SDs and paired tests. Both compare against original hashes, write fresh output directories and make **zero API calls**. Use `--output outputs/my_replay` for another fresh destination. See [reproduction details](docs/REPRODUCIBILITY.md).

For publication artwork, install `manuscript/revision8/requirements.txt`, change into that folder and run `python scripts/generate_figures.py`. `python scripts/build_word.py` rebuilds editable documents; Word must update its fields. Published manuscript files and numerical snapshots are included.

## Repository structure

| Path | Contents |
|---|---|
| `cadkg/` | Shared catalog, geometry, knowledge graph, retrieval, metrics, CLI and local annotation interface |
| `scripts/` | Versioned experiment preparation, acquisition, evaluation, audits and portable reproduction entry points |
| `configs/` | Frozen experiment settings and retrieval configurations |
| `annotations/` (restored) | Versioned labels, judgments, queries, engineering priors and provenance |
| `outputs/` (restored) | Curated frozen experiment inputs, journals, source evidence, rankings and results |
| `data/` | Lossless split research archive, hash manifest, encoder artifacts and representative STEP examples |
| `preprocessing/encoder/` | Original upstream encoder and preprocessing implementation/configurations |
| `manuscript/revision8/` | Current manuscript, supplement, figures, numerical snapshots and artwork scripts |
| `tests/` | Retrieval, geometry, annotation and evaluation contract checks |
| `docs/` | Methodology, study protocols, dataset/release and reproduction guides |

The study directories retain their original names because frozen protocols bind their paths and bytes. Common runtime operations already use `cadkg`; historical sealed implementations remain byte-identical. New entry points provide portable orchestration rather than modifying the experimental algorithms.

## Datasets and experimental workflows

[Data guide](data/README.md) describes the archived inputs and their relationships. [Experiment map](docs/EXPERIMENT_INDEX.md) distinguishes the publication study from historical expansions. The source corpus is **TMCAD**, with 9,799 parts in ten categories. Source geometry and category labels do not establish verified function, material, load, fit or mating compatibility. The complete raw STEP corpus is obtained separately from the original TMCAD source cited in the manuscript; representative unmodified examples are included with a part-ID/hash manifest.

The included catalog contains the original split membership, labels, normalized representations, classifier probabilities and geometry measurements. Original encoder checkpoint, embeddings, descriptors, cohort and preprocessing code are also included. Frozen result replay needs no encoder training, full STEP corpus, external service or credentials. Rebuilding geometry or retraining requires the original corpus and optional upstream dependencies; see the reproduction guide.

Live LLM runners remain available as historical implementations, but generating a fresh collection is a separate, paid experiment and will produce different observations. Read the corresponding frozen study protocol before using their explicit execution flags. Store keys only in process environment variables. No credentials are required for the documented replay.

## Verification and provenance

See [validation report](docs/VALIDATION.md) for checks actually performed for this release. `data/artifacts_manifest.json` maps every curated original file to its source location, SHA-256, size and storage location. Original experiments were preserved locally. Temporary files, repeated replay copies, editor state, private QA renders and redundant delivery archives are omitted. Historical reports retain their original limitations and must not be treated as current manuscript results.

No new blanket license is assigned in this consolidation. Dataset, upstream code and generated materials retain their original rights and attribution; consult [data provenance](data/README.md) before redistribution.
