# Validation

The publication layout was checked on Windows with Python 3.12.14.

| Check | Result |
|---|---|
| Core software and STEP geometry tests | 43 passed |
| Full offline reproduction through the public command | Nine artifacts match recorded SHA-256 hashes exactly; zero API calls |
| Clean tracked-file checkout | Data preparation and full nine-artifact reproduction pass without local untracked inputs |
| Statistical-table reproduction | Five tables match exactly |
| Data preparation | 3,490 archived files verified; 3,054 source inputs exposed under the public data layout |
| Figure generation in an isolated build | Figures 2–3 generated; all 64 numerical annotations match; no annotation collisions |
| Editable document build in an isolated build | Manuscript and supplement generated successfully from their final source paths |
| Supplied publication assets | Word documents, figures, and numerical tables retain their approved bytes |
| LLM usage examples | Both documented commands succeed; raw-response lookup also verified for all three observation designs |

Run `python -m scripts.validate` to verify the publication manifest, documentation links, and LaTeX figure/table paths. Machine-readable proofs are in [results/validation](../results/validation/).

The supplied reviewed Word files and artwork were preserved; isolated build outputs are not substituted for them. TeX compilation, Word pagination/field updates, encoder retraining, raw-corpus preprocessing, a fresh network dependency installation, and Linux execution were not rerun. Linux core/data/statistical checks are configured in CI. Tests for superseded experimental panels are retained with their implementations in the provenance archive.

These checks establish software and numerical consistency. They do not establish independent engineering validity of the AI-derived reference judgments.
