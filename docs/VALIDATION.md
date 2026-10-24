# Validation

The code, data workflows, numerical results, and documentation were checked on Windows with Python 3.12.14.

| Check | Result |
|---|---|
| Core software, STEP geometry, study evaluation, and extension tests | 52 passed |
| Full offline reproduction through the public command | Nine artifacts match recorded SHA-256 hashes exactly; zero API calls |
| Clean tracked-file checkout | Data preparation and full nine-artifact reproduction pass without local untracked inputs |
| Statistical-table reproduction | Five tables match exactly |
| Data preparation | 3,090 final-study inputs verified against the dataset manifest |
| Numerical result integrity | Checked-in result files match their recorded SHA-256 hashes |
| LLM usage examples | Both documented commands succeed; raw-response lookup also verified for all three observation designs |
| Baseline extension evaluator | Directed and holistic saved-response checks each match all 528 corresponding query results exactly |

Run `python -m scripts.validate` to verify the [result manifest](../results/manifest.json) and documentation links. Detailed validation records are in [results/validation](../results/validation/).

## Validation scope

The recorded checks cover software behavior and numerical consistency. They exclude encoder retraining, raw-corpus preprocessing, a fresh dependency installation from the network, and Linux execution. Core, data, and statistical checks for Linux are configured in CI.

Annotation provenance is documented in the [data guide](../data/README.md). Engineering relevance requires separate assessment from software and numerical validation.
