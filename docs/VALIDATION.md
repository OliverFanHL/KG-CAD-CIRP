# Validation

The code, data workflows, numerical results, and documentation were checked on Windows with Python 3.12.14.

| Check | Result |
|---|---|
| Core software and STEP geometry tests | 43 passed |
| Full offline reproduction through the public command | Nine artifacts match recorded SHA-256 hashes exactly; zero API calls |
| Clean tracked-file checkout | Data preparation and full nine-artifact reproduction pass without local untracked inputs |
| Statistical-table reproduction | Five tables match exactly |
| Data preparation | 3,490 archived files verified; 3,054 source inputs exposed under the public data layout |
| Numerical result integrity | Checked-in result files match their recorded SHA-256 hashes |
| LLM usage examples | Both documented commands succeed; raw-response lookup also verified for all three observation designs |

Run `python -m scripts.validate` to verify the [result manifest](../results/manifest.json) and documentation links. Machine-readable proofs are in [results/validation](../results/validation/).

Encoder retraining, raw-corpus preprocessing, a fresh network dependency installation, and Linux execution were not rerun. Linux core/data/statistical checks are configured in CI. Tests for superseded experimental panels are retained with their implementations in the provenance archive.

These checks establish software and numerical consistency. They do not establish independent engineering validity of the AI-derived reference judgments.
