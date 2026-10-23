# Release validation

Validation performed on 23 October 2026 with Python 3.12.14 on Windows, from the relocated `finalized_repository` checkout.

| Check | Observed result |
|---|---|
| Core tests, including actual primitive STEP geometry | 57 passed |
| CLI entry point | `python -m cadkg --help` succeeded |
| Distribution packaging | `cadkg-0.1.0-py3-none-any.whl` built successfully without installing dependencies |
| Split archive restoration | All 3,490 archived members restored and SHA-256 verified |
| Full relocated offline replay | Exact SHA-256 equality for all nine compared artifacts; zero API calls |
| Metrics-only replay | Exact SHA-256 equality for five aggregate/statistical CSVs |
| Credential-pattern review | No selected textual file matched standalone OpenAI/GitHub credential or private-key signatures; `.env`, editor state and private QA excluded |

Machine-readable full and metric replay proofs are in `docs/validation/`. Full replay compares graph support, graph evidence, per-query metrics, family metrics, run metrics, summary, paired tests, family differences and top-five ranking indices. Original results are unchanged. A hash inventory check records release file consistency in `inventory_check.json`.

Windows is the locally tested platform. The adapter preserves archived newline conventions for other platforms; Linux execution is configured in CI and was not tested locally. A clean network dependency installation, full raw-STEP preprocessing, encoder retraining, fresh model collection and manuscript Word rendering were not rerun. Historical version pins are environment records, not a claim of fresh-platform installation testing. The encoder and AI-reference limitations in the manuscript remain applicable.

Original files and older runs remain preserved in the source workspace. The release excludes temporary caches, replay duplicates, redundant delivery ZIPs, old manuscript drafts and large regenerable sensitivity arrays. Full primary-study replay uses only included artifacts after restoration.
