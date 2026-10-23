# Provenance archive

This directory is for source verification and maintenance. The final study is presented through the repository README, `publication/`, `results/`, and `data/`.

- `artifacts/`: unchanged lossless artifact parts and the original file/hash inventory.
- `workspace/`: sealed evaluation implementations and configurations, plus restored experimental inputs used internally by the public reproduction command. Its `cadkg/` snapshot preserves source-byte checks; the active library is at the repository root.
- `history/`: prior organizational documentation, manuscript packaging records, review decisions, and maintenance checks.

Original identifiers and paths remain here to preserve the recorded protocol hashes. They do not define the public directory layout. Archived manuscript scripts and historical run commands are not the supported public workflow. Use `python -m scripts.prepare_data` and `python -m scripts.reproduce` from the repository root.
