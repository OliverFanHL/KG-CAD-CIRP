# Study dataset packages

The numbered `study_inputs.zip.partNNN` files contain source data for the study: CAD input images, geometric evidence, model observations, graph data, encoder exports, and initial rankings. Each part is below GitHub's per-file size limit.

From the repository root, run:

```text
python -m scripts.prepare_data
```

The command verifies each package part and input against `data/manifest.json`, then extracts the listed study inputs to the paths in the [data guide](../README.md). The manifest's `source_member` field maps each archive entry to its destination. Preparation also verifies inputs supplied directly in the repository and stops if an existing input has been modified.
