# Study dataset packages

The numbered `study_inputs.zip.partNNN` files contain the packaged study inputs listed in `data/manifest.json`: CAD input images, geometric evidence, model observations, encoder exports, and initial rankings. Each part is below GitHub's per-file size limit.

From the repository root, run:

```text
python -m scripts.prepare_data
```

The command verifies each package part and input against `data/manifest.json`, then extracts the study inputs to the paths in the [data guide](../README.md). Archive paths match their destinations under `data/`. Preparation also verifies inputs supplied directly in the repository and stops if an existing input has been modified.
