# Study dataset packages

The numbered `study_inputs.zip.partNNN` files contain source data for the study: CAD input images, geometric evidence, model observations, graph data, encoder exports, and initial rankings. Each part is below GitHub's per-file size limit.

From the repository root, run:

```text
python -m scripts.prepare_data
```

The command verifies each part and every input against `data/manifest.json`, then prepares only the listed final-study inputs at the paths in the [data guide](../README.md). `source_member` maps a packaged record to its public destination. Unlisted package members are not extracted. The command refuses to overwrite modified inputs. The directly checked-in judgment release is verified and preserved.
