# Publication materials

**Knowledge-guided relational evidence for functional CAD retrieval**

- [Manuscript](manuscript/manuscript.docx) and [LaTeX source](manuscript/manuscript.tex).
- [Supplementary methods and results](supplement/supplement.docx) and [LaTeX source](supplement/supplement.tex).
- [Figures](figures/README.md), including vector artwork and editable framework graphics.
- [Tables](tables/README.md), with numerical sources in [results](../results/README.md).
- [Bibliography](manuscript/references.bib).

The supplied Word documents contain editable prose, tables, and LaTeX formula text. Their captions and cross-references use Word fields. Source layouts and scientific results are preserved.

To compile LaTeX, run your TeX compiler from `publication/manuscript/` with `manuscript.tex`, or from `publication/supplement/` with `supplement.tex`. Relative references resolve the shared figure/table folders. The sources require the packages declared in their preambles, including newtx, booktabs, and microtype. TeX compilation was not rerun for this structural update.

To regenerate Figures 2–3, install `publication/requirements.txt`, then run `python -m scripts.generate_figures` from the repository root. Their numerical sources are the final CSVs in `results/`. To rebuild the editable Word files, run `python -m scripts.build_documents`; this replaces the supplied documents, and Word must update its fields and pagination. Keep the supplied reviewed documents unless regeneration is needed.
