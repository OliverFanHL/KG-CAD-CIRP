# Revised manuscript and supplementary materials

**Knowledge-guided relational evidence for functional CAD retrieval**

Revision 8 contains the complete editable manuscript (nine Word pages) and
streamlined supplement (three Word pages, four sections, two tables). Text,
tables, formulas, captions and references are editable. Formulas are supplied as
unrendered LaTeX source text, including the seven numbered display equations.
All four tables use three horizontal rules: top, header separator and bottom,
without vertical rules, body-row rules or shading. Figure captions and
cross-references use Word fields.

Figures 2 and 3 use a consistent SciencePlots/Matplotlib style. Numerical
annotations occupy separate columns, with paired markers on distinct rows and
legends above the plotting area. Labels use three decimals. Numerical SD labels
below 0.001 are omitted; every error bar retains the measured SD. Tables retain
their original precision and the CSV sources retain full precision. Figures are
supplied as vector PDF/SVG and 450-dpi PNG files.

The revision changes presentation and prose only. The query panel, candidate
pools, observer collections, relevance judgments, scores, means, SDs and
statistical tests remain unchanged. No additional experiment or API request was
made. The main manuscript preserves the corrected, inconclusive role-level
comparisons and the distinction between acquisition variability and unmeasured
encoder-training variability.

The supplement retains geometric representations, numerical relevance rules,
the eleven roles and forty-four visual clauses, query selection, the primary
paired comparisons and additional interpretation. Acquisition logs, cost-ledger
history, duplicated initial-observer analyses, the auxiliary language-interface
study and unrelated implementation timings are omitted from the document. Their
original records remain available in the revision-7 study archive.

## Regenerating figures and editable documents

With Python 3.12 and the packages in `requirements.txt`, run from this folder:

```powershell
python scripts/generate_figures.py
python scripts/build_word.py
```

The figure script reads only the packaged CSVs. It performs no new experiment or
hypothesis test. The helper named `generate_revision7_assets.py` provides retained
metric-reading functions; its old figure entry point is not used by this revision.
The Word builder reads the revised LaTeX prose, native table sources, bibliography
and figure images. Open the generated documents in Word to update fields as
needed. The supplied final files already have updated fields and have been
rendered and visually reviewed in Microsoft Word.

`provenance/final_validation.json` records numerical and document checks;
`provenance/visual_review.json` binds the final reviewed files. The private `.qa`
directory contains rendering intermediates and is excluded from delivery archives.

The complete archived model responses and original source dataset are not
duplicated here. Full metric replay uses the unchanged source CAD/query/reference
workspace identified by the original study hashes; publication-artwork generation
needs only the packaged numerical data.
