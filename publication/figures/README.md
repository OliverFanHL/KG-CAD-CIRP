# Figures

| Figure | Content | Files |
|---|---|---|
| 1 | Retrieval and relational-evidence framework | [PNG](figure1_framework.png), [editable PowerPoint](figure1_framework.pptx) |
| 2 | Observer comparison: nDCG@5 and strong precision | [PDF](figure2_observer_comparison.pdf), [SVG](figure2_observer_comparison.svg), [PNG](figure2_observer_comparison.png) |
| 3 | Role-specific nDCG@5 differences | [PDF](figure3_role_differences.pdf), [SVG](figure3_role_differences.svg), [PNG](figure3_role_differences.png) |
| 4 | Source-linked evidence explaining a ranking change | [PDF](figure4_evidence_graph.pdf), [PNG](figure4_evidence_graph.png) |

Figures 2–3 use the [final summary](../../results/summary.csv) and [role-level metrics](../../results/family_metrics.csv). Error bars show sample SD across three observation collections; they are not role-level confidence intervals. `python -m scripts.generate_figures` regenerates them from those tables. Source geometry and explanatory observations are indexed in [data](../../data/README.md). Icon licensing is retained in `LUCIDE_LICENSE.txt`.
