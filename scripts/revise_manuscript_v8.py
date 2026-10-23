"""Presentation revision from immutable revision-7 results; no experiments."""
from pathlib import Path
import csv,hashlib,importlib.util,json,re,shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'manuscript/cirp_draft'
OUT=ROOT/'manuscript/revision8'
OUT.mkdir(exist_ok=True)
for name in ('figures','tables','data','scripts','provenance'):(OUT/name).mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,a,b):
    assert s.count(a)==1,a[:100]
    return s.replace(a,b)

# Keep this revision self-contained for figure/table reproduction.
for name in ('summary.csv','family_metrics.csv','paired_tests.csv','remaining_local_three_pass.csv'):
    shutil.copyfile(OLD/'data/robustness_v7'/name,OUT/'data'/name)
shutil.copyfile(OLD/'scripts/generate_revision7_assets.py',OUT/'scripts/generate_revision7_assets.py')
for name in ('framework_redrawn.png','llm_evidence_graph.pdf','llm_evidence_graph.png'):
    shutil.copyfile(OLD/'figures'/name,OUT/'figures'/name)
for name in ('component_ablation_v7.tex','repeated_paired_tests_v7.tex'):
    shutil.copyfile(OLD/'tables'/name,OUT/'tables'/name)
for name in ('references.bbl','references.bib'):
    shutil.copyfile(OLD/name,OUT/name)

main=(OLD/'main.tex').read_text(encoding='utf8')
main=main.replace('{\\small Author names and affiliations to be supplied by the authors\\par}\n\\vspace{9pt}\n','')
main=main.replace('\\resizebox{\\textwidth}{!}{\\input{figures/framework_overview.tex}}',
                  '\\includegraphics[width=\\textwidth]{framework_redrawn.png}')
main=main.replace('repeated_model_performance.pdf','figure2_observer_comparison.pdf')
main=main.replace('repeated_role_contrasts.pdf','figure3_role_differences.pdf')
main=main.replace('Supplementary Section~S8','Supplementary Section~S2')
main=main.replace('Supplementary Section~S9','Supplementary Section~S3')
main=main.replace('Supplementary Section~S4','Supplementary Section~S4')
main=main.replace('Supplementary Section~S2 details source exclusion, selection and the broader existing set of 440 requests.',
                  'Supplementary Section~S3 details exemplar selection and relevance assessment.')
main=main.replace('Their scopes and sources appear in Supplementary Section~S1.',
                  'Their scopes appear in Supplementary Section~S2; engineering sources accompany the role-association data.')
main=main.replace('Input images retain the original resolution and all CAD pixels; only independently verified title-only rectangles are cleared.',
                  'The 1500$\\times$1000 input montage preserves CAD geometry and view arrangement, with descriptive titles removed.')
main=main.replace('The full forty-four clause definitions accompany the methods.',
                  'Supplementary Table~S1 summarizes all forty-four clauses.')
main=main.replace('Source hashes bind each record to its image and model response.',
                  'Each observation records its source view and model provenance.')
start=main.index('Three collections are retained for each visual observer:')
end=main.index('The role-only observer receives',start)
main=main[:start]+'''Each observation design is evaluated in three response collections: KG-directed
GPT-6 Luna, KG-directed GPT-6.1 Sol and holistic role-only GPT-6 Luna. The gallery,
images, requested requirements, output schema and scoring weights are fixed.
Collections assess variation in model observations under a common representation
checkpoint. The first KG collection for each model precedes the interleaved
repetitions; the API provides no sampling-seed control. A failed assessment
receives neutral support of one half, as specified before acquisition.

'''+main[end:]
start=main.index('Each LLM collection produces its own rankings')
end=main.index('\\section{Evaluation}',start)
main=main[:start]+'''Rankings and metrics are calculated separately for each collection. Reported
SDs describe variation among the three collection means. Local retrieval uses a
fixed encoder and deterministic scoring; its repeated executions give identical
rankings. Variation from encoder retraining or surface sampling is not assessed.

'''+main[end:]
main=main.replace('Review of 126 sources from 156 ordered candidates gives 37 strong and seven conditional exemplars.',
                  'The selected exemplars comprise 37 strong and seven conditional forms under the source-view relevance rubric.')
main=main.replace('The original seven-method comparison, including\nreciprocal-rank fusion \\cite{cormack2009rrf}, remains a secondary analysis in\nSupplementary Section~S3.',
                  'A rank-fusion comparator \\cite{cormack2009rrf} provides an additional check on the initial score combination (Supplementary Section~S4).')
main=main.replace('The analysis concerns top-five refinement; complete arrays retain the appended tail for reproduction.',
                  'Evaluation focuses on the top-five results affected by visual refinement.')
main=main.replace('\\subsection{Statistical analysis and reproduction}', '\\subsection{Statistical analysis}')
start=main.index('Saved responses reconstruct observations')
end=main.index('\\section{Results and discussion}',start)
main=main[:start]+'''Source-linked observations, ranking arrays and metric data accompany the paper.
The numerical evaluation is independently reproduced from the retained responses,
including all admissible grades for the two unresolved reference judgments.

'''+main[end:]
main=main.replace('The repeated means support further investigation of the authored observation design, while the role-level tests leave its advantage uncertain.',
                  'The positive mean differences are consistent with a benefit from requirement-directed observation, although uncertainty across roles prevents a confirmed advantage after multiplicity correction.')
main=main.replace('The archived observations and post-hoc selection rule accompany Supplementary Section~S3.',
                  'The observations and qualitative selection procedure are supplied in Supplementary Section~S4.')
main=main.replace('Additional source-bound CAD comparisons appear in Supplementary Section~S4.',
                  'Supplementary Section~S4 examines an additional case involving dimensional-reference uncertainty.')
main=main.replace('The first KG\ncollections precede the interleaved repetitions, and three collections provide\na limited description of acquisition variation.',
                  'Three collections provide a limited estimate of acquisition variation.')
start=main.index('The repeated study adds 14,588 API calls.')
end=main.index('\\section{Conclusions}',start)
main=main[:start]+main[end:]
start=main.index('Accompanying materials include LaTeX sources')
end=main.index('\\section*{Declaration',start)
main=main[:start]+'''Accompanying materials provide numerical results, figure-generation scripts,
role requirements, observation contracts, source-linked model assessments and
ranking records. Retained responses support metric reconstruction without
additional model calls; complete replay uses the original source dataset.
TMCAD is available through its original source \\cite{zou2025brt}.

'''+main[end:]

# Directly labelled statistics occupy their own columns, away from error bars.
fig2cap=r'''Observer comparison on the common gallery and query panel: a, nDCG@5; b, Strong P@5. Circles and squares identify the embedding and descriptor representations. Points report three-collection means; capped error bars show sample SD across collection means. Numerical SDs below 0.001 are omitted for clarity; all error bars retain their actual values. The fixed $G+F+C$ control is deterministic. $J$ is holistic Luna support: Full+$J$ retains the initial score, whereas $J+C$ ranks the inherited eligible pool by visual support alone. KG-directed Luna and Sol use clause coverage $H$.'''
fig3cap=r'''Role-specific nDCG@5 differences from the Full+$J$ holistic Luna control: a, embedding; b, descriptor. Numerical labels show mean $\pm$ sample SD of three paired collection differences. SD labels below 0.001 are omitted; error bars retain their actual values. Each collection averages four exemplars and six size conditions within the role. Positive values favour KG-directed observation; the vertical reference marks zero. Luna compares observation designs with the same model. Sol also changes the observer and is descriptive. These SDs describe acquisition variation, not confidence intervals across roles.'''
for label,caption in [('fig:performance',fig2cap),('fig:families',fig3cap)]:
    pattern=r'(\\begin\{figure\*\}.*?\\caption\{)(.*?)(\}\s*\\label\{'+re.escape(label)+r'\}.*?\\end\{figure\*\})'
    # Restrict the match to the float owning this label.
    blocks=list(re.finditer(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',main,re.S))
    target=next(m for m in blocks if '\\label{'+label+'}' in m.group())
    block=target.group(); cs=block.index('\\caption{')+9; ce=block.index('\n\\label')-1
    new=block[:cs]+caption+block[ce:]
    main=main[:target.start()]+new+main[target.end():]

# In-flow figures follow the argument in the editable Word version.
floats={}
for label in ('fig:performance','tab:paired','fig:families','fig:evidence'):
    env='table' if label.startswith('tab') else 'figure'
    m=next(m for m in re.finditer(r'\\begin\{'+env+r'\*\}.*?\\end\{'+env+r'\*\}',main,re.S) if '\\label{'+label+'}' in m.group())
    floats[label]=m.group();main=main[:m.start()]+main[m.end():]
anchors={
    'fig:performance':'Each result uses the same queries, representation, size rules and candidate access.',
    'tab:paired':'holistic and clause-directed visual evidence on the same Full score.',
    'fig:families':'Figure~\\ref{fig:families} resolves the difference from holistic assessment by role.',
    'fig:evidence':'The graph-directed observations assign no to all four shaft clauses for this short annular body and yes to all four for splined shaft Z0972, giving $H=0$ and $H=1$.'}
for label,anchor in anchors.items():main=replace_once(main,anchor,anchor+'\n\n'+floats[label]+'\n')
(OUT/'main.tex').write_text(main,encoding='utf8')

supp=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath,booktabs,tabularx,array,graphicx,amsmath,microtype,caption}
\usepackage[margin=20mm]{geometry}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary methods and results\\Knowledge-guided relational evidence for functional CAD retrieval}
\author{}\date{}
\begin{document}\maketitle

\section{Geometric representations and numerical reference}

The thirteen descriptor coordinates are the smallest/largest and middle/largest
box-extent ratios; solid-volume/box-volume and surface-area/box-surface-area
ratios; face, edge and solid counts; planar, cylindrical, conical, toroidal and
B-spline face fractions; and circular-edge fraction. Missing values use training
medians. Training means and population standard deviations determine
standardization, with scale one for deviations below $10^{-8}$. The same
transformation applies to validation and test models.

The supervised local-surface encoder takes six input channels and uses
neighborhood edge blocks of widths 64 and 128. Global maximum, mean and
standard-deviation pooling produces 384 coordinates. A 384--256--128 projection
uses ReLU and dropout 0.2; a ten-class linear head supplies softmax probabilities.
Validation macro-F1 selects epoch 17. The selected encoder, classifier and
descriptor normalization remain fixed. Surface sampling and encoder training use
seed 42 and are not repeated; three retrieval executions with order seeds 17, 29
and 43 produce identical local ranking arrays. Thus no training-seed variance is
estimated.

STEP processing sets the system length unit to 1.0~mm before root transfer,
checks non-null shapes with BRepCheck and computes file-axis bounds with
BRepBndLib.AddOptimal. Triangulation and shape-tolerance expansion are disabled
for these bounds. Declared-unit, valid-shape measurements determine bounded-query
eligibility. Unknown or invalid dimensions exclude a candidate from a bounded
request. The independent tessellated-surface relevance reference applies
face-location transforms, angular deflection 0.25~rad and source-specific linear
deflection (0.05~mm for the illustrated candidate records).

An extent failure receives relevance zero. Missing or quarantined surface
evidence, or an extent within 0.2~mm of the requested bound, caps relevance at
grade 1. The two unassessable functional judgments retain all admissible grades
0--2 through this rule. Their metric bounds coincide for the reported primary
top-five endpoints. This numerical uncertainty treatment remains distinct from
uncertainty in the visual model's observations.

\section{Role priors and visual requirements}

Table~\ref{tab:requirements} specifies the category association and four visual
clauses for each role. All admitted class--role associations have unit relevance
and inclusion weights. Class probabilities therefore supply category-level
function support, while the clauses request candidate-specific interface and
binding evidence. Complete wording, subject--relation--object triples and
engineering-source references accompany the data.

\begin{table}[htbp]\centering\small
\begin{tabularx}{\textwidth}{@{}p{27mm}p{19mm}X@{}}
\toprule
Role & Prior class & Four visual clauses \\
\midrule
Shaft support & bearing & Circular seat; carrier mounting interface; seat--mount binding to one body; stationary carrier form. \\
Shaft connection & coupling & Two receiving ends; positive attachment at the first; positive attachment at the second; connection joining them. \\
Belt wheel & pulley & Rotating rim; circumferential belt track; belt-profile evidence; track bound to that rim. \\
Threaded clamping & bolt-screw & Connected head and shank; head bearing face; modeled external shank thread; head turning interface. \\
Nut retention & nut & Compact bored body; body turning feature; modeled internal thread; bore/turning-feature binding. \\
Bracket mounting & bracket & Two angled or offset faces; rigid connection; mounting feature on each region. \\
Flanged passage & flange & Open central passage; surrounding bolt pattern; flat mating face; coaxial register or neck. \\
Gear meshing & gear & Cylindrical wheel or gear band; external teeth; spur/helical form; exposed meshing band on that cylinder. \\
Shaft transmission & shaft & Elongated body; journal regions; drive feature on that body; rotary shaft form. \\
Compression return & spring & Helical coil; open pitch; axial seating ends; ends free of hooks or torsion arms. \\
Tension return & spring & Helical coil; first terminal hook; second terminal hook; opposite-end attachments on the same coil. \\
\bottomrule
\end{tabularx}
\caption{Category priors and visual clauses. Each role has four clauses; the bracket's two mounting clauses are expressed together for readability. Clauses specify preliminary visible support, not verification of materials, loading, fit or tolerance.}
\label{tab:requirements}
\end{table}

The input montage contains oblique, opposite, profile and axial views and two
uncapped display clips. Instructions distinguish clip openings from real
passages. KG-directed observers receive the four clauses and their bindings;
holistic Luna receives the plain role purpose and the same generic guidance.
Neither input contains class probabilities, rankings, relevance labels,
exemplar identities or numerical size constraints. KG states are yes, no or
unknown, each with cited views and a short literal description. Unknown covers
occlusion, simplified geometry and ambiguity. Both GPT models use low reasoning
effort, a 2,000-token output limit and a strict structured schema.

Holistic support uses $J\in\{0,1/8,\ldots,1\}$, with zero for visible
incompatibility, one for strong support and one half for unresolved support.
This matches the range and increments of four-clause coverage $H$. Both inputs
request view-linked evidence. Invalid or failed assessments receive whole-record
neutral support of one half. This fallback applies to 18 of the 18,756 observations
across all nine collections (0.10\%); all remain in their assigned collections.
The three-collection comparison retains the original KG collection for each
model and two further contract-identical collections. Consequently the comparison
is conditional on the existing query and relevance panel.

\section{Query design and statistical comparisons}

The primary 264 queries cross eleven roles, four training exemplars per role
and six size conditions. Eligible exemplars have distinct STEP bytes, with
validation/test exact-byte overlaps and previously used exemplars excluded.
Selection proceeds by farthest-point distance in training-normalized descriptor
space from development anchors. Neutral-view assessment retains the first four
grade-1/2 exemplars per role, including reserve candidates when necessary.
Assessment covers 126 of 156 ordered candidates and yields 37 strong and seven
conditional exemplars. Category labels, previous judgments and retrieval outcomes
are hidden during source assessment; reserve selection is aware of aggregate
shortages. Descriptor-conditioned sampling and prior dataset inspection limit
external generalization.

Each exemplar supplies one unbounded query and five upper bounds from validation
extent quantiles 0.10, 0.25, 0.50, 0.75 and 0.90. Linear interpolation and rounding
to six decimal places define the bounds; exact test-source overlaps are excluded
from the validation measurements. The panel therefore contains 44 unbounded and
220 bounded queries. The gallery, query exemplars, dimensional evidence, role
clauses, candidate pools and scoring weights are common to all methods.

Metrics first average six conditions within each exemplar, four exemplars
within each role and eleven roles equally. Means and sample SDs describe three
complete response collections; the SD denominator is two. For statistical
contrasts, collection effects are first averaged within each role. Eleven paired
role effects, rather than 33 role--collection observations, are the inferential
units. Exact two-sided sign flips enumerate 2,048 patterns; one Holm correction
covers the four nDCG@5 contrasts in Table~\ref{tab:contrasts}. Percentile intervals
use 20,000 role-bootstrap resamples with seed 20261023 and are unadjusted.
Sign exchangeability of the role effects is assumed. These intervals and
acquisition SDs quantify different sources of uncertainty.

\begin{table}[htbp]\centering\small
\resizebox{\textwidth}{!}{\input{tables/paired_contrasts_v8.tex}}
\caption{Primary paired contrasts from three-collection role means. Intervals are unadjusted 95\% role-bootstrap intervals. Holm correction covers the four exact sign-flip tests. W/T/L counts role wins, ties and losses.}
\label{tab:contrasts}
\end{table}

KG-directed Luna has positive mean effects relative to holistic Full+$J$ in
nine of eleven roles for each representation. Neither contrast meets the
Holm-adjusted 0.05 threshold (0.090820 for embeddings; 0.050781 for descriptors),
although the unadjusted bootstrap intervals exclude zero. Sol-minus-Luna
contrasts are unresolved and do not establish model equivalence.

\section{Additional comparisons and interpretation}

A secondary initial-ranker comparison replaces equal score fusion with
reciprocal-rank fusion of the same $G,A,F$ signals after dimensional screening.
It averages $1/(60+\operatorname{rank}_j)$ across the three signals, using average
ranks for ties. Base Full exceeds this comparator by 0.0467 and 0.1093 nDCG@5
for embeddings and descriptors. Both Holm-adjusted probabilities are 1 in the
separate twelve-contrast initial-ranker family. This comparison does not resolve
a preferred fusion rule; the main evidence concerns visual observation design.

An embedding belt-wheel query with a 42~mm bound illustrates the distinction
between visual support and dimensional-reference uncertainty. Its leading
candidates all have $H=1$, so coverage leaves their initial order unchanged.
Z0114 passes the serving screen but has two missing faces in the independent
surface reference and lies at the bound, capping its evaluated grade at 1.
The clear-pass candidate preferred by $G+F+C$ retains grade 2. The resulting
nDCG@5 is 1.0000 for $G+F+C$ and 0.9026 for Base Full and KG-directed Luna
collection 1. This case shows why apparently complete visible interfaces cannot
resolve uncertainty in dimensional relevance.

The main qualitative examples are post-hoc interpretations, not additional
feature-accuracy tests. For selected positive and negative roles, diagnostic
queries have effects closest to their role's three-collection mean. Candidate
Z0939 additionally illustrates a role-scope disagreement: the reference accepts
terminal tensile attachments on a connected paired-coil assembly, whereas the
authored tension binding requires opposite ends of one coil. Clauses and
reference grades are unchanged. Model observations and the AI-derived reference
require independent engineering adjudication before claims of feature truth.

The accompanying numerical snapshots and scripts regenerate the publication
figures and tables without additional model calls. Source-linked prompts,
responses, coverage records and ranking arrays support full analytical replay
using the original CAD gallery, queries and relevance data. Independent scalar
and saved-response calculations agree to floating-point precision. These checks
establish computational consistency, not independent validation of the relevance
judgments. Quantitative artwork uses SciencePlots with Matplotlib.

\end{document}
'''
(OUT/'supplement.tex').write_text(supp,encoding='utf8')
manifest=dict(revision=8,source_revision=7,experimental_results_changed=False,
    data_sha256={p.name:sha(p) for p in (OUT/'data').glob('*.csv')},
    source_main_sha256=sha(OLD/'main.tex'),source_supplement_sha256=sha(OLD/'supplement.tex'),
    retained_scope='Same fixed data, response collections, fallback policy, means, SDs and inference.',
    editorial_changes=['Remove acquisition/accounting logs from the scientific narrative.',
        'Consolidate supplement into four methodology/results sections.',
        'Keep reproducibility records in data archives rather than document prose.'])
(OUT/'provenance/editorial_revision.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(OUT),main_words=len(main.split()),supplement_words=len(supp.split())),indent=2))
