# Revision 7: protocol, disclosure and title review

Date: 22 October 2026. Bounded ARS scientific review and official OpenAI documentation check. No manuscript, protocol, prompt, study or bibliography files were edited. The proposals below concern decisions to freeze before new responses, not inferred outcomes.

## 1. Repeatability: actual API support

The fetched [official Responses create parameter reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) lists no `seed` argument. The full browser reference exceeded the retrieval size limit; its CLI counterpart was fetched in full and searched. The [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) pages do not document seed support. Low reasoning effort is supported for both. The [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) separately describes deprecated best-effort seeding; this cannot establish support for Responses.

**Conclusion:** do not send an undocumented seed field or describe these Responses collections as model-seeded/deterministic runs. Local pseudorandom seeds may control job ordering and resampling and should be named accordingly. Saved responses permit exact downstream replay, but cannot guarantee regeneration of the same model observations.

The current plan retains the original contract-identical KG collection as replicate 1 and collects two new KG passes for each model, giving **three total collections per model**. It adds three role-only Luna collections. This entails 14,588 new calls (seven collections ×2,084 pairs). Describe the prior collection as historical reuse, not three newly conducted collections or three independent dataset replications. Model identifiers, settings, output caps, view bytes and prompts remain fixed within an arm; record request timestamps, resolved model IDs and failures. Interleaving fresh collections helps distribute temporal/API conditions but does not retroactively randomize the historical run.

Report every retained response once, including invalid and failed outcomes with fixed neutral fallback. No best-of-three response selection, majority-vote substitution, post-result prompt repair or hidden retries. Report between-collection mean and sample SD/range descriptively; three repeats give limited evidence about stochastic variability. They do not increase the number of distinct roles, CADs, exemplars or query designs.

## 2. Fair direct-role visual baseline

The parent's proposed baseline is scientifically defensible with these explicit boundaries:

- Use the same 2,084 role–CAD pairs, original neutral six-view image bytes, fixed eligible pools, Luna identity, low reasoning effort, output cap, strict schema, view citations and failure policy. The baseline should not receive a geometric exemplar, grades, class labels/probabilities, source IDs, ranks, pool order, bound or performance information.
- Supply the **requested role's semantic description alone**, rather than its four authored geometric clauses, triples or same-body bindings. Provide equivalent generic instructions on visible evidence, ambiguity, display clips and the inability to determine materials, load, tolerance or scale. The model can naturally infer what a role usually requires; do not prohibit relevant reasoning merely to weaken the baseline.
- **Approved before protocol freeze:** request one holistic visible-role support score J from the nine-value domain {0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1}, cited views and a short literal rationale. No four-clause decomposition or triples enter this prompt. An invalid whole record receives J=0.5. This replaces the earlier three-state proposal and matches the possible values of H.
- Evaluate two uses of the same J observations: **J+C** ranks the shared eligible refinement pool by J alone, without G/A/F score fusion; **Full+J** uses S=(3s+J)/4 inside the same pool. Apply frozen deterministic tie breaking and appended-tail policy. Clearly state the pool provenance: J+C is a visual-score-only reranker of a computationally acquired pool, not exhaustive LLM retrieval over all 1,816 candidates.
- Avoid copying the AI qrel grading rubric into the baseline, including “strong/conditional/rejected” grades or their reference definitions. A generic instruction to report visible support is sufficient; prompts must not ask the model to predict the benchmark label.
- Match baseline Luna against KG-directed Luna for the causal contrast. Sol KG versus Luna role-only mixes observer identity with observation design and is descriptive, not a matched decomposition contrast.

**Names:** “Role-only LLM, J+C” for J-only scoring under inherited eligibility, and “Full+J” for the hybrid control. A shortened “LLM-only” legend must say that it denotes J-only scoring inside the same shared screened pool; pool acquisition still uses the initial retrieval branches. It is not an end-to-end pure-LLM/full-gallery retriever.

**Identified contrast:** Full+H versus Full+J compares explicit authored geometric decomposition, component bindings, observation format and fixed aggregation **jointly**, relative to direct holistic role assessment under matched fusion. Both signals now have the same nine-value domain, removing the earlier score-granularity difference. This does not make their calibration or informational content identical. J+C separately tests visual-score-only reranking within the shared pool and is not the primary matched-fusion ablation. Positive differences cannot isolate graph storage, same-part binding alone, relational topology alone or visual perception accuracy. A requirement-information-matched flat representation would still be needed to isolate a graph representation advantage. Avoid calling Full+J “without KG” as if every KG-derived input were removed: its shared Full score still contains class-derived F.

## 3. Analysis and integration safeguards

Freeze the actual comparison family, primary endpoint, averaging, tolerance and failure accounting before collecting the new responses. Suggested priority is the same-model KG-directed versus direct-role contrast for both representations. Any additional Sol-versus-Luna and observer-versus-GFC inferential tests need an explicit multiplicity family; do not silently reuse historical adjusted p values as if they belonged to the new repeated analysis.

For an analysis of average collection performance, first compute the complete ranking/metric separately for each response collection. Then average each role's 24 query conditions within each collection and average those three role values. Family inference remains based on eleven paired role means, not 33 nominally independent role×run units or 792 independent queries. The same deterministic GFC score can repeat for display, but its three copies do not add information. If plotting collection SD, identify it as variation across three API collections; do not confuse it with role-bootstrap uncertainty or combine the two into an unlabeled error bar.

Keep a single-observer historical result identifiable alongside repeated means. Recompute any updated main result from all retained repeats with the same fixed procedure. Do not update only favorable cells or silently replace original worked examples/graphs with observations from another repeat or model. Every qualitative example needs model/collection provenance.

Figure 2 can display Base Full, GFC, J+C (Luna), KG-Luna and KG-Sol for both endpoints, with compact direct labels and a caption specifying model, three-collection summaries and error-bar meaning. Include Full+J if its extra series remains legible, or reserve it for the component table. This displays the visual-score-only comparator and both observers without implying Sol superiority.

**Table 2 is a compact component ablation table**, as requested, not a paired-inference table. Use seven rows: G+C; G+A+C; G+F+C; Full without C; Full; Full+J; Full+H. Use four metric columns: embedding nDCG@5 and Strong P@5, descriptor nDCG@5 and Strong P@5. Show mean±sample SD across the three complete observation collections for stochastic methods. Deterministic methods have SD=0 by construction (or show a dash explained in the caption); their replicated display values do not represent fresh experiments. Define Full's retained dimensional screen C, the unfiltered “without C” contrast, and Luna identity for Full+J/Full+H in the caption. Place formal paired tests, intervals, multiplicity families and complete role effects in the supplement. Do not place per-query and per-family uncertainty on the same visual scale without identifying the unit.

Independent audit should verify: request-arm differences; all image/source hashes; no hidden predicate leakage into role-only payloads; one attempt/terminal response per declared collection/pair; fallback counts; direct-J and decomposed-H mappings; control continuity; pool and appended-tail preservation; all relevance envelopes; metric aggregation; exact family sign flips/bootstrap/multiplicity; all positive and negative family effects. The three-run SD is a repeatability description, not a feature-accuracy validation or independent engineering reference.

## 4. CIRP/Elsevier requirements: mandatory versus recommended

Primary sources read on 22 October 2026:

1. [CIRP Annals official submission instructions](https://cirp.net/mainmenu-publications/cirp-annals-col-200/authors-submit-a-paper.html), particularly Policies and submission documents.
2. [Elsevier full generative AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals), Disclosure, explanatory images, reproducible research methods and declaration-location FAQ.
3. [Elsevier policy update, 18 August 2026](https://www-prod.elsevier.com/connect/updated-generative-ai-policies-for-journals-supporting-responsible-use-while-protecting-trust), read as a summary rather than an override of the full policy.

**Required by the fetched policy:** CIRP applies the publisher's current AI policies. Elsevier requires a manuscript-preparation disclosure **in the manuscript**, immediately before references. It must identify tools/purposes and human responsibility; substantive writing and organization assistance is covered. Research AI belongs in reproducible Methods. AI-assisted explanatory figures require caption disclosure identifying the tool/version/use and the general declaration. AI-assisted data visualizations must derive reproducibly from data and be described in Methods. CIRP also requires Declaration of Interest and CRediT submission forms. [CIRP instructions](https://cirp.net/mainmenu-publications/cirp-annals-col-200/authors-submit-a-paper.html); [full publisher policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals).

**Recommended, not mandatory wording:** the long suggested section title is an example. A short declaration can replace the present broad “Use of generative AI” section; a separate upload alone does not satisfy the fetched manuscript-location rule. Six pages is strongly recommended, not absolute. The live CIRP page currently requests 3–5 keywords, first two from the unified list; an indexed snippet differs, so use the fetched page. The publisher journal-specific Guide for Authors was inaccessible (403); no exception was verified. [CIRP instructions](https://cirp.net/mainmenu-publications/cirp-annals-col-200/authors-submit-a-paper.html).

The August update's phrase “separate AI declaration statement at submission” does not mean disclosure is excluded from the paper. The full policy explicitly specifies manuscript location. Explanatory schematics are permitted, while general-purpose generative graphical abstracts are not. Figure 1 is an explanatory methods figure, not a graphical abstract. [Full publisher policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals).

### Concise declaration draft for authors to verify

**Proposed section title:** Declaration of AI-assisted manuscript preparation

**Submission draft, to be approved and completed by the authors after their review:** “OpenAI Codex assisted with manuscript preparation, reference checking, visualization code and figure development. GPT-Image 2 supplied a layout concept for Figure 1, which was rebuilt as editable graphics using source CAD renders. [After completing their review, the authors confirm that they reviewed and edited the assisted material and take responsibility for the article.]”

The bracketed sentence is conditional author-confirmation text and must be replaced by a truthful approved statement before submission. This draft does not assert that final human review has occurred. Add exact available tool/model versions from the retained creation records rather than inventing them. The research-method uses of Luna, Sol, GPT-5 Mini and AI-derived reference judgments remain described in Methods/supplement, distinct from the preparation disclosure. If some models do not enter the revised study/report, do not list them as research methods in the main body merely to make the declaration comprehensive.

**Possible short caption addition:** “The schematic was developed with OpenAI Codex from a GPT-Image 2 layout concept and rebuilt as editable graphics; CAD inserts retain source geometry.” This describes actual assistance without implying that the scientific CAD surfaces were generated. The authors should verify exact version and provenance. Lucide license attribution belongs in provenance/source package and is separate from AI disclosure.

## 5. Six grounded title options

1. **Knowledge-guided relational evidence for functional CAD retrieval** — recommended; concise, role-centered and independent of one observer brand.
2. **Knowledge-directed visual reasoning for functional CAD retrieval** — stronger emphasis on the observation process; “reasoning” is broader than measured predicate observation.
3. **Relational geometric evidence for intent-driven CAD retrieval** — clear engineering problem; KG and model mechanism become less visible.
4. **Knowledge-guided CAD retrieval through relational visual observation** — most literal mechanism, slightly less natural phrasing.
5. **Functional CAD retrieval with knowledge-directed visual language models** — names visual modality and the two evaluated observers; no “first” or superiority claim.
6. **Knowledge-guided interface evidence for functional CAD retrieval** — precise for hooks/journals/drives, but narrower than the entire eleven-role scope.

Option 1 best foregrounds the contribution that the planned role-only baseline can assess: requested role→authored relational requirements→observations→deterministic retrieval evidence. Explain the LLM module in the abstract's first method sentence. If the authors specifically want the title to name the model class, option 5 is balanced. Neither title presupposes a favorable repeatability/ablation outcome or graph-storage superiority.

The parent selected option 1 for revision 7; option 5 remains an optional alternative. This is an editorial selection, not an outcome-dependent claim.

## Material passport

- Origin: ARS bounded scientific/policy review with OpenAI Docs verification.
- Status: primary-source policy/API documentation verified to the access depth stated; protocol recommendations prospective, no outcome claimed.
- Changes: this review file only; no API calls, credentials, experiments, manuscript edits or source-label changes.

## Presentation refinement during acquisition

Table 2 adds the already-existing No-G (A+F+C) control from its verified three
local executions. This gives eight compact configurations and exposes each
initial-score component omission alongside the visual-evidence controls. The
change concerns presentation of an existing result, not the frozen API arms,
queries, scoring, primary comparisons or inference family. The remaining-local
CSV and Full anchor checks bind that row to the same panel.
