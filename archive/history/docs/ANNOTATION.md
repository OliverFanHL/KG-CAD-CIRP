# Human review: what to annotate and how

The reviewer judges visible shape evidence for a **preliminary engineering shortlist**. This is not certification of strength, actual use, thread fit, gear mating or compatibility. Class labels are not functional ground truth.

## Three separate sections

| Section | Human decision | Scope |
|---|---|---|
| Task definitions | Accept the request/criteria or explain a needed change | 11 draft tasks, reviewed before completing the study |
| CAD relevance | Shortlist, Conditional, Reject or Cannot assess, with visible evidence | 674 task–part records over 424 source CADs |
| Generic knowledge | Accept, reject or request changes to a scoped class–function prior | 11 priors; needed before claiming expert-curated KG knowledge |

Independent retrieval evaluation requires relevance assessment. Generic knowledge review is separate: test-part labels never enter the serving KG. An expert need not annotate every part, measurement or possible pair in TMCAD.

## Expanded cohort status

**Current canonical release:** `annotations/intent_candidates1816_v4/judgments.json` contains complete AI shape judgments for all 1,816 eligible held-out CADs (19,976 task–part records). See [the release](../annotations/intent_candidates1816_v4/README.md) and [audit/evaluation](../outputs/candidate_scale_v4/REPORT.md). The v4 audit checked every parent record's integrity, visually rereviewed 135 parent CADs and corrected 32 grades before scoring. All 466 additions were visually assessed under the unchanged scopes. This is shared-context AI assessment, not independent human validation; unsampled parent CADs were not revalidated. Existing human app queues and exports remain distinct and must not be represented as reviewing the whole v4 release.

The following describes historical expansion queues and reproduction inputs:

The separate `annotations/intent_batch2_expanded_2x/judgments.json` queue targets
1,350 eligible held-out CADs × 11 tasks = 14,850 assessments. Its unreviewed
pairs have null grades and must not be counted as complete labels or rejects.
See `outputs/intent_batch2_expanded_2x/coverage.json` for actual completion.
The earlier 7,425-record expansion contained 6,751 category-derived proxy
grades. Those added grades have been withdrawn; their archived artifact is
explicitly invalid for relevance evaluation. The original 674 AI inspections
remain available separately. Class labels may balance sampling, but cannot
establish a candidate's relevance.

The fresh matrix reproduction under `outputs/intent_batch2_matrix_reproduction/`
uses the original 674 AI judgments. It is not an evaluation of a completed
14,850-assessment dataset.

## Use the local app

```powershell
python -m cadkg annotate --open
```

Substitute the neighboring Python environment from the README if needed. The server binds only to `127.0.0.1`, normally port 8765; `--port 8766` changes the port. No upload or external service is needed.

1. Enter a stable reviewer ID such as `expert_a` and confirm you are doing the review yourself. This is self-attestation, not authentication or a credentials check. Work resumes with the same ID.
2. In **Task definitions**, read the request and rubric. Accept only if clear and appropriate. Requested changes are saved as notes; they do not modify the frozen experiment. A changed request requires a new frozen protocol before new labels are collected.
3. In **CAD relevance**, select a task/candidate. Rotate/zoom the source surface; Shift-drag pans. Standard views and a neutral STEP download are available. Cutaway is an uncapped visual slice, not an exact measured section. Double-click a surface for an optional marker; its normalized coordinates are evidence locators, not dimensions or interface facts.
4. Mark features **Visible / Visibly absent / Unclear**. Choose a decision and evidence reason. Add a short note when an explanation is needed. Valid CAD decisions autosave; **Save & next** checks validity. Incomplete edits stay unsaved and show an explanation. Queues distinguish not reviewed from Cannot assess.
5. In **Generic knowledge**, inspect the scope, limitations and source. Use **Save review**. Approval of a generic relation does not assert that every CAD in that category performs the function.

No AI grade, predicted class, method identity, rank or score is displayed. Pool order is deterministic and unrelated to rank. STEP/evidence hashes are checked. Original STEP content may contain identifying metadata; this is not a claim of fully blinded review.

## Decisions

| Decision | Meaning | Grade |
|---|---|---:|
| Shortlist | All required visible shape features support the role | 2 |
| Conditional | Positive evidence, but a specific requirement is unverified | 1 |
| Reject | Visible shape conflicts with the request | 0 |
| Cannot assess | Another view or document is needed to decide | null, never 0 |

Illustrative instructions, **not new verified TMCAD annotations**:

- Compression coil with clear seating ends and axial travel → Shortlist; record each feature.
- Wrenchable bored nut-like body with unresolved internal thread → Conditional; name the missing evidence.
- Extension hooks conflict with the requested compression seating form → Reject; record the end-feature mismatch.
- Hidden mating region prevents interpretation → Cannot assess; request another view/drawing.

Do not estimate material, spring rate, gear module, thread designation, pressure rating or load capacity from shape alone. Similar-looking parts are not proven interchangeable.

## Pilot and disagreement review

Start with approximately 20–30 shared cases across tasks and ambiguous forms. Two reviewers use different IDs without seeing each other's answers. Resolve task clarity before scaling up; the software does not manufacture a human pilot.

```powershell
python -m cadkg annotate --action compare --reviewers expert_a expert_b
```

This saves overlap counts, agreement, quadratic-weighted kappa and a disagreement CSV under `outputs/expert_review/comparisons/`. Cannot assess is excluded from numerical kappa, with its count reported. Kappa is undefined when assessed overlap has no category variation. This is descriptive agreement, not retrieval effectiveness.

For adjudication, inspect the disagreement CSV together, reopen the corresponding task/CAD, and document resolved decisions under a separate adjudicator ID. Preserve both original reviews. The app does not choose a majority grade or expose peer answers during independent annotation. Any rubric change after the pilot must be frozen and reviewed again before final evaluation.

## Export and evaluate

```powershell
python -m cadkg annotate --action export --reviewer expert_a
python -m cadkg expert-evaluate --review outputs/expert_review/exports/expert_a/review.json
```

**Download review JSON** also exports the current source-bound record. CLI export produces `review.json`, `judgments.csv`, task/knowledge reviews and `knowledge/{functions,class_functions,interfaces}.csv`. Only accepted generic priors enter the knowledge folder. Relevance labels never do. Revision history, reviewer, time, observations and source bindings are automatic; humans do not type hashes or IDs.

Expert evaluation uses frozen full-gallery rankings and the same metric engine as the AI study, writing a new directory. Every pooled candidate in a family must be assessable and its task accepted. Missing/unresolved families block the default run; `coverage.json` lists remaining work. `--allow-partial` includes **complete families only**, with all exclusions reported. Partial results must not be compared directly with all-family averages. No unjudged part is filled with grade 0.

Family labels can be reused across variants only because they request the same role; numeric envelopes are applied separately. A different function, mate or specific shape preference requires separate judgments. Current automatic screening covers stored-axis largest extent with a provisional 0.2 mm boundary guard, not optimized packing or independent metrology.

Reviewed knowledge can build a **new development bundle** through `build --annotations .../knowledge`; preserve the frozen test KG and never tune weights on test judgments. Ideally different experts author KG priors and judge test relevance.

Optional additions: reviewed bore/shaft roles, drawing-backed dimensions/tolerances, actual function metadata and narrowly specified compatibility cases. Blank `annotations/templates/interfaces.csv` and `clearance_cases.json` support the existing clearance screen. The visual marker does not generate these engineering facts.

Legacy CSV packs remain usable. Do not replace AI provenance with a human source string. Automated UI validation uses `--validation-fixture`; those records are non-human and expert evaluation rejects them.
