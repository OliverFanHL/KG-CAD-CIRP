# Matched evaluation of KG-directed geometric evidence

The single proposed addition is an internal visual evidence module. The KG
specifies four role-dependent geometric predicates and their component bindings.
GPT observes their status in six fixed CAD views. Stored observations retain
view provenance and uncertainty; they are model observations, not independently
verified geometric facts. Graph rules aggregate yes=1, no=0, unknown=0.5 into
H, the mean of the four clauses. A failed request or invalid response has H=0.5.
The existing largest-axis dimensional filter is unchanged and never delegated
to a language model.

The matched panel is exactly the latest v6 primary population: 264 queries,
44 training exemplars, eleven roles, six conditions per exemplar, two geometry
representations and the same 1816 held-out CAD candidates. No new query, label,
embedding, classifier, measurement or CAD rendering is generated. A fixed pool
is the union of the top ten candidates from Full, G+F+C and G+C per query and
representation. Its membership is frozen before API execution. Each
role-candidate pair is observed once and cached across all relevant queries.
Only requested geometric predicates and neutral views are disclosed to GPT;
no grades, class labels, model scores, rankings or exemplars are disclosed.

Exactly three ranking rows are evaluated: original Full, the existing strongest
G+F+C control, and Full plus the new evidence. The proposed pool score is
(3 Full + H)/4, giving H the same weight as each of G, A and F. The reranked
pool precedes the unchanged Full tail. This is bounded retrieval with evidence
refinement, so it cannot recover a candidate outside the frozen pool. There is
no weight sweep, prompt selection, alternate model comparison or outcome-based
pool expansion.

Primary outcomes are nDCG@5 and strong P@5 under unchanged AI-reference v5
judgments and unchanged numerical transformation (0.2 mm guard). Null judgments
retain their full grade envelopes. Six conditions are averaged within exemplar,
four exemplars within role, and eleven roles equally. The statistical unit is
the role, not the 528 repeated query/representation evaluations. Four exact
two-sided role sign-flip tests compare the proposed method against Full and
G+F+C separately in each representation, with one Holm correction. Unadjusted
95% role-bootstrap intervals use 20000 resamples. No test is interpreted when
the relevant primary nDCG is not point-identified.

Before responses or new outcomes are inspected, the manuscript-update gate is
fixed: at least +0.02 mean nDCG@5 over G+F+C in BOTH representations, Holm
p<=0.05 for BOTH such comparisons, and no decrease in strong P@5 against G+F+C
in either representation. Otherwise all results stay in this separate study.
Passing the gate permits an exploratory manuscript addition; it does not
establish independent industrial validity, graph-storage novelty, or a causal
benefit beyond this fixed computational intervention. The historical collection
has already been examined, and the relevance reference itself is AI-derived;
another GPT may share its visual assumptions. Independent expert adjudication
is still needed for an external engineering claim.

Use GPT-6 Luna with low reasoning and a fixed 2000-token output cap, Responses
API, store=false, at most twelve simultaneous requests and a USD10 spending ceiling. Keep
the exact returned model identifiers, token usage, cost estimates, durable
request journal, raw responses, hashes, rules, per-query rankings and metrics.
No automatic resampling of failed or ambiguous requests is permitted. API
credentials are process-local and excluded from all artifacts.
