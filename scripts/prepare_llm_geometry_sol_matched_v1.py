"""Prepare a single fixed Sol candidate, preserving every Luna visual input.

This script is local-only. It reads no credentials and sends no request.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil

from scripts import evaluate_llm_geometry_sol_matched_v1 as evaluation
from scripts import run_llm_geometry_sol_api_v1 as runner

ROOT=Path('outputs/llm_geometry_sol_matched_v1')
LUNA=Path('outputs/llm_geometry_matched_v1')

PROTOCOL="""# Matched GPT-6.1 Sol candidate protocol

## Material Passport

- Origin Skill: ARS experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-22
- Verification Status: UNVERIFIED (input freeze; outcomes pending)
- Version Label: llm_geometry_sol_matched_v1

## Fixed intervention and input continuity

The user requested GPT-6.1 Sol as one additional candidate after observing the
completed Luna study. This is a prospectively frozen extension of an historically
examined benchmark, not a preregistration of the original study. It keeps all
264 queries,44 training exemplars,11roles,two representations,1,816gallery CADs,
528fixed pools,8,427pool memberships and2,084role-CAD observation jobs. There is
no additional query, gallery candidate, prompt, clause, weight, image rendering
or ground-truth annotation. Original Luna files remain unchanged.

Sol receives byte-identical six-view PNGs, instruction strings, per-role predicate
payloads and per-role strict schemas. Its request changes only the model identifier.
Role semantics are necessarily visible in the predicates; part/family identifiers,
classifier outputs, scores, ranks, reference identities, size bounds and qrels are
not supplied to the model. Responses are cached once per role-CAD pair.

Requested model is gpt-6.1-sol, low reasoning, image detail auto, a2,000-output-token
cap and store=false through the Responses API. All2,084jobs are attempted once.
The initial single request is included in the final benchmark; its observation is
not used for prompt/schema tuning. Continuation attempts only unattempted jobs.
Failed, invalid or ambiguous attempts are never automatically resampled. A durable
attempt precedes each call; unresolved attempts block resume.

## Budget and official sources

The maximum is USD50 under the recorded standard prices: input2.00,cached input
0.10,30-minute cache-write2.50 and output10.00 per million tokens. Each in-flight
job reserves16,000input tokens at the maximum declared input/cache-write rate
plus2,000output tokens (USD0.06). Actual recorded usage releases unused reserve.
This rolling reservation does not reserve all remaining jobs in advance. A usage
overrun halts new dispatch. Sixfold margin over the observed Luna input size
is conservative for the unchanged compact prompts and1500x1000PNGs, but it is a
reservation assumption rather than a provider-enforced input-token cap. Any
overrun/failure and any incomplete coverage are disclosed rather than repaired.
Credentials remain process-local and are absent from all study artifacts.

Official pages checked2026-10-22:
- https://developers.openai.com/api/docs/models/gpt-6.1-sol
- https://developers.openai.com/api/docs/pricing

## Evidence and rankings

The observation contract and four component-binding clauses per role are unchanged
from Luna. Yes/no/unknown map to1/0/.5; H is their fixed arithmetic mean. A whole-job
transport/schema failure or asserted yes/no without a view gives H=.5. Observations
are model reports with provenance, not independently verified geometric facts.

For each representation and query, use the unchanged union of original Full,
G+F+C and G+C top10eligible results. Rank this pool by(3*Full_score+H)/4 with original
gallery identity breaking ties; append the unchanged Full tail. Numerical serving
eligibility is unchanged. Four displayed methods are Full,G+F+C,frozen Luna and Sol.

## Prespecified statistical comparisons

Primary comparisons are Sol minus frozen Luna and Sol minus G+F+C, separately for
embedding and descriptor: four tests in one new Holm family. Use nDCG@5 as the
tested endpoint and StrongP@5 descriptively. Both retain original v5 AI-reference
grades, the original nominal numerical transformation and0.2mm guard. Preserve
both raw null judgments and every admissible grade completion. Withhold all
primary point tests if any primary nDCG endpoint is not identified.

Average six conditions within each of four exemplars, then eleven role means
equally. Enumerate all2^11sign patterns for exact two-sided tests. Use20,000role
bootstrap resamples,seed20261022,and unadjusted95percentile intervals. Report all
four comparisons, states, failures, cost and latency regardless of model ordering.
There is no prompt/model/weight sweep, automatic selection or manuscript-update gate.
The previous Luna four-test and component twelve-test analyses remain separate
historical families; no joint familywise guarantee across the whole research
history is asserted.

## Interpretation and completion

Agreement with fixed AI-reference relevance does not establish expert engineering
truth. Shared model-family/visual assumptions, historical collection exposure,
authored role dependence, fixed-pool access and lack of independent expert truth
remain explicit. This comparison isolates a model substitution within the fixed
visual-observation-and-coverage intervention; it does not identify graph storage,
requirement decomposition or relation-binding benefits separately.

After all calls, seal the evidence, reconstruct all rankings and primary metrics,
replay in an isolated directory with no API, and compare artifacts byte-for-byte.
Independently check scalar full-gallery DCG, grade bounds and all four tests.
No manuscript file is modified by these scripts.
"""


def main():
    assert not (ROOT/'protocol_freeze.json').exists(),'Never reprepare a sealed study'
    assert not any((ROOT/n).exists() for n in ('api_attempts.jsonl','api_responses.jsonl'))
    ROOT.mkdir(parents=True,exist_ok=True)
    original=json.loads((LUNA/'study.json').read_text(encoding='utf8'))
    oldfreeze=json.loads((LUNA/'protocol_freeze.json').read_text(encoding='utf8'))
    assert runner.file_digest(LUNA/'study.json')==oldfreeze['study_sha256']
    (ROOT/'protocol.md').write_text(PROTOCOL,encoding='utf8')
    for name in ('pool_manifest.json','knowledge_rules.json','image_neutralization_audit.json'):
        shutil.copyfile(LUNA/name,ROOT/name)
    study=dict(original)
    study.update(version='llm-geometry-sol-matched-v1',model='gpt-6.1-sol',max_usd=50,
      pricing_source='https://developers.openai.com/api/docs/pricing',pricing_verified_date='2026-10-22')
    required=list(original['required_files'])
    for path in [Path(__file__).resolve(),*(Path('scripts')/n for n in (
      'run_llm_geometry_sol_api_v1.py','complete_llm_geometry_sol_evidence_v1.py',
      'evaluate_llm_geometry_sol_matched_v1.py','replay_llm_geometry_sol_matched_v1.py')),
      ROOT/'protocol.md',LUNA/'study.json',LUNA/'protocol_freeze.json',LUNA/'results.json',
      LUNA/'full_rankings.npz',LUNA/'graph_support.json']:
        required.append({'path':str(path.resolve()),'sha256':runner.file_digest(path)})
    study['required_files']=required
    assert study['instructions']==original['instructions'] and study['jobs']==original['jobs']
    assert len(study['jobs'])==2084
    for job in study['jobs']:
        assert runner.file_digest(Path(job['image_path']))==job['image_sha256']
    (ROOT/'study.json').write_text(json.dumps(study,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    preparation={'source_luna_study_sha256':runner.file_digest(LUNA/'study.json'),
      'source_luna_protocol_sha256':runner.file_digest(LUNA/'protocol_freeze.json'),
      'source_luna_results_sha256':runner.file_digest(LUNA/'results.json'),
      'image_payload_schema_job_records_exactly_equal':True,
      'instructions_exactly_equal':True,'unique_jobs':2084,
      'changed_request_fields':['model'],'credentials_read':False,'api_calls':0}
    (ROOT/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf8')
    evaluation.prepare_freeze(ROOT.resolve())
    _,jobs,seal=runner.prepare((ROOT/'study.json').resolve(),ROOT.resolve())
    print(json.dumps({'protocol_sha256':seal,'jobs':len(jobs),'max_usd':50,'reservation_per_request_usd':.06}))


if __name__=='__main__':main()
