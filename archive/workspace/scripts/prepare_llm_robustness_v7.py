"""Freeze three-acquisition benchmark and nine-level role-only baseline."""
import copy,json,random,shutil
from pathlib import Path
from scripts import run_llm_robustness_v7_api as runner
from scripts import evaluate_llm_robustness_v7 as evaluator

ROOT=Path('outputs/llm_robustness_v7');LUNA=Path('outputs/llm_geometry_matched_v1');SOL=Path('outputs/llm_geometry_sol_matched_v1')
ROLES={
 'support_body':'stationary support for a rotating shaft',
 'shaft_connector':'mechanical connection joining two shafts',
 'belt_wheel':'a rotating wheel that engages a drive belt',
 'threaded_clamp':'threaded clamping using a headed externally threaded fastener',
 'nut_retention':'retaining an externally threaded fastener using a separate nut',
 'bracket_mount':'fixed mounting between angled or offset attachment faces',
 'flanged_passage':'joining a central passage using a flanged connection',
 'gear_mesh':'parallel-axis rotational transmission using cylindrical gear meshing',
 'shaft_drive':'transmitting rotation using a drive shaft',
 'compression_return':'axial compression return using a helical spring',
 'tension_return':'axial tension return using a helical spring',
}
ROLE_INSTRUCTIONS=(
 'Inspect only the supplied neutral CAD surface views and assess visible geometric support for the requested role as a whole. '
 'Do not rank the part, infer its dataset class, assign reference relevance grades, or predict engineering performance. '
 'The six panels in row-major order are oblique, opposite, profile, axial, cut_X, cut_Y. '
 'The last two are uncapped visual clips: apparent clip openings are not proof of actual through-passages. All panels show the same CAD. '
 'Return one holistic role_support score from 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1. '
 'Use 0 for visibly incompatible form,1 for strong visible geometric support for the requested role, and0.5 when support is unresolved or ambiguous. '
 'Intermediate levels express graded visible support between those anchors; the score is not a calibrated probability. '
 'Occlusion, simplified representation, too-small details and ambiguity must retain uncertainty rather than invented features. '
 'A missing modeled thread must not be invented. For absence statements retain uncertainty unless the relevant region is visible. '
 'Material, loads, tolerances, real functionality and scale cannot be established from these renderings. '
 'Cite supporting views and give a short literal geometric observation (at most22words); explain what cannot be seen when uncertain. '
 'Do not invent additional task requirements. Return only the required structured object.'
)
ROLE_SCHEMA={'type':'object','properties':{'role_support':{'type':'object','properties':{
 'score':{'type':'number','enum':[0,.125,.25,.375,.5,.625,.75,.875,1]},
 'views':{'type':'array','items':{'type':'string','enum':['oblique','opposite','profile','axial','cut_X','cut_Y']}},
 'observation':{'type':'string'}},'required':['score','views','observation'],'additionalProperties':False}},
 'required':['role_support'],'additionalProperties':False}

PROTOCOL="""# Repeated model acquisition and holistic role-only comparison

## Material Passport
- Origin Skill: ARS experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-22
- Verification Status: UNVERIFIED (prospective acquisition/analysis seal)
- Version Label: llm_robustness_v7

## Research scope
Three total API acquisitions are evaluated for each fixed candidate: KG-directed
GPT-6 Luna, KG-directed GPT-6.1 Sol and a holistic role-only GPT-6 Luna control.
The original completed Luna and Sol acquisitions count as historical run1 for
their identical frozen contracts. Each gets two new passes. The role-only control
gets three new full passes. Thus14,588newcalls plus4,168historicalcalls supply
18,756role-CAD observations. These are repeated stochastic API acquisitions
conditional on fixed encoder/classifier weights, not retrained seeds or three
fresh dataset replications. Historical run1 order is unseeded and unaltered.

There is no documented Responses API sampling-seed parameter in this protocol.
Local seeds29/43 shuffle new KG passes;17/29/43 shuffle the role-only passes.
These seeds control local job order only. Seven collection queues are interleaved
round-robin after deterministic independent shuffles, reducing temporal block
confounding among new calls. Historical runs remain temporally separated.
The seed values are not sent to the API and cannot guarantee regeneration.

The fixed matrix is264queries,44references,11roles,1816galleryCADs,528pools and
2084role-CAD jobs per pass. The pools contain8427memberships and are exactly the
prior union of top10eligible Full,G+F+C,G+C rankings. No query, model weights,
candidate, label, bound, view or dimensional rule is changed. All original
frozen records remain unchanged. Existing neutral image bytes are reused.

## Arms and fair role-only control
KG arms receive exactly the original prompts, four clause definitions, triples,
strict schemas and image bytes, with low reasoning and2000outputtokens.
Their H mapping and fixed mean remain unchanged. Their historical first runs
are reused only because their complete contracts match byte-for-byte.

The role-only Luna control receives a plain requested-role name/purpose from
the query scope, the same six views and generic uncertainty/uncapped-clip guidance.
It receives no authored four-clause decomposition, component-binding triples,
class probabilities, prior associations, ranks, reference IDs, bounds or qrels.
It returns one holistic J from{0,.125,.25,.375,.5,.625,.75,.875,1}, cited views
and a literal geometric explanation. This matches H's value domain, avoiding
a three-value quantization disadvantage.0means incompatible visible form,
1means strong visible role support, and.5means unresolved/ambiguous support;
intermediates express graded support. It is not calibrated engineering truth.
No source-grade rubric, candidate labels or examples are disclosed.

For hybrid comparison, rank each inherited eligible pool by(3Full+J)/4 or
(3Full+H)/4, followed by the unchanged Full tail. Score-only J+C and H+C ranks
are also computed inside the identical inherited pool by their observation score
alone, with lexicographic part-ID ties and the original tail. They remove G,A,F
from the serving score, but their pool was produced by the original KG-aware
branches; they are bounded score-only controls, not full-gallery KG-free discovery.
The primary role-only baseline is named role_only_hybrid, not unqualified LLM-only.

Whole-job failed/invalid KG responses, or yes/no assertions without a cited view,
map to H=.5. Failed/invalid role-only responses or J!=.5 without a cited view map
to J=.5. All calls remain in the analysis; no retries or resampling.

## Execution and budget
Use Responses API,store=false,lowreasoning,image detailauto,2000outputtokens,
up to12concurrentcalls and aUSD75totalnew-spending ceiling. Model rates per million
tokens: Luna input.1,cached.01,cachewrite.125,output.5; Sol input2,cached.1,
cachewrite2.5,output10. Shared pending reservation isUSD.06perjob at the higher
Sol rate for16000input+2000outputtokens. Actual model-specific usage releases
unused reserve. Missing usage/failures retain the conservative reservation.
Reservation overruns halt further dispatch; this is not an API input-token cap.
No credentials are persisted. A first included request verifies execution, then
the same frozen queue continues. A401or429stops dispatch without retry; ambiguous
attempts block continuation. Any exception needing resampling requires user input.

## Analysis fixed before new outcomes
Retain deterministic G+C,G+A+C,G+F+C,Full withoutC,Full,KG-Luna,KG-Sol,
role-onlyhybrid,J-only+C,H-Luna-only+C,H-Sol-only+C. The deterministic initial
components use unchanged original arrays; their three-run variation is zero
conditional on the frozen model. Local reordered replays are separate checks,
not evidence of training robustness.

Primary four tests: three-run-mean KG-Luna minus role-onlyhybrid, and three-run-mean
KG-Sol minus KG-Luna, separately for both representations. Average conditions and
exemplars within role, then three acquisitions within role, then11roles equally.
Use exact two-sided11role sign-flips and one4test Holm family. Bootstrap20000role
resamples,seed20261023,unadjusted95percentileintervals. StrongP@5is descriptive.
Report all contrasts; there is no automatic model selection, favorable-run choice
or manuscript-update gate. Show sampleSD(ddof1) across the3aggregate run means,
separately from role heterogeneity and statistical uncertainty. Do not treat33role
byrun values as independent observations. Preserve raw null labels and uncertainty
envelopes; withhold primary point tests if any tested endpoint is unidentified.

Original v5 AI qrels and independent numerical transformation remain unchanged,
including missing-surface precedence and.2mmguard. Full-gallery idealDCG is retained.
The two missing judgments are not completed from new model observations.

## Interpretation and reproducibility
This tests the authored decomposition/bindings/coverage prompting package against
holistic role judgment under the same model/views/pool and score domain. It does
not isolate graph storage or bindings alone. Direct compatibility judgments can
share AI-reference assumptions; no independent human truth is created. Historical
test exposure, sequential extensions, fixed pool and author-defined roles remain
explicit. Earlier test families do not supply research-history-wide error control.
All11ARSfallacytypes are checked after completion. Report API model identities,
failures,cost and run-level variation. Deterministic replay of saved observations
must match evidence,rankings and metrics without additionalAPI calls.
"""

def main():
    assert not (ROOT/'protocol_freeze.json').exists();assert not (ROOT/'api_attempts.jsonl').exists()
    ROOT.mkdir(parents=True,exist_ok=True)
    source=json.loads((LUNA/'study.json').read_text(encoding='utf8'))
    sol=json.loads((SOL/'study.json').read_text(encoding='utf8'))
    assert source['jobs']==sol['jobs'] and source['instructions']==sol['instructions']
    (ROOT/'protocol.md').write_text(PROTOCOL,encoding='utf8')
    contract=dict(roles=ROLES,instructions=ROLE_INSTRUCTIONS,schema=ROLE_SCHEMA,score_domain=[i/8 for i in range(9)])
    (ROOT/'role_only_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
    shutil.copyfile(LUNA/'pool_manifest.json',ROOT/'pool_manifest.json')
    queues=[];manifest=[]
    for arm,model,rep,seeds in [('kg_luna','gpt-6-luna',2,29),('kg_sol','gpt-6.1-sol',2,29),('role_only_luna','gpt-6-luna',1,17),('kg_luna','gpt-6-luna',3,43),('kg_sol','gpt-6.1-sol',3,43),('role_only_luna','gpt-6-luna',2,29),('role_only_luna','gpt-6-luna',3,43)]:
        queue=[]
        for oldjob in source['jobs']:
            j=copy.deepcopy(oldjob);j.update(source_job_id=oldjob['job_id'],job_id=arm+'__r'+str(rep)+'__'+oldjob['job_id'],arm=arm,replicate=rep,local_order_seed=seeds,model=model)
            if arm=='role_only_luna':j.update(instructions=ROLE_INSTRUCTIONS,instruction_payload={'requested_role':ROLES[j['family_id']]},schema=ROLE_SCHEMA,predicate_ids=['role_support'])
            queue.append(j)
        random.Random(seeds).shuffle(queue);queues.append(queue);manifest.append(dict(arm=arm,replicate=rep,model=model,local_order_seed=seeds,jobs=len(queue)))
    jobs=[queues[k][i] for i in range(2084) for k in range(7)]
    assert len(jobs)==len({j['job_id'] for j in jobs})==14588
    files=[Path('scripts')/n for n in ['prepare_llm_robustness_v7.py','run_llm_robustness_v7_api.py','complete_llm_robustness_v7.py','evaluate_llm_robustness_v7.py','replay_llm_robustness_v7.py']]
    files += [ROOT/n for n in ['protocol.md','role_only_contract.json','pool_manifest.json']]
    for d in [LUNA,SOL]:files += [d/n for n in ['study.json','protocol_freeze.json','api_responses.jsonl','evidence_completion.json','results.json']]
    files += [Path('annotations/CURRENT_ANNOTATIONS.json')]
    required=[dict(path=str(p.resolve()),sha256=runner.file_digest(p)) for p in files]
    study=dict(version='llm-robustness-v7',model='per_job_frozen',instructions=source['instructions'],schema={},jobs=jobs,max_output_tokens=2000,reasoning_effort='low',max_usd=75,concurrency=12,required_files=required)
    (ROOT/'study.json').write_text(json.dumps(study,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
    preparation=dict(new_jobs=len(jobs),collections=manifest,historical_run1=[dict(arm='kg_luna',local_order_seed=None,source=str(LUNA)),dict(arm='kg_sol',local_order_seed=None,source=str(SOL))],queue='roundrobin of7independentlyseededlocalpermutations',encoder='unchanged',role_only_domain='same9possibleHvalues',api_seed_parameter_sent=False)
    (ROOT/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf8')
    evaluator.prepare_freeze();_,prepared,seal=runner.prepare((ROOT/'study.json').resolve(),ROOT.resolve())
    print(json.dumps(dict(protocol_sha256=seal,jobs=len(prepared),budget=75,concurrency=12)))

if __name__=='__main__':main()
