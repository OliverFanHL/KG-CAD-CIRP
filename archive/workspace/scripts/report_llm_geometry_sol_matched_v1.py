"""Generate a concise, complete Sol candidate report from existing artifacts."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path('outputs/llm_geometry_sol_matched_v1')
LUNA=Path('outputs/llm_geometry_matched_v1')
LABELS={'kg_full':'Base Full','geometry_intent_filter':'G+F+C',
        'kg_full_llm':'KG-directed Luna','kg_full_sol':'KG-directed Sol'}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def fmt(value,signed=False):return 'withheld' if value is None else format(value,'+.6f' if signed else '.6f')


def interval(lo,hi):return fmt(lo) if lo==hi else f'[{fmt(lo)}, {fmt(hi)}]'


def main():
    r=read(ROOT/'results.json');c=read(ROOT/'evidence_completion.json')
    replay=read(ROOT/'independent_replay_audit.json')
    old=read(LUNA/'evidence_completion.json')
    sol={g['job_id']:g for g in map(json.loads,(ROOT/'graph_evidence.jsonl').read_text(encoding='utf8').splitlines())}
    luna={g['job_id']:g for g in map(json.loads,(LUNA/'graph_evidence.jsonl').read_text(encoding='utf8').splitlines())}
    state_pairs=Counter();h_equal=h_up=h_down=0
    for jid,a in sol.items():
        b=luna[jid]
        h_equal+=a['H']==b['H'];h_up+=a['H']>b['H'];h_down+=a['H']<b['H']
        ob={v['predicate_id']:v for v in b['observations']}
        for v in a['observations']:state_pairs[ob[v['predicate_id']]['status'],v['status']]+=1
    rows=['# Matched GPT-6.1 Sol candidate results','',
      '## Material Passport','',
      '- Origin Skill: ARS experiment-agent',
      '- Origin Mode: run + validate',
      '- Origin Date: 2026-10-22',
      '- Verification Status: '+('VERIFIED (saved-response deterministic replay)' if replay['status']=='PASS' else 'ANALYZED'),
      '- Version Label: llm_geometry_sol_matched_v1','',
      '## Comparison and scope','',
      'This single additional model candidate follows the completed Luna study. All 2,084 jobs, '
      '1,213 CADs, 528 fixed pools, 264 queries and both representations are unchanged. Every reconstructed '
      'request differs only in model identifier. Prompts, view bytes, predicates, schemas, score weights, '
      'eligibility and immutable AI-reference judgments remain fixed. The first one-request execution '
      'is included in the benchmark; continuation skips that terminal job. There is no response selection '
      'or resampling, and no automatic model-selection or manuscript-update gate.','',
      '## Primary metrics','',
      '| Representation | Method | nDCG@5 | Strong P@5 |',
      '|---|---|---:|---:|']
    for x in r['summary']:
        rows.append(f"| {x['geometry_kind']} | {LABELS[x['method']]} | {interval(x['ndcg@5_lower'],x['ndcg@5_upper'])} | {interval(x['strong_precision@5_lower'],x['strong_precision@5_upper'])} |")
    rows+=['','Means average 24 conditions within each role, then 11 roles equally. The complete gallery supplies '
      'the dimensional-reference ideal DCG. Bounds retain both unresolved raw judgments.','',
      '## Four prespecified paired comparisons','',
      '| Representation | Comparator | Sol minus comparator | Unadjusted 95% interval | Exact raw p | Holm p | Role W/T/L |',
      '|---|---|---:|---:|---:|---:|---:|']
    for x in r['paired_tests']:
        ci='withheld' if x['ci95_lower'] is None else f"[{fmt(x['ci95_lower'],True)}, {fmt(x['ci95_upper'],True)}]"
        w='withheld' if x['family_wins'] is None else f"{x['family_wins']}/{x['family_ties']}/{x['family_losses']}"
        rows.append(f"| {x['geometry_kind']} | {LABELS[x['comparator']]} | {interval(x['delta_lower'],x['delta_upper'])} | {ci} | {fmt(x['exact_sign_flip_p'])} | {fmt(x['holm_p'])} | {w} |")
    rows+=['','Each exact two-sided test enumerates 2,048 role sign patterns. One Holm family covers '
      'these four model-comparison tests. Bootstrap intervals use 20,000 role resamples, seed 20261022, '
      'and are unadjusted. The earlier Luna four-test and component twelve-test analyses are historical '
      'families; no joint guarantee across the sequential research history is claimed. A non-rejected '
      'Sol–Luna difference does not establish model equivalence.','']
    with (ROOT/'family_differences.csv').open(encoding='utf8',newline='') as f:
        negative=[x for x in csv.DictReader(f) if x['comparator']=='geometry_intent_filter' and float(x['ndcg@5_upper'])<0]
    for x in negative:
        rows.append(f"The negative role effect versus G+F+C for {x['geometry_kind']} is {x['family_id']}: "
          f"nDCG difference {float(x['ndcg@5_lower']):+.6f}, Strong P difference {float(x['strong_precision@5_lower']):+.6f}.")
    rows+=['',
      '## Collection and evidence','',
      f"There are {c['completed_jobs']:,} terminal jobs with status counts {json.dumps(c['status_counts'],sort_keys=True)}. "
      f"Returned model identities are {json.dumps(c['resolved_models'],sort_keys=True)}. "
      f"The token-based Sol cost estimate is USD {c['estimated_cost_usd']:.8f} versus Luna USD {old['estimated_cost_usd']:.8f}; "
      f"the Sol ceiling was USD 50. Mean client request duration is {c['latency_seconds']['mean']:.3f} s. "
      'These are reported-token estimates and collection timings, not invoices or complete online-service costs.','',
      'Predicate counts: '+json.dumps(c['predicate_status_counts'],sort_keys=True)+'.',
      f"Role-CAD H equals the prior model in {h_equal} jobs, increases in {h_up}, and decreases in {h_down}. "
      'State agreement is descriptive because neither model observation set is independent feature truth.','',
      '| Luna state | Sol state | Clauses |','|---|---|---:|']
    for (a,b),n in sorted(state_pairs.items()):rows.append(f'| {a} | {b} | {n} |')
    rows+=['','## Interpretation boundaries and validation','',
      'This is an exploratory model substitution after known Luna outcomes, on the same historically '
      'examined CAD collection. Fixed AI-derived qrels may share visual assumptions with both models. '
      'Neither a larger model nor a favorable ranking difference establishes engineering feature truth, '
      'industrial compatibility, graph-storage advantage or generalization to new roles. H remains '
      'soft model-observed evidence coverage; verified numerical measurements retain eligibility control. '
      'The fixed pool limits recovery and conditionally defines this comparison.','',
      'All 11 ARS statistical-pattern checks are documented in the computational audit: Simpson direction '
      'and heterogeneity, ecological inference, selected-population/Berkson effects, collider conditioning, '
      'base-rate neglect, regression to mean, survivorship, look-elsewhere effects, forking paths, '
      'unsupported causal attribution and reverse causality. Role-level differences and fixed sequential '
      'history constrain interpretation.','',
      'Saved observations and numerical outputs replay without API calls. See `independent_replay_audit.json`, '
      '`post_execution_audit.json`, `protocol_independent_audit.md` and the frozen `protocol.md`. '
      'Complete response journals, source hashes, per-query metrics, family metrics and tests are retained. '
      'No manuscript, original Luna record or relevance label is changed by this report.','']
    (ROOT/'study_report.md').write_text('\n'.join(rows),encoding='utf8')
    comparison={'H_equal':h_equal,'H_increase':h_up,'H_decrease':h_down,
      'state_pairs':[{'luna':a,'sol':b,'count':n} for (a,b),n in sorted(state_pairs.items())]}
    (ROOT/'observation_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'report':str(ROOT/'study_report.md'),'replay_status':replay['status']}))


if __name__=='__main__':main()
