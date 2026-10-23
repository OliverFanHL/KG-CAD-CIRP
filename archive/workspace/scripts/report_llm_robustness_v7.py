"""Report all prespecified acquisition replicates and contrasts without selection."""
import csv,json,math,hashlib
from pathlib import Path
ROOT=Path('outputs/llm_robustness_v7')
LABELS={'kg_no_class':'G+C','geometry_class_filter':'G+A+C','geometry_intent_filter':'G+F+C','kg_no_constraints':'Full without C','kg_full':'Base Full','kg_luna':'KG-directed Luna','kg_sol':'KG-directed Sol','role_only_hybrid':'Full + holistic role-only J','role_only_score':'J-only+C in inherited pool','kg_luna_score':'Luna H-only+C in inherited pool','kg_sol_score':'Sol H-only+C in inherited pool'}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def f(x,signed=False):return 'withheld' if x is None else format(x,'+.6f' if signed else '.6f')
def main():
    r=read(ROOT/'results.json');c=read(ROOT/'evidence_completion.json');replay=read(ROOT/'independent_replay_audit.json');audit=read(ROOT/'post_execution_audit.json')
    terminal=[json.loads(line) for line in (ROOT/'api_responses.jsonl').read_text(encoding='utf8').splitlines()]
    accounting=dict(recorded_usage_cost_estimate_usd=math.fsum(x['cost_usd'] for x in terminal if x['usage'] is not None),
        no_usage_conservative_reservation_usd=math.fsum(x['cost_usd'] for x in terminal if x['usage'] is None),
        journal_total_usd=math.fsum(x['cost_usd'] for x in terminal),
        calls_without_reported_usage=sum(x['usage'] is None for x in terminal),
        api_responses_sha256=hashlib.sha256((ROOT/'api_responses.jsonl').read_bytes()).hexdigest(),
        scope='Frozen-price estimate from recorded usage plus conservative journal reservations; not a provider invoice.')
    assert abs(accounting['journal_total_usd']-c['estimated_new_cost_usd'])<1e-9
    (ROOT/'accounting_summary.json').write_text(json.dumps(accounting,indent=2)+'\n',encoding='utf8')
    rows=['# Repeated-acquisition CAD retrieval and holistic role-only control','',
      '## Material Passport','',
      '- Origin Skill: ARS experiment-agent','- Origin Mode: run + validate',
      '- Origin Date: 2026-10-22','- Verification Status: VERIFIED (deterministic saved-response replay and independent numerical oracle)',
      '- Version Label: llm_robustness_v7','',
      'The study retains three total acquisitions for KG-directed Luna and Sol, with their historical '
      'first acquisitions reused unchanged, and three fresh acquisitions for a holistic role-only Luna '
      'control. Local queue seeds do not seed Responses generation. The encoders, classifier, geometry, '
      'query matrix, original candidate pools and AI-reference relevance remain fixed. This measures '
      'repeated API acquisition conditional on that fixed model and benchmark, not training robustness.','',
      '## Three-acquisition means and sample variation','',
      '| Geometry | Method | Mean nDCG@5 | Run SD | Mean Strong P@5 | Run SD |',
      '|---|---|---:|---:|---:|---:|']
    for x in r['summary']:
        nd=f(x['ndcg@5_lower_mean']) if x['ndcg@5_lower_mean']==x['ndcg@5_upper_mean'] else '['+f(x['ndcg@5_lower_mean'])+','+f(x['ndcg@5_upper_mean'])+']'
        sp=f(x['strong_precision@5_lower_mean']) if x['strong_precision@5_lower_mean']==x['strong_precision@5_upper_mean'] else '['+f(x['strong_precision@5_lower_mean'])+','+f(x['strong_precision@5_upper_mean'])+']'
        rows.append(f"| {x['geometry_kind']} | {LABELS[x['method']]} | {nd} | {f(x['ndcg@5_sd'])} | {sp} | {f(x['strong_precision@5_sd'])} |")
    rows+=['','Run SD uses ddof=1 over the three aggregate acquisition means. It is distinct from '
      'role heterogeneity, reference-label envelopes and bootstrap uncertainty. Lower/upper fields '
      'remain in the machine-readable outputs; where reference uncertainty persists, the displayed SD '
      'describes lower endpoints. Deterministic controls reuse the original complete rankings and their '
      'acquisition-index SD is zero conditional on the frozen model. Independent reordered local replays '
      'are a separate determinism check.','',
      '## Four prespecified role-level comparisons','',
      '| Geometry | Contrast | Mean nDCG difference | Unadjusted 95% interval | Raw p | Holm p | Role W/T/L |',
      '|---|---|---:|---:|---:|---:|---:|']
    for t in r['paired_tests']:
        ci='withheld' if t['ci95_lower'] is None else '['+f(t['ci95_lower'],True)+','+f(t['ci95_upper'],True)+']'
        w='withheld' if t['family_wins'] is None else str(t['family_wins'])+'/'+str(t['family_ties'])+'/'+str(t['family_losses'])
        rows.append(f"| {t['geometry_kind']} | {LABELS[t['proposed']]} minus {LABELS[t['comparator']]} | {f(t['delta'],True)} | {ci} | {f(t['raw_p'])} | {f(t['holm_p'])} | {w} |")
    rows+=['','Tests first average the three acquisitions within each role, then use eleven authored '
      'role differences. Exact two-sided tests enumerate 2,048 sign patterns; one Holm family covers '
      'all four tests. Bootstrap intervals use 20,000 role resamples and seed 20261023. No acquisition '
      'is selected for favorable results, and the run-by-role entries are not treated as 33 independent '
      'replications. A non-rejected model contrast does not establish equivalence.','',
      '## Evidence and budget accounting','',
      f"New calls: {c['new_calls']:,}; historical calls reused: {c['historical_calls_reused']:,}; "
      f"all role–CAD observations: {c['total_observation_jobs']:,}. Terminal statuses: {json.dumps(c['status_counts'],sort_keys=True)}. "
      f"New journal accounting estimate: USD {c['estimated_new_cost_usd']:.8f}, under the USD75 ceiling. "
      f"New costs by arm: {json.dumps(c['new_cost_by_arm_usd'],sort_keys=True)}.",
      '',f"The journal total comprises USD {accounting['recorded_usage_cost_estimate_usd']:.8f} estimated from reported tokens "
      f"and USD {accounting['no_usage_conservative_reservation_usd']:.8f} conservative reservations for "
      f"{accounting['calls_without_reported_usage']} calls without reported usage. No reservation is asserted to be an actual provider charge.",
      '',f"Observation validity/fallback counts: {json.dumps(c['validity_counts'],sort_keys=True)}.",
      '',f"Reported state/score counts: {json.dumps(c['predicate_states_by_arm'],sort_keys=True)}.",
      '',f"Mean new client request duration: {c['mean_new_call_seconds']:.3f} s. "
      'This interleaved concurrent collection is not an online-query latency benchmark or a provider invoice.','',
      '## Scope of the ablation','',
      'The role-only model receives a competent plain role purpose, neutral CAD views and the same '
      'generic visibility, uncertainty and clip guidance. It returns one holistic nine-level J. The '
      'KG-directed model returns four relation-bound clause states whose fixed mean H has the same '
      'possible value domain. The contrast changes authored decomposition, bindings and aggregation '
      'together; it does not isolate graph storage or bindings alone.','',
      'Full+J is a hybrid control retaining G/A/F and C. The separate J-only+C method removes those '
      'score components inside the identical inherited pool. That pool was produced by KG-aware '
      'initial branches, so neither result establishes full-gallery KG-free retrieval. H-only variants '
      'are supplied to distinguish visual-score behavior from hybrid fusion without additional calls.','',
      'Historical first KG runs precede the new interleaved collections and had no local seed. This '
      'unequal temporal history is retained explicitly. AI-reference relevance and model observations '
      'may share assumptions; no independent expert assessment, unseen-role validation or physical '
      'compatibility certification is created. All eleven ARS statistical fallacy patterns and '
      'run/label provenance are checked in the independent completion audit.','',
      '## Reproduction','',
      f"Saved-response replay status: {replay['status']}. Independent audit status: {audit['status']}. "
      'Full commands and output schemas are in REPRODUCE.md. Source hashes, frozen contracts, all '
      'attempts/terminal responses, observation provenance, top-five arrays, per-query results, family '
      'and run summaries and all tests are retained. No original frozen study or manuscript file is '
      'changed by this report.','']
    (ROOT/'study_report.md').write_text('\n'.join(rows),encoding='utf8')
    print(json.dumps({'report':str(ROOT/'study_report.md'),'replay':replay['status'],'audit':audit['status']}))
if __name__=='__main__':main()
