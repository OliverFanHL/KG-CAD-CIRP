"""Validate genuine reviewer records; evaluate only fully resolved frozen pools."""
from pathlib import Path

from ..io import object_hash, read_csv, read_json, receipt, write_csv, write_json
from ..review import DECISIONS, Review
from .intent import evaluate_observations


def evaluate_expert(run, review_file, output, allow_partial=False, tasks=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Use a separate expert results directory; preserve the frozen AI run')
    state = read_json(review_file)
    review = Review(run, output/'unused_review_storage', tasks)
    if state.get('source') != 'independent_expert' or state.get('human_review_attested') is not True or not state.get('reviewer'):
        raise ValueError('Expert evaluation requires genuine self-attested reviewer records; AI labels are forbidden')
    if state.get('binding') != review.binding:
        raise ValueError('Reviewer records belong to different frozen inputs or instructions')
    previous = output/'review_provenance.json'
    if previous.exists() and read_json(previous)['input_review_sha256'] != object_hash(state):
        raise ValueError('Use a new output directory for a changed review; earlier results are preserved')
    observations, coverage, complete = {}, [], []
    records = state.get('judgments', {})
    required = {fid+'|'+pid for fid,ids in review.pools.items() for pid in ids}
    if set(records)-required:
        raise ValueError('Review contains unknown task/part records')
    for fid, ids in review.pools.items():
        pending, unknown = [], []
        accepted = state.get('task_reviews',{}).get(fid,{}).get('decision') == 'accept'
        for pid in ids:
            record = records.get(fid+'|'+pid)
            if record is None:
                pending.append(pid); continue
            if record.get('decision') not in DECISIONS or record.get('grade') != DECISIONS[record['decision']]:
                raise ValueError('Decision and grade disagree; Cannot assess must remain null')
            if record.get('family_id') != fid or record.get('part_id') != pid:
                raise ValueError('Reviewer record belongs to a different task or part')
            if record['grade'] is not None and type(record['grade']) is not int:
                raise ValueError('Reviewer grade must be integer 0, 1 or 2, or null for Cannot assess')
            if record.get('source')!='independent_expert' or record.get('human_review_attested') is not True or record.get('reviewer')!=state['reviewer']:
                raise ValueError('Missing actual reviewer provenance')
            if record.get('source_sha256') != review.manifest[pid]['source_sha256']:
                raise ValueError('Annotation source hash mismatch')
            if not record.get('observation','').strip() or not record.get('rationale','').strip() or not record.get('evidence','').strip():
                raise ValueError('Missing reviewer evidence')
            if record['grade'] is None:
                unknown.append(pid)
            else:
                observations[fid,pid] = record
        resolved = accepted and not pending and not unknown
        if resolved:
            complete.append(fid)
        coverage.append({'family_id':fid,'pool_size':len(ids),'pending':len(pending),'cannot_assess':len(unknown),
                         'task_accepted':accepted,'included_in_paired_evaluation':resolved})
    write_csv(output/'coverage.csv',coverage)
    write_json(output/'coverage.json',{'reviewer':state['reviewer'],'families':coverage,
                                      'complete_families':complete,'all_families_complete':len(complete)==len(review.pools),
                                      'unknown_policy':'Unresolved families are blocked, never filled with grade 0.'})
    if len(complete)!=len(review.pools) and not allow_partial:
        raise ValueError('Human review is incomplete. See coverage.json; resolve unknowns or explicitly use --allow-partial for complete families only')
    if not complete:
        raise ValueError('No complete family yet; coverage saved, no effectiveness metrics computed')
    for fid in complete:
        for pid in review.pools[fid]:
            review.evidence(review.manifest[pid]['inspection_id'])
    settings=review.protocol['settings']
    result=evaluate_observations(run,settings,observations,result_dir=output,source='independent_expert',families=complete)
    write_json(output/'review_provenance.json',{'reviewer':state['reviewer'],'source':'independent_expert',
               'human_review_attested':True,'expert_credentials_verified':False,'binding':review.binding,
               'input_review_sha256':object_hash(state),'partial_family_subset':len(complete)!=len(review.pools),
               'evaluation_labels_used_for_retrieval':False})
    write_json(output/'run_manifest.json',receipt('expert-evaluate',[review_file,run/'rankings.csv',run/'protocol.json'],
               {'included_families':complete,'partial_requested':allow_partial}))
    # Keep the expert report short; all eight methods use the same shared summary.
    rows=read_csv(output/'summary.csv')
    table='| Representation | Method | Pool nDCG@5 | P@5 | Strong P@5 |\n|---|---|---:|---:|---:|\n'
    from .full import LABELS
    for row in rows:
        if row['subset']=='combined' and row['variant']=='exemplar':
            table+=f"|{row['geometry_kind']}|{LABELS[row['method']]}|{float(row['macro_pool_ndcg@5']):.4f}|{float(row['macro_precision@5']):.4f}|{float(row['macro_strong_precision@5']):.4f}|\n"
    text=f'''# Human reviewer evaluation

Reviewer: **{state['reviewer']}**. Human review is self-attested; credentials are not programmatically authenticated. This report uses that reviewer's decisions, not AI labels. Included families: **{len(complete)} / {len(review.pools)}**. See coverage.csv for every excluded or unresolved family.

{table}

Metrics are averaged within families and then across included families. All methods use the same frozen full-gallery rankings and the same fully judged pools. Cannot assess is never converted to rejection. Partial-family results describe the completed subset and must not be compared directly with the all-family AI average. Pooled nDCG does not establish full-gallery recall. Shape shortlist grades do not certify load capacity or actual mating compatibility.

Files: per_query.csv, paired_differences.csv, family_differences.csv, expert_judgments.csv, sensitivity.csv, coverage.csv/json, review_provenance.json, run_manifest.json. Generic knowledge reviews are a separate export and were not added to the frozen retrieval graph.
'''
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    return result|{'included_families':len(complete),'report':str(output/'REPORT.md')}
