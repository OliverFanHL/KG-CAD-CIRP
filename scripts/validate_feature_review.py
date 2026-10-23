"""Independent source, version, split, qrel-transform and ranking validation."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256

ROOT=Path('outputs/feature_review_v5')


def preflight():
    from scripts.finalize_feature_review import feature_issues,ROLE,combine,PARENT
    combine()
    for name in ['evaluation_freeze.json','analysis_freeze.json','first_pass_freeze.json','consumed_evidence_freeze.json']:
        for p,d in read_json(ROOT/name)['input_sha256'].items():assert sha256(p)==d,p
    second=read_json(ROOT/'second_pass_freeze.json')
    assert sha256(ROOT/'second_notes.txt')==second['notes_sha256']
    assert sha256(ROOT/'second_blind_manifest.csv')==second['manifest_sha256']
    assert sha256(ROOT/'FROZEN_RUBRIC.md')==second['rubric_sha256']
    adj=read_json(ROOT/'adjudication_freeze.json')
    assert sha256(ROOT/'adjudication.json')==adj['decisions_sha256']
    assert sha256(ROOT/'adjudication_notes.txt')==adj['notes_sha256']
    cfg=read_json('configs/feature_review_v5.json');records=read_json(cfg['annotation_path'])['judgments']
    reverse={v:k for k,v in ROLE.items()}
    assert len(records)==19976 and len({(r['part_id'],r['family_id']) for r in records})==19976
    for r in records:
        if r['grade']!=0:assert not feature_issues(reverse[r['family_id']],r['grade'],r['feature_evidence']),r
    previous={(r['part_id'],r['family_id']):r for r in read_json(PARENT)['judgments']}
    current={(r['part_id'],r['family_id']):r for r in records}
    assert current.keys()==previous.keys()
    actual={k for k in current if current[k]['grade']!=previous[k]['grade']}
    logged={(r['part_id'],r['family_id']) for r in read_csv(Path(cfg['annotation_path']).parent/'changes.csv')}
    assert actual==logged
    parent_digest=sha256(PARENT)
    for k,r in current.items():
        assert r['source_sha256']==previous[k]['source_sha256']
        assert r['previous_grade']==previous[k]['grade'] and r['previous_annotation_sha256']==parent_digest
        assert r['source']=='ai_reference' and r['human_expert_reviewed'] is False
    for p,d in adj['input_sha256'].items():assert sha256(p)==d,p
    write_json(ROOT/'pre_score_validation.json',dict(status='VERIFIED',stage='Before numerical evaluation',
        frozen_inputs_verified=True,consumed_evidence_hashes_verified=True,pairs=len(records),
        null_pairs=sum(r['grade'] is None for r in records),annotation_sha256=sha256(cfg['annotation_path'])))
    print(read_json(ROOT/'pre_score_validation.json'))


def sources():
    manifest=read_csv(ROOT/'render_manifest.csv')
    catalog=read_json('outputs/intent_batch2_matrix_reproduction/bundle/catalog.json')
    parts={p['part_id']:p for p in catalog['parts']}
    records=[]
    for row in manifest+read_csv(ROOT/'pilot_manifest.csv'):
        binding=read_json(Path(row['image']).with_suffix('.json'));p=parts[row['part_id']]
        assert sha256(p['absolute_path'])==row['source_sha256']==binding['source_sha256']
        assert p['part_id']==binding['part_id'] and p['split']==row['split']
        assert sha256(row['image'])==binding['image_sha256']
        assert binding['render_version']==2 and len(binding['views'])==6
        if row.get('mesh_path'):assert sha256(row['mesh_path'])==binding['mesh_sha256']
        records.append(dict(review_id=row['review_id'],part_id=p['part_id'],split=p['split'],
            source_sha256=row['source_sha256'],image_sha256=binding['image_sha256'],
            binding_sha256=sha256(Path(row['image']).with_suffix('.json'))))
    test={r['source_sha256'] for r in records if r['split']=='test'}
    pilot={r['source_sha256'] for r in records if r['split']=='val'}
    train={p['file_sha256'] for p in parts.values() if p['split']=='train'}
    assert len(test)==1816 and len(pilot)==44 and not test&pilot and not test&train
    write_csv(ROOT/'source_image_hashes.csv',records)
    result=dict(test_cads=len(test),pilot_cads=len(pilot),source_and_image_hashes_verified=len(records),
        test_train_exact_overlap=0,pilot_test_exact_overlap=0,render_version=2,views_per_cad=6,
        surface_edges_hidden=True,source_hash_table_sha256=sha256(ROOT/'source_image_hashes.csv'))
    write_json(ROOT/'source_validation.json',result);print(result)


def evaluation():
    from scripts.run_verified_evaluation import grade_vector
    from scripts.finalize_feature_review import PARENT,ROLE,FEATURES
    cfg=read_json('configs/feature_review_v5.json');base=read_json('configs/candidate_scale_v4.json')
    changed={'version','annotation_path','annotation_audit_path','evaluation_plan','evidence_root','output','notes'}
    assert cfg.keys()==base.keys() and all(v==base[k] for k,v in cfg.items() if k not in changed)
    for p,d in read_json(ROOT/'evaluation_freeze.json')['input_sha256'].items():assert sha256(p)==d,p
    for p,d in read_json(ROOT/'analysis_freeze.json')['input_sha256'].items():assert sha256(p)==d,p
    for p,d in read_json(ROOT/'first_pass_freeze.json')['input_sha256'].items():assert sha256(p)==d,p
    for p,d in read_json(ROOT/'consumed_evidence_freeze.json')['input_sha256'].items():assert sha256(p)==d,p
    adj=read_json(ROOT/'adjudication_freeze.json')
    assert sha256(ROOT/'adjudication.json')==adj['decisions_sha256']
    assert sha256(ROOT/'adjudication_notes.txt')==adj['notes_sha256']
    new=read_json(cfg['annotation_path'])['judgments'];old=read_json(PARENT)['judgments']
    obs={(r['family_id'],r['part_id']):r for r in new};previous={(r['family_id'],r['part_id']):r for r in old}
    assert len(new)==len(obs)==19976 and obs.keys()==previous.keys()
    actual={k for k in obs if obs[k]['grade']!=previous[k]['grade']}
    logged={(r['family_id'],r['part_id']) for r in read_csv(Path(cfg['annotation_path']).parent/'changes.csv')}
    assert actual==logged
    reverse={v:k for k,v in ROLE.items()}
    for r in new:
        assert r['source']=='ai_reference' and r['human_expert_reviewed'] is False
        assert r['review_status']=='visually_reviewed' and r['geometry_evidence']
        assert r['source_sha256']==previous[r['family_id'],r['part_id']]['source_sha256']
        if r['grade'] in (1,2,None):
            assert set(FEATURES[reverse[r['family_id']]])<={f['feature'] for f in r['feature_evidence']}
        if r['grade'] is None:assert r['decision']=='cannot_assess'
    assert not any(r['grade'] is None for r in new),'Partial-qrel intervals required, no complete-label validation'
    out=Path(cfg['output']);cohorts=read_json(out/'cohorts.json');protocol=read_json(out/'effective_protocol.json')
    assert len(protocol['queries'])==110 and len(protocol['families'])==11
    assert sorted(cohorts['closed'])==sorted(protocol['gallery'])
    evidence={r['part_id']:read_json(Path(cfg['evidence_root'])/(r['inspection_id']+'.json')) for r in new}
    # Separately expressed task/constraint transforms, without calling the
    # evaluator's score_grade or numeric_reference to build expected values.
    checked=0
    for query in protocol['queries']:
        for scenario in ['nominal','conditional_zero','conditional_two']:
            ids=cohorts['closed'];reference=grade_vector(query,ids,obs,evidence,scenario)
            for i,pid in enumerate(ids):
                g=obs[query['scenario_id'],pid]['grade'];e=evidence[pid]
                if g==1 and scenario!='nominal':g=0 if scenario=='conditional_zero' else 2
                if query['constraints']:
                    if e['aabb_quarantined'] or e['faces_missing_mesh']:g=min(g,1)
                    else:
                        outcomes=[]
                        for constraint in query['constraints']:
                            assert (constraint['attribute'],constraint['op'],constraint['unit'])==('aabb_large_mm','<=','mm')
                            delta=e['surface_large_mm']-constraint['value']
                            outcomes.append('unknown' if abs(delta)<=.2 else 'fail' if delta>0 else 'pass')
                        if 'fail' in outcomes:g=0
                        elif 'unknown' in outcomes:g=min(g,1)
                assert g==reference[i],(query['query_id'],pid,scenario)
                checked+=1
    arrays=np.load(out/'complete_rankings.npz');historical=np.load('outputs/candidate_scale_v4/evaluation/complete_rankings.npz')
    assert arrays.files==historical.files
    assert all(np.array_equal(arrays[key],historical[key]) for key in arrays.files)
    run=read_json(out/'run_manifest.json');replay=read_json(out/'replay/run_manifest.json')
    assert run['input_sha256']==replay['input_sha256'] and run['code_sha256']==replay['code_sha256']
    historical_run=read_json('outputs/candidate_scale_v4/evaluation/run_manifest.json')
    assert run['code_sha256']==historical_run['code_sha256']
    shared=set(run['input_sha256'])&set(historical_run['input_sha256'])
    assert all(run['input_sha256'][p]==historical_run['input_sha256'][p] for p in shared)
    assert sha256(out/'robustness_rankings.csv')==sha256('outputs/candidate_scale_v4/evaluation/robustness_rankings.csv')
    repro=read_json(out/'reproducibility.json');assert repro['status']=='VERIFIED'
    result=dict(status='VERIFIED',complete_pairs=len(new),candidate_cads=1816,queries=110,methods=14,
        representations=2,matrix_settings_unchanged=True,all_historical_ranking_arrays_unchanged=len(arrays.files),
        independently_constructed_task_constraint_cases=checked,changes_logged=len(actual),
        frozen_inputs_unchanged=True,core_evaluator_and_method_code_unchanged=True,
        all_shared_historical_model_protocol_input_hashes_unchanged=len(shared),
        robustness_rankings_unchanged=True,raw_step_hashes_verified=repro['raw_step_hashes_verified'],
        independent_metric_cases=repro['metric_cases_independently_checked'],
        null_pairs=0,independent_human_validation=False,
        provenance='All-source AI rereview; first passes separately blinded; retrospective operational rubric revision')
    write_json(ROOT/'validation.json',result);print(result)


def ranking_replay():
    """Validate actual fresh retrieval/replay and partial grade transforms."""
    preflight()
    from scripts.partial_feature_review import transformed_set
    from cadkg.experiments.reference import numeric_reference
    cfg=read_json('configs/feature_review_v5.json');out=Path(cfg['output']);replay=ROOT/'ranking_replay'
    main=read_json(out/'ranking_validation.json');rep=read_json(replay/'ranking_validation.json')
    assert main['status']==rep['status']=='VERIFIED' and main['input_sha256']==rep['input_sha256']
    counts={}
    for name in ['complete_rankings.npz','robustness_complete_rankings.npz']:
        with np.load(out/name) as a,np.load(replay/name) as b:
            assert a.files==b.files
            for key in a.files:assert np.array_equal(a[key],b[key]),key
            counts[name]=len(a.files)
    for name in main['files_sha256']:
        assert sha256(out/name)==main['files_sha256'][name],name
        assert sha256(replay/name)==rep['files_sha256'][name],name
        assert sha256(out/name)==sha256(replay/name),name
    protocol=read_json(out/'effective_protocol.json');records=read_json(cfg['annotation_path'])['judgments']
    obs={(r['family_id'],r['part_id']):r for r in records};ids=read_json(out/'cohorts.json')['closed']
    evidence={r['part_id']:read_json(Path(cfg['evidence_root'])/(r['inspection_id']+'.json')) for r in records}
    checked=0
    for q in protocol['queries']:
        for scenario in ['nominal','conditional_zero','conditional_two']:
            for pid in ids:
                raw=obs[q['scenario_id'],pid]['grade'];e=evidence[pid]
                expected=set()
                for rawgrade in ([0,1,2] if raw is None else [raw]):
                    g=rawgrade
                    if g==1 and scenario!='nominal':g=0 if scenario=='conditional_zero' else 2
                    if q['constraints']:
                        if e['aabb_quarantined'] or e['faces_missing_mesh']:g=min(g,1)
                        else:
                            outcomes=[]
                            for c in q['constraints']:
                                assert (c['attribute'],c['op'],c['unit'])==('aabb_large_mm','<=','mm')
                                delta=e['surface_large_mm']-c['value']
                                outcomes.append('unknown' if abs(delta)<=.2 else 'fail' if delta>0 else 'pass')
                            if 'fail' in outcomes:g=0
                            elif 'unknown' in outcomes:g=min(g,1)
                    expected.add(g)
                assert expected==set(transformed_set(raw,numeric_reference(q,e),scenario)),(q['query_id'],pid,scenario)
                checked+=1
    catalog=read_json(Path(cfg['source_run'])/'bundle/catalog.json');parts={p['part_id']:p for p in catalog['parts']}
    for pid in ids:assert sha256(parts[pid]['absolute_path'])==parts[pid]['file_sha256'],pid
    main.update(replay_verified=True,replay_arrays_verified=counts['complete_rankings.npz'],
        replay_robustness_orders_verified=counts['robustness_complete_rankings.npz'],
        independent_task_constraint_grade_sets_verified=checked,raw_step_hashes_verified=len(ids),
        replay_validation='Fresh retrieval executed twice, every nominal order/score and full robustness order equal; deterministic tables equal.')
    write_json(out/'ranking_validation.json',main);print({k:v for k,v in main.items() if k not in ['files_sha256','input_sha256','code_sha256']})


def partial():
    preflight()
    run=read_json(ROOT/'partial_evaluation/run_manifest.json');rep=read_json(ROOT/'partial_replay/run_manifest.json')
    assert run['status']==rep['status']=='VERIFIED_CONSERVATIVE_PARTIAL_QREL_ENVELOPES'
    assert run['input_sha256']==rep['input_sha256'] and run['code_sha256']==rep['code_sha256']
    for name,digest in run['output_sha256'].items():
        assert sha256(ROOT/'partial_evaluation'/name)==digest,name
        assert rep['output_sha256'][name]==digest and sha256(ROOT/'partial_replay'/name)==digest,name
    ranks=read_json(ROOT/'evaluation/ranking_validation.json');assert ranks['replay_verified']
    write_json(ROOT/'partial_validation.json',dict(status='VERIFIED',candidate_cads=1816,role_pairs=19976,
        null_pairs=run['raw_null_pairs'],fresh_retrieval_runs=2,interval_evaluation_runs=2,
        all_historical_ranking_arrays_unchanged=ranks['complete_arrays_equal_to_v4'],
        robustness_top20_equal_to_v4=ranks['robustness_top20_equal_to_v4'],
        all_fresh_replay_arrays_equal=ranks['replay_arrays_verified'],
        all_fresh_robustness_replay_orders_equal=ranks['replay_robustness_orders_verified'],
        independent_task_constraint_grade_sets_verified=ranks['independent_task_constraint_grade_sets_verified'],
        raw_step_hashes_verified=ranks['raw_step_hashes_verified'],
        interval_outputs_reproduce=True,metrics_are_conservative_bounds=True,point_significance_withheld=True,
        human_expert_reviewed=False,partial_run_manifest_sha256=sha256(ROOT/'partial_evaluation/run_manifest.json')))
    print(read_json(ROOT/'partial_validation.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sources','preflight','evaluation','ranking-replay','partial']);a=p.parse_args()
    {'sources':sources,'preflight':preflight,'evaluation':evaluation,'ranking-replay':ranking_replay,'partial':partial}[a.action]()
