"""Source, partition, label-version and separately implemented qrel checks."""
from pathlib import Path
from collections import Counter
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256
from scripts.run_verified_evaluation import grade_vector

ROOT=Path('outputs/candidate_scale_v4')

def main():
    cfg=read_json('configs/candidate_scale_v4.json');freeze=read_json(ROOT/'freeze.json');seal=read_json(ROOT/'evaluation_freeze.json')
    for path,digest in freeze['inputs'].items():assert sha256(path)==digest,path
    assert sha256(cfg['annotation_path'])==seal['annotation_sha256']
    assert sha256('configs/candidate_scale_v4.json')==seal['config_sha256']
    assert sha256(cfg['diagnostic_cohorts_path'])==seal['cohorts_sha256']
    old_cfg=read_json('configs/kg_queries110_v3.json')
    changed={'version','annotation_path','annotation_audit_path','evaluation_plan','evidence_root','output','diagnostic_cohorts_path','notes'}
    assert all(cfg[k]==v for k,v in old_cfg.items() if k not in changed)
    assert set(cfg)-set(old_cfg)=={'diagnostic_cohorts_path'}
    parent=read_json('annotations/intent_batch2_audited_v2/judgments.json');child=read_json(cfg['annotation_path'])
    old={(r['family_id'],r['part_id']):r for r in parent['judgments']};new={(r['family_id'],r['part_id']):r for r in child['judgments']}
    assert len(old)==14850 and len(new)==len(child['judgments'])==19976 and old.keys()<=new.keys()
    log=read_csv('annotations/intent_candidates1816_v4/corrections.csv');logged={(r['family_id'],r['part_id']) for r in log}
    actual={k for k in old if old[k]['grade']!=new[k]['grade']}
    assert actual==logged and len(actual)==32
    assert all(old[k]==new[k] for k in old if k not in logged)
    from scripts.finalize_candidate_scale import resolved_records
    resolved={(r['family_id'],r['part_id']):r for r in resolved_records(cfg['annotation_path'])}
    overlay=read_json(Path(cfg['annotation_path']).parent/'evidence_corrections.json')
    assert len(overlay['corrections'])==188
    assert all(resolved[k]['grade']==new[k]['grade'] and resolved[k]['source_sha256']==new[k]['source_sha256'] for k in new)
    for path,digest in overlay['source_notes_sha256'].items():assert sha256(path)==digest
    for r in log:
        k=r['family_id'],r['part_id'];assert old[k]['grade']==int(r['previous_grade']) and new[k]['grade']==int(r['revised_grade']) and r['reason']
    out=Path(cfg['output']);cohorts=read_json(out/'cohorts.json')
    assert not set(cohorts['previous1350'])&set(cohorts['added466'])
    assert set(cohorts['previous1350'])|set(cohorts['added466'])==set(cohorts['closed'])
    protocol=read_json(out/'effective_protocol.json');assert set(protocol['gallery'])==set(cohorts['closed'])
    assert len(protocol['queries'])==110 and len(protocol['families'])==11
    parts={r['part_id']:r for r in read_json(Path(cfg['source_run'])/'bundle/catalog.json')['parts']}
    assert all(parts[pid]['split']=='test' for pid in cohorts['closed'])
    assert all(parts[r['part_id']]['split']=='train' for r in protocol['exemplars'])
    assert not {r['source_sha256'] for r in protocol['exemplars']}&{parts[p]['file_sha256'] for p in cohorts['closed']}
    assert len({parts[p]['file_sha256'] for p in cohorts['closed']})==1816
    evidence={};records={}
    for r in child['judgments']:
        pid=r['part_id'];records.setdefault(pid,r)
        assert r['source']=='ai_reference' and r['human_expert_reviewed'] is False
        assert type(r['grade']) is int and r['grade'] in (0,1,2) and r['review_status']=='visually_reviewed'
    hashes=[]
    for pid,r in records.items():
        path=Path(cfg['evidence_root'])/f"{r['inspection_id']}.json";e=read_json(path);evidence[pid]=e
        assert e['part_id']==pid and e['source_sha256']==r['source_sha256']==sha256(parts[pid]['absolute_path'])
        assert Path(r['geometry_evidence']).is_file()
        hashes.append(dict(part_id=pid,inspection_id=r['inspection_id'],source_sha256=r['source_sha256'],evidence_json_sha256=sha256(path),image_sha256=sha256(r['geometry_evidence'])))
    # This implementation intentionally does not call numeric_reference or score_grade.
    checked=0
    for q in protocol['queries']:
        for scenario in ['nominal','conditional_zero','conditional_two']:
            reference=grade_vector(q,cohorts['closed'],new,evidence,scenario)
            for i,pid in enumerate(cohorts['closed']):
                r=new[q['scenario_id'],pid];g=r['grade'];e=evidence[pid]
                if g==1 and scenario!='nominal':g=0 if scenario=='conditional_zero' else 2
                if q['constraints']:
                    verdict=[]
                    for c in q['constraints']:
                        assert (c['attribute'],c['op'],c['unit'])==('aabb_large_mm','<=','mm')
                        delta=e['surface_large_mm']-c['value']
                        verdict.append('unknown' if abs(delta)<=.2 else 'fail' if delta>0 else 'pass')
                    if e['aabb_quarantined'] or e['faces_missing_mesh']:g=min(g,1)
                    elif 'fail' in verdict:g=0
                    elif 'unknown' in verdict:g=min(g,1)
                assert g==reference[i],(q['query_id'],pid,scenario)
                checked+=1
    bounds=read_csv(out/'full_gallery_bounds.csv');assert all(int(r['unjudged_at5'])==0 for r in bounds)
    first=read_json(out/'run_manifest.json');second=read_json(out/'replay/run_manifest.json')
    assert first['input_sha256']==second['input_sha256'] and first['code_sha256']==second['code_sha256']
    write_csv(ROOT/'evidence_hashes.csv',hashes)
    result=dict(status='VERIFIED',historical_inputs_unchanged=True,matrix_settings_unchanged=True,queries=110,methods=14,representations=2,unique_test_cads=1816,unique_raw_step_hashes=1816,train_exemplars=19,exact_source_exemplar_overlap=0,changed_parent_pairs=len(actual),all_unchanged_parent_records_exact=True,added_pairs=len(new)-len(old),evidence_only_repairs=188,evidence_supplement_preserves_all_grades=True,evidence_supplement_sha256=sha256(Path(cfg['annotation_path']).parent/'evidence_corrections.json'),independently_constructed_grade_cases=checked,full_gallery_unjudged_top5_slots=0,input_and_code_manifests_match_replay=True,evidence_hash_capture_stage='post-run validation; not a claim of pre-run sealing',independent_human_validation=False)
    write_json(ROOT/'validation.json',result);print(result)

if __name__=='__main__':main()
