"""Synthetic fixtures test provenance and unknown handling, never expert ground truth."""
import json
from pathlib import Path

import numpy as np
import pytest

from cadkg.io import object_hash, read_csv, read_json, sha256, write_csv, write_json
from cadkg.review import Review, agreement
from cadkg.experiments.expert import evaluate_expert
from cadkg.experiments.reference import REFERENCE_METHODS


@pytest.fixture
def review_fixture(tmp_path):
    run=tmp_path/'run'; knowledge=tmp_path/'knowledge'; ids=['part_a','part_b']
    manifest=[]; parts=[]; tasks=[]; families=[]; queries=[]
    for i,pid in enumerate(ids):
        step=tmp_path/f'{pid}.step'; step.write_text('synthetic test source '+pid)
        item={'part_id':pid,'inspection_id':f'N{i+1:03}','source_sha256':sha256(step)}
        manifest.append(item);parts.append({'part_id':pid,'absolute_path':str(step)})
        write_json(run/'evidence'/f"{item['inspection_id']}.json",{**item,'surface_large_mm':10.,'aabb_quarantined':False,'faces_missing_mesh':0})
        (run/'evidence'/f"{item['inspection_id']}.png").write_bytes(b'synthetic fixture, not rendered CAD')
        np.savez(run/'evidence'/f"{item['inspection_id']}.npz",triangles=np.array([[[0,0,0],[1,0,0],[0,1,0]]]))
    for fid,batch in [('a','previous'),('b','new')]:
        scope='Synthetic task '+fid
        families.append({'family_id':fid,'label':fid,'scope':scope,'batch':batch})
        tasks.append({'family_id':fid,'label':fid,'request':scope,'features':[{'id':'shape','label':'Fixture shape'}],'rules':{}})
        queries.append({'query_id':fid,'scenario_id':fid,'reference_part_id':'train_fixture','constraints':[]})
    functions=[{'function_id':'f','label':'Fixture function','aliases':'','evidence':'https://example.invalid/fixture','status':'source_supported','source_id':'fixture','reviewer':''}]
    priors=[{'class_id':'spring','function_id':'f','relevance':'1','confidence':'1','scope':'fixture only','caveat':'fictional test','reviewer':'','evidence':'https://example.invalid/fixture','status':'source_supported','source_id':'fixture'}]
    write_csv(knowledge/'functions.csv',functions);write_csv(knowledge/'class_functions.csv',priors)
    write_json(run/'bundle/catalog.json',{'dataset_type':'synthetic_fixture','parts':parts})
    settings={'knowledge':str(knowledge),'k':5,'geometry_kinds':['embedding','descriptor']}
    write_json(run/'protocol.json',{'families':families,'queries':queries,'settings':settings,
                                   'query_sha256':object_hash(queries),'scope_sha256':object_hash(families)})
    write_csv(run/'pool_manifest.csv',manifest);write_json(run/'family_pools.json',{'a':ids,'b':ids})
    ranks=[{'geometry_kind':kind,'query_id':q['query_id'],'method':method,'rank':i+1,'part_id':pid}
           for kind in settings['geometry_kinds'] for q in queries for method in REFERENCE_METHODS for i,pid in enumerate(ids)]
    write_csv(run/'rankings.csv',ranks)
    task_file=tmp_path/'tasks.json';write_json(task_file,{'tasks':tasks})
    return Review(run,tmp_path/'reviews',task_file),task_file


def save(review, fid='a', pid='part_a', decision='shortlist', reviewer='fixture_a'):
    features={} if decision=='cannot_assess' else {'shape':'present' if decision=='shortlist' else 'absent'}
    reason={'shortlist':'features_visible','reject':'feature_missing','conditional':'feature_unverified','cannot_assess':'views_insufficient'}[decision]
    return review.save({'reviewer':reviewer,'human_review_attested':True,'kind':'judgments','key':fid+'|'+pid,
                        'record':{'family_id':fid,'part_id':pid,'decision':decision,'features':features,'reason':reason,'notes':''}})


def approve(review,fid,reviewer='fixture_a'):
    review.save({'reviewer':reviewer,'human_review_attested':True,'kind':'task_reviews','key':fid,'record':{'decision':'accept'}})


def test_unknown_is_null_and_revisions_and_exports_preserve_work(review_fixture,tmp_path):
    review,_=review_fixture
    result=save(review,decision='cannot_assess')
    assert result['record']['grade'] is None
    assert result['record']['observation']
    save(review,decision='reject')
    state=review.load('fixture_a')
    assert state['history'][-1]['previous']['grade'] is None
    review.export('fixture_a',tmp_path/'export')
    assert read_csv(tmp_path/'export/judgments.csv')[0]['grade']=='0'
    assert read_csv(tmp_path/'export/knowledge/class_functions.csv')==[]
    # Reviewer records survive new sessions rather than being reset by preparation.
    again=Review(review.run,review.workspace,review_fixture[1])
    assert again.load('fixture_a')==state


def test_generic_knowledge_export_contains_no_relevance_judgments(review_fixture,tmp_path):
    review,_=review_fixture
    save(review)
    review.save({'reviewer':'fixture_a','human_review_attested':True,'kind':'knowledge_reviews','key':'spring|f','record':{'decision':'accept'}})
    result=review.export('fixture_a',tmp_path/'export')
    assert result['accepted_priors']==1
    assert read_csv(tmp_path/'export/knowledge/class_functions.csv')[0]['status']=='curated'
    assert not (tmp_path/'export/knowledge/judgments.csv').exists()


def test_source_change_and_false_shortlist_and_instruction_change_are_rejected(review_fixture):
    review,tasks=review_fixture
    with pytest.raises(ValueError,match='confirm all'):
        review.save({'reviewer':'fixture_a','human_review_attested':True,'kind':'judgments','key':'a|part_a',
                     'record':{'family_id':'a','part_id':'part_a','decision':'shortlist','features':{'shape':'unclear'},'reason':'features_visible'}})
    source=review.check_source('part_a');source.write_text('changed source')
    with pytest.raises(ValueError,match='STEP changed'):
        save(review)
    state=review.load('fixture_a');write_json(review.path('fixture_a'),state)
    changed=read_json(tasks);changed['tasks'][0]['features'][0]['label']='Changed instructions';write_json(tasks,changed)
    with pytest.raises(ValueError,match='different source inputs'):
        Review(review.run,review.workspace,tasks).load('fixture_a')


def test_expert_evaluation_blocks_incomplete_families_and_does_not_fill_unknown(review_fixture,tmp_path):
    review,tasks=review_fixture
    approve(review,'a');approve(review,'b')
    save(review,'a','part_a');save(review,'a','part_b','reject')
    save(review,'b','part_a');save(review,'b','part_b','cannot_assess')
    exported=tmp_path/'review.json';write_json(exported,review.load('fixture_a'))
    with pytest.raises(ValueError,match='incomplete'):
        evaluate_expert(review.run,exported,tmp_path/'strict',tasks=tasks)
    assert not (tmp_path/'strict/summary.csv').exists()
    result=evaluate_expert(review.run,exported,tmp_path/'partial',True,tasks)
    assert result['included_families']==1
    assert {r['family_id'] for r in read_csv(tmp_path/'partial/per_query.csv')}=={'a'}
    coverage=read_json(tmp_path/'partial/coverage.json')
    assert coverage['families'][1]['cannot_assess']==1
    assert not coverage['families'][1]['included_in_paired_evaluation']
    assert {r['source'] for r in read_csv(tmp_path/'partial/expert_judgments.csv')}=={'independent_expert'}


def test_ai_and_ui_fixture_cannot_be_passed_as_human_truth(review_fixture,tmp_path):
    review,tasks=review_fixture
    record=review.load('fixture_a');record['source']='ai_reference';record['human_review_attested']=True
    target=tmp_path/'fake.json';write_json(target,record)
    with pytest.raises(ValueError,match='AI labels are forbidden'):
        evaluate_expert(review.run,target,tmp_path/'out',tasks=tasks)
    qa=Review(review.run,tmp_path/'qa',tasks,validation_fixture=True);save(qa)
    record=qa.load('fixture_a')
    assert record['source']=='validation_fixture' and record['human_review_attested'] is False
    write_json(target,record)
    with pytest.raises(ValueError,match='AI labels are forbidden'):
        evaluate_expert(review.run,target,tmp_path/'out',tasks=tasks)


def test_agreement_excludes_unknown_without_calling_it_rejection(review_fixture,tmp_path):
    review,_=review_fixture
    save(review,decision='shortlist',reviewer='fixture_a');save(review,decision='shortlist',reviewer='fixture_b')
    save(review,pid='part_b',decision='cannot_assess',reviewer='fixture_a');save(review,pid='part_b',decision='reject',reviewer='fixture_b')
    result=agreement(review,['fixture_a','fixture_b'],tmp_path/'agreement')
    assert result['common_records']==2 and result['both_assessable']==1
    assert result['agreement']==1 and result['quadratic_weighted_kappa'] is None
    assert result['cannot_assess_in_either']==1
    assert len(read_csv(tmp_path/'agreement/disagreements.csv'))==1
