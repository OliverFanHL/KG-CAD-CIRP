"""Synthetic contracts for the new evaluator; never engineering truth."""
import numpy as np
from scripts.run_verified_evaluation import rank_components, average_ranks, measure, near_duplicate_gallery
from cadkg.experiments.verified_statistics import _flip_p, _holm

def test_diagnostic_cohorts_cannot_admit_unjudged_or_duplicate_ids(tmp_path):
    import pytest
    from cadkg.io import write_json
    from scripts.run_verified_evaluation import diagnostic_cohorts
    path=tmp_path/'cohorts.json'
    write_json(path,{'added':['b','a']})
    assert diagnostic_cohorts(path,['a','b','c'],{'closed':[]})=={'added':['a','b']}
    for definition in [{'added':['train']},{'added':['a','a']},{'closed':['a']},{'added':[]}]:
        write_json(path,definition)
        with pytest.raises(ValueError):diagnostic_cohorts(path,['a','b'],{'closed':[]})

def test_average_rank_ties():
    np.testing.assert_array_equal(average_ranks(np.array([1.,1.,0.])),[1.5,1.5,3.])

def test_missing_geometry_abstains():
    order,_=rank_components({'intent':np.ones(3)},np.ones(3,dtype=bool),'geometry',['a','b','c'])
    assert not len(order)

def test_filter_and_equal_weight_arithmetic():
    comp={'geometry':np.array([.8,.2,.6]),'reference':np.array([0.,1.,.5]),'intent':np.array([1.,0.,.5])}
    order,score=rank_components(comp,np.array([False,True,True]),'kg_full',['a','b','c'])
    np.testing.assert_array_equal(order,[2,1]);np.testing.assert_allclose(score,[1.6/3,1.2/3])

def test_rrf_uses_only_available_filtered_branches():
    comp={'intent':np.array([1.,1.,0.])}
    order,score=rank_components(comp,np.array([True,True,False]),'rrf_filter',['a','b','c'])
    np.testing.assert_array_equal(order,[0,1]);np.testing.assert_allclose(score,[1/61.5,1/61.5])

def test_metrics_and_abstention():
    result=measure(np.array([0,1]),np.array([2,0,1]),['pass']*3,[5])
    assert result['precision@5']==.2 and result['strong_precision@5']==.2
    assert result['recall@5']==.5 and result['ap']==.5
    assert result['ap@20']==.5
    empty=measure(np.array([],dtype=int),np.array([2,0,1]),['pass']*3,[5])
    assert empty['ndcg@5']==0 and empty['ap']==0

def test_exact_sign_flip_and_holm():
    assert _flip_p([1.,1.,1.])==.25
    assert _flip_p([0.,0.])==1.
    np.testing.assert_allclose(_holm([.01,.04,.03]),[.03,.06,.06])

def test_numeric_signature_groups():
    class Fixture:
        vectors={'a':np.array([0.,1.]),'b':np.array([0.,1.]),'c':np.array([2.,1.]),'train':np.array([2.,1.])}
    kept,audit=near_duplicate_gallery(['a','b','c'],['train'],Fixture(),1e-6)
    assert kept==['a'] and audit['train_matches']==1 and audit['duplicate_groups']==[['a','b']]
