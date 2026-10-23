import pytest
from cadkg.experiments.intent import score_grade, summarize
from cadkg.experiments.reference import REFERENCE_METHODS


def test_grade_sensitivity_cannot_override_dimension_failure_or_uncertainty():
    assert score_grade(1, 'fail', 'conditional_two') == 0
    assert score_grade(1, 'unknown', 'conditional_two') == 1
    assert score_grade(2, 'unknown') == 1
    assert score_grade(1, 'pass', 'conditional_zero') == 0
    with pytest.raises(ValueError):
        score_grade(True, 'pass')


def test_macro_family_averaging_does_not_overweight_more_query_variants():
    families=[{'family_id':'a','batch':'previous'},{'family_id':'b','batch':'new'}]
    cases=[]
    for kind in ['embedding','descriptor']:
        for method in REFERENCE_METHODS:
            for family,n,value in [('a',2,0.),('b',4,1.)]:
                for i in range(n):
                    cases.append({'family_id':family,'geometry_kind':kind,'method':method,'has_exemplar':True,
                                  'pool_ndcg@5':value,'precision@5':value,'strong_precision@5':value,'dimension_violations':0})
                cases.append({'family_id':family,'geometry_kind':kind,'method':method,'has_exemplar':False,
                              'pool_ndcg@5':value,'precision@5':value,'strong_precision@5':value,'dimension_violations':0})
    result=next(r for r in summarize(cases,families) if r['geometry_kind']=='embedding' and r['subset']=='combined' and r['variant']=='exemplar' and r['method']=='kg_full')
    assert result['macro_pool_ndcg@5']==.5
    assert result['paired_queries']==6
    assert result['families']==2
