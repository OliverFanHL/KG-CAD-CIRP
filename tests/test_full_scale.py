import numpy as np

from cadkg.experiments.full import byte_unique, paired_scores, topk_stable, extent
from cadkg.retrieval import METHODS


def test_stable_topk_handles_saturated_classifier_ties():
    # Columns already have lexicographically sorted source IDs.
    scores=np.array([[1.,1.,.5,1.],[.2,.9,.9,.1]])
    assert topk_stable(scores,3).tolist()==[[0,1,3],[1,2,0]]


def test_ablation_removes_only_exemplar_cosine_and_denominator():
    g,c,ref=np.array([[.1]]),np.array([[.8]]),np.array([[.6]])
    assert np.isclose(paired_scores(g,c,ref,'kg_full',{}),.5)
    assert np.isclose(paired_scores(g,c,ref,'kg_no_geometry',{}),.7)
    assert METHODS['kg_no_geometry']['constraints']
    assert 'class_reference' in METHODS['kg_no_geometry']['signals']


def test_source_duplicates_collapse_without_category_information():
    parts={'b':{'file_sha256':'same'},'a':{'file_sha256':'same'},'c':{'file_sha256':'different'}}
    assert byte_unique(['c','b','a'],parts)==['a','c']


def test_unattested_measurements_cannot_enter_known_dimension_gallery():
    p={'measurements':{'aabb_large_mm':{'value':5,'unit':'unknown','status':'imported_unverified'}}}
    assert extent(p) is None
    p['measurements']['aabb_large_mm'].update(unit='mm',status='computed')
    assert extent(p)==5


def test_pool_annotations_cannot_be_reused_for_changed_function_scope(tmp_path):
    import pytest
    from cadkg.experiments.full_report import evaluate_pool
    from cadkg.io import write_json,write_csv
    write_csv(tmp_path/'functional_pool_manifest.csv',[{'inspection_id':'F001','part_id':'part','source_sha256':'hash'}])
    write_json(tmp_path/'functional_protocol.json',{'query_sha256':'new_scope'})
    annotations=tmp_path/'judgments.json'
    write_json(annotations,{'source':'ai_reference','human_expert_reviewed':False,'query_object_sha256':'old_scope',
                           'parts':[{'inspection_id':'F001'}]})
    with pytest.raises(ValueError,match='different functional scopes'):
        evaluate_pool(tmp_path,annotations)
