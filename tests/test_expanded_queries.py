from collections import Counter
from copy import deepcopy
from pathlib import Path
import pytest
from cadkg.io import read_json, object_hash
from scripts.prepare_expanded_queries import build, SOURCE


@pytest.fixture(scope='module')
def inputs():
    catalog=read_json(SOURCE/'bundle/catalog.json')
    catalog={'parts':[{k:p[k] for k in ['part_id','split','label','measurements','file_sha256']} for p in catalog['parts']]}
    return read_json(SOURCE/'protocol.json'),catalog,read_json(SOURCE/'bundle/knowledge.json')


def test_preserves_originals_and_balances_queries(inputs):
    protocol,catalog,knowledge=inputs
    before=object_hash(protocol)
    queries,manifest,sources=build(*inputs)
    assert object_hash(protocol)==before
    assert queries[:49]==protocol['queries']
    assert len(queries)==len({q['query_id'] for q in queries})==110
    assert set(Counter(q['scenario_id'] for q in queries).values())=={10}
    for family in protocol['families']:
        local=[q for q in queries if q['scenario_id']==family['family_id']]
        assert sum(bool(q['reference_part_id']) for q in local)==7
    assert sum(r['origin']=='added_validation_quantile' for r in manifest)==61
    train={p['part_id'] for p in catalog['parts'] if p['split']=='train'}
    assert {q['reference_part_id'] for q in queries if q['reference_part_id']}<=train
    assert {q['reference_part_id'] for q in queries if q['reference_part_id']}=={e['part_id'] for e in protocol['exemplars']}


def test_test_set_changes_cannot_change_selection(inputs):
    protocol,catalog,knowledge=inputs
    changed=deepcopy(catalog)
    for p in changed['parts']:
        if p['split']=='test':
            p['label']='arbitrary';p['measurements']={}
    assert build(protocol,changed,knowledge)==build(*inputs)


def test_frozen_extension_reproduces_and_sources_are_validation(inputs):
    queries,manifest,sources=build(*inputs)
    root=Path('annotations/intent_queries110_v3')
    assert queries==read_json(root/'queries.json')
    assert sources==read_json(root/'validation_sources.json')
    catalog={p['part_id']:p for p in inputs[1]['parts']}
    for group in sources.values():
        for value in group['measurements']:
            assert catalog[value['part_id']]['split']=='val'
    assert object_hash(queries)==read_json(root/'protocol_extension.json')['query_sha256']
