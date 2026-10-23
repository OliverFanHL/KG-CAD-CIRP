"""Freeze a balanced variant extension using validation sizes, never test outcomes."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from cadkg.io import read_json, write_json, write_csv, sha256, object_hash

SOURCE = Path('outputs/intent_batch2_matrix_reproduction')
OUT = Path('annotations/intent_queries110_v3')


def signature(q):
    return object_hash({k:q[k] for k in ('scenario_id','function_ids','category_ids','reference_part_id','constraints')})


def build(protocol, catalog, knowledge):
    queries=deepcopy(protocol['queries']); manifest=[]; sources={}
    signatures={signature(q) for q in queries}
    if len(signatures)!=len(queries):raise ValueError('Original structured queries are not distinct')
    original_ids={q['query_id'] for q in queries}
    for fi,family in enumerate(protocol['families']):
        fid=family['family_id']; function=family['function_id']
        links=[r for r in knowledge['class_functions'] if r['function_id']==function]
        if len(links)!=1 or links[0]['relevance']*links[0]['confidence']!=1:raise ValueError('Unexpected class-function mapping')
        category=links[0]['class_id'];values=[]
        for p in catalog['parts']:
            m=p['measurements'].get('aabb_large_mm',{})
            v=m.get('value')
            if p['split']=='val' and p['label']==category and v is not None and m.get('status')=='computed' and np.isfinite(v) and v>0:
                values.append(dict(part_id=p['part_id'],source_sha256=p['file_sha256'],value_mm=v))
        values.sort(key=lambda x:x['part_id'])
        if not values:raise ValueError('Missing validation measurements')
        sources[fid]=dict(category_for_validation_sampling=category,split='val',measurements=values)
        sizes=np.asarray([x['value_mm'] for x in values])
        current=[q for q in queries if q['scenario_id']==fid]
        refs=sorted({q['reference_part_id'] for q in current if q['reference_part_id']})
        refs=refs[fi%len(refs):]+refs[:fi%len(refs)]
        def add(ref,quantile,mode):
            bound=round(float(np.quantile(sizes,quantile,method='linear')),6)
            q=dict(query_id=f'{fid}_v3_{mode}_{int(round(quantile*100)):02d}'+('' if ref is None else f'_{ref[:8]}'),
                   scenario_id=fid,split='test',text=family['scope']+f' Largest stored-axis extent must be at most {bound:g} mm.'
                   +(' Use the supplied training exemplar for geometric similarity.' if ref else '')
                   +' This is a hypothetical dimension screening requirement.',function_ids=[function],category_ids=[],reference_part_id=ref,
                   constraints=[dict(attribute='aabb_large_mm',op='<=',value=bound,unit='mm')])
            sig=signature(q)
            if sig in signatures:return False
            if q['query_id'] in {x['query_id'] for x in queries}:raise ValueError('Duplicate query ID')
            signatures.add(sig);queries.append(q)
            manifest.append(dict(query_id=q['query_id'],family_id=fid,origin='added_validation_quantile',mode=mode,
                reference_part_id=ref,quantile=quantile,bound_mm=bound,validation_n=len(values),validation_category=category,
                query_object_sha256=object_hash(q)))
            return True
        for quantile in [.25,.75]:
            if not add(None,quantile,'text'):raise ValueError('Collapsed text quantiles require a new prespecified design')
        needed=10-sum(q['scenario_id']==fid for q in queries)
        grid=[.25,.75,.10,.90,.50,.05,.95]+[x/100 for x in range(1,100)]
        # Each step pairs a quantile with a reference; duplicates advance the grid.
        for i,quantile in enumerate(grid):
            if not needed:break
            if add(refs[i%len(refs)],quantile,'exemplar'):needed-=1
        if needed:raise ValueError('Insufficient distinct validation-derived variants')
    old=[dict(query_id=q['query_id'],family_id=q['scenario_id'],origin='original49',mode='exemplar' if q['reference_part_id'] else 'text',
              reference_part_id=q['reference_part_id'],query_object_sha256=object_hash(q)) for q in protocol['queries']]
    assert len(queries)==110 and len(manifest)==61
    assert Counter(q['scenario_id'] for q in queries)=={f['family_id']:10 for f in protocol['families']}
    assert [q for q in queries if q['query_id'] in original_ids]==protocol['queries']
    return queries,old+manifest,sources


def main():
    protocol=read_json(SOURCE/'protocol.json');catalog=read_json(SOURCE/'bundle/catalog.json');knowledge=read_json(SOURCE/'bundle/knowledge.json')
    queries,manifest,sources=build(protocol,catalog,knowledge)
    write_json(OUT/'queries.json',queries);write_csv(OUT/'query_manifest.csv',manifest);write_json(OUT/'validation_sources.json',sources)
    write_json(OUT/'protocol_extension.json',dict(version='queries110-v3',original_query_sha256=protocol['query_sha256'],
        query_sha256=object_hash(queries),scope_sha256=protocol['scope_sha256'],original_queries=49,added_queries=61,total_queries=110,
        family_count=11,queries_per_family=10,annotation_transfer='Unchanged family shape grades plus per-query numeric verdict; not new independent judgments',
        input_sha256={str(p):sha256(p) for p in [SOURCE/'protocol.json',SOURCE/'bundle/catalog.json',SOURCE/'bundle/knowledge.json',
            Path('docs/QUERY_EXPANSION_110.md'),Path(__file__)]},
        file_sha256={str(OUT/p):sha256(OUT/p) for p in ['queries.json','query_manifest.csv','validation_sources.json']},
        selection_uses_test_grades=False,selection_uses_rankings=False,new_task_families=False))
    print(dict(queries=len(queries),families=11,has_exemplar=sum(bool(q['reference_part_id']) for q in queries),
        constrained=sum(bool(q['constraints']) for q in queries),constrained_exemplar=sum(bool(q['constraints']) and bool(q['reference_part_id']) for q in queries)))


if __name__=='__main__':main()
