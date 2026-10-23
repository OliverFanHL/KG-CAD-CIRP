"""Build a balanced, development-only query extension after neutral source audit."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256,object_hash
from scripts.finalize_feature_review import notes,feature_issues,ROLE
from scripts.prepare_expanded_queries import signature

ROOT=Path('outputs/query_diversity_v6');POOL=ROOT/'exemplar_pool';OUT=Path('annotations/intent_queries440_v6')
BASE=Path('outputs/intent_batch2_matrix_reproduction')

def candidate_pool():
    """Load immutable original candidates plus the separately frozen addition."""
    original=read_json(POOL/'pool_freeze.json');paths=[POOL/'pool_freeze.json']
    for name,h in original['outputs_sha256'].items():
        path=POOL/name;assert sha256(path)==h,(path,'original pool hash');paths.append(path)
    pool=read_csv(POOL/'private/candidate_manifest.csv');assert len(pool)==132
    expansion_path=POOL/'expansion_freeze.json'
    if expansion_path.exists():
        expansion=read_json(expansion_path);paths.append(expansion_path)
        assert expansion['original_132_prefixes_identical'] and expansion['original_normalization_reused']
        for name,h in expansion['inputs_sha256'].items():assert sha256(name)==h,(name,'expansion input')
        for name,h in expansion['outputs_sha256'].items():
            path=POOL/name;assert sha256(path)==h,(path,'expansion output');paths.append(path)
        check=read_json(POOL/'private/expansion_selection_audit.json')
        before={(r['family_id'],int(r['selection_step'])):r for r in pool}
        assert len(check['prefix_checks'])==132 and check['all_original_132_prefixes_identical']
        assert {(r['family_id'],r['selection_step']) for r in check['prefix_checks']}==set(before)
        for r in check['prefix_checks']:
            old=before[r['family_id'],r['selection_step']]
            assert (old['part_id'],old['source_sha256'],old['review_id'])==(r['part_id'],r['source_sha256'],r['original_review_id'])
        added=read_csv(POOL/'private/expansion_candidate_manifest.csv')
        assert len(added)==24 and {r['family_id'] for r in added}=={'support_body','tension_return'}
        for fid in {'support_body','tension_return'}:
            assert sorted(int(r['selection_step']) for r in added if r['family_id']==fid)==list(range(13,25))
        assert {r['review_id'] for r in added}=={f'Q{i:04}' for i in range(133,157)}
        pool+=added
    assert len({r['review_id'] for r in pool})==len(pool)==len({r['source_sha256'] for r in pool})
    hashes={str(p):sha256(p) for p in paths}
    if expansion_path.exists():hashes.update(expansion['inputs_sha256'])
    return pool,hashes

def audit_inputs():
    pool,hashes=candidate_pool();known={r['review_id'] for r in pool};combined={}
    for path in sorted(POOL.glob('*_notes.txt')):
        report_path=path.with_name(path.name.replace('_notes.txt','_coverage.json'))
        if not report_path.exists():continue
        report=read_json(report_path)
        if not (report.get('complete') or report.get('status')=='complete'):continue
        chunk=notes(path)
        inspected=report.get('inspected_ids',report.get('visually_reviewed_ids'))
        assert inspected is not None and set(inspected)==set(chunk),(path,'coverage')
        assert report['notes_sha256']==sha256(path),(path,'notes hash')
        assert report['rubric_sha256']==sha256('outputs/feature_review_v5/FROZEN_RUBRIC.md')
        assert not set(chunk)&set(combined),'Duplicate source audit'
        assert set(chunk)<=known,(path,'source outside frozen pools')
        for code,item in chunk.items():
            for role,grade in item['grades'].items():
                assert not feature_issues(role,grade,[f for f in item['features'] if f['role']==role]),(code,role)
        combined.update(chunk);hashes.update({str(p):sha256(p) for p in [path,report_path]})
    initial={r['review_id'] for r in read_csv(POOL/'blind_manifest.csv')}
    assert initial<=set(combined),f'Initial audit incomplete: {len(initial&set(combined))}/66'
    return combined,hashes

def seal_initial():
    path=ROOT/'initial_source_audit_freeze.json';assert not path.exists()
    audited,hashes=audit_inputs();initial={r['review_id'] for r in read_csv(POOL/'blind_manifest.csv')}
    assert set(audited)==initial
    write_json(path,dict(stage='All66 initial sources reviewed before intended-family unblinding and query selection',
        cads=66,human_reviewed=False,input_sha256=hashes,rubric_sha256=sha256('outputs/feature_review_v5/FROZEN_RUBRIC.md')))
    print('Initial66 source audit sealed')

def select():
    for path,h in read_json(ROOT/'initial_source_audit_freeze.json')['input_sha256'].items():assert sha256(path)==h
    audited,hashes=audit_inputs();pool,_=candidate_pool();selected=[];needed=[];counts={}
    for fid in sorted(ROLE.values()):
        candidates=sorted([r for r in pool if r['family_id']==fid],key=lambda r:int(r['selection_step']))
        eligible=[];pending=[]
        for row in candidates:
            code=row['review_id']
            if len(eligible)>=4:break
            if code not in audited:
                pending.extend(r for r in candidates if int(r['selection_step'])>=int(row['selection_step']))
                break
            grade=audited[code]['grades'].get(row['role'],0)
            if grade in (1,2):eligible.append(dict(row,source_grade=grade,source_observation=audited[code]['observation'],source_notes=audited[code]['source_notes']))
        counts[fid]=dict(eligible_reviewed=len(eligible),selected=[r['review_id'] for r in eligible],reserve_needed=len(eligible)<4)
        if len(eligible)<4:needed.extend(pending)
        else:selected.extend(eligible)
    write_csv(POOL/'needed_reserve_manifest.csv',[dict(review_id=r['review_id'],image=r['image']) for r in needed])
    write_json(ROOT/'source_selection_progress.json',dict(families=counts,selected_complete_families=len(selected)//4,needed_sources=len(needed),audited_sources=len(audited),frozen_candidate_sources=len(pool)))
    print({k:v for k,v in counts.items() if v['reserve_needed']})
    return selected,needed,audited,hashes

def build():
    assert not (OUT/'queries.json').exists(),'Query release already exists'
    selected,needed,audited,hashes=select();assert not needed,'Complete frozen reserve source audit before building'
    if (POOL/'expansion_freeze.json').exists():
        expanded={r['review_id'] for r in read_csv(POOL/'expansion_blind_manifest.csv')}
        assert expanded<=set(audited),'Complete all 24 frozen additive source audits before query release'
    assert len(selected)==44 and len({r['part_id'] for r in selected})==44
    catalog=read_json(BASE/'bundle/catalog.json');parts={p['part_id']:p for p in catalog['parts']}
    knowledge=read_json(BASE/'bundle/knowledge.json');old=read_json('annotations/intent_queries110_v3/queries.json')
    protocol=read_json(BASE/'protocol.json');queries=deepcopy(old);manifest=[];sources={};sigs={signature(q) for q in queries}
    oldrefs={q['reference_part_id'] for q in old if q['reference_part_id']};testhashes={p['file_sha256'] for p in parts.values() if p['split']=='test'}
    heldouthashes={p['file_sha256'] for p in parts.values() if p['split'] in {'val','test'}}
    for row in selected:
        p=parts[row['part_id']]
        assert p['split']=='train' and p['part_id'] not in oldrefs and p['file_sha256'] not in heldouthashes
        assert sha256(p['absolute_path'])==row['source_sha256']==p['file_sha256']
        binding=read_json(Path(row['image']).with_suffix('.json'));assert binding['image_sha256']==sha256(row['image'])
        assert binding['source_sha256']==p['file_sha256']
    for q in old:manifest.append(dict(query_id=q['query_id'],family_id=q['scenario_id'],origin='legacy110',condition='legacy',reference_review_id='',reference_grade='',validation_quantile='',bound_mm=q['constraints'][0]['value'] if q['constraints'] else '',query_object_sha256=object_hash(q)))
    for family in protocol['families']:
        fid=family['family_id'];function=family['function_id'];links=[r for r in knowledge['class_functions'] if r['function_id']==function]
        assert len(links)==1 and links[0]['relevance']*links[0]['confidence']==1
        category=links[0]['class_id'];values=[]
        for p in parts.values():
            m=p['measurements'].get('aabb_large_mm',{});v=m.get('value')
            if p['split']=='val' and p['label']==category and p['file_sha256'] not in testhashes and m.get('status')=='computed' and v is not None and np.isfinite(v) and v>0:
                assert sha256(p['absolute_path'])==p['file_sha256']
                values.append(dict(part_id=p['part_id'],source_sha256=p['file_sha256'],value_mm=v))
        values.sort(key=lambda x:x['part_id']);assert values
        sizes=np.asarray([r['value_mm'] for r in values]);sources[fid]=dict(split='val',category=category,exact_test_hashes_excluded=True,measurements=values)
        def add(ref,quantile,origin,index,review_id='',grade=''):
            bound=round(float(np.quantile(sizes,quantile,method='linear')),6) if quantile is not None else None
            constraints=[] if bound is None else [dict(attribute='aabb_large_mm',op='<=',value=bound,unit='mm')]
            q=dict(query_id=f'{fid}_v6_{origin}_{index}',scenario_id=fid,split='test',text=family['scope']+(' Use the supplied training exemplar for geometric similarity.' if ref else '')+(f' Largest stored-axis extent must be at most {bound:g} mm. This is a hypothetical dimension screening requirement.' if bound is not None else ''),function_ids=[function],category_ids=[],reference_part_id=ref,constraints=constraints)
            sig=signature(q)
            if sig in sigs:return False
            sigs.add(sig);queries.append(q);manifest.append(dict(query_id=q['query_id'],family_id=fid,origin=origin,condition='unconstrained' if quantile is None else f'val_q{int(round(quantile*100)):02}',reference_review_id=review_id,reference_grade=grade,validation_quantile='' if quantile is None else quantile,bound_mm='' if bound is None else bound,query_object_sha256=object_hash(q)))
            return True
        refs=sorted([r for r in selected if r['family_id']==fid],key=lambda r:int(r['selection_step']))
        for i,row in enumerate(refs,1):
            for j,quantile in enumerate([None,.10,.25,.50,.75,.90]):
                assert add(row['part_id'],quantile,'new_exemplar',f'r{i}_c{j}',row['review_id'],row['source_grade']),'Collapsed exemplar bounds; do not inflate duplicate queries'
        count=0
        for quantile in [.05,.10,.35,.50,.65,.90]+[i/100 for i in range(1,100)]:
            if count==6:break
            if add(None,quantile,'new_text',f'q{count+1}'):count+=1
        assert count==6
    assert len(queries)==len(sigs)==440 and Counter(q['scenario_id'] for q in queries)=={f['family_id']:40 for f in protocol['families']}
    assert queries[:110]==old and len({q['query_id'] for q in queries})==440
    write_json(OUT/'queries.json',queries);write_csv(OUT/'query_manifest.csv',manifest);write_csv(OUT/'exemplars.csv',selected);write_json(OUT/'validation_sources.json',sources)
    inputs=[Path(__file__),Path('docs/QUERY_DIVERSITY_V6.md'),POOL/'pool_freeze.json',POOL/'private/candidate_manifest.csv',ROOT/'initial_source_audit_freeze.json',Path('outputs/feature_review_v5/FROZEN_RUBRIC.md'),Path('annotations/intent_queries110_v3/queries.json'),BASE/'protocol.json',BASE/'bundle/catalog.json',BASE/'bundle/knowledge.json']
    input_hashes={str(p):sha256(p) for p in inputs};input_hashes.update(hashes)
    for row in selected:
        for p in [Path(row['image']),Path(row['image']).with_suffix('.json')]:input_hashes[str(p)]=sha256(p)
    write_json(OUT/'protocol_extension.json',dict(version='queries440-v6',total_queries=440,queries_per_family=40,old_query_objects_preserved=True,old_queries=110,new_queries=330,new_reference_queries=264,new_training_references=44,primary_subset='new_exemplar',primary_tests=12,source='ai_reference',human_expert_reviewed=False,scope_sha256=protocol['scope_sha256'],annotation_path='annotations/intent_feature_review_v5/judgments.json',annotation_sha256=sha256('annotations/intent_feature_review_v5/judgments.json'),query_sha256=object_hash(queries),selection_uses_test_grades=False,selection_uses_ranks=False,geometry_sampling='training-normalized descriptor farthest point; not probability sampled or backend-neutral',input_sha256=input_hashes,file_sha256={str(OUT/p):sha256(OUT/p) for p in ['queries.json','query_manifest.csv','exemplars.csv','validation_sources.json']}))
    print(dict(queries=len(queries),new_refs=44,exemplar=sum(bool(q['reference_part_id']) for q in queries),constrained_exemplar=sum(bool(q['reference_part_id']) and bool(q['constraints']) for q in queries)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal-initial','needed','build']);a=p.parse_args();{'seal-initial':seal_initial,'needed':select,'build':build}[a.action]()
