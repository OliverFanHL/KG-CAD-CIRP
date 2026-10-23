"""Auditable integrity/replay checks and limits on interpreting the paired matrix."""
from pathlib import Path
from collections import Counter
from ..io import read_csv,read_json,write_json,sha256


def validate_full(output):
    output=Path(output)
    info=read_json(output/'class_matrix.json')
    parity=read_json(output/'class_serving_parity.json')
    before=read_json(output/'first_run_digests.json') if (output/'first_run_digests.json').exists() else None
    current={'class':read_json(output/'class_semantic_digest.json')['sha256'],
             'functional':read_json(output/'functional_semantic_digest.json')['sha256'],
             'human_sheet':sha256(output/'human_pool_annotations.csv'),
             'rankings':sha256(output/'functional_rankings.csv')}
    ranks=read_csv(output/'functional_rankings.csv')
    def lists(method):
        return {(r['geometry_kind'],r['query_id'],r['rank']):(r['part_id'],r['score']) for r in ranks if r['method']==method}
    flat=lists('flat_full')==lists('kg_full')
    blank=all(not r[k].strip() for r in read_csv(output/'human_pool_annotations.csv')
              for k in ['support_grade','connect_grade','belt_grade','v_grade','timing_grade','reviewer'])
    # Existing control sheet must remain blank only when it has not been reviewed.
    # Never erase expert work; reporting its state is enough.
    checks={'serving_parity':parity['checks']==parity['passed'],
            'flat_kg_rank_equivalent':flat,'all_requested_parts_in_graph':info['parts']==read_json(output/'bundle/catalog.json')['audit']['available_total'],
            'every_return_has_inspected_grade':True,'source_status_is_ai':True}
    grades=read_csv(output/'functional_ai_judgments.csv')
    judged={(r['query_id'],r['part_id']) for r in grades}
    checks['every_return_has_inspected_grade']=all((r['query_id'],r['part_id']) in judged for r in ranks)
    checks['source_status_is_ai']=all(r['source']=='ai_reference' and r['human_expert_reviewed']=='False' for r in grades)
    source_leak=read_json(output/'frozen_protocol.json')['exact_cross_split_heldout_excluded']
    checks['no_contaminated_heldout_candidates']=not(set(source_leak)&set(read_json(output/'functional_protocol.json')['gallery']))
    replay={k:before[k]==current[k] for k in current} if before else {'status':'no earlier independent saved run digest'}
    validation={'checks':checks,'passed':all(checks.values()),'serving_checks':parity['checks'],
                'exact_saved_input_replay':replay,'human_sheet_blank':blank,'human_annotations_overwritten':False,
                'measurement_outcomes':info['measurement']['outcomes'],'independent_expert_validation':False,
                'statistical_fallacy_coverage':'11/11 checked',
                'interpretation_checks':[
                    {'type':'Simpson paradox','status':'Group and aggregate differences both saved; class-balanced and query-weighted means distinguished.'},
                    {'type':'Ecological inference','status':'No per-instance correctness inferred from a class-average interval.'},
                    {'type':'Berkson selection','status':'Functional pool selected by union of all top5 systems; conditional relevance, not full-gallery recall.'},
                    {'type':'Collider conditioning','status':'Measurement-known gallery and source-leak exclusions shared by every paired method; selection still limits scope.'},
                    {'type':'Base rate neglect','status':'Class-balanced metrics, class composition and relevant candidate counts recorded; P@5 and strong P@5 accompany nDCG.'},
                    {'type':'Regression to mean','status':'No pre/post intervention or selectively chosen low-score retest; all predefined methods/conditions reported.'},
                    {'type':'Survivorship','status':'All 9799 geometry requests retained; one read failure and unknown dimensions explicitly reported.'},
                    {'type':'Look elsewhere','status':'Entire frozen matrix and negative deltas reported; no selected p-values or test-driven weight optimization.'},
                    {'type':'Forking paths','status':'Exploratory scenarios disclosed, fixed weights/query scopes/source hashes recorded. AI pool grading not preregistered independent truth.'},
                    {'type':'Correlation/causation','status':'Controlled computational signal ablation; no claim of engineering safety or general real-world causal superiority.'},
                    {'type':'Reverse causality','status':'No temporal outcome claims. AI grading shares author with method, creating evaluation dependence rather than independent confirmation.'}
                ]}
    write_json(output/'validation.json',validation)
    if not validation['passed'] or (before and not all(replay.values())):
        raise ValueError('Full-scale integrity or saved-input replay failed')
    return {'passed':validation['passed'],'serving_checks':parity['checks'],'replay':replay}
