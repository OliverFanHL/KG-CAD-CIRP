"""Distinguish inactive ablations, identical rankings and equal coarse metric values."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from ..io import read_csv,read_json,write_csv,write_json
from ..retrieval import METHODS


def equality_audit(output):
    output=Path(output)
    protocol=read_json(output/'functional_protocol.json')
    ranks=defaultdict(list)
    for row in read_csv(output/'functional_rankings.csv'):
        ranks[row['geometry_kind'],row['query_id'],row['method']].append(row)
    grades={(r['query_id'],r['part_id']):int(r['grade']) for r in read_csv(output/'functional_ai_judgments.csv')}
    values={(r['geometry_kind'],r['query_id'],r['method']):r for r in read_csv(output/'functional_per_query.csv')}
    rows=[]
    def signature(query,method):
        if method=='category_from_intent':
            return ('class_function_prior',),False
        recipe=METHODS[method]
        active=[]
        for s in recipe['signals']:
            if s in {'geometry','class_reference'} and not query['reference_part_id']:
                continue
            if s=='category' and not query['category_ids']:
                continue
            if s=='semantic' and not query['function_ids']:
                continue
            active.append('class_function_prior' if s=='semantic' else s)
        return tuple(sorted(active)),bool(recipe['constraints'] and query['constraints'])
    for kind in ['embedding','descriptor']:
        for query in protocol['queries']:
            qid=query['query_id']
            for a,b in combinations(protocol['methods'],2):
                left=sorted(ranks[kind,qid,a],key=lambda r:int(r['rank']))
                right=sorted(ranks[kind,qid,b],key=lambda r:int(r['rank']))
                ids_a=[r['part_id'] for r in left];ids_b=[r['part_id'] for r in right]
                gain_a=[grades[qid,pid] for pid in ids_a];gain_b=[grades[qid,pid] for pid in ids_b]
                va,vb=values[kind,qid,a],values[kind,qid,b]
                nd_a,nd_b=float(va['pool_ndcg@5']),float(vb['pool_ndcg@5'])
                exact=np.isclose(nd_a,nd_b,rtol=0,atol=1e-12)
                if not left and not right:
                    reason='Both abstain: requested ranking signals unavailable'
                elif signature(query,a)==signature(query,b):
                    reason='Same effective signals/filter; difference inactive for this query'
                elif ids_a==ids_b:
                    reason='Different formulas, same ordered top5 on this query'
                elif gain_a==gain_b:
                    reason='Different retrieved parts, same judged grade at each rank'
                elif exact:
                    reason='Equal aggregate gain under this metric, despite different retrieved lists'
                elif round(nd_a,3)==round(nd_b,3):
                    reason='Three-decimal display hides a small numerical difference'
                else:
                    reason='Different metric values'
                rows.append({'geometry_kind':kind,'query_id':qid,'method_a':a,'method_b':b,
                    'same_effective_recipe':signature(query,a)==signature(query,b),'same_ordered_ids':ids_a==ids_b,
                    'same_grade_sequence':gain_a==gain_b,'same_ndcg_to_1e12':bool(exact),
                    'same_ndcg_at_3_decimals':round(nd_a,3)==round(nd_b,3),
                    'same_precision':float(va['precision@5'])==float(vb['precision@5']),
                    'both_abstain':not left and not right,'ndcg_a':nd_a,'ndcg_b':nd_b,'reason':reason})
    write_csv(output/'functional_equality_audit.csv',rows)
    report={'gallery':len(protocol['gallery']),'functional_query_families':len({q['scenario_id'] for q in protocol['queries']}),
            'pair_query_representation_comparisons':len(rows),
            'exact_same_ndcg':sum(r['same_ndcg_to_1e12'] for r in rows),
            'identical_ordered_ids':sum(r['same_ordered_ids'] for r in rows),
            'same_ndcg_but_different_ids':sum(r['same_ndcg_to_1e12'] and not r['same_ordered_ids'] for r in rows),
            'rounding_only_at_3_decimals':sum(r['same_ndcg_at_3_decimals'] and not r['same_ndcg_to_1e12'] for r in rows),
            'limitation':'Descriptive pair-count audit; pairs share queries and are not independent trials.',
            'core_reason':'Inactive signals, matched-information controls and coarse relevance grades. Gallery is full-scale; functional query diversity remains limited.'}
    write_json(output/'equality_audit.json',report)
    return report
