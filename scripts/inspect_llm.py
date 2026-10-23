"""Inspect saved LLM evidence or compare observations without making API calls."""
from pathlib import Path
import argparse,json

ROOT=Path(__file__).resolve().parents[1]

def records(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    identity=parser.add_mutually_exclusive_group(required=True)
    identity.add_argument('--inspection-id',help='Source-review identifier, e.g. Z0430')
    identity.add_argument('--part-id',help='Stable CAD part ID')
    parser.add_argument('--role',required=True,help='Role identifier, e.g. shaft_drive')
    parser.add_argument('--arm',choices=['kg_luna','kg_sol','role_only_luna'],default='kg_luna')
    parser.add_argument('--collection',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--compare',action='store_true',help='Show support across all observation designs and collections')
    parser.add_argument('--raw',action='store_true',help='Include the saved parsed model output and usage')
    args=parser.parse_args()
    base=ROOT/'data/observations'
    path=base/'repeated_observations/graph_evidence.jsonl'
    if not path.exists():parser.error('Run python -m scripts.prepare_data first.')
    part=args.part_id
    if args.inspection_id:
        annotations=json.loads((ROOT/'data/annotations/judgments.json').read_text(encoding='utf-8-sig'))
        ids={r['part_id'] for r in annotations['judgments'] if r['inspection_id']==args.inspection_id}
        if len(ids)!=1:parser.error('Inspection ID must identify exactly one CAD part.')
        part=ids.pop()
    matches=[r for r in records(path) if r['part_id']==part and r['family_id']==args.role]
    if args.compare:
        selected=sorted(matches,key=lambda r:(r['arm'],r['replicate']))
        result=dict(part_id=part,role=args.role,observations=[dict(arm=r['arm'],collection=r['replicate'],support=r.get('H',r.get('J')),valid_response=r['valid_response']) for r in selected])
    else:
        selected=[r for r in matches if r['arm']==args.arm and r['replicate']==args.collection]
        if len(selected)!=1:parser.error('No unique saved observation for this role, part, design, and collection; the CAD may be outside the refinement pool.')
        row=selected[0];source='repeated_observations'
        if row['historical']:source='directed_luna' if args.arm=='kg_luna' else 'directed_sol'
        result=dict(evidence=row,image=str(base/'images'/(part+'.png')),raw_response_file=str(base/source/'api_responses.jsonl'))
        if args.raw:
            raw_job=row['source_job_id'] if row['historical'] else row['job_id']
            raw=next((r for r in records(base/source/'api_responses.jsonl') if r['job_id']==raw_job),None)
            if raw is None:parser.error('Saved raw response was not found.')
            result['model_output']={k:raw.get(k) for k in ['job_id','response_id','model_resolved','output_json','output_text','usage_tokens','status']}
    if not selected:parser.error('No saved observations for this role and CAD part.')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
