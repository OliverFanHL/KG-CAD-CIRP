"""Local visual review, source-bound human records, exports and rater comparison.

No prediction, retrieval rank or AI judgment is read by the review interface.
Reviewer identity and human review are self-attested, not authenticated credentials.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .bundle import ROOT
from .io import object_hash, read_csv, read_json, sha256, write_csv, write_json

DECISIONS = {'shortlist': 2, 'conditional': 1, 'reject': 0, 'cannot_assess': None}
REASONS = {
    'features_visible': 'Required shape features are visible',
    'feature_missing': 'A required feature is visibly absent',
    'wrong_form': 'Visible form conflicts with the requested role',
    'feature_unverified': 'A required feature remains unverified',
    'views_insufficient': 'Additional views or documentation are needed',
    'other': 'Other evidence, explained in the note',
}


def reviewer_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{1,63}', value):
        raise ValueError('Use a stable reviewer ID of 2–64 letters, digits, dots, hyphens or underscores')
    return value


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class Review:
    def __init__(self, run, workspace, tasks=None, validation_fixture=False):
        self.run, self.workspace = Path(run).resolve(), Path(workspace).resolve()
        self.validation_fixture = validation_fixture
        self.protocol = read_json(self.run/'protocol.json')
        self.manifest = {r['part_id']: r for r in read_csv(self.run/'pool_manifest.csv')}
        self.bycode = {r['inspection_id']: r for r in self.manifest.values()}
        self.pools = read_json(self.run/'family_pools.json')
        self.tasks = read_json(tasks or ROOT/'data/annotations/review_tasks.json')['tasks']
        self.bytask = {t['family_id']: t for t in self.tasks}
        self.families = {f['family_id']: f for f in self.protocol['families']}
        if set(self.bytask) != set(self.families) or set(self.pools) != set(self.families):
            raise ValueError('Review tasks must cover exactly the frozen experiment families')
        for fid, task in self.bytask.items():
            if task['request'] != self.families[fid]['scope']:
                raise ValueError('Task request changed; freeze a new experiment before collecting new labels')
        self.knowledge_dir = Path(self.protocol['settings']['knowledge']).resolve()
        self.functions = {r['function_id']: r for r in read_csv(self.knowledge_dir/'functions.csv')}
        self.priors = read_csv(self.knowledge_dir/'class_functions.csv')
        self.prior_keys = {r['class_id']+'|'+r['function_id']: r for r in self.priors}
        self.binding = {'query_sha256': self.protocol['query_sha256'], 'scope_sha256': self.protocol['scope_sha256'],
                        'task_instructions_sha256': object_hash(self.tasks),
                        'pool_manifest_sha256': sha256(self.run/'pool_manifest.csv'),
                        'rankings_sha256': sha256(self.run/'rankings.csv'),
                        'knowledge_sha256': object_hash({'functions': self.functions, 'priors': self.priors}),
                        'validation_fixture': validation_fixture}
        self.lock = threading.RLock()
        self.catalog = None
        self.checked_sources = {}

    def path(self, reviewer):
        identifier = reviewer_id(reviewer)
        # A digest also avoids Windows device names and case-folding collisions.
        return self.workspace/'reviewers'/(object_hash(identifier)+'.json')

    def load(self, reviewer):
        target = self.path(reviewer)
        if target.exists():
            state = read_json(target)
            if state['binding'] != self.binding or state['reviewer'] != reviewer:
                raise ValueError('Review belongs to different source inputs or instructions')
            return state
        return {'schema_version': 1, 'source': 'validation_fixture' if self.validation_fixture else 'independent_expert', 'reviewer': reviewer,
                'human_review_attested': False, 'binding': self.binding, 'run': str(self.run),
                'task_reviews': {}, 'knowledge_reviews': {}, 'judgments': {}, 'history': []}

    def check_source(self, pid):
        with self.lock:
            if self.catalog is None:
                catalog = read_json(self.run/'bundle/catalog.json')
                integrity_path = self.run/'bundle/bundle_manifest.json'
                if integrity_path.exists() and object_hash(catalog) != read_json(integrity_path)['catalog_sha256']:
                    raise ValueError('Source catalog was modified; rebuild the frozen bundle')
                self.catalog = {p['part_id']: p for p in catalog['parts']}
            part = self.catalog[pid]
            path = Path(part['absolute_path'])
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if self.checked_sources.get(pid) != signature:
                if sha256(path) != self.manifest[pid]['source_sha256']:
                    raise ValueError('Original STEP changed; stop review and rebuild evidence')
                self.checked_sources[pid] = signature
        return path

    def evidence(self, code):
        if code not in self.bycode:
            raise ValueError('Unknown neutral CAD ID')
        item = self.bycode[code]
        self.check_source(item['part_id'])
        data = read_json(self.run/'evidence'/f'{code}.json')
        if data['part_id'] != item['part_id'] or data['source_sha256'] != item['source_sha256']:
            raise ValueError('Rendered evidence is not bound to the original STEP')
        return item

    def bootstrap(self, reviewer=None):
        state = self.load(reviewer) if reviewer else None
        return {'tasks': self.tasks, 'reasons': REASONS, 'state': state,
                'parts': [{'family_id': fid, 'part_id': pid, 'inspection_id': self.manifest[pid]['inspection_id']}
                          for fid, ids in self.pools.items() for pid in sorted(ids, key=lambda x: object_hash((fid,x)))],
                'knowledge': [{'key': key, **r, 'function_label': r['function_id'].replace('_',' ')}
                              for key,r in self.prior_keys.items()]}

    def save(self, payload):
        reviewer = reviewer_id(payload.get('reviewer'))
        if payload.get('human_review_attested') is not True:
            raise ValueError('Confirm that these are your own human review decisions')
        kind, key = payload.get('kind'), payload.get('key')
        record = payload.get('record')
        if not isinstance(record, dict):
            raise ValueError('Missing review decision')
        notes = record.get('notes', '').strip()
        if len(notes) > 4000:
            raise ValueError('Please keep the review note under 4,000 characters')
        if kind in {'task_reviews', 'knowledge_reviews'}:
            allowed = self.bytask if kind == 'task_reviews' else self.prior_keys
            if key not in allowed or record.get('decision') not in {'accept','request_change','reject','cannot_assess'}:
                raise ValueError('Unknown task/prior or review decision')
            if record['decision'] != 'accept' and not notes:
                raise ValueError('Explain the change, rejection or missing evidence')
            cleaned = {'decision': record['decision'], 'notes': notes}
        elif kind == 'judgments':
            fid, pid = record.get('family_id'), record.get('part_id')
            if fid not in self.pools or pid not in self.pools[fid] or key != fid+'|'+pid:
                raise ValueError('Part does not belong to this task pool')
            decision, reason = record.get('decision'), record.get('reason')
            if decision not in DECISIONS or reason not in REASONS:
                raise ValueError('Choose a shortlist decision and an evidence reason')
            features = record.get('features', {})
            expected = {f['id'] for f in self.bytask[fid]['features']}
            if set(features) - expected or any(v not in {'present','absent','unclear'} for v in features.values()):
                raise ValueError('Invalid feature observation')
            if decision != 'cannot_assess' and not features and not notes:
                raise ValueError('Record visible feature evidence or a specific observation')
            if decision == 'shortlist' and (set(features) != expected or any(v != 'present' for v in features.values())):
                raise ValueError('For Shortlist, confirm all required visible features; use Conditional if one is unclear')
            if reason in {'other','wrong_form'} and not notes:
                raise ValueError('Add a short description of the visible evidence')
            item = self.evidence(self.manifest[pid]['inspection_id'])
            point = record.get('point')
            if point is not None and (not isinstance(point, list) or len(point)!=3 or
                                      any(type(v) not in {int,float} or not np.isfinite(v) or abs(v)>2 for v in point)):
                raise ValueError('Invalid optional surface marker')
            labels = {f['id']: f['label'] for f in self.bytask[fid]['features']}
            cleaned = {'family_id':fid, 'part_id':pid, 'inspection_id':item['inspection_id'],
                       'source_sha256':item['source_sha256'], 'decision':decision, 'grade':DECISIONS[decision],
                       'features':features, 'reason':reason, 'notes':notes, 'point':point,
                       'observation': '; '.join(labels[k]+': '+v for k,v in features.items()) or notes or REASONS[reason],
                       'rationale': REASONS[reason]+('. '+notes if notes else ''),
                       'evidence':f"source STEP and evidence/{item['inspection_id']}.png",
                       'source':'validation_fixture' if self.validation_fixture else 'independent_expert',
                       'reviewer':reviewer, 'human_review_attested':not self.validation_fixture}
        else:
            raise ValueError('Unknown review section')
        cleaned['reviewed_at'] = now()
        with self.lock:
            state = self.load(reviewer)
            previous = state[kind].get(key)
            state['history'].append({'kind':kind,'key':key,'previous':previous,'saved_at':cleaned['reviewed_at']})
            state[kind][key] = cleaned
            state['human_review_attested'] = not self.validation_fixture
            target = self.path(reviewer)
            temporary = target.with_suffix('.tmp')
            write_json(temporary, state)
            temporary.replace(target)
        return {'saved': True, 'record': cleaned}

    def export(self, reviewer, output):
        state = self.load(reviewer)
        output = Path(output)
        write_json(output/'review.json', state)
        rows = list(state['judgments'].values())
        write_csv(output/'judgments.csv', [{k:v for k,v in r.items() if k not in {'features','point'}} for r in rows],
                  ['family_id','part_id','inspection_id','source_sha256','decision','grade','reason','notes',
                   'observation','rationale','evidence','source','reviewer','human_review_attested','reviewed_at'])
        write_json(output/'task_reviews.json', state['task_reviews'])
        write_json(output/'knowledge_reviews.json', state['knowledge_reviews'])
        # Only accepted generic priors are exported. Evaluation labels never enter this directory.
        functions, priors = {}, []
        for key,r in state['knowledge_reviews'].items():
            if r['decision'] != 'accept' or self.validation_fixture:
                continue
            prior = self.prior_keys[key]
            fid = prior['function_id']
            functions[fid] = {**self.functions[fid], 'status':'curated', 'reviewer':reviewer}
            priors.append({**prior, 'status':'curated', 'reviewer':reviewer})
        write_csv(output/'knowledge/functions.csv', list(functions.values()), list(next(iter(self.functions.values())).keys()))
        write_csv(output/'knowledge/class_functions.csv', priors, list(self.priors[0].keys()))
        write_csv(output/'knowledge/interfaces.csv', [], ['part_id','interface_id','role','diameter_mm','min_mm','max_mm','unit','reviewer','evidence','status'])
        return {'reviewer':reviewer,'decisions':len(rows),'cannot_assess':sum(r['grade'] is None for r in rows),
                'accepted_priors':len(priors),'output':str(output.resolve())}


def agreement(review, reviewers, output):
    a,b = [review.load(r)['judgments'] for r in reviewers]
    common = sorted(set(a)&set(b))
    valid = [k for k in common if a[k]['grade'] is not None and b[k]['grade'] is not None]
    counts = np.zeros((3,3),dtype=int)
    for key in valid:
        counts[a[key]['grade'],b[key]['grade']] += 1
    n=int(counts.sum())
    weights=np.square(np.arange(3)[:,None]-np.arange(3)[None,:])/4
    expected=np.outer(counts.sum(1),counts.sum(0))/n if n else np.zeros((3,3))
    denom=float((weights*expected).sum())
    result={'reviewers':reviewers,'common_records':len(common),'both_assessable':n,
            'cannot_assess_in_either':len(common)-n,'agreement':float(np.trace(counts)/n) if n else None,
            'quadratic_weighted_kappa':1-float((weights*counts).sum())/denom if denom else None,
            'confusion_matrix_grade012':counts.tolist(),
            'interpretation':'Descriptive overlap only; unresolved cases excluded from kappa, not coded as rejection.'}
    write_json(Path(output)/'agreement.json',result)
    rows=[{'key':k,'family_id':a[k]['family_id'],'inspection_id':a[k]['inspection_id'],
           'reviewer_a_decision':a[k]['decision'],'reviewer_b_decision':b[k]['decision'],
           'reviewer_a_reason':a[k]['rationale'],'reviewer_b_reason':b[k]['rationale']}
          for k in common if a[k]['decision']!=b[k]['decision'] or a[k]['grade'] is None]
    write_csv(Path(output)/'disagreements.csv',rows,['key','family_id','inspection_id','reviewer_a_decision','reviewer_b_decision','reviewer_a_reason','reviewer_b_reason'])
    return result


def serve(review, port=8765, open_browser=False):
    assets=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def response(self, body, status=200, mime='application/json', filename=None):
            if not isinstance(body, bytes):
                body=json.dumps(body,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            if filename:
                self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'")
            self.end_headers(); self.wfile.write(body)

        def guarded(self):
            expected={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in expected:
                raise ValueError('Local host access only')
            origin=self.headers.get('Origin')
            if origin and origin not in {'http://'+h for h in expected}:
                raise ValueError('Cross-origin review writes are forbidden')

        def do_GET(self):
            try:
                self.guarded()
                parsed=urlparse(self.path); query=parse_qs(parsed.query); path=parsed.path
                if path in {'/','/app.js','/style.css'}:
                    name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
                    return self.response((assets/name).read_bytes(),mime={'index.html':'text/html; charset=utf-8','app.js':'text/javascript','style.css':'text/css'}[name])
                if path=='/api/bootstrap':
                    return self.response(review.bootstrap(query.get('reviewer',[None])[0]))
                if path=='/api/export':
                    identifier=reviewer_id(query['reviewer'][0])
                    state=review.load(identifier)
                    return self.response(state,filename=identifier+'_review.json')
                if path.startswith('/api/evidence/'):
                    name=path.rsplit('/',1)[1]; code,suffix=name.rsplit('.',1)
                    item=review.evidence(code)
                    if suffix=='png':
                        return self.response((review.run/'evidence'/name).read_bytes(),mime='image/png')
                    if suffix=='step':
                        return self.response(review.check_source(item['part_id']).read_bytes(),mime='application/octet-stream')
                    if suffix=='bin':
                        triangles=np.load(review.run/'evidence'/f'{code}.npz')['triangles']
                        center=(triangles.max(axis=(0,1))+triangles.min(axis=(0,1)))/2
                        radius=max(float(np.ptp(triangles.reshape(-1,3),axis=0).max()/2),1e-9)
                        data=np.asarray((triangles-center)/radius,dtype='<f4')
                        return self.response(data.tobytes(),mime='application/octet-stream')
                self.response({'error':'Not found'},404)
            except (ValueError,KeyError,OSError) as exc:
                self.response({'error':str(exc)},400)

        def do_POST(self):
            try:
                self.guarded()
                if self.path!='/api/save':
                    return self.response({'error':'Not found'},404)
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=32_000:
                    raise ValueError('Invalid review request size')
                payload=json.loads(self.rfile.read(size))
                self.response(review.save(payload))
            except (ValueError,KeyError,OSError,TypeError) as exc:
                self.response({'error':str(exc)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    address=f'http://127.0.0.1:{server.server_port}'
    print(f'Visual annotation: {address}\nSaved human work: {review.workspace}\nPress Ctrl+C to stop.',flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(args):
    review=Review(args.run,args.workspace,args.tasks,args.validation_fixture)
    if args.action=='serve':
        serve(review,args.port,args.open)
    elif args.action=='export':
        print(review.export(args.reviewer,args.output or review.workspace/'exports'/reviewer_id(args.reviewer)))
    else:
        if not args.reviewers or len(args.reviewers)!=2 or args.reviewers[0]==args.reviewers[1]:
            raise ValueError('Provide two distinct reviewer IDs for comparison')
        print(agreement(review,args.reviewers,args.output or review.workspace/'comparisons'/('_'.join(args.reviewers))))
