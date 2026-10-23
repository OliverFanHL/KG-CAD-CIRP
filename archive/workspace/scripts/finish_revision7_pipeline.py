"""Complete sealed saved-response analysis after the existing runner finishes.

Makes no network request and never retries jobs. Stops if acquisition is partial.
The resulting paper still requires scientific and visual review before packaging.
"""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time

REPO=Path(__file__).resolve().parents[1]
STUDY=REPO/'outputs/llm_robustness_v7'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def status(stage,**extra):
    record=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    (STUDY/'pipeline_state.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record),flush=True)

def run(module,*args):
    status('RUNNING_SAVED_RESPONSE_STAGE',module=module)
    target=STUDY/(module.rsplit('.',1)[-1]+'.log')
    with target.open('w',encoding='utf8') as log:
        result=subprocess.run([sys.executable,'-m',module,*args],cwd=REPO,
                              stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status('STOPPED_REVIEW_REQUIRED',module=module,returncode=result.returncode,log=str(target))
        raise SystemExit(result.returncode)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-runner',action='store_true')
    parser.add_argument('--integrate',action='store_true');args=parser.parse_args()
    lock=STUDY/'.api_runner.lock'
    if lock.exists():
        assert args.wait_for_runner,'Active acquisition: wait explicitly or return later'
        status('WAITING_FOR_EXISTING_API_RUNNER')
        while lock.exists():time.sleep(30)
    attempt_rows=(STUDY/'api_attempts.jsonl').read_bytes().splitlines()
    response_rows=(STUDY/'api_responses.jsonl').read_bytes().splitlines()
    assert len(attempt_rows)==len(response_rows)==14588,'Acquisition is incomplete; no missing results will be imputed'
    assert len({json.loads(x)['job_id'] for x in response_rows})==14588
    assert len({json.loads(x)['job_id'] for x in attempt_rows})==14588
    for filename,module,args2 in [
        ('evidence_completion.json','scripts.complete_llm_robustness_v7',[]),
        ('results.json','scripts.evaluate_llm_robustness_v7',['--evaluate']),
        ('independent_replay_audit.json','scripts.replay_llm_robustness_v7',[]),
        ('post_execution_audit.json','scripts.audit_llm_robustness_v7',[])]:
        if not (STUDY/filename).exists():run(module,*args2)
    assert read(STUDY/'independent_replay_audit.json')['status']=='PASS'
    assert read(STUDY/'post_execution_audit.json')['status']=='PASS'
    run('scripts.report_llm_robustness_v7')
    if args.integrate:
        run('scripts.integrate_revision7_assets')
        status('INTEGRATED_AWAITING_SCIENTIFIC_AND_VISUAL_REVIEW')
    else:status('AUDITED_RESULTS_READY_FOR_INTEGRATION')

if __name__=='__main__':main()
