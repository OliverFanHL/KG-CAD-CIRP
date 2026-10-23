"""Install audited snapshot-only visuals, then revise prose from actual outcomes."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
STUDY=REPO/'outputs/llm_robustness_v7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    audit=read(STUDY/'post_execution_audit.json')
    assert str(audit.get('status','')).upper().startswith('PASS')
    assert read(STUDY/'independent_replay_audit.json')['status']=='PASS'
    data=PAPER/'data/robustness_v7';data.mkdir(parents=True,exist_ok=True)
    names=['summary.csv','per_query.csv','family_metrics.csv','run_metrics.csv','paired_tests.csv',
        'family_differences.csv','top5_rankings.json','graph_support.json','graph_evidence.jsonl',
        'results.json','evidence_completion.json','protocol.md','role_only_contract.json',
        'preparation.json','evaluation_freeze.json','post_execution_audit.json',
        'post_execution_audit.md','independent_replay_audit.json','protocol_independent_audit.md',
        'study_report.md','accounting_summary.json']
    snapshots={}
    for name in names:
        path=STUDY/name;assert path.is_file(),name
        shutil.copyfile(path,data/name);snapshots[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    for path,name in [(REPO/'outputs/robustness_local_v7/verification.json','local_verification.json'),
        (REPO/'outputs/robustness_local_v7/compact_ablation_three_pass.csv','local_ablation_three_pass.csv'),
        (REPO/'outputs/robustness_remaining_local_v7/verification.json','remaining_local_verification.json'),
        (REPO/'outputs/robustness_remaining_local_v7/three_pass_summary.csv','remaining_local_three_pass.csv')]:
        shutil.copyfile(path,data/name);snapshots[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(data/'snapshot_manifest.json',dict(source='outputs/llm_robustness_v7 and local repeatability audits',
        protocol_sha256=sha(STUDY/'protocol_freeze.json'),files=snapshots,
        raw_API_records='Complete records and original images are in the separate repeated-observation experiment archive.'))
    generator=REPO/'outputs/robustness_local_v7/generate_revision7_assets.py'
    script=PAPER/'scripts/generate_revision7_assets.py';shutil.copyfile(generator,script)
    out=PAPER/'publication_v7'
    subprocess.run([sys.executable,str(script),'--data',str(data),'--output',str(out)],check=True)
    outputs={}
    for name in ('repeated_model_performance','repeated_role_contrasts'):
        for ext in ('pdf','png','svg'):
            file=name+'.'+ext;shutil.copyfile(out/file,PAPER/'figures'/file)
            outputs['figures/'+file]=sha(PAPER/'figures'/file)
    for name in ('component_ablation_v7.tex','repeated_paired_tests_v7.tex'):
        shutil.copyfile(out/name,PAPER/'tables'/name);outputs['tables/'+name]=sha(PAPER/'tables'/name)
    for name in ('captions_v7.json','role_contrast_plot_data.json','publication_manifest_v7.json'):
        shutil.copyfile(out/name,PAPER/'provenance'/name)
    record=dict(revision=7,generator='scripts/generate_revision7_assets.py',generator_sha256=sha(script),
        snapshot_manifest_sha256=sha(data/'snapshot_manifest.json'),outputs=outputs,
        primary_data='data/robustness_v7/summary.csv',primary_data_sha256=sha(data/'summary.csv'),
        comparison_data_sha256=sha(data/'paired_tests.csv'),source_audit_sha256=sha(data/'post_execution_audit.json'))
    write(PAPER/'provenance/repeated_observation_publication_v7.json',record)
    subprocess.run([sys.executable,str(REPO/'scripts/revise_manuscript_v7.py'),'--integrate'],check=True)
    # README and title options are authored from completed, retained results.
    original=(REPO/'manuscript/revision_archive/revision_6_before_repeated_observation/README.md').read_text(encoding='utf8')
    original=original.replace('Knowledge-guided visual evidence for CAD retrieval with large language models',
        'Knowledge-guided relational evidence for functional CAD retrieval')
    original=original.replace('Revision 6,','Revision 7,')
    original=original.replace('Table 2 gives paired differences and\nuncertainty.',
        'Table 2 gives the compact component ablation.')
    start=original.index('## Verified matched results');end=original.index('## Presentation and scope')
    brief='''## Repeated primary comparison

`data/robustness_v7/` supplies the revised primary data. It contains the actual
three-collection means and sample SDs for KG-directed Luna, KG-directed Sol and
the focused holistic role-only Luna control. Historical Luna/Sol collections are
retained as collection1; two additional collections per model and three new
role-only collections add14,588calls. Complete records are supplied in the
separate experiment archive. All outcomes, including null or negative contrasts,
are retained. `data/llm_geometry/` is the historical first-Luna snapshot.

The encoder, classifier, gallery, queries, pools, size rules and source judgments
are fixed. Local seeds17/29/43 control execution order. Responses has no sampling
seed in this protocol. Thus SD measures repeated acquisition under one encoder,
not variability from retraining. All historical local methods also have three
actual deterministic executions; their complete arrays repeat exactly.

Figure2 includes both GPT models and role-only controls. Table2 is now a compact
stagewise component ablation; detailed paired tests are in SupplementaryS9.
New figures retain external captions, plain panel letters and source-bound
SciencePlots/Matplotlib output. Role-onlyJ has the same nine-value domain as
coverageH. Full+J controls the observation design with other scoring terms held
constant; J+C omits those terms from scoring inside the inherited eligible pool.

The lengthy preparation discussion is replaced by the concise declaration
required by the current publisher policy. Human authors must verify this draft
statement and complete names, affiliations, funding, interests and CRediT before
submission. AI research methods and the AI reference are documented separately.

'''
    # Normal prose spacing is intentional; names/equations remain exact.
    replacements={'collection1':'collection 1','add14,588calls':'add 14,588 calls',
        'seeds17/29/43':'seeds 17/29/43','Figure2':'Figure 2','Table2':'Table 2',
        'SupplementaryS9':'Supplementary S9','Role-onlyJ':'Role-only J','coverageH':'coverage H'}
    for a,b in replacements.items():brief=brief.replace(a,b)
    original=original[:start]+brief+original[end:]
    old='## Separate Sol model candidate'
    if old in original:original=original[:original.index(old)]
    original+='''\n## Reproduce current quantitative artwork

Run `python scripts/generate_revision7_assets.py` with the packages in
`requirements-figures.txt`. It reads only `data/robustness_v7` and writes a new
`publication_v7` staging folder. No model calls or original CAD dataset are
required to regenerate figures and tables. The retained source/audit hashes bind
the displayed means, SDs and statistical comparisons to the completed study.

Six title alternatives and their rationale, policy sources and repeatability
scope are in the accompanying revision notes. The selected title foregrounds
relational evidence without claiming superiority of graph storage.
'''
    (PAPER/'README.md').write_text(original,encoding='utf8')
    for name in ('revision7_policy_and_scientific_review.md','revision7_local_stochasticity_audit.md'):
        shutil.copyfile(REPO/'docs'/name,PAPER/'provenance'/name)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
