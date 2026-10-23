"""Bind blind source-feature reviews to immutable prior records; no score tuning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import re

import numpy as np

from cadkg.io import read_csv, read_json, write_csv, write_json, sha256
from scripts.visual_annotation_review import ROLE

ROOT = Path('outputs/feature_review_v5')
ANN = Path('annotations/intent_feature_review_v5')
PARENT = Path('annotations/intent_candidates1816_v4/judgments.json')
FEATURES = dict(S='body seat mounting', C='connection two_ends retention',
    B='wheel belt_contact profile', T='head shank external_helix',
    N='body turning internal_helix', M='rigid two_faces fixing',
    F='passage bolt_pattern register shaft_interface', G='cylindrical teeth profile',
    D='elongated journal drive', P='coil travel axial_ends', E='coil hooks tensile_ends')
FEATURES = {k: v.split() for k, v in FEATURES.items()}
FACT = re.compile(r'(?P<role>[SCBTNMFGDPE])\.(?P<feature>[a-z_]+)(?P<state>[+?!-])@(?P<view>oblique|opposite|profile|axial|cut_X|cut_Y)\s*=\s*(?P<detail>[^;]+)')


def notes(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        code, marks, facts = line.split('|', 2)
        code = code.strip()
        if code in result:
            raise ValueError(f'Duplicate ID {code} in {path}:{number}')
        grades = {}
        for mark in filter(None, (x.strip() for x in marks.split(','))):
            if not re.fullmatch(r'[SCBTNMFGDPE][12?]', mark) or mark[0] in grades:
                raise ValueError(f'Bad or duplicate role {mark}, {code}')
            grades[mark[0]] = None if mark[1] == '?' else int(mark[1])
        parsed = [m.groupdict() for m in FACT.finditer(facts)]
        if not facts.strip():
            raise ValueError(f'Missing source observation {code}')
        result[code] = dict(grades=grades, observation=facts.strip(), features=parsed,
                            source_notes=str(path), line=number)
    return result


def feature_issues(role,grade,facts):
    issues=[]
    names={f['feature'] for f in facts}
    for missing in set(FEATURES[role])-names:issues.append(('missing_feature',missing))
    for f in facts:
        if (f['role']!=role or f['feature'] not in FEATURES[role] or f['state'] not in '+-?!'
            or f['view'] not in {'oblique','opposite','profile','axial','cut_X','cut_Y'} or not f['detail'].strip()):
            issues.append(('invalid_feature',str(f)))
    if not any(f['view'] in {'oblique','opposite','profile','axial'} for f in facts):
        issues.append(('no_full_surface_fact',str(facts)))
    if grade==2:
        for f in facts:
            expected='-' if role=='F' and f['feature']=='shaft_interface' else '+'
            if f['state']!=expected:issues.append(('strong_grade_nonpositive_fact',str(f)))
    elif grade==1 and not any(f['state'] in '-?!' for f in facts):
        issues.append(('conditional_without_limitation',str(facts)))
    elif grade is None and not any(f['state']=='!' for f in facts):
        issues.append(('null_without_unreadable_feature',str(facts)))
    if grade==1 and not any(f['state']=='+' for f in facts):
        issues.append(('conditional_without_visible_positive_form',str(facts)))
    return issues


def freeze_plan_and_second_sample():
    """Selection uses historical outcomes only; review packet has no outcomes."""
    freeze_path = ROOT/'analysis_freeze.json'
    if freeze_path.exists():
        raise ValueError('Analysis/sample already frozen; never replace it')
    manifest = read_csv(ROOT/'render_manifest.csv')
    bypart = {r['part_id']:r for r in manifest}
    old = read_json(PARENT)['judgments']
    positive = {r['part_id'] for r in old if r['grade'] == 2}
    conditional = {r['part_id'] for r in old if r['grade'] == 1}
    top = defaultdict(set)
    for r in read_csv('outputs/candidate_scale_v4/evaluation/rankings_top20.csv'):
        if r['cohort']=='closed' and int(r['rank'])<=5:
            top[r['geometry_kind'],r['query_id'],r['method']].add(r['part_id'])
    disagreement=set()
    for kind,qid,method in top:
        if method=='kg_full':
            for other in ['geometry_class_filter','geometry_intent_filter','intent_filter','rrf_filter']:
                disagreement |= top[kind,qid,method] ^ top.get((kind,qid,other),set())
    # Coordinator accidentally saw the first three shard1 notes while inspecting
    # format; those IDs are excluded from this second blind review sample.
    exposed = {'Z0001','Z0004','Z0007'}
    eligible = {p for p,r in bypart.items() if r['review_id'] not in exposed}
    rng=np.random.default_rng(20261028); selected=[]; strata=[]
    for name,pool,count in [('method_disagreement',disagreement,40),
                           ('historical_conditional',conditional,20),
                           ('historical_positive',positive,12),
                           ('other_control',set(bypart),8)]:
        options=sorted((pool & eligible)-set(selected))
        if len(options)<count:raise ValueError('Insufficient audit stratum')
        picks=list(rng.choice(options,count,replace=False));selected.extend(picks)
        strata.extend(dict(review_id=bypart[p]['review_id'],selection_stratum=name) for p in picks)
    rng.shuffle(selected)
    packet=[dict(review_id=bypart[p]['review_id'],image=bypart[p]['image'],
                 binding_json=str(Path(bypart[p]['image']).with_suffix('.json'))) for p in selected]
    write_csv(ROOT/'second_blind_manifest.csv',packet)
    write_csv(ROOT/'second_sample_selection_private.csv',strata)
    inputs=[Path('docs/FEATURE_REVIEW_EVALUATION.md'), ROOT/'FROZEN_RUBRIC.md',
            ROOT/'rubric_freeze.json', ROOT/'initial_freeze.json',
            ROOT/'second_blind_manifest.csv',Path('configs/candidate_scale_v4.json'),PARENT]
    write_json(freeze_path,dict(stage='Before new-label comparisons and evaluation',
        seed=20261028,input_sha256={str(p):sha256(p) for p in inputs},
        second_sample_size=80,selection_strata=Counter(x['selection_stratum'] for x in strata),
        second_sample_excluded_previously_exposed_ids=sorted(exposed),
        interpretation='Targeted AI second review; not a probability annotation-error estimate or human agreement.',
        proposed_method='kg_full',matrix_unchanged=True))
    from scripts.prepare_feature_review import contact
    contact(packet,ROOT/'second_pages')


def combine():
    combined={}; hashes={};coverage={}
    for shard in (1,2,3):
        path=ROOT/f'reviewer{shard}_notes.txt'
        chunk=notes(path)
        wanted={r['review_id'] for r in read_csv(ROOT/f'blind_manifest_{shard}.csv')}
        report=read_json(ROOT/f'reviewer{shard}_coverage.json')
        if set(chunk)!=wanted:
            raise ValueError(f'Incomplete shard{shard}: {len(chunk)}/{len(wanted)}')
        if set(combined)&set(chunk):raise ValueError('Overlapping shards')
        if report.get('status')!='complete' and report.get('complete') is not True:
            raise ValueError(f'Reviewer{shard} does not attest completed visual review')
        inspected=report.get('visually_reviewed_ids',report.get('inspected_ids',report.get('reviewed_ids')))
        if inspected is None or set(inspected)!=wanted or len(inspected)!=len(wanted):
            raise ValueError(f'Visual coverage IDs mismatch shard{shard}')
        enlarged=report.get('individually_enlarged_ids',report.get('enlarged_ids',[]))
        if len(enlarged)!=len(set(enlarged)) or not set(enlarged)<=wanted:raise ValueError('Invalid enlargement coverage')
        if report.get('missing_ids'):raise ValueError('Reviewer reports missing sources')
        if report.get('notes_sha256')!=sha256(path):raise ValueError('Coverage note hash mismatch')
        if report.get('rubric_sha256',report.get('frozen_rubric_sha256'))!=sha256(ROOT/'FROZEN_RUBRIC.md'):
            raise ValueError('Coverage rubric hash mismatch')
        manifest_path=ROOT/f'blind_manifest_{shard}.csv'
        if report.get('manifest_sha256',sha256(manifest_path))!=sha256(manifest_path):raise ValueError('Shard manifest changed')
        combined.update(chunk);hashes[str(path)]=sha256(path)
        hashes[str(ROOT/f'reviewer{shard}_coverage.json')]=sha256(ROOT/f'reviewer{shard}_coverage.json')
        coverage[str(shard)]=report
    return combined,hashes,coverage


def diagnostics():
    if (ROOT/'first_pass_freeze.json').exists():
        raise ValueError('First pass already frozen; preserve its original seal')
    allnotes,hashes,coverage=combine()
    findings=[];details=defaultdict(list)
    for code,note in allnotes.items():
        for f in note['features']:details[f['detail'].strip().lower()].append(code)
        for role,grade in note['grades'].items():
            facts=[f for f in note['features'] if f['role']==role]
            findings.extend(dict(review_id=code,role=role,issue=issue,detail=detail)
                            for issue,detail in feature_issues(role,grade,facts))
    for observation,codes in _group_observations(allnotes).items():
        if len(codes)>1:
            for code in codes:
                role=next(iter(allnotes[code]['grades']),'S')
                findings.append(dict(review_id=code,role=role,issue='duplicate_observation_source_check',
                    detail='Identical text must be verified against this individual source, not assumed copied or correct: '+observation))
    write_csv(ROOT/'feature_diagnostics.csv',findings)
    write_json(ROOT/'first_pass_freeze.json',dict(input_sha256=hashes,coverage=coverage,
        cases=len(allnotes),role_pairs=len(allnotes)*11,features=sum(len(n['features']) for n in allnotes.values()),
        unresolved_pairs=sum(g is None for n in allnotes.values() for g in n['grades'].values()),
        exact_observation_duplicates=[dict(observation=k,ids=v) for k,v in _group_observations(allnotes).items() if len(v)>1],
        repeated_feature_phrases=[dict(detail=k,ids=sorted(set(v))) for k,v in details.items() if len(set(v))>=20],
        findings=len(findings),human_expert_reviewed=False))
    print(dict(cads=len(allnotes),findings=len(findings)))


def _group_observations(allnotes):
    result=defaultdict(list)
    for code,note in allnotes.items():result[note['observation']].append(code)
    return result


def compare():
    seal=read_json(ROOT/'first_pass_freeze.json')
    for path,digest in seal['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('First-pass notes changed after freezing')
    first,_,_=combine();manifest=read_csv(ROOT/'render_manifest.csv')
    old={(r['part_id'],r['family_id']):r for r in read_json(PARENT)['judgments']}
    proposals=[]
    for item in manifest:
        code=item['review_id'];note=first[code]
        for role,fid in ROLE.items():
            previous=old[item['part_id'],fid];new=note['grades'].get(role,0)
            if previous['grade']!=new:
                proposals.append(dict(review_id=code,part_id=item['part_id'],family_id=fid,role=role,
                    previous_grade=previous['grade'],blind_proposed_grade=new,
                    previous_observation=previous['observation'],blind_observation=note['observation'],
                    image=item['image']))
    write_csv(ROOT/'grade_proposals.csv',proposals)
    print(dict(proposals=len(proposals),cads=len({r['review_id'] for r in proposals})))


def second_comparison():
    """Open the already sealed blind judgments; never change a first-pass file."""
    first,_,_=combine();second=notes(ROOT/'second_notes.txt')
    seal=read_json(ROOT/'second_pass_freeze.json')
    if sha256(ROOT/'second_notes.txt')!=seal['notes_sha256']:
        raise ValueError('Second-pass notes changed after sealing')
    wanted={r['review_id'] for r in read_csv(ROOT/'second_blind_manifest.csv')}
    if set(second)!=wanted or len(second)!=80:raise ValueError('Second-review sample mismatch')
    if sha256(ROOT/'second_blind_manifest.csv')!=seal['manifest_sha256'] or sha256(ROOT/'FROZEN_RUBRIC.md')!=seal['rubric_sha256']:
        raise ValueError('Second-review source packet changed')
    rows=[]
    for code,note in second.items():
        for role in ROLE:
            a=first[code]['grades'].get(role,0);b=note['grades'].get(role,0)
            rows.append(dict(review_id=code,role=role,first_grade=a,second_grade=b,
                equal=a==b,nontrivial=a!=0 or b!=0,first_observation=first[code]['observation'],
                second_observation=note['observation']))
    numeric=[r for r in rows if r['first_grade'] is not None and r['second_grade'] is not None]
    matrix=np.zeros((3,3),int)
    for r in numeric:matrix[r['first_grade'],r['second_grade']]+=1
    weights=(np.arange(3)[:,None]-np.arange(3)[None,:])**2/4
    expected=np.outer(matrix.sum(1),matrix.sum(0))/len(numeric)
    denominator=float((weights*expected).sum())
    nontrivial=[r for r in rows if r['nontrivial']]
    result=dict(cads=len(second),role_pairs=len(rows),exact_grade_agreement=sum(r['equal'] for r in rows)/len(rows),
        nontrivial_pairs=len(nontrivial),nontrivial_pair_agreement=sum(r['equal'] for r in nontrivial)/len(nontrivial),
        disagreements=sum(not r['equal'] for r in rows),
        disagreement_cads=len({r['review_id'] for r in rows if not r['equal']}),
        confusion_matrix_0_1_2=matrix.tolist(),numeric_pairs=len(numeric),
        null_pairs_excluded_from_kappa=len(rows)-len(numeric),
        quadratic_weighted_kappa=1-float((weights*matrix).sum())/denominator if denominator else None,
        interpretation='Targeted difficult-case AI consistency; class-imbalanced overall agreement is not human reliability or accuracy.',
        first_pass_notes_frozen=True,second_pass_notes_frozen=True,independent_human_validation=False)
    write_csv(ROOT/'second_review_pairs.csv',rows);write_json(ROOT/'second_review_comparison.json',result)
    manifest={r['review_id']:r for r in read_csv(ROOT/'render_manifest.csv')}
    ids={r['review_id'] for r in read_csv(ROOT/'grade_proposals.csv')}
    ids|={r['review_id'] for r in rows if not r['equal']}
    ids|={r['review_id'] for r in read_csv(ROOT/'feature_diagnostics.csv')}
    packet=[dict(review_id=code,image=manifest[code]['image']) for code in sorted(ids)]
    write_csv(ROOT/'adjudication_manifest.csv',packet)
    from scripts.prepare_feature_review import contact
    contact(packet,ROOT/'adjudication_pages')
    print(result)


def adjudicate():
    """Expand explicitly source-reviewed per-CAD final decisions, never scores."""
    if (ROOT/'adjudication_freeze.json').exists():raise ValueError('Adjudication is already sealed; create a new version for later changes')
    first,_,_=combine();second=notes(ROOT/'second_notes.txt');final=notes(ROOT/'adjudication_notes.txt')
    for path,digest in read_json(ROOT/'first_pass_freeze.json')['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Sealed first pass changed')
    if sha256(ROOT/'second_notes.txt')!=read_json(ROOT/'second_pass_freeze.json')['notes_sha256']:
        raise ValueError('Sealed second pass changed')
    packet=read_csv(ROOT/'adjudication_manifest.csv');wanted={r['review_id'] for r in packet}
    if set(final)!=wanted:raise ValueError(f'Incomplete adjudication {len(final)}/{len(wanted)}')
    origins={};input_hashes={};assignment=read_json(ROOT/'adjudication_assignment_freeze.json')
    for who in ('adjudicator1','adjudicator2','coordinator'):
        manifest=ROOT/(who+'_manifest.csv');path=ROOT/(who+'_notes.txt');coverage_path=ROOT/(who+'_coverage.json')
        report=read_json(coverage_path);chunk=notes(path);expected={r['review_id'] for r in read_csv(manifest)}
        if (not (report.get('complete') or report.get('status')=='complete') or set(report['inspected_ids'])!=expected or set(chunk)!=expected
            or report.get('inspected_count',report.get('actual_inspected_count'))!=len(expected) or report['notes_sha256']!=sha256(path)
            or report['rubric_sha256']!=sha256(ROOT/'FROZEN_RUBRIC.md')
            or sha256(manifest)!=assignment['manifest_sha256'][who]):raise ValueError('Unverified fresh adjudication packet '+who)
        for code,note in chunk.items():
            if code in origins or note['grades']!=final[code]['grades'] or note['observation']!=final[code]['observation']:
                raise ValueError('Combined fresh adjudication differs from sealed source notes')
            origins[code]=who
        input_hashes.update({str(p):sha256(p) for p in [manifest,path,coverage_path]})
    if set(origins)!=wanted:raise ValueError('Fresh source coverage mismatch')
    required={(r['review_id'],r['role']) for r in read_csv(ROOT/'grade_proposals.csv')}
    required|={(r['review_id'],r['role']) for r in read_csv(ROOT/'second_review_pairs.csv') if r['equal']=='False'}
    required|={(r['review_id'],r['role']) for r in read_csv(ROOT/'feature_diagnostics.csv')}
    for code,note in final.items():
        for role in ROLE:
            if note['grades'].get(role,0)!=first[code]['grades'].get(role,0):required.add((code,role))
    decisions=[]
    for code,role in sorted(required):
        note=final[code];g=note['grades'].get(role,0)
        facts=[f for f in note['features'] if f['role']==role]
        evidence_origin=origins[code]+' fresh source review'
        if set(FEATURES[role])-{f['feature'] for f in facts}:
            for origin,source in [('sealed first-pass source-feature evidence',first),('sealed second-pass source-feature evidence',second)]:
                if code in source and source[code]['grades'].get(role,0)==g:
                    alternative=[f for f in source[code]['features'] if f['role']==role]
                    if not set(FEATURES[role])-{f['feature'] for f in alternative}:
                        facts=alternative;evidence_origin=origin;break
        if g in (1,2,None) and set(FEATURES[role])-{f['feature'] for f in facts}:
            raise ValueError(f'Adjudicated role needs source features: {code}/{role}/{g}')
        if g in (1,2,None) and feature_issues(role,g,facts):
            raise ValueError(f'Adjudication feature contradiction {code}/{role}: {feature_issues(role,g,facts)}')
        decisions.append(dict(review_id=code,role=role,family_id=ROLE[role],final_grade=g,
            reason=note['observation'],source_reviewed=True,adjudicator=origins[code]+'; AI, not human',
            observation=note['observation']+' | Retained feature evidence: '+evidence_origin,
            features=facts,feature_evidence_origin=evidence_origin,
            decision_provenance='Visual source adjudication after sealed blind passes, before evaluation; no score criterion'))
    write_json(ROOT/'adjudication.json',decisions)
    write_json(ROOT/'adjudication_freeze.json',dict(cads=len(final),role_dispositions=len(decisions),
        notes_sha256=sha256(ROOT/'adjudication_notes.txt'),decisions_sha256=sha256(ROOT/'adjudication.json'),
        input_sha256=input_hashes,assignment_sha256=sha256(ROOT/'adjudication_assignment_freeze.json'),
        stage='Source adjudication complete, before numeric scoring',human_expert_reviewed=False))
    print(dict(cads=len(final),role_dispositions=len(decisions)))


def publish():
    """Requires explicit source adjudication, never selects a grade from a score."""
    if (ROOT/'evaluation_freeze.json').exists():
        raise ValueError('This release is already sealed; never overwrite it')
    first,hashes,coverage=combine();seal=read_json(ROOT/'first_pass_freeze.json')
    for path,digest in seal['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'Frozen review changed {path}')
    for path,digest in read_json(ROOT/'analysis_freeze.json')['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'Analysis plan changed {path}')
    second_seal=read_json(ROOT/'second_pass_freeze.json')
    for path,key in [(ROOT/'second_notes.txt','notes_sha256'),(ROOT/'second_blind_manifest.csv','manifest_sha256'),(ROOT/'FROZEN_RUBRIC.md','rubric_sha256')]:
        if sha256(path)!=second_seal[key]:raise ValueError('Sealed second review changed before publication')
    decisions=read_json(ROOT/'adjudication.json')
    adjudication_seal=read_json(ROOT/'adjudication_freeze.json')
    if (sha256(ROOT/'adjudication.json')!=adjudication_seal['decisions_sha256'] or
        sha256(ROOT/'adjudication_notes.txt')!=adjudication_seal['notes_sha256']):
        raise ValueError('Adjudication changed after sealing')
    for path,digest in adjudication_seal['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Fresh source adjudication changed '+path)
    adjudicated={(r['review_id'],r['role']):r for r in decisions}
    if len(adjudicated)!=len(decisions):raise ValueError('Duplicate adjudication')
    proposals=read_csv(ROOT/'grade_proposals.csv')
    required={(r['review_id'],r['role']) for r in proposals}
    required|={(r['review_id'],r['role']) for r in read_csv(ROOT/'second_review_pairs.csv') if r['equal']=='False'}
    required|={(r['review_id'],r['role']) for r in read_csv(ROOT/'feature_diagnostics.csv')}
    if not required<=set(adjudicated):
        raise ValueError('Every grade disagreement, second-pass disagreement and feature diagnostic needs explicit adjudication')
    packet_ids={r['review_id'] for r in read_csv(ROOT/'adjudication_manifest.csv')}
    if set(notes(ROOT/'adjudication_notes.txt'))!=packet_ids or adjudication_seal['cads']!=len(packet_ids):
        raise ValueError('Incomplete source adjudication coverage')
    for code,role in adjudicated:
        if code not in packet_ids or role not in ROLE:raise ValueError('Extraneous adjudication ID/role')
    parent=read_json(PARENT);old={(r['part_id'],r['family_id']):r for r in parent['judgments']}
    oldcfg=read_json('configs/candidate_scale_v4.json');oldevidence={}
    for r in parent['judgments']:oldevidence.setdefault(r['part_id'],r)
    manifest=read_csv(ROOT/'render_manifest.csv');records=[];changes=[];features=[]
    for item in manifest:
        code=item['review_id'];note=deepcopy(first[code]);pid=item['part_id']
        binding=read_json(Path(item['image']).with_suffix('.json'))
        if binding['source_sha256']!=item['source_sha256'] or binding['image_sha256']!=sha256(item['image']):
            raise ValueError('Rendered source binding mismatch')
        e=deepcopy(read_json(Path(oldcfg['evidence_root'])/(oldevidence[pid]['inspection_id']+'.json')))
        e.update(inspection_id=code,source_view_binding=binding,source_view_binding_sha256=sha256(Path(item['image']).with_suffix('.json')))
        write_json(ROOT/'evidence'/f'{code}.json',e)
        for role,fid in ROLE.items():
            decision=adjudicated.get((code,role));grade=note['grades'].get(role,0)
            observation=note['observation'];factlist=[f for f in note['features'] if f['role']==role]
            if decision:
                if decision.get('final_grade') not in (0,1,2,None) or not decision.get('reason') or not decision.get('source_reviewed'):
                    raise ValueError('Unjustified adjudication')
                grade=decision['final_grade']
                if decision.get('observation'):observation=decision['observation']
                if decision.get('features') is not None:factlist=decision['features']
            if grade in (1,2,None):
                if not set(FEATURES[role])<={f['feature'] for f in factlist}:
                    raise ValueError(f'Incomplete final feature evidence {code}/{role}')
                if feature_issues(role,grade,factlist):raise ValueError(f'Invalid final feature evidence {code}/{role}: {feature_issues(role,grade,factlist)}')
            term={0:'reject',1:'conditional',2:'shortlist',None:'cannot_assess'}[grade]
            previous=old[pid,fid]
            rationale=(f'{term}: '+('; '.join('{feature}{state}@{view}={detail}'.format(**f) for f in factlist)
                if factlist else 'Observed incompatible form; all roles explicitly considered. '+observation))
            row=dict(family_id=fid,part_id=pid,inspection_id=code,source_sha256=item['source_sha256'],
                grade=grade,decision=term,observation=observation,feature_evidence=factlist,
                rationale=rationale,geometry_evidence=item['image'],source='ai_reference',human_expert_reviewed=False,
            reviewer='Separate-context blind AI reviewer; flagged cases freshly adjudicated in another AI context or by coordinator',
                review_status='visually_reviewed',review_origin=note['source_notes'],previous_grade=previous['grade'],
                previous_annotation_sha256=sha256(PARENT),rubric_sha256=sha256(ROOT/'FROZEN_RUBRIC.md'),
                adjudication=decision)
            records.append(row)
            features.extend(dict(review_id=code,part_id=pid,family_id=fid,**f) for f in factlist)
            if grade!=previous['grade']:
                changes.append(dict(review_id=code,part_id=pid,family_id=fid,previous_grade=previous['grade'],
                    revised_grade=grade,blind_proposed_grade=first[code]['grades'].get(role,0),
                    previous_observation=previous['observation'],revised_observation=observation,
                    reason=decision['reason'],source_sha256=item['source_sha256'],geometry_evidence=item['image']))
    release={k:deepcopy(v) for k,v in parent.items() if k!='judgments'}
    release.update(judgments=sorted(records,key=lambda r:(r['part_id'],r['family_id'])),
        release='intent-feature-review-v5',parent_annotation_sha256=sha256(PARENT),
        quality_audit='All1816 source-view rereview with validation-only pilot, separate-context blind AI reviewers, per-feature evidence and explicit adjudication; not human validation.',
        source='ai_reference',human_expert_reviewed=False)
    write_json(ANN/'judgments.json',release);write_csv(ANN/'changes.csv',changes);write_csv(ANN/'feature_evidence.csv',features)
    cfg=deepcopy(oldcfg);cfg.update(version='feature-review-v5',annotation_path=str(ANN/'judgments.json'),
        annotation_audit_path=str(ROOT/'quality_gate.json'),evaluation_plan='docs/FEATURE_REVIEW_EVALUATION.md',
        evidence_root=str(ROOT/'evidence'),output=str(ROOT/'evaluation'),
        notes='Same frozen candidate-scale matrix; source-feature AI rereview after validation-only operational pilot. No human validation or new configurations.')
    write_json('configs/feature_review_v5.json',cfg)
    gate=dict(cads=len(manifest),pairs=len(records),grade_counts=Counter(str(r['grade']) for r in records),
        changes=len(changes),changed_cads=len({r['part_id'] for r in changes}),
        parent_sha256=sha256(PARENT),annotation_sha256=sha256(ANN/'judgments.json'),
        null_pairs=sum(r['grade'] is None for r in records),feature_observations=len(features),
        all_prior_records_preserved_at=str(PARENT),review_notes_sha256=hashes,
        rubric_sha256=sha256(ROOT/'FROZEN_RUBRIC.md'),adjudication_sha256=sha256(ROOT/'adjudication.json'),
        independent_human_validation=False,retrospective_rubric_refinement=True,
        matrix_unchanged=True,frozen_before_scoring=True)
    write_json(ROOT/'quality_gate.json',gate)
    consumed={}
    for item in manifest:
        for path in [Path(item['image']),Path(item['image']).with_suffix('.json'),ROOT/'evidence'/(item['review_id']+'.json')]:
            consumed[str(path)]=sha256(path)
    write_json(ROOT/'consumed_evidence_freeze.json',dict(stage='Before numerical evaluation',input_sha256=consumed))
    inputs=['configs/feature_review_v5.json',ANN/'judgments.json',ROOT/'quality_gate.json',ROOT/'adjudication.json',
            Path('scripts/run_verified_evaluation.py'),Path('cadkg/experiments/verified_statistics.py'),
            Path('docs/FEATURE_REVIEW_EVALUATION.md'),ROOT/'first_pass_freeze.json',
            ROOT/'second_pass_freeze.json',ROOT/'second_notes.txt',ROOT/'source_image_hashes.csv',
            ROOT/'PLAN_CLARIFICATION.md',ROOT/'METHOD_REVIEW.md',ROOT/'exemplar_review_freeze.json',
            ROOT/'exemplar_notes.txt',ROOT/'exemplar_role_checks.csv',ROOT/'consumed_evidence_freeze.json',Path(__file__),
            Path('scripts/report_feature_review.py'),Path('scripts/validate_feature_review.py')]
    inputs.extend([ROOT/'adjudication_freeze.json',ROOT/'adjudication_notes.txt',ROOT/'adjudication_assignment_freeze.json',
        Path('scripts/partial_feature_review.py'),Path('scripts/rerun_feature_rankings.py'),Path('scripts/report_partial_feature_review.py'),
        Path('cadkg/experiments/intent.py'),Path('cadkg/experiments/reference.py'),
        Path('configs/candidate_scale_v4.json'),ROOT/'PARTIAL_QREL_EXECUTION.md',ROOT/'adjudicator1_correction_log_001.json'])
    inputs.extend(Path(p) for p in adjudication_seal['input_sha256'])
    inputs.extend(Path('outputs/candidate_scale_v4/evaluation')/p for p in
        ['complete_rankings.npz','effective_protocol.json','cohorts.json','run_manifest.json'])
    write_json(ROOT/'evaluation_freeze.json',dict(input_sha256={str(p):sha256(p) for p in inputs}))
    print(gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','diagnostics','compare','second','adjudicate','publish']);a=p.parse_args()
    {'freeze':freeze_plan_and_second_sample,'diagnostics':diagnostics,'compare':compare,'second':second_comparison,'adjudicate':adjudicate,'publish':publish}[a.action]()
