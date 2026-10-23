"""Prepare label-blind CAD evidence requests for a single matched GPT study.

No relevance annotation, grade, class label or classifier output is read here.
Existing frozen rankings determine a bounded pool; numeric CAD records bind
source images but are not disclosed to the vision model.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from scripts.neutralize_llm_cad_views import neutralize, verify

ROOT = Path('outputs/llm_geometry_matched_v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            assert not (ROOT/'protocol_freeze.json').exists()
            path.write_text(text, encoding='utf-8')
    else:
        path.write_text(text, encoding='utf-8')

# Four equally weighted evidence clauses per role. The requirements are
# operational geometric task definitions, not learned from candidate labels.
# Each clause explicitly binds its feature to the relevant body/interface.
RULES = {
 'support_body': [
  ('circular_seat', 'A distinguishable circular bearing seat or carrier recess is visibly formed within a surrounding support body; a generic unfeatured through-bore alone is ambiguous.', ['part','has_feature','shaft_seat']),
  ('mounting_interface', 'Holes, slots or feet suitable as attachment geometry are visible on that same surrounding carrier.', ['mounting_interface','belongs_to','carrier']),
  ('seat_mount_binding', 'The seat is carried by the same rigid body that bears the mounting interface, rather than by a separate shaft or loose bearing ring.', ['shaft_seat','carried_by','mountable_carrier']),
  ('stationary_carrier_form', 'The surrounding body has a nonrotating carrier/housing form beyond a plain rotating ring, wheel or generic pipe-flange disk.', ['carrier','has_form','stationary_housing'])],
 'shaft_connector': [
  ('two_receiving_ends', 'Two distinct shaft-receiving ends are visible on the body; they may be coaxial or mechanically articulated. An unfeatured pipe bore alone is ambiguous.', ['receiver_a','paired_with','receiver_b']),
  ('attachment_at_first', 'A keyway, spline, clamp split, tightening screw arrangement or equivalent positive shaft attachment is visible at the first receiving end.', ['attachment_a','belongs_to','receiver_a']),
  ('attachment_at_second', 'A keyway, spline, clamp split, tightening screw arrangement or equivalent positive shaft attachment is visible at the other receiving end.', ['attachment_b','belongs_to','receiver_b']),
  ('connected_ends', 'The two shaft attachment regions are visibly joined by a continuous rigid, flexible or articulated mechanical connection body.', ['receiver_a','mechanically_connected_to','receiver_b'])],
 'belt_wheel': [
  ('rotating_rim', 'A substantially circular wheel rim is visible about a rotation axis.', ['rim','coaxial_with','rotation_axis']),
  ('belt_track', 'A circumferential belt-contact track is visible: a continuous belt groove, V-channel, or timing-belt-like repeated tooth band.', ['rim','has_feature','belt_track']),
  ('track_form', 'The track has identifiable belt guidance/profile geometry (groove flanks, edge flanges, or belt-type transverse teeth), rather than an unqualified gear-tooth pattern.', ['belt_track','has_form','belt_profile']),
  ('track_binding', 'The identified belt-contact profile extends circumferentially around the same wheel rim, rather than appearing on an unrelated component.', ['belt_track','runs_around','rim'])],
 'threaded_clamp': [
  ('headed_shank', 'A head and a shank are connected as one fastener-shaped body; short headed screws also qualify.', ['head','attached_to','shank']),
  ('bearing_face', 'An underside bearing surface on the head is visible around the shank.', ['bearing_face','belongs_to','head']),
  ('external_thread', 'A helical external thread is visibly modeled on the shank; a smooth cylinder alone is not positive evidence.', ['external_thread','lies_on','shank']),
  ('head_actuation', 'The head has a turning/drive feature such as wrench flats, socket, slot or recess on the same body.', ['turning_interface','belongs_to','head'])],
 'nut_retention': [
  ('compact_bored_body', 'A standalone compact body with an axial bore is visible, rather than an externally threaded bolt or long sleeve.', ['bore','passes_through','compact_body']),
  ('turning_feature', 'External flats, wings, notches or another explicit turning feature are visible on that bored body.', ['turning_interface','belongs_to','bored_body']),
  ('internal_thread', 'An internal helical thread is visibly modeled inside the bore; an untextured hole alone is not positive evidence.', ['internal_thread','lies_in','bore']),
  ('thread_turning_binding', 'The internally threaded bore and the external turning interface belong to the same compact body.', ['threaded_bore','contained_by','turnable_body'])],
 'bracket_mount': [
  ('two_faces', 'Two distinct orthogonal, angled or offset attachment face regions are visibly present on one structural body.', ['attachment_a','offset_or_angled_to','attachment_b']),
  ('rigid_connection', 'The two regions are rigidly joined, rather than linked by an articulating hinge pin.', ['attachment_a','rigidly_joined_to','attachment_b']),
  ('mount_on_first', 'A mounting hole or slot lies on the first attachment region.', ['mounting_feature_a','lies_on','attachment_a']),
  ('mount_on_second', 'A mounting hole or slot lies on the other attachment region; it must not be the same hole counted twice.', ['mounting_feature_b','lies_on','attachment_b'])],
 'flanged_passage': [
  ('open_passage', 'A central through-passage is visibly open through the body; a blind pocket is insufficient.', ['central_passage','passes_through','body']),
  ('bolt_pattern', 'Several peripheral mounting holes form a pattern around that same central passage.', ['bolt_pattern','surrounds','central_passage']),
  ('flat_mating_face', 'A flat annular mating-face region surrounds the passage and is associated with the peripheral bolt pattern.', ['mating_face','surrounds','central_passage']),
  ('coaxial_register', 'A raised or recessed coaxial mating register/neck is visibly formed around the passage on the same body.', ['mating_register','coaxial_with','central_passage'])],
 'gear_mesh': [
  ('cylindrical_wheel', 'A cylindrical wheel or local cylindrical gear band integral with a shaft is visible about a rotation axis, rather than only a conical bevel or worm form.', ['wheel','has_form','cylindrical_body']),
  ('external_teeth', 'Repeated projecting teeth are visible on the outer circumference of that wheel.', ['external_teeth','lie_on','wheel_rim']),
  ('spur_helical_form', 'The tooth geometry visibly has spur or helical gear form; ambiguous fine timing-belt teeth alone are insufficient.', ['external_teeth','have_form','spur_or_helical']),
  ('mesh_track_binding', 'The teeth form an exposed circumferential meshing band on the cylindrical wheel, rather than being confined as fine belt teeth between belt flanges.', ['gear_band','runs_around','cylindrical_wheel'])],
 'shaft_drive': [
  ('elongated_axis', 'An elongated, predominantly axial solid/journal-like body is visible rather than a short hollow hub.', ['part','has_form','elongated_shaft_body']),
  ('journals', 'Cylindrical journal or stepped shaft regions lie along the same elongated body.', ['journal_regions','lie_along','shaft_body']),
  ('drive_feature', 'A keyway, spline, integral gear, crank/eccentric or noncircular drive feature is visibly formed on the body; simple shoulders or screw threads alone are insufficient.', ['drive_feature','belongs_to','shaft_body']),
  ('rotary_body', 'The visible form is a rotary drive body rather than a headed fastening screw or a thread-only linear screw.', ['part','has_form','rotary_shaft'])],
 'compression_return': [
  ('helical_coil', 'A helical wire coil body is visibly present rather than a solid threaded rod or flat clip.', ['part','has_component','helical_coil']),
  ('open_pitch', 'Axial gaps between multiple adjacent turns are visible, providing geometric coil travel.', ['adjacent_turns','separated_by','axial_gaps']),
  ('axial_seating', 'Both terminal regions terminate as axial seating ends of that coil rather than extension hooks or long torsion arms.', ['terminal_ends','form','axial_seats']),
  ('hook_free_binding', 'The same coil has no terminal extension hooks/loops or projecting torsion arms; assert yes only when both terminal regions are visible.', ['coil_ends','lack_feature','hooks_or_torsion_arms'])],
 'tension_return': [
  ('helical_coil', 'A helical wire coil body is visibly present rather than a solid threaded rod or flat clip.', ['part','has_component','helical_coil']),
  ('first_hook', 'A terminal hook or loop is visibly attached to one end of that coil; a straight torsion arm alone does not qualify.', ['hook_a','terminal_of','coil']),
  ('second_hook', 'A second terminal hook or loop is visibly attached to the other end of the same coil.', ['hook_b','terminal_of','coil']),
  ('opposed_attachments', 'The two hooks/loops are bound to opposite axial ends of the same coil to form two tensile attachment regions, rather than unrelated loops or torsion arms.', ['hook_a','opposite_end_of_same_coil_as','hook_b'])]
}

INSTRUCTIONS = '''Inspect only the supplied neutral CAD surface views and report geometric observations for the requested atomic predicates. These observations will be inserted into a knowledge graph; do not rank the part, infer its dataset class, assign suitability/relevance grades, or predict engineering performance. The six panels in row-major order are oblique, opposite, profile, axial, cut_X, cut_Y. The last two are uncapped visual clips: apparent clip openings are not proof of actual through-passages. All panels show the same CAD. For each predicate return exactly one status: yes when the claimed geometric relation is positively visible; no when visible geometry contradicts it; unknown when occluded, simplified, too small, or ambiguous. A missing modeled thread must not be invented. For absence statements use unknown unless the relevant region is visible. Material, loads, tolerances, real functionality and scale cannot be established from these renderings. Follow component bindings: features on unrelated bodies or different interfaces cannot satisfy a same-body relation. Cite supporting views and give a short literal geometric observation (at most 22 words); for unknown explain what cannot be seen. Return every requested predicate once and no additional facts.'''

def schema_for(predicates):
    entry = {'type':'object','properties': {
        'status': {'type':'string','enum':['yes','no','unknown']},
        'views': {'type':'array','items':{'type':'string','enum':['oblique','opposite','profile','axial','cut_X','cut_Y']}},
        'observation': {'type':'string'}},
        'required':['status','views','observation'],'additionalProperties':False}
    return {'type':'object','properties':{p:entry for p in predicates},
            'required':predicates,'additionalProperties':False}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'protocol_freeze.json').exists():
        raise ValueError('Study already sealed; do not regenerate preparations.')
    cfg = read('configs/query_diversity_v6.json')
    rows = list(csv.DictReader(Path(cfg['query_manifest_path']).open(encoding='utf-8')))
    selected_ids = {r['query_id'] for r in rows if r['origin']=='new_exemplar'}
    queries = read(cfg['queries_path'])
    if isinstance(queries, dict): queries = queries['queries']
    queries = [q for q in queries if q['query_id'] in selected_ids]
    assert len(queries)==264 and len({q['scenario_id'] for q in queries})==11
    ids = read(cfg['cohorts_path'])['closed']
    assert len(ids)==1816 and len(set(ids))==1816
    pool = {kind:{} for kind in ('embedding','descriptor')}
    pairs = set()
    with np.load('outputs/query_diversity_v6/evaluation/complete_rankings.npz', allow_pickle=False) as archive:
        for kind in pool:
            for q in queries:
                indexes = set()
                for method in ('kg_full','geometry_intent_filter','kg_no_class'):
                    key = f"{kind}__{q['query_id']}__closed__{method}__order"
                    indexes.update(map(int,archive[key][:10]))
                selected = [ids[i] for i in sorted(indexes)]
                pool[kind][q['query_id']] = selected
                pairs.update((q['scenario_id'],p) for p in selected)
    bindings = {}
    for path in sorted(Path('outputs/feature_review_v5/views').glob('Z*.json')):
        b = read(path)
        if b['part_id'] in ids:
            assert b['part_id'] not in bindings
            bindings[b['part_id']] = (path,b)
    jobs, source_hashes, neutral_audit = [], {}, {}
    for family, pid in sorted(pairs):
        binding_path,b = bindings[pid]
        source = binding_path.with_suffix('.png')
        assert sha(source)==b['image_sha256']
        target = ROOT/'images'/f'{pid}.png'
        if pid not in neutral_audit:
            neutral_audit[pid] = neutralize(source,target)
            neutral_audit[pid]['verification'] = verify(source,target)
        predicates = [{'predicate_id':p,'geometric_claim':desc,'required_triple':triple}
                      for p,desc,triple in RULES[family]]
        jobs.append(dict(job_id=f'{family}__{pid}',family_id=family,part_id=pid,
            image_path=target.as_posix(),image_sha256=sha(target),
            predicate_ids=[p[0] for p in RULES[family]],
            schema=schema_for([p[0] for p in RULES[family]]),
            instruction_payload={'atomic_predicates':predicates}))
        source_hashes[str(source)] = sha(source)
        source_hashes[str(binding_path)] = sha(binding_path)
    rules = {family:[{'predicate_id':p,'geometric_claim':desc,'required_triple':triple}
                     for p,desc,triple in definitions] for family,definitions in RULES.items()}
    study = dict(version='llm-geometry-matched-v1',model='gpt-6-luna',
        max_output_tokens=2000,reasoning_effort='low',max_usd=10,concurrency=12,
        instructions=INSTRUCTIONS,schema={},jobs=jobs,
        score='(3*Full_score + H)/4 within fixed union pool; append Full outside pool',
        support='H = mean of four graph predicate values; yes=1,no=0,unknown=0.5; transport/schema failure whole-job H=0.5',
        declared_fields_sent=['atomic geometric predicates and required triples','six original neutral CAD views with only pixel-verified title rectangles removed; original opaque identifiers retained if safe removal is unprovable'],
        excluded_fields=['part IDs','family IDs','class labels','classifier outputs','exemplar IDs','rankings','scores','relevance grades','candidate feature annotations','size constraints'],
        pricing_source='https://developers.openai.com/api/docs/models/gpt-6-luna',pricing_verified_date='2026-10-22')
    write(ROOT/'pool_manifest.json',pool)
    write(ROOT/'knowledge_rules.json',rules)
    write(ROOT/'image_neutralization_audit.json',neutral_audit)
    write(ROOT/'preparation.json',dict(query_count=len(queries),representations=list(pool),
        family_candidate_pairs=len(pairs),unique_cads=len({p for _,p in pairs}),
        source_image_hashes=source_hashes,
        source_note='Latest CURRENT_ANNOTATIONS pointer verified; grades not read during preparation.',
        manuscript_sha256={p.as_posix():sha(p) for p in Path('manuscript/cirp_draft').glob('*')
                           if p.is_file() and p.suffix in ('.tex','.pdf','.bib')},
        studies_already_examined=['query_diversity_v6'],
        evaluation_status='Exploratory extension on historically exposed collection with AI-only qrels; no independent human truth.'))
    required = [Path(__file__), Path('scripts/neutralize_llm_cad_views.py'),
        Path('scripts/complete_llm_geometry_evidence.py'),Path('scripts/evaluate_llm_geometry_matched.py'),
        Path('docs/LLM_GEOMETRY_MATCHED_V1.md'),Path('annotations/CURRENT_ANNOTATIONS.json'),
        ROOT/'pool_manifest.json',ROOT/'knowledge_rules.json',ROOT/'preparation.json',ROOT/'image_neutralization_audit.json']
    study['required_files'] = [{'path':p.as_posix(),'sha256':sha(p)} for p in required]
    write(ROOT/'study.json',study)
    print(json.dumps(dict(jobs=len(jobs),unique_cads=len({p for _,p in pairs}),queries=len(queries),pool_entries=sum(len(v) for a in pool.values() for v in a.values()))))

if __name__=='__main__': main()
