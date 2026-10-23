"""Write a compact separate results report from the completed fixed comparison."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('outputs/llm_geometry_matched_v1')
LABELS={'kg_full':'Full','geometry_intent_filter':'G+F+C','kg_full_llm':'Full + KG-directed LLM evidence'}
FAMILIES={'belt_wheel':'belt wheels','bracket_mount':'bracket mounting',
 'compression_return':'compression return','flanged_passage':'flanged passages',
 'gear_mesh':'gear meshing','nut_retention':'nut retention','shaft_connector':'shaft connection',
 'shaft_drive':'shaft transmission','support_body':'shaft support','tension_return':'tension return',
 'threaded_clamp':'threaded clamping'}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def number(value, signed=False):
    return 'withheld' if value is None else format(value,'+.4f' if signed else '.4f')
def interval(lo,hi,signed=False):
    return number(lo,signed) if lo==hi else f'[{number(lo,signed)}, {number(hi,signed)}]'

def main():
    result=read(ROOT/'results.json');evidence=read(ROOT/'evidence_completion.json')
    assert result['status']=='COMPLETED_FROZEN_MATCHED_EVIDENCE_STUDY'
    assert sha(ROOT/'protocol_freeze.json')==result['main_protocol_freeze_sha256']
    gate=result['gate']['passed']
    prep=read(ROOT/'preparation.json')
    paper={p:sha(p)==digest for p,digest in prep['manuscript_sha256'].items()}
    verdict=('The addition meets the prespecified criteria for considering an exploratory manuscript revision.' if gate else
             'The addition does not meet the prespecified criteria for revising the manuscript. The new results are retained separately.')
    s=['# KG-directed LLM evidence for CAD retrieval: matched results','',
       verdict,'',
       '## Matched comparison','',
       'The proposed addition uses the KG to specify four geometric predicates and their component bindings for each requested role. '
       'GPT-6 Luna observes these relations in six existing CAD views. Yes, no and unknown observations map to 1, 0 and 0.5; '
       'their fixed mean is the soft evidence-coverage score H. The LLM supplies observations, while the KG specifies the evidence '
       'requirements and the fixed aggregation. Existing deterministic size checks retain sole control of dimensional eligibility.','',
       'The evaluation uses all 264 primary v6 exemplar queries, four training references per each of eleven roles and six size '
       'conditions per reference, with both existing geometry representations and the same 1,816 test CADs. The three compared '
       'methods are Full (G+A+F+C), the strongest retained control G+F+C, and Full plus H. Geometry encodings, classifier outputs, '
       'measurements, query objects and relevance judgments are unchanged.','',
       'For each query and representation, the candidate pool is the fixed union of the top ten results from Full, G+F+C and G+C. '
       'The proposed score is (3 S_Full + H)/4 within that pool; the original Full tail follows. Every baseline top-five candidate '
       'is retained. H is cached once per role–CAD pair and shared across references, size conditions and representations. '
       'No additional model, weight sweep, prompt selection or outcome-based pool expansion is included.','',
       '## Quantitative results','',
       'Figure 1 compares absolute retrieval quality; Table 1 reports the paired effects and their statistical uncertainty. '
       'The complete numerical means are retained in [summary.csv](summary.csv).','',
       '![Matched primary metrics](figures/matched_summary.png)','',
       '**Figure 1.** Family-balanced primary metrics: a, nDCG@5; b, Strong P@5. Each mean averages six conditions per reference, '
       'four references per role and eleven roles equally. Full + H denotes the proposed addition. Open points and nonzero '
       'segments, if present, propagate unresolved reference labels; they are not sampling confidence intervals. All methods '
       'share the same queries, measurements and candidate gallery.','',
       '**Table 1.** Proposed-minus-control nDCG@5 differences. Bootstrap intervals use 20,000 role resamples and are unadjusted. '
        'Exact two-sided sign-flip tests use eleven role differences with one Holm correction over four comparisons. '
        'Point inference is withheld if the new primary endpoint is not identified.','',
        '| Geometry | Comparator | Difference | 95% bootstrap interval | Holm p | Role wins/ties/losses |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['paired_tests']:
        ci='withheld' if r['ci95_lower'] is None else f"[{number(r['ci95_lower'],True)}, {number(r['ci95_upper'],True)}]"
        wtl='withheld' if r['family_wins'] is None else f"{r['family_wins']}/{r['family_ties']}/{r['family_losses']}"
        s.append(f"| {r['geometry_kind'].capitalize()} | {LABELS[r['comparator']]} | {interval(r['delta_lower'],r['delta_upper'],True)} | {ci} | {number(r['holm_p'])} | {wtl} |")
    s+=['','![Role-level changes](figures/family_differences.png)','',
        '**Figure 2.** Role-level nDCG@5 changes relative to G+F+C: a, embedding; b, descriptor. Each effect averages '
        'the 24 matched queries within a role. Positive effects favour the addition. Nonzero horizontal segments retain '
        'unresolved-label envelopes and do not represent sampling uncertainty. Panel labels, axes and legends are inside '
        'the figure; this caption is external.','',
        '## Interpretation and decision','',
        'Figure 2 separates the aggregate change into the eleven role-specific effects.','']
    differences=rows(ROOT/'family_differences.csv')
    for kind in ('embedding','descriptor'):
        local=[r for r in differences if r['geometry_kind']==kind and r['comparator']=='geometry_intent_filter']
        ordered=sorted(local,key=lambda r:float(r['ndcg@5_lower']))
        low,high=ordered[0],ordered[-1]
        gains=sum(float(r['ndcg@5_lower'])>1e-12 for r in local)
        losses=sum(float(r['ndcg@5_upper'])<-1e-12 for r in local)
        s.append(f"For {kind} geometry, {gains} roles have positive lower-bound effects and {losses} have negative upper-bound effects against G+F+C. "
            f"The largest lower-bound increase occurs for {FAMILIES[high['family_id']]} ({interval(float(high['ndcg@5_lower']),float(high['ndcg@5_upper']),True)}); "
            f"the lowest effect occurs for {FAMILIES[low['family_id']]} ({interval(float(low['ndcg@5_lower']),float(low['ndcg@5_upper']),True)}).")
        s.append('')
    s+=['The update rule was fixed before any API response: at least +0.02 nDCG@5 over G+F+C in both representations, '
        'Holm p≤0.05 for both comparisons, and no Strong P@5 decrease against G+F+C in either representation. '
        'The complete gate and every condition are recorded in results.json. '+verdict,'',
        'The intervention adds task-specific visual observations and fixed requirement coverage. The three-method comparison '
        'estimates the utility of that combined addition. It does not separately isolate the value of the LLM, the authored '
        'predicates or graph storage relative to information-matched alternatives. H is evidence coverage rather than a '
        'probability of mechanical suitability. Component bindings make the observations inspectable but do not establish '
        'that a model observation is geometrically correct.','',
        '## Collection, provenance and scope','',
        f"All {evidence['completed_jobs']:,} predeclared role–CAD jobs have terminal records across 1,213 unique CADs. "
        f"Response status counts are {json.dumps(evidence['status_counts'],sort_keys=True)}. "
        f"The token-based cost estimate is US${evidence['estimated_cost_usd']:.6f}, below the US$10 ceiling. "
        'The request, response, model, usage and source hashes are retained. Credentials are absent from all artifacts.','',
        'Source views preserve the original 1500×1000 geometry pixels; only exact verified title rectangles are cleared. '
        'The model receives requested predicate semantics and neutral views, without role or part identifiers, candidate grades, '
        'classifier outputs, ranking scores, exemplar identities or numerical constraints. Role semantics are necessarily '
        'present in the geometric requirements. No failed response is resampled and no reference judgment is filled from the new model.','',
        'The v5 relevance reference is AI-derived and has not received independent expert adjudication. The collection and '
        'earlier retrieval outcomes have already been examined, so this is an exploratory extension. Another GPT can share '
        'the reference annotator’s visual assumptions. Effects concern visible role evidence and top-five retrieval on the '
        'fixed known-role pools; they do not establish industrial compatibility or unseen-domain performance. Pool-first '
        'ordering can change deeper ranks even with neutral H, so this comparison reports only the prespecified @5 metrics.','',
        '## Reproducibility','',
        'The [replay instructions](REPRODUCE.md), [fixed protocol](../../docs/LLM_GEOMETRY_MATCHED_V1.md), '
        '[per-query metrics](per_query.csv), [paired tests](paired_tests.csv), [rule definitions](knowledge_rules.json), '
        '[saved graph observations](graph_evidence.jsonl) and [input audit](pre_execution_audit.md) accompany the report. '
        'SciencePlots scripts provide vector PDF/SVG and 450 dpi PNG figures, with captions stored separately.','',
        f"Manuscript source/PDF/bibliography hashes unchanged at report creation: {sum(paper.values())}/{len(paper)}.",'']
    (ROOT/'study_report.md').write_text('\n'.join(s),encoding='utf-8')
    (ROOT/'report_provenance.json').write_text(json.dumps(dict(results_sha256=sha(ROOT/'results.json'),
        evidence_sha256=sha(ROOT/'evidence_completion.json'),script_sha256=sha(Path(__file__)),
        manuscript_unchanged=paper,gate_passed=gate),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(ROOT/'study_report.md'),'gate_passed':gate,'manuscript_unchanged':all(paper.values())}))

if __name__=='__main__': main()
