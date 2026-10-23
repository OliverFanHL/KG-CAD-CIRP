"""Labeled scientific figures and observed-family interpretation for intent batches."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..io import read_csv, read_json, write_json, receipt
from .full import LABELS
from .reference import REFERENCE_METHODS


def report(output, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .reference import draw_surface
    output = Path(output)
    directory = output/'report'; directory.mkdir(parents=True, exist_ok=True)
    summary = read_csv(output/'summary.csv')
    cases = read_csv(output/'per_query.csv')
    pairs = read_csv(output/'family_differences.csv')
    protocol = read_json(output/'protocol.json')
    ranks = read_csv(output/'rankings.csv')
    judged = {(r['query_id'],r['part_id']):r for r in read_csv(output/'ai_judgments.csv')}
    manifest = {r['part_id']:r for r in read_csv(output/'pool_manifest.csv')}
    pools = read_json(output/'family_pools.json')
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.path import Path as PlotPath
    fig, ax=plt.subplots(figsize=(13,6),layout='constrained'); ax.set_xlim(0,13); ax.set_ylim(0,6); ax.axis('off')
    boxes=[(0.2,3.9,2.5,1.5,'TMCAD: 9,799 STEP parts\nGeometry, units, hashes\nLabels excluded from ranking','#e4f1f5'),
           (3.1,4.15,2.5,1.25,'Frozen geometry model\nEmbeddings + category probabilities\nDescriptors: train-normalized','#e4f1f5'),
           (3.1,2.35,2.5,1.25,'Measured CAD + generic priors\n11 source-supported functions\nNo verified instance function','#fff0da'),
           (6.1,3.9,2.5,1.5,'Structured intent query\nOptional train exemplar\nOptional extent bound','#edf3e5'),
           (9.15,3.9,3.5,1.5,'Full-gallery ranking + filtering\n1,816 held-out candidates\n8 methods × 2 representations\n49 queries = 784 runs','#e4f1f5'),
           (6.1,.35,2.5,1.6,'Separate inspection evidence\nUnion top5 + controls\n424 actual source-CAD renders\n674 family–part AI records','#f4e8f3'),
           (9.15,.35,3.5,1.6,'Paired evaluation\nFamily-balanced P@5 / pool nDCG\nAll wins and losses + sensitivity\nIndependent expert review pending','#f4e8f3')]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#55707c'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    model_route=PlotPath([(5.7,4.65),(5.85,4.65),(5.85,5.7),(10.9,5.7),(10.9,5.48)],
                         [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=model_route,arrowstyle='-|>',mutation_scale=14,color='#55707c',lw=1.4))
    for a,b in [((2.8,4.7),(3,4.7)),((2.4,3.9),(3,3.1)),((5.7,3.0),(9.02,3.9)),((8.75,4.65),(9.02,4.65)),((10.9,3.8),(7.4,2.1)),((8.75,1.15),(9.03,1.15))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#55707c',lw=1.4))
    ax.text(3.5,.9,'AI grades never enter KG,\ntraining or retrieval scoring',ha='center',va='center',fontsize=10,color='#714873')
    ax.set_title('Expanded intent-retrieval experiment: fixed method, broader evaluation\nSource-supported priors and provisional AI relevance remain distinct',fontsize=14)
    fig.savefig(directory/'expanded_method.png',dpi=170); fig.savefig(directory/'expanded_method.pdf'); plt.close(fig)
    def row(kind, subset, method, variant='exemplar'):
        return next(r for r in summary if r['geometry_kind']==kind and r['subset']==subset and r['method']==method and r['variant']==variant)
    fig, axes = plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    y=np.arange(len(REFERENCE_METHODS))
    for ax,metric,title in zip(axes,['macro_pool_ndcg@5','macro_strong_precision@5'],['Pool nDCG@5','Strong shape P@5 (grade 2)']):
        for kind,offset,color in [('embedding',-.18,'#267b97'),('descriptor',.18,'#d48538')]:
            values=[float(row(kind,'combined',m)[metric]) for m in REFERENCE_METHODS]
            ax.barh(y+offset,values,height=.33,label=kind,color=color)
            for yi,value in zip(y+offset,values):
                ax.text(value+.008,yi,f'{value:.4f}',va='center',fontsize=9)
        ax.set_yticks(y,[LABELS[m] for m in REFERENCE_METHODS]); ax.invert_yaxis()
        ax.set_xlim(0,1.18); ax.set_title(title+' — higher is better'); ax.legend(loc='upper right',fontsize=9)
        ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('11 families · 38 exemplar cases · 1,816 held-out candidates\nFamily-balanced averages; provisional AI shape judgments',fontsize=13)
    fig.savefig(directory/'method_comparison.png',dpi=170); fig.savefig(directory/'method_comparison.pdf'); plt.close(fig)
    families=protocol['families']
    fig,ax=plt.subplots(figsize=(12,6.2),layout='constrained')
    for kind,offset,color in [('embedding',-.18,'#267b97'),('descriptor',.18,'#d48538')]:
        values=[float(next(r for r in pairs if r['family_id']==f['family_id'] and r['geometry_kind']==kind)['delta_pool_ndcg@5']) for f in families]
        ax.barh(np.arange(len(families))+offset,values,height=.33,label=kind,color=color)
        for yi,value in zip(np.arange(len(families))+offset,values):
            ax.text(value+(.008 if value>=0 else -.008),yi,f'{value:+.4f}',va='center',ha='left' if value>=0 else 'right',fontsize=9)
    ax.axvline(0,color='#333',lw=1); ax.invert_yaxis()
    ax.set_yticks(np.arange(len(families)),[f['label']+(' [new]' if f['batch']=='new' else '') for f in families])
    ax.set_xlim(-1.12,1.12); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    ax.set_xlabel('Full − remove exemplar cosine: mean paired pool nDCG@5 difference')
    ax.set_title('Where exemplar geometry helps or hurts\nPositive favors full; negative favors removing cosine · observed families, no significance claim')
    ax.legend(); fig.savefig(directory/'family_differences.png',dpi=170); fig.savefig(directory/'family_differences.pdf'); plt.close(fig)
    # Predefined cases, including both within-class spring intents. No selection by winning score.
    qualitative=[]
    examples=[(qid,'embedding') for qid in ['threaded_clamp_envelope2','gear_mesh_envelope1','compression_return_example1','tension_return_example1','nut_retention_example1','compression_return_example2']]
    examples.append(('tension_return_example1','descriptor'))
    for qid,kind in examples:
        fig=plt.figure(figsize=(13,8.7)); records=[]
        for i,method in enumerate(['geometry','kg_no_geometry','kg_full']):
            selected=sorted([r for r in ranks if r['geometry_kind']==kind and r['query_id']==qid and r['method']==method],key=lambda r:int(r['rank']))[:3]
            for j,item in enumerate(selected):
                m=manifest[item['part_id']]; truth=judged[qid,item['part_id']]
                triangles=np.load(output/f"evidence/{m['inspection_id']}.npz")['triangles']
                draw_surface(fig.add_subplot(3,3,i*3+j+1,projection='3d'),triangles)
                ax=fig.axes[-1]
                ax.set_title(f"{LABELS[method]} · rank {item['rank']}\n{m['inspection_id']} · AI grade {truth['grade']} · dimension {truth['constraint_verdict']}",fontsize=9)
                records.append({'method':method,'rank':int(item['rank']),'part_id':item['part_id'],'inspection_id':m['inspection_id'],
                                'grade':int(truth['grade']),'observation':truth['observation'],'rationale':truth['rationale'],'constraint_verdict':truth['constraint_verdict']})
        query=next(q for q in protocol['queries'] if q['query_id']==qid)
        filename=qid if kind=='embedding' else f'{qid}_{kind}'
        fig.suptitle(qid+f' · {kind}\nActual STEP surfaces; AI grades: 0 mismatch · 1 conditional · 2 strong shape evidence',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(directory/f'{filename}.png',dpi=155); plt.close(fig)
        write_json(directory/f'{filename}.json',{'query':query,'geometry_kind':kind,'records':records})
        qualitative.append(f'![{qid}: {kind}]({filename}.png)\n\nSource observations and rationales: [{filename}.json]({filename}.json).')
    table='| Method | Embedding pool nDCG | Descriptor pool nDCG | Embedding P@5 | Descriptor P@5 | Embedding strong P@5 | Descriptor strong P@5 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for method in REFERENCE_METHODS:
        a,b=row('embedding','combined',method),row('descriptor','combined',method)
        table+=f"|{LABELS[method]}|{float(a['macro_pool_ndcg@5']):.4f}|{float(b['macro_pool_ndcg@5']):.4f}|{float(a['macro_precision@5']):.4f}|{float(b['macro_precision@5']):.4f}|{float(a['macro_strong_precision@5']):.4f}|{float(b['macro_strong_precision@5']):.4f}|\n"
    batch_table='| Batch | Families | Exemplar cases | Embedding full | Embedding remove cosine | Difference | Descriptor full | Descriptor remove cosine | Difference |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for subset in ['previous','new','combined']:
        a,b=row('embedding',subset,'kg_full'),row('embedding',subset,'kg_no_geometry')
        c,d=row('descriptor',subset,'kg_full'),row('descriptor',subset,'kg_no_geometry')
        batch_table+=f"|{subset}|{a['families']}|{a['paired_queries']}|{float(a['macro_pool_ndcg@5']):.4f}|{float(b['macro_pool_ndcg@5']):.4f}|{float(a['macro_pool_ndcg@5'])-float(b['macro_pool_ndcg@5']):+.4f}|{float(c['macro_pool_ndcg@5']):.4f}|{float(d['macro_pool_ndcg@5']):.4f}|{float(c['macro_pool_ndcg@5'])-float(d['macro_pool_ndcg@5']):+.4f}|\n"
    ftable='| Engineering family | Batch | Exemplar cases | Embedding ΔnDCG | Descriptor ΔnDCG | Family judged parts |\n|---|---|---:|---:|---:|---:|\n'
    for f in families:
        a=next(r for r in pairs if r['family_id']==f['family_id'] and r['geometry_kind']=='embedding')
        b=next(r for r in pairs if r['family_id']==f['family_id'] and r['geometry_kind']=='descriptor')
        ftable+=f"|{f['label']}|{f['batch']}|{a['paired_queries']}|{float(a['delta_pool_ndcg@5']):+.4f}|{float(b['delta_pool_ndcg@5']):+.4f}|{len(pools[f['family_id']])}|\n"
    sensitivity=read_csv(output/'sensitivity.csv')
    stable=[]
    for scenario in ['conditional_zero','nominal','conditional_two']:
        selected=[{**r,'has_exemplar':r['has_exemplar']=='True',**{m:float(r[m]) for m in ['pool_ndcg@5','precision@5','strong_precision@5']},'dimension_violations':int(r['dimension_violations'])} for r in sensitivity if r['grading_scenario']==scenario]
        from .intent import summarize
        summaries=summarize(selected,families)
        for kind in settings['geometry_kinds']:
            full=next(r for r in summaries if r['geometry_kind']==kind and r['subset']=='combined' and r['variant']=='exemplar' and r['method']=='kg_full')
            other=next(r for r in summaries if r['geometry_kind']==kind and r['subset']=='combined' and r['variant']=='exemplar' and r['method']=='kg_no_geometry')
            stable.append({'scenario':scenario,'geometry_kind':kind,'macro_delta_pool_ndcg@5':full['macro_pool_ndcg@5']-other['macro_pool_ndcg@5']})
    write_json(directory/'sensitivity_summary.json',stable)
    st='| Conditional-grade interpretation | Embedding ΔnDCG | Descriptor ΔnDCG |\n|---|---:|---:|\n'
    for s in ['conditional_zero','nominal','conditional_two']:
        values=[next(r for r in stable if r['scenario']==s and r['geometry_kind']==k)['macro_delta_pool_ndcg@5'] for k in settings['geometry_kinds']]
        st+=f'|{s}|{values[0]:+.4f}|{values[1]:+.4f}|\n'
    decisions=[]
    for kind in settings['geometry_kinds']:
        values=[float(r['delta_pool_ndcg@5']) for r in pairs if r['geometry_kind']==kind]
        decisions.append(f"{kind}: {sum(v>1e-10 for v in values)} family wins, {sum(abs(v)<=1e-10 for v in values)} ties, {sum(v < -1e-10 for v in values)} losses for full versus removal of exemplar cosine.")
    # Equality remains possible even on a much larger intent set; audit exact IDs vs metric ties.
    grouped=defaultdict(list)
    for r in ranks:
        grouped[r['geometry_kind'],r['query_id'],r['method']].append(r)
    equality=[]
    for kind in settings['geometry_kinds']:
        for q in protocol['queries']:
            a=sorted(grouped[kind,q['query_id'],'kg_full'],key=lambda r:int(r['rank']))
            b=sorted(grouped[kind,q['query_id'],'kg_no_geometry'],key=lambda r:int(r['rank']))
            ca=next(r for r in cases if r['geometry_kind']==kind and r['query_id']==q['query_id'] and r['method']=='kg_full')
            cb=next(r for r in cases if r['geometry_kind']==kind and r['query_id']==q['query_id'] and r['method']=='kg_no_geometry')
            equality.append({'geometry_kind':kind,'query_id':q['query_id'],'has_exemplar':bool(q['reference_part_id']),
                             'identical_ordered_ids':[r['part_id'] for r in a]==[r['part_id'] for r in b],
                             'identical_grade_sequence':[judged[q['query_id'],r['part_id']]['grade'] for r in a]==[judged[q['query_id'],r['part_id']]['grade'] for r in b],
                             'equal_pool_ndcg':float(ca['pool_ndcg@5'])==float(cb['pool_ndcg@5'])})
    from ..io import write_csv
    write_csv(output/'equality_audit.csv',equality)
    ef=row('embedding','combined','kg_full'); en=row('embedding','combined','kg_no_geometry')
    df=row('descriptor','combined','kg_full'); dn=row('descriptor','combined','kg_no_geometry')
    text=f'''# Expanded intent-family experiments — batch 2

The broader test changes the interpretation: full with learned embeddings is lower than removal of exemplar cosine on family-balanced pool nDCG@5 (**{float(ef['macro_pool_ndcg@5']):.4f} vs {float(en['macro_pool_ndcg@5']):.4f}**), while full with descriptors is higher (**{float(df['macro_pool_ndcg@5']):.4f} vs {float(dn['macro_pool_ndcg@5']):.4f}**). The learned-embedding combination also has lower strong-shape precision, despite higher precision when conditional matches count. Geometry's value depends on the intent and representation; a universal improvement is unsupported.

## Material Passport

Artifact: computational experiment and provisional AI inspection report. Status: executed; independent human expert validation pending. Inputs: TMCAD STEP geometry, frozen local geometry encoder/classifier exports, manufacturer-supported generic priors. Author: Codex AI assistant, also implementation author. No AI grade is claimed as expert ground truth.

## What changed

Engineering families increased from **3 to 11**. This batch adds **8 families**, **16 train-only inspected exemplars**, and **40 structured queries**. The combined primary matrix contains **49 queries** (11 text-only, 38 with exemplars), two geometric representations and eight methods: **784 ranking runs** against the same **1,816 deduplicated held-out candidates**. The KG still covers all **9,799 corpus parts**. Geometry measurements and model exports were reused; no classifier retraining or scoring-weight tuning.

The new scopes cover headed threaded clamping, nut retention, fixed bracket mounting, bolted passage flanges, external spur/helical gear meshing, rotary shaft transmission, compression springs and tension-return springs. Two spring tasks share the spring category but need different end geometry. Every task asks for a preliminary geometric shortlist. Threads, intended service, mating fit, strength and load capacity remain unverified unless independently documented.

New envelope bounds were fixed to each category's validation-split median before test ranking. Compression and tension tasks share the same bound. Exemplar selection used 40 train-only source-CAD inspections, selecting two role-relevant shape forms per new family. This is an authored exploratory benchmark, not a random sample of real engineering demands.

![Expanded experiment schematic](expanded_method.png)

## Paired results

Higher is better for all metrics. Δ = full minus remove exemplar cosine. Results are averaged within each family and then across families; adding four variants for a new family does not give it twice the weight of an old family. Text-only cases are reported separately in summary.csv because exemplar cosine is unavailable there. The three old capability-gap queries remain in the original report; they are excluded from this primary comparison.

{batch_table}

{table}

![Numerically labeled comparison](method_comparison.png)

{' '.join(decisions)} All family results, including negative differences, follow:

{ftable}

![Signed family differences](family_differences.png)

## Interpretation and limits

More intent families make the test more informative; they do not guarantee that full wins. Singleton class–function priors can behave exactly like category lookup. The matched flat table has the same information and scoring as the KG, so identical flat/full results are expected: this implementation tests geometry, priors and constraint filtering, **not a unique advantage of graph storage**.

“Remove exemplar cosine” removes only g=(cosine+1)/2 from the weighted score and its denominator. It retains classifier probabilities, exemplar category-distribution agreement and dimension filtering. The classifier itself was trained on geometry; this ablation is **not geometry-free**. The no-classifier/function branch instead uses exemplar cosine plus the same filter. With no exemplar or no bound, corresponding terms or filters are inactive.

The structured query schema carries generic function IDs, references and numeric bounds. Fine task distinctions in prose (thread visibility, hook form, spur versus bevel teeth) are **evaluation criteria**, not parsed symbolic geometry predicates. Two spring functions both map to the spring category; with no exemplar their rankings are identical. With exemplars, generic priors plus similarity can still prefer a coil with the wrong ends. This is a concrete remaining model/representation gap, not cured merely by adding more function names to the KG.

The nut failures favor visually similar but unverified inner-thread forms over clearer threaded candidates. The compression example2 ranks a hook-ended extension-like coil ahead of valid compression forms; the simpler spring class branch cannot penalize the hook mismatch. These observations suggest that reviewed geometric interface/subtype information and validation-selected signal weighting are worth testing in a future **separate** development experiment. This batch keeps the original weights and priors fixed; its negative results are retained.

Only nominal visible shape evidence is graded: 0 mismatch/insufficient evidence, 1 conditional, 2 strong visible shortlist evidence. All method/representation top5 outputs are inspected within each family, plus two seeded random controls for each new family. There are **{len(manifest)} unique rendered candidates**, **{sum(map(len,pools.values()))} family–part records** and **{len(judged)} query–part judgments** after numeric screening. The original families retain their original complete 111-part judgment pools, making their old results directly comparable. New family pools have different sizes. Pool nDCG uses these finite pools as IDCG; full-gallery recall and exhaustive relevance are unknown. P@5 does not need a complete gallery relevance census, and strong P@5 avoids treating conditional shape matches as firm matches.

The AI that implemented the framework also inspected and graded the CADs. Neutral views conceal categories, method identity, rank and score, but this does not create independent or blinded human validation. Family variants are correlated, and compression/tension share a category. No p-values, population confidence intervals, safety claims or statistical generalization are made.

Conditional-grade sensitivity, a hypothetical reinterpretation rather than an uncertainty interval:

{st}

## Qualitative examples

The initial four cases were chosen by task identity, including both spring roles. Two additional failure cases (nut retention example1 and compression example2) were added after seeing family losses, explicitly to explain those losses. The same tension-return case is also shown with descriptors to contrast the two representations. These are diagnostic examples rather than an unbiased qualitative sample. Representation names appear on each figure. The same gallery and source shapes are used in every row. Full top5 grades and explanations are saved in the structured result files.

{chr(10).join(qualitative)}

## Reproduce

From the repository root, with the existing environment:

```powershell
$cadkgPython = 'D:/Postdoc/research/jepa_cad/.venv/Scripts/python.exe'
& $cadkgPython -m cadkg intent-batch --stage all
```

Stages: search, evidence, evaluate, report, validate. Config: configs/intent_batch2.json. Inputs: annotations/intent_batch2/families.json, queries.json, exemplars.json, knowledge/*.csv and judgments.json. The existing full-scale bundle must be present; create it with `python -m cadkg full-scale --stage all` if starting from raw STEP and legacy exports. STEP files and predecessor artifacts remain source-hash checked. Cached evidence is reused only when the source matches. The fixed AI judgments are replayed; expert work in human_annotations.csv is never overwritten.

Structured outputs: summary.csv (previous/new/combined; exemplar/text/all), per_query.csv, paired_differences.csv, family_differences.csv, ai_judgments.csv, sensitivity.csv, equality_audit.csv, protocol.json, run_manifest.json, validation.json. Full rankings were computed against the held-out gallery, not restricted to the judged pools. Independent experts can fill the blank human_annotations.csv using docs/ANNOTATION.md.
'''
    (directory/'REPORT.md').write_text(text,encoding='utf-8')
    write_json(directory/'report_manifest.json',receipt('intent batch report',[output/'summary.csv',output/'per_query.csv',output/'ai_judgments.csv'],settings))
    return {'report':str((directory/'REPORT.md').resolve()),'comparison':'family-balanced, AI-authored, no independent expert claim'}
