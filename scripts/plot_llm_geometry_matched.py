"""Plot the separately frozen matched geometric-evidence study.

--dry-validate checks synthetic inputs only and reads no study outcomes.
--plot requires completed evaluator outputs plus their original evidence hashes.
No API calls, evaluation changes or manuscript edits are performed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'manuscript/cirp_draft/.dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D

ROOT=REPO/'outputs/llm_geometry_matched_v1'
KINDS=('embedding','descriptor')
METHODS=('kg_full','geometry_intent_filter','kg_full_llm')
LABELS={'kg_full':'Base Full\n'+r'$G+A+F+C$',
        'geometry_intent_filter':r'$G+F+C$',
        'kg_full_llm':'KG-directed LLM\n'+r'Full + $H$'}
COLORS={'embedding':'#0072B2','descriptor':'#D55E00'}
MARKERS={'embedding':'o','descriptor':'s'}
FAMILY_LABELS={'belt_wheel':'Belt wheel','bracket_mount':'Bracket mount',
    'compression_return':'Compression return','flanged_passage':'Flanged passage',
    'gear_mesh':'Gear mesh','nut_retention':'Nut retention',
    'shaft_connector':'Shaft connector','shaft_drive':'Shaft drive',
    'support_body':'Support body','tension_return':'Tension return',
    'threaded_clamp':'Threaded clamp'}
METRICS=('ndcg@5','strong_precision@5')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def need(condition,message):
    if not condition:raise ValueError(message)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def select(data,**keys):
    selected=[r for r in data if all(r[k]==v for k,v in keys.items())]
    need(len(selected)==1,f'Expected one row:{keys}')
    return selected[0]


def endpoints(row,metric):
    lo=float(row[metric+'_lower']);hi=float(row[metric+'_upper'])
    need(np.isfinite(lo) and np.isfinite(hi) and lo<=hi+1e-12,'Invalid metric envelope')
    return lo,hi


def endpoint_label(lo,hi):
    return f'{lo:.4f}' if lo==hi else f'[{lo:.4f}, {hi:.4f}]'


def style():
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,
        'ytick.labelsize':8,'axes.linewidth':.6,'lines.linewidth':1.2,
        'savefig.pad_inches':.025,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.dpi':150,'axes.unicode_minus':False,'xtick.top':False,'ytick.right':False,
        'xtick.minor.visible':False,'ytick.minor.visible':False})


def save(fig,out,name):
    for ext in ['pdf','png','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=450,facecolor='white',
                    metadata={'Creator':'Matplotlib + SciencePlots'})
    plt.close(fig)


def point(ax,x,y,lo,hi,kind,size=4.5):
    color=COLORS[kind]
    if lo!=hi:
        ax.plot([lo,hi],[y,y],color=color,lw=1.7,zorder=2)
        ax.plot([lo,lo],[y-.028,y+.028],color=color,lw=.8,zorder=2)
        ax.plot([hi,hi],[y-.028,y+.028],color=color,lw=.8,zorder=2)
    ax.plot(x,y,marker=MARKERS[kind],ms=size,color=color,
            mfc=color if lo==hi else 'white',mec=color,mew=.9,zorder=3)


def summary_figure(data,out):
    need(len(data)==6,'Summary requires three methods and two representations')
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),sharey=True)
    fig.subplots_adjust(left=.21,right=.985,bottom=.20,top=.83,wspace=.16)
    for letter,ax,metric in zip('ab',axes,METRICS):
        for y,method in enumerate(METHODS):
            for offset,kind in [(-.14,'embedding'),(.14,'descriptor')]:
                row=select(data,geometry_kind=kind,method=method)
                lo,hi=endpoints(row,metric);need(0<=lo<=hi<=1+1e-12,'Summary values outside0..1')
                yy=y+offset;point(ax,(lo+hi)/2,yy,lo,hi,kind)
                place_left=hi>.70
                xpos=lo-.025 if place_left else hi+.025
                ax.text(xpos,yy,endpoint_label(lo,hi),fontsize=8,color=COLORS[kind],
                        ha='right' if place_left else 'left',va='center')
        ax.set(xlim=(0,1),ylim=(2.52,-.52),xticks=[0,.25,.5,.75,1],
               xticklabels=['0','0.25','0.50','0.75','1.00'],yticks=range(3),
               yticklabels=[LABELS[m] for m in METHODS],
               xlabel='nDCG@5' if metric=='ndcg@5' else 'Strong P@5')
        ax.text(0,1.065,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=8)
        ax.grid(axis='x',color='.92',lw=.5,zorder=0)
    handles=[Line2D([],[],color=COLORS[k],marker=MARKERS[k],linestyle='None',markersize=4.5,label=k.capitalize()) for k in KINDS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,1.02),ncol=2,
               frameon=False,fontsize=8,handletextpad=.4,columnspacing=1.5)
    save(fig,out,'matched_summary')


def family_figure(data,out):
    selected=[r for r in data if r['comparator']=='geometry_intent_filter']
    need(len(selected)==22,'Family plot needs eleven paired differences per representation')
    families=list(FAMILY_LABELS)
    all_ends=[v for r in selected for v in endpoints(r,'ndcg@5')]
    limit=min(1.,max(.1,np.ceil(max(abs(v) for v in all_ends)*10)/10))
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.95),sharey=True)
    fig.subplots_adjust(left=.225,right=.985,bottom=.14,top=.92,wspace=.18)
    for letter,ax,kind in zip('ab',axes,KINDS):
        for y,fid in enumerate(families):
            row=select(selected,geometry_kind=kind,family_id=fid,comparator='geometry_intent_filter')
            lo,hi=endpoints(row,'ndcg@5');point(ax,(lo+hi)/2,y,lo,hi,kind,size=4.5)
            place_left=hi>.65*limit
            label=f'{lo:+.3f}' if lo==hi else f'[{lo:+.3f}, {hi:+.3f}]'
            ax.text(lo-.04*limit if place_left else hi+.04*limit,y,label,
                    ha='right' if place_left else 'left',va='center',fontsize=8,color=COLORS[kind])
        ax.axvline(0,color='.5',lw=.7,zorder=1)
        ax.set(xlim=(-limit,limit),ylim=(10.6,-.6),yticks=range(11),
               yticklabels=[FAMILY_LABELS[f] for f in families],xlabel=r'$\Delta$ nDCG@5')
        ax.text(0,1.045,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
        handle=Line2D([],[],color=COLORS[kind],marker=MARKERS[kind],linestyle='None',markersize=4.5,label=kind.capitalize())
        ax.legend(handles=[handle],loc='upper right',bbox_to_anchor=(1,1.12),frameon=False,fontsize=8,handletextpad=.3)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.grid(axis='x',color='.92',lw=.5,zorder=0)
    save(fig,out,'family_differences')


def dry_validate():
    synthetic=[dict(geometry_kind=k,method=m,**{f'{metric}_{end}':.5 for metric in METRICS for end in ['lower','upper']})
               for k in KINDS for m in METHODS]
    need(len(synthetic)==6 and select(synthetic,geometry_kind='embedding',method='kg_full')['ndcg@5_lower']==.5,'Synthetic selection failed')
    need(endpoints({'ndcg@5_lower':'.4','ndcg@5_upper':'.6'},'ndcg@5')==(.4,.6),'Envelope parsing failed')
    need(endpoint_label(.5,.5)=='0.5000' and endpoint_label(.4,.6)=='[0.4000, 0.6000]','Envelope labels failed')
    print(json.dumps({'status':'SYNTHETIC_PLOT_VALIDATION_PASSED','study_outcomes_read':False,
                      'figure_files_written':False,'scienceplots':importlib.metadata.version('SciencePlots')},indent=2))


def publication_tables(root,results):
    summary=rows(root/'summary.csv');tests=rows(root/'paired_tests.csv')
    names={'kg_full':r'Base Full: $G+A+F+C$',
           'geometry_intent_filter':r'Strong control: $G+F+C$',
           'kg_full_llm':r'Proposed KG-directed LLM: Full+$H$'}
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
           r'Method & \multicolumn{2}{c}{Embedding} & \multicolumn{2}{c}{Descriptor} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
           r' & nDCG@5 & Strong P@5 & nDCG@5 & Strong P@5 \\',r'\midrule']
    for method in METHODS:
        values=[]
        for kind in KINDS:
            row=select(summary,geometry_kind=kind,method=method)
            values.extend(endpoint_label(*endpoints(row,m)) for m in METRICS)
        lines.append(' & '.join([names[method],*values])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    table_performance='\n'.join(lines)+'\n'
    lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
           r'Representation & Comparator & $\Delta$ nDCG@5 & 95\% interval & Raw $p$ & Holm $p$ & W/T/L \\',r'\midrule']
    for kind in KINDS:
        for comparator in METHODS[:2]:
            row=select(tests,geometry_kind=kind,comparator=comparator)
            identified=str(row['point_inference_eligible']).lower()=='true'
            if identified:
                effect=f"{float(row['delta']):+.4f}"
                interval=f"[{float(row['ci95_lower']):+.4f}, {float(row['ci95_upper']):+.4f}]"
                raw=f"{float(row['exact_sign_flip_p']):.6f}";holm=f"{float(row['holm_p']):.6f}"
                wtl='/'.join(row[k] for k in ['family_wins','family_ties','family_losses'])
            else:
                effect=f"[{float(row['delta_lower']):+.4f}, {float(row['delta_upper']):+.4f}]"
                interval=raw=holm=wtl='--'
            label=r'Base Full: $G+A+F+C$' if comparator=='kg_full' else r'$G+F+C$'
            lines.append(' & '.join([kind.capitalize(),label,effect,interval,raw,holm,wtl])+r' \\')
        if kind=='embedding':lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    return table_performance,'\n'.join(lines)+'\n'


def publish_manuscript(root,results):
    need(results['gate']['passed'],'The frozen full gate must pass before publication copies are prepared')
    manuscript=REPO/'manuscript/cirp_draft'
    for name in ['figures','tables','provenance']:(manuscript/name).mkdir(exist_ok=True)
    copied={}
    for source,target in [('matched_summary','llm_geometry_summary'),('family_differences','llm_geometry_families')]:
        for ext in ['pdf','png','svg']:
            origin=root/f'figures/{source}.{ext}';destination=manuscript/f'figures/{target}.{ext}'
            shutil.copyfile(origin,destination)
            need(sha(origin)==sha(destination),'Figure copy changed content')
            copied[str(destination.relative_to(manuscript)).replace('\\','/')]=sha(destination)
    perf,comp=publication_tables(root,results)
    for name,text in [('llm_geometry_performance.tex',perf),('llm_geometry_comparisons.tex',comp)]:
        destination=manuscript/'tables'/name;destination.write_text(text,encoding='utf-8')
        copied['tables/'+name]=sha(destination)
    captions=read(root/'figures/captions.json')
    captions['table_performance']={
        'caption':('Matched KG-directed geometric-evidence study on the fixed 264 new-reference queries, 11 authored task families '
            'and 1,816 held-out candidates. Values use the unchanged nominal AI relevance rubric and independent dimensional transform, '
            'averaged over 24 query conditions within each family and then equally across families. Base Full retains G+A+F+C; '
            'the strong G+F+C control omits exemplar-category agreement. The proposed method reranks the same fixed candidate pool '
            'using the KG-directed geometric-evidence score H. Strong P@5 counts grade-2 returned candidates with a denominator of five.'),
        'source':'summary.csv'}
    captions['table_comparisons']={
        'caption':('Paired nominal nDCG@5 effects of the proposed KG-directed LLM method relative to Base Full and the strong G+F+C '
            'control. Effects average 24 matched queries within each of 11 families and then equally across families. Intervals are '
            'unadjusted exploratory 95% percentile intervals from 20,000 family bootstrap replicates. Raw p-values use exact two-sided '
            'enumeration of 2,048 family sign assignments; Holm correction covers these four comparisons together. W/T/L denotes '
            'positive, tied and negative family differences. The benchmark remains AI-graded and the extension exploratory.'),
        'source':'paired_tests.csv'}
    destination=manuscript/'provenance/llm_geometry_captions.json'
    destination.write_text(json.dumps(captions,indent=2)+'\n',encoding='utf-8')
    copied['provenance/llm_geometry_captions.json']=sha(destination)
    provenance={'script':'scripts/plot_llm_geometry_matched.py','script_sha256':sha(Path(__file__)),
        'source_results':'outputs/llm_geometry_matched_v1/results.json','source_results_sha256':sha(root/'results.json'),
        'source_summary_sha256':sha(root/'summary.csv'),'source_tests_sha256':sha(root/'paired_tests.csv'),
        'source_plot_manifest_sha256':sha(root/'figures/plot_manifest.json'),
        'source_gate_passed':results['gate']['passed'],'manuscript_text_edited':False,
        'table_values':'Literal formatting of preserved CSV endpoints/effects; no new metrics, tests or estimates.',
        'publication_outputs_sha256':copied}
    (manuscript/'provenance/llm_geometry_publication.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Copied six publication figure files and generated two LaTeX tables; manuscript text unchanged.')


def plot(root,publish=False):
    result_path=root/'results.json';results=read(result_path)
    need(results['status']=='COMPLETED_FROZEN_MATCHED_EVIDENCE_STUDY','Study evaluation is not completed')
    need(sha(root/'protocol_freeze.json')==results['main_protocol_freeze_sha256'],'Main protocol differs from completed evaluation')
    need(sha(root/'evaluation_freeze.json')==results['source_freeze_sha256'],'Evaluator freeze differs from completed evaluation')
    need(sha(root/'graph_support.json')==results['graph_support_sha256'],'Support differs from completed evaluation')
    inputs={}
    for name in ['summary.csv','family_differences.csv']:
        path=root/name;digest=sha(path)
        need(digest==results['artifact_sha256'][name],f'Completed plot input changed:{name}')
        inputs[name]=digest
    data=rows(root/'summary.csv');family=rows(root/'family_differences.csv')
    out=root/'figures';out.mkdir(exist_ok=True)
    style();summary_figure(data,out);family_figure(family,out)
    captions={
       'matched_summary':{'caption':(
           'Matched geometric-evidence comparison on 264 frozen new-reference queries and 1,816 held-out CAD candidates. '
           'Panel a shows nDCG@5 and b strong P@5, averaged over 24 conditions within each of 11 families and then equally across families. '
           'Both fixed representations and all three retained methods are shown. Full + H reranks the fixed union of the top 10 candidates '
           'from Full, G+F+C and G+C by (3S_Full + H)/4, then appends the baseline tail. Points denote identified metric values; '
           'open points and horizontal segments denote unknown-label metric envelopes, whose numerical endpoints are printed. '
           'These envelopes are not sampling confidence intervals. No significance is inferred from this absolute-performance plot.'),
         'panel_labels':['a','b'],'source':'summary.csv'},
       'family_differences':{'caption':(
           'Family-level geometric-evidence effects relative to the strongest retained control G+F+C. Panels a and b show embedding '
           'and descriptor results, respectively. Each point or envelope is Full + H minus G+F+C in nominal nDCG@5 after averaging '
           'the 24 frozen reference/condition queries within that authored family. Positive values favor Full + H. Horizontal segments '
           'preserve unknown-label metric envelopes; they are not sampling confidence intervals. Statistical comparisons, when '
           'identified, use the separately reported four-test family-paired Holm procedure.'),
         'panel_labels':['a','b'],'source':'family_differences.csv'},
       'scope':'Separate exploratory study; plots do not authorize manuscript updates. The prespecified full gate is evaluated in results.json.',
       'gate_passed':results['gate']['passed']}
    (out/'captions.json').write_text(json.dumps(captions,indent=2)+'\n',encoding='utf-8')
    manifest={'script':'scripts/plot_llm_geometry_matched.py','script_sha256':sha(Path(__file__)),
       'results_sha256':sha(result_path),'input_sha256':inputs,'scienceplots':importlib.metadata.version('SciencePlots'),
       'matplotlib':matplotlib.__version__,'style':['science','no-latex'],
       'uncertainty':'Unknown-label metric envelopes; not sampling confidence intervals.',
       'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.pdf','.png','.svg']
                  and p.stem in ['matched_summary','family_differences']}}
    (out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if publish:publish_manuscript(root,results)
    print('Generated two separate-study figures in PDF,450dpi PNG and SVG; captions remain outside figures.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-validate',action='store_true');group.add_argument('--plot',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--publish-manuscript',action='store_true',help='Prepare gated publication figure copies and LaTeX tables, without editing manuscript text.')
    args=p.parse_args();root=args.root.resolve()
    need(root.is_relative_to(ROOT.resolve()),'Outputs must remain within the separate matched-study root')
    if args.dry_validate:dry_validate()
    else:plot(root,args.publish_manuscript)


if __name__=='__main__':main()
