"""Portable revision-7 publication assets from completed CSV snapshots only.

No APIs, evaluation reruns, prompt changes, bootstrap or hypothesis tests occur.
Examples:
  python -B generate_revision7_assets.py --dry-validate
  python -B generate_revision7_assets.py --data PATH --output PATH
When copied to manuscript/scripts, defaults read ../data/robustness_v7 and write
../publication_v7. Outputs deliberately do not replace active manuscript assets.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
for parent in (HERE,*HERE.parents):
    for deps in (parent/'.dependencies',parent/'manuscript/cirp_draft/.dependencies'):
        if deps.is_dir():sys.path.insert(0,str(deps))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

KINDS=('embedding','descriptor')
METRICS=('ndcg@5','strong_precision@5')
FIG_METHODS=('geometry_intent_filter','role_only_score','role_only_hybrid','kg_luna','kg_sol')
TABLE_METHODS=('kg_no_class','kg_no_geometry','geometry_class_filter','geometry_intent_filter',
               'kg_no_constraints','kg_full','role_only_hybrid','kg_luna')
LOCAL_METHODS={'kg_no_class','kg_no_geometry','geometry_class_filter','geometry_intent_filter','kg_no_constraints','kg_full'}
LABELS={'geometry_intent_filter':r'$G+F+C$',
    'role_only_score':r'Role-only: $J+C$',
    'role_only_hybrid':r'Full + $J$',
    'kg_luna':'KG-directed (Luna)',
    'kg_sol':'KG-directed (Sol)'}
TABLE_LABELS={'kg_no_class':r'$G+C$','kg_no_geometry':r'$A+F+C$','geometry_class_filter':r'$G+A+C$',
    'geometry_intent_filter':r'$G+F+C$','kg_no_constraints':r'$G+A+F$',
    'kg_full':r'Full: $G+A+F+C$','role_only_hybrid':r'Full+$J$ (role-only)',
    'kg_luna':r'Full+$H$ (KG, Luna)'}
MODEL_COLORS={'kg_luna':'#148477','kg_sol':'#A16B20'}
KIND_COLORS={'embedding':'#3E6A95','descriptor':'#A76C32'}
MARKERS={'embedding':'o','descriptor':'s'}
FAMILIES={'belt_wheel':'Belt wheel','bracket_mount':'Bracket mount',
    'compression_return':'Compression return','flanged_passage':'Flanged passage',
    'gear_mesh':'Gear mesh','nut_retention':'Nut retention','shaft_connector':'Shaft connector',
    'shaft_drive':'Shaft drive','support_body':'Support body','tension_return':'Tension return',
    'threaded_clamp':'Threaded clamp'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def select(data,**keys):
    found=[r for r in data if all(str(r[k])==str(v) for k,v in keys.items())]
    need(len(found)==1,f'Expected exactly one row:{keys}; got {len(found)}')
    return found[0]


def numbers(row,metric):
    """Mean and across-collection SD of each identification-envelope endpoint."""
    values=tuple(float(row[metric+s]) for s in ('_lower_mean','_upper_mean','_lower_sd','_upper_sd'))
    lo,hi,slo,shi=values
    need(all(np.isfinite(values)) and -1e-12<=lo<=hi+1e-12<=1+2e-12 and min(slo,shi)>=0,
         f'Invalid summary metric:{row}')
    if metric+'_mean' in row:
        need(abs(float(row[metric+'_mean'])-lo)<1e-12,'Mean alias differs from lower endpoint')
    if metric+'_sd' in row:
        need(abs(float(row[metric+'_sd'])-slo)<1e-12,'SD alias differs from lower endpoint')
    return values


def style():
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,
        'ytick.labelsize':8,'axes.linewidth':.6,'lines.linewidth':1.1,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'cadkg-revision7',
        'figure.dpi':150,'axes.unicode_minus':False,'xtick.top':False,'ytick.right':False,
        'xtick.minor.visible':False,'ytick.minor.visible':False})


def save(fig,out,name):
    for ext in ('pdf','png','svg'):
        meta={'Creator':'Matplotlib + SciencePlots'}
        if ext=='pdf':meta.update(CreationDate=None,ModDate=None)
        if ext=='svg':meta['Date']=None
        fig.savefig(out/f'{name}.{ext}',dpi=450,facecolor='white',metadata=meta)
    plt.close(fig)


def interval_point(ax,y,lo,hi,slo,shi,color,marker,ms=4.2,hollow=False):
    # Thin connector is an identification envelope; capped error bars are SD.
    if hi>lo+1e-12:
        # No point estimate is drawn when source grades leave it unidentified.
        ax.plot([lo,hi],[y,y],color=color,lw=2.3,alpha=.75,zorder=3)
        for endpoint,sd in ((lo,slo),(hi,shi)):
            ax.plot([endpoint,endpoint],[y-.055,y+.055],color=color,lw=1.,zorder=3)
            if sd>0:ax.errorbar(endpoint,y,xerr=sd,fmt='none',color=color,alpha=.55,
                               capsize=2.5,elinewidth=.7,zorder=2)
        return
    if slo<1e-12:
        ax.plot(lo,y,marker=marker,linestyle='None',mfc='none' if hollow else color,mec=color,
                color=color,ms=ms,mew=.8,zorder=4)
        return
    ax.errorbar(lo,y,xerr=slo,fmt=marker,mfc='none' if hollow else color,mec=color,color=color,
                ms=ms,capsize=2,elinewidth=.8,mew=.8,zorder=4)


def summary_figure(data,out):
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.15),sharey=True)
    fig.subplots_adjust(left=.245,right=.985,bottom=.16,top=.86,wspace=.14)
    for letter,ax,metric in zip('ab',axes,METRICS):
        for y,method in enumerate(FIG_METHODS):
            for kind,offset in [('embedding',-.15),('descriptor',.15)]:
                row=select(data,geometry_kind=kind,method=method)
                values=numbers(row,metric)
                need(.5<=values[0]-values[2] and values[1]+values[3]<=.9,'Summary SD exceeds displayed axis range')
                interval_point(ax,y+offset,*values,KIND_COLORS[kind],MARKERS[kind],ms=3.4,hollow=True)
        ax.set(xlim=(.5,.9),ylim=(4.55,-.55),xticks=np.linspace(.5,.9,5),
               yticks=range(5),yticklabels=[LABELS[m] for m in FIG_METHODS],
               xlabel='nDCG@5' if metric=='ndcg@5' else 'Strong P@5')
        ax.text(0,1.04,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=8)
        ax.grid(axis='x',color='.92',lw=.5,zorder=0)
    handles=[Line2D([],[],color=KIND_COLORS[k],marker=MARKERS[k],markerfacecolor='none',linestyle='None',
        markersize=4.2,label=k.capitalize()) for k in KINDS]
    if any(numbers(select(data,geometry_kind=k,method=m),metric)[1]-numbers(select(data,geometry_kind=k,method=m),metric)[0]>1e-12
           for k in KINDS for m in FIG_METHODS for metric in METRICS):
        handles.append(Line2D([],[],color='.45',lw=2.3,label='Unresolved-grade range'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.62,.995),ncol=len(handles),
        frameon=False,fontsize=8,handletextpad=.4,columnspacing=1.6)
    save(fig,out,'repeated_model_performance')


def paired_family_values(data,kind,method,family,metric):
    low=[];high=[]
    for rep in (1,2,3):
        a=select(data,geometry_kind=kind,method=method,family_id=family,replicate=rep)
        b=select(data,geometry_kind=kind,method='role_only_hybrid',family_id=family,replicate=rep)
        # Difference of identification envelopes; no point identified by fiat.
        low.append(float(a[metric+'_lower'])-float(b[metric+'_upper']))
        high.append(float(a[metric+'_upper'])-float(b[metric+'_lower']))
    return float(np.mean(low)),float(np.mean(high)),float(np.std(low,ddof=1)),float(np.std(high,ddof=1))


def family_figure(data,out):
    allvals={}
    for kind in KINDS:
        for method in ('kg_luna','kg_sol'):
            for family in FAMILIES:
                allvals[kind,method,family]=paired_family_values(data,kind,method,family,'ndcg@5')
    bound=max(max(abs(lo-slo),abs(hi+shi)) for lo,hi,slo,shi in allvals.values())
    limit=max(.1,np.ceil(bound*10)/10)
    fig,axes=plt.subplots(1,2,figsize=(7.15,4.15),sharey=True)
    fig.subplots_adjust(left=.24,right=.985,bottom=.14,top=.86,wspace=.16)
    for letter,ax,kind in zip('ab',axes,KINDS):
        for y,family in enumerate(FAMILIES):
            for method,offset,marker in [('kg_luna',-.14,'o'),('kg_sol',.14,'s')]:
                interval_point(ax,y+offset,*allvals[kind,method,family],MODEL_COLORS[method],marker,ms=4.)
        ax.axvline(0,color='.45',lw=.8,zorder=1)
        ax.set(xlim=(-limit,limit),ylim=(10.55,-.55),yticks=range(11),
               yticklabels=list(FAMILIES.values()),xlabel=r'$\Delta$ nDCG@5')
        ax.text(0,1.04,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
        # Representation is an essential panel identifier, not a prose heading.
        ax.text(.5,1.04,kind.capitalize(),transform=ax.transAxes,ha='center',va='bottom',fontsize=8)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.grid(axis='x',color='.92',lw=.5,zorder=0)
    handles=[Line2D([],[],color=MODEL_COLORS[m],marker=mark,linestyle='None',markersize=4,
             label=label) for m,mark,label in [('kg_luna','o','KG-directed (Luna)'),('kg_sol','s','KG-directed (Sol)')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.61,.995),ncol=2,
               frameon=False,fontsize=8,handletextpad=.4,columnspacing=1.4)
    save(fig,out,'repeated_role_contrasts')
    return [{'geometry_kind':k,'method':m,'family_id':f,'comparator':'role_only_hybrid',
             'ndcg@5_lower_mean':v[0],'ndcg@5_upper_mean':v[1],
             'ndcg@5_lower_sd':v[2],'ndcg@5_upper_sd':v[3]}
            for (k,m,f),v in allvals.items()]


def tex_cell(row,metric):
    lo,hi,slo,shi=numbers(row,metric)
    if hi>lo+1e-12:return rf'$[{lo:.3f}\!\pm\!{slo:.3f},\,{hi:.3f}\!\pm\!{shi:.3f}]$'
    return rf'${lo:.3f}\!\pm\!{slo:.3f}$'


def component_table(data):
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
        r'Scoring configuration & \multicolumn{2}{c}{Embedding} & \multicolumn{2}{c}{Descriptor} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r' & nDCG@5 & Strong P@5 & nDCG@5 & Strong P@5 \\',r'\midrule']
    for method in TABLE_METHODS:
        if method=='role_only_hybrid':lines.append(r'\midrule')
        cells=[TABLE_LABELS[method]]
        for kind in KINDS:
            row=select(data,geometry_kind=kind,method=method)
            # Execution-order repeats do not estimate training uncertainty.
            cells.extend((rf'${numbers(row,m)[0]:.3f}$' if method in LOCAL_METHODS else tex_cell(row,m)) for m in METRICS)
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(lines)+'\n'


def ablation_rows(summary,local):
    """Add the already repeated No-G control without changing primary analysis."""
    combined=list(summary)
    for kind in KINDS:
        anchor=select(local,geometry_kind=kind,method='kg_full')
        primary=select(summary,geometry_kind=kind,method='kg_full')
        for metric in METRICS:
            for end in ('lower','upper'):
                need(abs(float(anchor[metric+'_'+end+'_mean'])-float(primary[metric+'_'+end+'_mean']))<1e-12,
                     'Historical local Full anchor disagrees with current primary source')
        no_g=select(local,geometry_kind=kind,method='kg_no_geometry')
        need(int(no_g['runs'])==3 and no_g['seeds']=='17|29|43','No-G source must contain the three verified executions')
        combined.append({**no_g,'replicates':3,'sd_ddof':1})
    return combined


def paired_table(data):
    # Copies frozen paired-inference outputs. It never runs a significance test.
    lines=[r'\begin{tabular}{@{}lllrrrrr@{}}',r'\toprule',
      r'Representation & Method & Comparator & $\Delta$ nDCG@5 & 95\% interval & Raw $p$ & Holm $p$ & W/T/L \\',r'\midrule']
    need(len(data)==4,'Expected exactly four predeclared paired comparisons')
    data=[{**r,'proposed_method':r.get('proposed_method',r.get('proposed')),
            'point_inference_eligible':r.get('point_inference_eligible',r.get('point_identified','true')),
            'exact_sign_flip_p':r.get('exact_sign_flip_p',r.get('raw_p'))} for r in data]
    for kind in KINDS:
        for proposed,comparator in [('kg_luna','role_only_hybrid'),('kg_sol','kg_luna')]:
            row=select(data,geometry_kind=kind,proposed_method=proposed,comparator=comparator)
            identified=str(row.get('point_inference_eligible','true')).lower()=='true'
            if identified:
                effect=f"{float(row['delta']):+.4f}"
                ci=f"[{float(row['ci95_lower']):+.4f}, {float(row['ci95_upper']):+.4f}]"
                raw=f"{float(row['exact_sign_flip_p']):.6f}";holm=f"{float(row['holm_p']):.6f}"
                wtl='/'.join(str(row[k]) for k in ('family_wins','family_ties','family_losses'))
            else:
                effect=f"[{float(row['delta_lower']):+.4f}, {float(row['delta_upper']):+.4f}]"
                ci=raw=holm=wtl='--'
            lines.append(' & '.join([kind.capitalize(),LABELS[proposed],LABELS[comparator],effect,ci,raw,holm,wtl])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(lines)+'\n'


CAPTIONS={
 'repeated_model_performance':'Repeated model comparison on the same 264 queries and 1,816-candidate gallery. (a) nDCG@5; (b) strong precision at five. Identified points show means over three response collections, each averaged equally over eleven families; capped error bars show sample SD across the three collection means, not confidence intervals. Representation checkpoints, queries, evidence and candidate pools remain fixed. The deterministic G+F+C control was executed in three independently ordered local passes and has zero execution SD. J is a nine-level role-only LLM score; Full+J adds it to Base Full, whereas J+C uses that score alone within the same KG-derived candidate pool with the same feasibility handling. H is the structured KG-directed LLM score. Where grades remain unresolved, a thick interval with endpoint ticks replaces the point estimate; thin capped whiskers show the SD of each endpoint across collections. Such identification envelopes are not confidence intervals. Neither pool construction nor J+C is a full-gallery, KG-free retrieval baseline.',
 'repeated_role_contrasts':'Role-specific KG-directed evidence effects relative to the role-only Full+J hybrid, with both methods evaluated on the same fixed candidate pools. (a) Embedding geometry; (b) descriptor geometry. For each model and role, identified points show the mean of three paired collection differences, each first averaged over the same 24 query conditions; capped error bars show sample SD of those three paired differences. The vertical line marks zero difference. Where labels remain unresolved, a thick identification interval with endpoint ticks replaces the point estimate, and thin whiskers show the SD of each endpoint. These envelopes and SD bars are not confidence intervals. Sol-minus-role-only differences are descriptive contrasts, not additional formal tests. All contrasts use the same fixed encoder and query panel.',
 'component_ablation_v7':'Component ablation on the fixed 264-query panel and 1,816-candidate gallery. LLM entries show mean plus or minus sample SD over three response collections, with equal family weighting. Deterministic rows show fixed-checkpoint scores without an SD. G is geometric similarity, A predicted-category agreement with the exemplar, F category-derived functional compatibility, and C the hard dimensional constraint. Full averages G, A and F before C. Full+J and Full+H use the frozen role-only and KG-directed LLM combinations, respectively. Execution-order checks reproduce the deterministic scores; encoder-training and mesh-sampling variability are not measured. If an identification envelope is nonzero, the cell reports lower and upper endpoint means, each with its corresponding SD; it is not a confidence interval.',
 'repeated_paired_tests_v7':'Predeclared paired family comparisons after averaging each role over three response collections and its 24 query conditions. Intervals are the saved 20,000-resample family-bootstrap percentile intervals; raw probabilities are the saved two-sided exact family sign-flip results, and Holm correction covers the four listed comparisons. W/T/L counts family wins, ties and losses. No inference is produced for an unidentified point effect.'}


def validate(data):
    for kind in KINDS:
        for method in set(FIG_METHODS)|set(TABLE_METHODS):
            row=select(data,geometry_kind=kind,method=method)
            need(int(row['replicates'])==3 and int(row['sd_ddof'])==1,'Exactly three collections with sample SD required')
            for metric in METRICS:
                lo,hi,slo,shi=numbers(row,metric)
                # np.std can leave ~1e-16 from summing three identical floats.
                # Preserve source values and treat only machine roundoff as zero.
                if method in LOCAL_METHODS:need(max(slo,shi)<1e-12,'Local fixed-checkpoint baseline SD exceeds floating-point roundoff')


def dry_validate():
    synthetic=[]
    for kind in KINDS:
        for method in set(FIG_METHODS)|set(TABLE_METHODS):
            row=dict(geometry_kind=kind,method=method,replicates=3,sd_ddof=1)
            for m in METRICS:row.update({m+'_lower_mean':.5,m+'_upper_mean':.5,m+'_lower_sd':0.,m+'_upper_sd':0.})
            synthetic.append(row)
    validate(synthetic)
    need('0.500' in component_table(synthetic),'Synthetic table failed')
    need(numbers({m+s:v for m in METRICS for s,v in [('_lower_mean',.4),('_upper_mean',.6),('_lower_sd',.01),('_upper_sd',.02)]},'ndcg@5')==(.4,.6,.01,.02),'Envelope failed')
    print(json.dumps(dict(status='SYNTHETIC_SCHEMA_VALIDATED',outcomes_read=False,figures_written=False,
                         scienceplots=importlib.metadata.version('SciencePlots')),indent=2))


def generate(data,out):
    need((data/'summary.csv').exists() and (data/'family_metrics.csv').exists(),'Completed metric snapshots required')
    out.mkdir(parents=True,exist_ok=True)
    summary=rows(data/'summary.csv');family=rows(data/'family_metrics.csv')
    table_data=ablation_rows(summary,rows(data/'remaining_local_three_pass.csv'));validate(table_data)
    style();summary_figure(summary,out);contrasts=family_figure(family,out)
    (out/'component_ablation_v7.tex').write_text(component_table(table_data),encoding='utf-8')
    if (data/'paired_tests.csv').exists():
        (out/'repeated_paired_tests_v7.tex').write_text(paired_table(rows(data/'paired_tests.csv')),encoding='utf-8')
    (out/'captions_v7.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n',encoding='utf-8')
    (out/'role_contrast_plot_data.json').write_text(json.dumps(contrasts,indent=2)+'\n',encoding='utf-8')
    source_hashes={p.name:sha(p) for p in sorted(data.iterdir()) if p.is_file() and p.suffix in ('.csv','.json')}
    files=[p for p in out.iterdir() if p.suffix in ('.pdf','.png','.svg','.tex','.json') and p.name!='publication_manifest_v7.json']
    manifest=dict(generator_sha256=sha(__file__),source_sha256=source_hashes,
        output_sha256={p.name:sha(p) for p in sorted(files)},
        matplotlib=matplotlib.__version__,scienceplots=importlib.metadata.version('SciencePlots'),
        scope='Presentation from frozen metric CSVs only; no evaluation/inference/API calls.',
        summary_methods=list(FIG_METHODS),ablation_methods=list(TABLE_METHODS),
        role_contrast='kg_luna and kg_sol minus role_only_hybrid, paired within collection and family',
        uncertainty='Sample SD across three collection/execution means; identification endpoints retained independently; no training variability claim.')
    (out/'publication_manifest_v7.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PUBLICATION_ASSETS_GENERATED',output=str(out),files=sorted(manifest['output_sha256'])),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-validate',action='store_true')
    p.add_argument('--data',type=Path,default=HERE.parent/'data/robustness_v7')
    p.add_argument('--output',type=Path,default=HERE.parent/'publication_v7');args=p.parse_args()
    if args.dry_validate:dry_validate()
    else:generate(args.data.resolve(),args.output.resolve())

if __name__=='__main__':main()
