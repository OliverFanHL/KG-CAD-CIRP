"""Shared plotting and table-reading helpers for publication figures."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

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
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'cadkg-publication',
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



def paired_family_values(data,kind,method,family,metric):
    low=[];high=[]
    for rep in (1,2,3):
        a=select(data,geometry_kind=kind,method=method,family_id=family,replicate=rep)
        b=select(data,geometry_kind=kind,method='role_only_hybrid',family_id=family,replicate=rep)
        # Difference of identification envelopes; no point identified by fiat.
        low.append(float(a[metric+'_lower'])-float(b[metric+'_upper']))
        high.append(float(a[metric+'_upper'])-float(b[metric+'_lower']))
    return float(np.mean(low)),float(np.mean(high)),float(np.std(low,ddof=1)),float(np.std(high,ddof=1))


