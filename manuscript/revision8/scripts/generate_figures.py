"""Redraw Figures 2 and 3 from fixed revision-7 CSVs with SciencePlots.

No model calls, experimental evaluation or statistical testing. Annotations and
error bars come from the retained collection means and paired differences.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('frozen_metrics',HERE/'generate_revision7_assets.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
np=g.np;plt=g.plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
g.style()
plt.rcParams.update({'font.size':8.2,'axes.labelsize':8.2,'ytick.labelsize':8.2,
    'xtick.labelsize':8,'svg.hashsalt':'cadkg-revision8'})
METHOD_LABELS=[r'$G+F+C$',r'Role-only $J+C$',r'Full + $J$','KG-directed Luna','KG-directed Sol']
audit=[];plotdata=[]

def panels(height,limits,yticks,ylabels,ylim):
    fig=plt.figure(figsize=(7.15,height))
    sets=[]
    for plot_x,stat_x in ((.195,.438),(.610,.853)):
        ax=fig.add_axes([plot_x,.135,.232,.715])
        stats=fig.add_axes([stat_x,.135,.143,.715],sharey=ax)
        ax.set(xlim=limits,ylim=ylim,yticks=yticks,yticklabels=ylabels)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.grid(axis='x',color='.91',lw=.5,zorder=0)
        stats.set(xlim=(0,1),ylim=ylim);stats.axis('off')
        stats.text(.5,1.025,r'Mean $\pm$ SD',ha='center',va='bottom',transform=stats.transAxes,fontsize=7.8)
        sets.append((ax,stats))
    sets[1][0].tick_params(axis='y',labelleft=False)
    return fig,sets

def finish(fig,name,labels):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[(t,t.get_window_extent(renderer)) for t in labels]
    for i,(t,b) in enumerate(boxes):
        assert t.axes.get_window_extent(renderer).contains(b.x0,b.y0),t.get_text()
        assert t.axes.get_window_extent(renderer).contains(b.x1,b.y1),t.get_text()
        for u,c in boxes[:i]:assert not b.overlaps(c),(t.get_text(),u.get_text())
    audit.append(dict(figure=name,numeric_annotations=len(labels),annotation_collisions=0,
        separate_statistic_columns=True,native_width_inches=7.15))
    g.save(fig,OUT,name)

summary=g.rows(ROOT/'data/summary.csv')
fig,axes=panels(3.75,(.5,.9),np.arange(5)*1.2,METHOD_LABELS,(5.6,-.8))
labels=[]
for letter,(ax,stats),metric in zip('ab',axes,g.METRICS):
    for i,method in enumerate(g.FIG_METHODS):
        for kind,offset in [('embedding',-.22),('descriptor',.22)]:
            mean,upper,sd,usd=g.numbers(g.select(summary,geometry_kind=kind,method=method),metric)
            assert abs(mean-upper)<1e-12 and abs(sd-usd)<1e-12
            y=i*1.2+offset;color=g.KIND_COLORS[kind]
            g.interval_point(ax,y,mean,upper,sd,usd,color,g.MARKERS[kind],ms=4.4,hollow=True)
            fixed=method=='geometry_intent_filter'
            value=f'{mean:.3f}' if fixed or sd<.001 else f'{mean:.3f} ± {sd:.3f}'
            labels.append(stats.text(.5,y,value,ha='center',va='center',fontsize=7.8,color=color))
            plotdata.append(dict(figure=2,panel=letter,geometry_kind=kind,method=method,metric=metric,mean=mean,sd=sd))
    ax.set_xticks([.5,.6,.7,.8,.9]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('nDCG@5' if metric=='ndcg@5' else 'Strong P@5',labelpad=7)
    ax.text(-.02,1.055,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
handles=[Line2D([],[],color=g.KIND_COLORS[k],marker=g.MARKERS[k],mfc='white',mec=g.KIND_COLORS[k],
    linestyle='none',markersize=4.5,label=k.capitalize()) for k in g.KINDS]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.985),ncol=2,frameon=False,
    handletextpad=.5,columnspacing=2.8,fontsize=8.3)
finish(fig,'figure2_observer_comparison',labels)

family=g.rows(ROOT/'data/family_metrics.csv')
values={(kind,method,f):g.paired_family_values(family,kind,method,f,'ndcg@5')
    for kind in g.KINDS for method in ('kg_luna','kg_sol') for f in g.FAMILIES}
lo=min(v[0]-v[2] for v in values.values());hi=max(v[1]+v[3] for v in values.values())
limits=(float(np.floor((lo-.012)*20)/20),float(np.ceil((hi+.012)*20)/20))
role_labels=['Belt wheel','Bracket mounting','Compression return','Flanged passage',
    'Gear meshing','Nut retention','Shaft connection','Shaft transmission',
    'Shaft support','Tension return','Threaded clamping']
fig,axes=panels(6.7,limits,np.arange(11)*1.2,role_labels,(12.8,-.8))
labels=[]
for letter,(ax,stats),kind in zip('ab',axes,g.KINDS):
    for i,f in enumerate(g.FAMILIES):
        for method,offset,marker in [('kg_luna',-.22,'o'),('kg_sol',.22,'s')]:
            mean,upper,sd,usd=values[kind,method,f]
            assert abs(mean-upper)<1e-12 and abs(sd-usd)<1e-12
            color=g.MODEL_COLORS[method];y=i*1.2+offset
            g.interval_point(ax,y,mean,upper,sd,usd,color,marker,ms=4.2)
            value=f'{mean:+.3f}' if sd<.001 else f'{mean:+.3f} ± {sd:.3f}'
            labels.append(stats.text(.5,y,value,ha='center',va='center',fontsize=7.8,color=color))
            plotdata.append(dict(figure=3,panel=letter,geometry_kind=kind,method=method,role=f,
                comparator='role_only_hybrid',mean=mean,sd=sd))
    ax.axvline(0,color='.35',lw=.8,zorder=1)
    ax.set_xticks([v for v in [-.1,0,.1,.2] if limits[0]-1e-9<=v<=limits[1]+1e-9])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel(r'$\Delta$ nDCG@5',labelpad=7)
    ax.text(-.02,1.035,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
    ax.text(.5,1.035,kind.capitalize(),transform=ax.transAxes,ha='center',va='bottom',fontsize=8.5)
handles=[Line2D([],[],color=g.MODEL_COLORS[m],marker=marker,linestyle='none',markersize=4.5,label=label)
    for m,marker,label in [('kg_luna','o','KG-directed Luna'),('kg_sol','s','KG-directed Sol')]]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.987),ncol=2,frameon=False,
    handletextpad=.5,columnspacing=2.8,fontsize=8.3)
finish(fig,'figure3_role_differences',labels)
(ROOT/'data/figure_annotations.json').write_text(json.dumps(plotdata,indent=2)+'\n')
record=dict(status='PASS',results_changed=False,annotation_layout_checks=audit,
    numerical_SD_omission_threshold=.001,annotation_decimals=3,error_bars_unchanged=True,
    generator_sha256=sha(__file__),source_sha256={p.name:sha(p) for p in (ROOT/'data').glob('*.csv')},
    output_sha256={p.name:sha(p) for p in OUT.glob('figure[23]*')},
    scienceplots=g.importlib.metadata.version('SciencePlots'),matplotlib=g.matplotlib.__version__)
(ROOT/'provenance/figure_generation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
