"""One source-bound SciencePlots figure for the separate matched model report."""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'manuscript/cirp_draft/.dependencies'))
import scienceplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    study=ROOT/'outputs/llm_geometry_sol_matched_v1'
    rows=list(csv.DictReader((study/'summary.csv').open()))
    assert len(rows)==8
    methods=['kg_full','geometry_intent_filter','kg_full_llm','kg_full_sol']
    lookup={(r['geometry_kind'],r['method']):r for r in rows}
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':9,'savefig.dpi':360,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.15),sharey=True)
    for j,(metric,label) in enumerate([('ndcg@5','nDCG@5'),('strong_precision@5','Strong P@5')]):
        ax=axes[j]
        for kind,off,color,mark in [('embedding',-.11,'#0072B2','o'),('descriptor',.11,'#D55E00','s')]:
            yy=[3-k+off for k in range(4)]
            xx=[float(lookup[kind,m][metric+'_lower']) for m in methods]
            assert all(lookup[kind,m][metric+'_lower']==lookup[kind,m][metric+'_upper'] for m in methods)
            ax.plot(xx,yy,linestyle='none',marker=mark,markersize=4.5,color=color,label=kind.title())
            for x,y in zip(xx,yy):ax.text(x-.016,y,f'{x:.4f}',ha='right',va='center',fontsize=7.2,color=color)
        ax.set_xlim(0,1);ax.set_ylim(-.45,3.55);ax.set_xlabel(label)
        ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='x',color='#DADADA',linewidth=.45)
        ax.tick_params(axis='y',length=0);ax.spines[['left','right','top']].set_visible(False)
        ax.set_yticks([3,2,1,0],['Base Full','G + F + C','KG-directed Luna','KG-directed Sol'])
        ax.text(0,1.025,'ab'[j],transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
    axes[0].legend(loc='lower center',bbox_to_anchor=(1.08,1.04),ncol=2,frameon=False)
    fig.subplots_adjust(left=.185,right=.985,bottom=.18,top=.82,wspace=.20)
    out=study/'figures';out.mkdir(exist_ok=True)
    paths=[]
    for ext in ('pdf','png','svg'):
        path=out/('matched_model_comparison.'+ext)
        fig.savefig(path,metadata={'Creator':'Matplotlib with SciencePlots'} if ext=='pdf' else None)
        paths.append(path)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(Path(__file__)),
        source='summary.csv',source_sha256=sha(study/'summary.csv'),
        outputs={p.name:sha(p) for p in paths},caption_in_artwork=False)
    (out/'figure_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    brief='''# Matched visual-observer comparison

GPT-6.1 Sol completed the same 2,084 role–CAD observations as GPT-6 Luna.
The requested model name resolves to `gpt-6.1-sol`. Only model identity changed
in the API request; clauses, image bytes, candidate pools and score weights
remained fixed. The comparison uses 264 queries and eleven role families.

![Matched model results](figures/matched_model_comparison.png)

**Figure 1.** Family-balanced retrieval performance for the fixed methods:
a, nDCG@5; b, Strong P@5. Points average six size conditions per exemplar,
four exemplars per role, then eleven roles equally. All metric bounds have
coincident endpoints. Baselines and frozen Luna results are retained.

Sol reaches nDCG@5 of 0.827162 with embeddings and 0.812562 with descriptors.
Differences from Luna are −0.002032 and +0.000906, with Holm-adjusted p=1 for
both in the prespecified four-comparison family. The analysis does not resolve
superiority and does not establish equivalence. Sol exceeds G+F+C by 0.160380
and 0.087921, both adjusted p=0.0078125. Estimated collection costs are $18.36
for Sol and $1.10 for Luna, based on returned usage and recorded rates.

The manuscript retains Luna's main results. Sol is reported separately as the
additional matched model candidate. This sequential comparison concerns the
same AI-graded CAD collection. All 2,112 metric rows passed an independent
calculation, and saved-response replay produced identical numerical outputs.
Full intervals, role effects, provenance and the AI-reference scope appear in
[the detailed report](study_report.md) and [reproduction instructions](REPRODUCE.md).
'''
    (study/'results_brief.md').write_text(brief,encoding='utf8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
