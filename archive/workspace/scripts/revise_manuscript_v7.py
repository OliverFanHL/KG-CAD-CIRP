"""Revise the paper only after all frozen three-collection outcomes are audited.

Writes main/supplement from archived revision6 sources, never from an API stream.
"""
from pathlib import Path
import csv,hashlib,json,re,shutil
REPO=Path(__file__).resolve().parents[1]
PAPER=REPO/'manuscript/cirp_draft'
STUDY=REPO/'outputs/llm_robustness_v7'
ARCHIVE=REPO/'manuscript/revision_archive/revision_6_before_repeated_observation'
TITLE='Knowledge-guided relational evidence for functional CAD retrieval'
OLDTITLE='Knowledge-guided visual evidence for CAD retrieval with large language models'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def archive():
    if ARCHIVE.exists():return
    ARCHIVE.mkdir(parents=True)
    for name in ('main.tex','main.pdf','supplement.tex','supplement.pdf','README.md','package_manifest.json'):
        shutil.copyfile(PAPER/name,ARCHIVE/name)
    shutil.copyfile(REPO/'manuscript/cirp_annals_manuscript_package.zip',ARCHIVE/'cirp_annals_manuscript_package.zip')

def section(text,start,end,replacement):
    i=text.index(start);j=text.index(end,i)
    return text[:i]+replacement.strip()+'\n\n'+text[j:]

def prepare_sources():
    archive()
    # The text layer is completed after the outcome schema and independent audit
    # are available. No placeholder or inferred result is written into the paper.
    return (ARCHIVE/'main.tex').read_text(encoding='utf8'),(ARCHIVE/'supplement.tex').read_text(encoding='utf8')

REPEATED_METHODS=r'''
\subsection{Repeated acquisition and role-only controls}

Three collections are retained for each visual observer: KG-directed GPT-6 Luna,
KG-directed GPT-6.1 Sol and role-only GPT-6 Luna. The original KG collection
for each model is reused with its original request contract; two further collections
use identical clauses, image bytes and schemas. The role-only observer has three
new collections. Fresh requests from the seven collections are interleaved. Local
seeds 17, 29 and 43 control their execution order; the Responses API exposes no
sampling-seed control. The repeats assess model-acquisition variability with
fixed representations. All responses
enter the analysis once, with neutral support for failed assessments.

The role-only observer receives the requested role's purpose and the same six views,
without the four authored clauses, triples or component-binding instructions.
Generic guidance on uncertainty, display clips and the limits of visible geometry
is retained. It reports holistic visual support $J_i$ from
$\{0,\frac18,\ldots,1\}$, with cited views and a literal explanation. This matches
the range and increments of $H_i$. The same-model control substitutes $J_i$ for
$H_i$ in Eq.~\ref{eq:refined}, preserving the initial score, pool and dimensional
screen. A score-only LLM control ranks the same eligible pool by $J_i$ alone.
Its serving score omits $G,A,F$, while its candidate access remains conditional
on the common pool. These controls distinguish holistic role assessment from the
combined authored decomposition, bindings and deterministic evidence aggregation.

Each collection produces its own rankings and metrics before averaging. Figures
and tables report the mean and sample standard deviation of the three aggregate
collection scores. Three fresh local executions with shuffled query order also
verify deterministic initial ranking; identical outputs have zero execution SD.
The fixed encoder, query construction, judgments and pool membership remain unchanged.
'''

ANALYSIS=r'''
\subsection{Statistical analysis and reproduction}

Inference uses eleven role means after averaging the three collections within
each role. The primary contrasts are KG-directed Luna minus its role-only hybrid
control and KG-directed Sol minus KG-directed Luna, for each representation.
Exact two-sided sign-flip tests enumerate $2^{11}$ patterns, with one Holm
correction across these four nDCG@5 contrasts \cite{holm1979procedure}.
Unadjusted role-bootstrap intervals use 20,000 resamples. Collection SD describes
repeated acquisition and is distinct from uncertainty across roles; repeated calls
do not create additional independent roles. The tests condition on the authored
roles and assume sign exchangeability of their paired effects. Complete comparisons and source
identities are supplied in Supplementary Section~S9.

Saved responses reconstruct observations, coverage, ranking orders and metrics
without further model calls. Independent checks compare the numerical calculation
and preserved source arrays. Both unresolved reference judgments retain their
admissible grades. The earlier component analysis and auxiliary request-grounding
assessment remain separate in Supplementary Sections~S3 and S7.
'''

DISCUSSION_SCOPE=r'''
\subsection{Interpretation and limitations}

The controls locate the contribution at the level of evidence acquisition.
Full and the role-only hybrid share the same geometric and category-derived score;
the proposed module changes how visible support is requested and aggregated.
The graph organizes the requirements, their bindings and observation provenance.
An information-matched flat implementation can reproduce the same initial algebra.
The comparison tests the requirement-guided acquisition and aggregation procedure.

Repeated collection assesses stochastic observer variability conditional on a
single validation-selected encoder and an unchanged query design. The first KG
collections precede the interleaved repetitions, and three collections provide
a limited description of acquisition variation. The AI-derived reference has no
independent expert adjudication and may share visual assumptions with the tested
observers. Historical inspection, descriptor-conditioned exemplar selection and
related CAD designs constrain generalization. Materials, loads, tolerances and
mating compatibility require subsequent engineering assessment. The bounded
refinement pool also limits recovery to candidates reached by the initial branches.
'''

DECLARATION=r'''
\section*{Declaration of AI-assisted manuscript preparation}
OpenAI Codex assisted with manuscript organization, reference checking,
visualization code and figure development. OpenAI image generation supplied a
layout concept for Figure~\ref{fig:framework}, which was reconstructed with
editable graphics and source CAD renders. The authors retain responsibility
for final verification, editing and approval of the article.
'''

def integrate():
    main,sup=prepare_sources()
    result=json.loads((STUDY/'results.json').read_text())
    audit=json.loads((STUDY/'post_execution_audit.json').read_text())
    replay=json.loads((STUDY/'independent_replay_audit.json').read_text())
    assert replay['status']=='PASS'
    assert str(audit.get('status','')).upper().startswith('PASS'),audit.get('status')
    assert result['primary_ndcg_identified']
    summary={(r['geometry_kind'],r['method']):r for r in rows(STUDY/'summary.csv')}
    tests={(r['geometry_kind'],r['proposed'],r['comparator']):r for r in rows(STUDY/'paired_tests.csv')}
    def val(k,m,metric='ndcg@5'):return float(summary[k,m][metric+'_mean'])
    def sd(k,m,metric='ndcg@5'):return float(summary[k,m][metric+'_sd'])
    def cell(k,m,metric='ndcg@5'):
        row=summary[k,m]
        lo=float(row[metric+'_lower_mean']);hi=float(row[metric+'_upper_mean'])
        if hi-lo>1e-12:
            return (f"$[{lo:.4f}\\pm{float(row[metric+'_lower_sd']):.4f},"
                    f"\\,{hi:.4f}\\pm{float(row[metric+'_upper_sd']):.4f}]$")
        return f"${val(k,m,metric):.4f}\\pm{sd(k,m,metric):.4f}$"
    def diff(k,a,b):return val(k,a)-val(k,b)
    def pair(a,b):return ' and '.join(f'{diff(k,a,b):+.4f}' for k in ('embedding','descriptor'))
    # Source snapshots and generated publication assets must be present first.
    data=PAPER/'data/robustness_v7'
    assert (data/'summary.csv').exists()
    for name in ('summary.csv','paired_tests.csv','family_metrics.csv','run_metrics.csv'):
        assert sha(data/name)==sha(STUDY/name)
    main=main.replace('Knowledge-guided visual evidence for CAD retrieval\\\\with large language models',
                      'Knowledge-guided relational evidence\\\\for functional CAD retrieval')
    abstract=(
      'Functional CAD reuse requires evidence of role-specific interfaces. A knowledge graph (KG) specifies relational '
      'geometric requirements, visual language models assess candidate views, and deterministic coverage refines '
      'retrieval under measured size constraints. Evaluation uses 1,816 CADs and 264 queries across eleven roles '
      'with AI-derived relevance judgments. Three response collections per observer assess acquisition variability. '
      f"KG-directed GPT-6 Luna achieves nDCG@5 of {val('embedding','kg_luna'):.4f} and {val('descriptor','kg_luna'):.4f} for embeddings and descriptors. "
      f"Differences from matched holistic role assessment are {pair('kg_luna','role_only_hybrid')}. "
      'Component ablation and source-linked observations relate retrieval behaviour to numerical eligibility, role priors and interface evidence.')
    assert len(abstract.split())<=100,len(abstract.split())
    ai=main.index('\n\\small\n');aj=main.index('\n\\par\\vspace{6pt}',ai)
    main=main[:ai]+'\n\\small\n'+abstract+'\n'+main[aj:]
    main=main.replace('and a matched evaluation of its ranking contribution across eleven design-role families.',
        'and a repeated-collection comparison with holistic visual assessment across eleven design-role families.')
    main=main.replace('GPT-6 Luna returns a structured observation for every clause:',
        'The evaluated GPT-6 Luna and GPT-6.1 Sol observers return a structured observation for every clause:')
    main=main.replace("Every comparator's original top five is available to refinement.",
        'Each pool includes the original top five of these three branches.')
    main=main.replace('The new primary nDCG@5 and Strong P@5 values are point-identified under these bounds.',
        'Metric envelopes propagate both unassessable pairs; identified endpoints are reported as point values.')
    main=main.replace('The view montage reproduces an archived input, including two display clips.',
        'The view montage reproduces an archived input, including two display clips. OpenAI image generation assisted the layout concept; the final diagram uses editable graphics and source CAD renders.')
    # Add the matched observer control at the end of the mathematical method.
    pos=main.index('\\section{Evaluation}')
    main=main[:pos]+REPEATED_METHODS.strip()+'\n\n'+main[pos:]
    main=main.replace('Methods & Base Full, $G+F+C$, proposed KG--LLM',
        'Methods & Component controls; role-only Luna; KG-directed Luna and Sol')
    main=main.replace('Visual acquisition & 2,084 role--CAD pairs; 1,213 CADs',
        'Visual acquisition & 2,084 role--CAD pairs per collection; three collections per observer')
    main=main.replace('Inference & Eleven role differences; four Holm-adjusted tests',
        'Inference & Eleven role means across collections; four paired contrasts')
    start='The three retained methods test the addition against both the complete initial ranker'
    stop='\\subsection{Reference relevance and metrics}'
    repl=r'''The component controls isolate the available scoring signals and the dimensional screen.
The role-only and KG-directed observers then use the same 528 query/representation
pools, each containing 10--26 candidates. Across these pools, 2,084 distinct
role--CAD pairs cover 1,213 CAD identities. Requirements, pool membership and
weights precede acquisition. The original seven-method comparison, including
reciprocal-rank fusion \cite{cormack2009rrf}, remains a secondary analysis in
Supplementary Section~S3.'''
    main=section(main,start,stop,repl)
    main=section(main,'\\subsection{Matched analysis and reproduction}','\\section{Results and discussion}',ANALYSIS)
    # Replace aggregate results and the former inference table with an ablation.
    lead=(r'''\subsection{Retrieval performance and observer variability}

Figure~\ref{fig:performance} compares the observation designs and both GPT models
using means and sample SDs across three collections. With embeddings and descriptors,
respectively, KG-directed Luna achieves '''+
      cell('embedding','kg_luna')+' and '+cell('descriptor','kg_luna')+
      r''' nDCG@5. Sol achieves '''+cell('embedding','kg_sol')+' and '+cell('descriptor','kg_sol')+
      r'''. Corresponding Strong P@5 values are '''+
      cell('embedding','kg_luna','strong_precision@5')+' and '+cell('descriptor','kg_luna','strong_precision@5')+
      ' for Luna, and '+cell('embedding','kg_sol','strong_precision@5')+' and '+cell('descriptor','kg_sol','strong_precision@5')+
      r''' for Sol. Each result uses the same queries, representation, size rules and candidate access.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{repeated_model_performance.pdf}
\caption{Repeated observer comparison: a, nDCG@5; b, Strong P@5. Points are means over three collections, with sample-SD error bars. Each collection averages six conditions per exemplar, four exemplars per role and eleven roles equally. $J$ denotes holistic role-only Luna assessment; Full+$J$ retains the initial score, while $J+C$ ranks the inherited eligible pool by $J$ alone. KG-directed observers use four authored clauses and coverage $H$. Candidate pools, representations and measured eligibility are common. The deterministic $G+F+C$ control has zero execution SD.}
\label{fig:performance}
\end{figure*}

''')
    lead+=('Relative to the same-model Full+$J$ control, KG-directed Luna changes nDCG@5 by '+
           pair('kg_luna','role_only_hybrid')+'. ')
    unresolved=any(float(r['ndcg@5_upper_mean'])-float(r['ndcg@5_lower_mean'])>1e-12 or
                   float(r['strong_precision@5_upper_mean'])-float(r['strong_precision@5_lower_mean'])>1e-12
                   for r in summary.values() if r['method']=='role_only_score')
    if unresolved:
        lead=lead.replace('The deterministic $G+F+C$ control has zero execution SD.',
            'The deterministic $G+F+C$ control has zero execution SD. Thick ranges denote unresolved-reference identification intervals, with endpoint-SD whiskers in place of point estimates.')
    ps=[float(tests[k,'kg_luna','role_only_hybrid']['holm_p']) for k in ('embedding','descriptor')]
    if all(diff(k,'kg_luna','role_only_hybrid')>0 and ps[i]<.05 for i,k in enumerate(('embedding','descriptor'))):
        lead+=r'The positive differences survive the four-comparison Holm correction. '
    else:
        significant=[k for i,k in enumerate(('embedding','descriptor')) if ps[i]<.05]
        if significant:
            lead+='The '+ ' and '.join(significant)+' contrast survives the four-comparison Holm correction. '
        else:
            lead+='Neither contrast survives the four-comparison Holm correction. '
    lead+=('Sol-minus-Luna changes are '+pair('kg_sol','kg_luna')+'. ')
    sol_ps=[float(tests[k,'kg_sol','kg_luna']['holm_p']) for k in ('embedding','descriptor')]
    if all(p>=.05 for p in sol_ps):
        lead+='Neither model difference is statistically resolved by the role-level tests. '
    else:
        significant=[k for i,k in enumerate(('embedding','descriptor')) if sol_ps[i]<.05]
        lead+='The '+ ' and '.join(significant)+' model contrast survives the four-comparison correction. '
    lead+=(r'The four role-level comparisons and their intervals appear in Supplementary Section~S9.'+'\n\n')
    lead+=r'''\subsection{Contribution of the scoring components}

Table~\ref{tab:paired} presents a stagewise component ablation. The screened
$G+C$ ranker provides the geometric anchor. Removing one component from Full
gives $A+F+C$, $G+A+C$, $G+F+C$ and $G+A+F$, exposing the respective contributions
of geometry, function support, exemplar-category agreement and size eligibility.
The final two rows then compare
holistic and clause-directed visual evidence on the same Full score.

\begin{table*}[t]
\centering\small
\resizebox{\textwidth}{!}{\input{tables/component_ablation_v7.tex}}
\caption{Stagewise component ablation on the same 264 queries. Entries are mean $\pm$ sample SD across three executions or response collections. $G,A,F$ are geometric similarity, exemplar-category agreement and category-derived role support; $C$ is measured-size screening. Active score components have equal weights. $J$ is holistic role-only Luna support, while $H$ is KG-directed Luna coverage. Deterministic configurations have zero execution SD with the encoder held fixed. Formal statistical comparisons are supplied separately in the supplement.}
\label{tab:paired}
\end{table*}

'''
    lead+=('Measured screening increases Full nDCG@5 over its unscreened score by '+pair('kg_full','kg_no_constraints')+
      '. The $G+F+C$ control exceeds Full by '+pair('geometry_intent_filter','kg_full')+
      ', showing that adding exemplar-category agreement can reduce the initial score\'s effectiveness. '
      'These component effects motivate instance-level visual evidence that examines the requested interfaces directly.\n\n')
    main=section(main,'\\subsection{Contribution of instance-level visual evidence}',
                 '\\subsection{Within-category functional discrimination}',lead)
    # Replace historical role aggregates with current, source-bound contrasts.
    family=rows(STUDY/'family_metrics.csv')
    groups={}
    for r in family:groups.setdefault((r['geometry_kind'],r['method'],r['family_id']),[]).append(float(r['ndcg@5_lower']))
    import numpy as np
    role_text=[]
    for kind in ('embedding','descriptor'):
        effects={fid:float(np.mean(groups[kind,'kg_luna',fid])-np.mean(groups[kind,'role_only_hybrid',fid]))
                 for k,m,fid in groups if k==kind and m=='kg_luna'}
        pos=sum(v>1e-12 for v in effects.values());neg=sum(v< -1e-12 for v in effects.values())
        role_text.append(f'{kind.capitalize()} Luna differences are positive in {pos} roles and negative in {neg}')
    role_block=r'''\subsection{Role-dependent evidence effects}

Figure~\ref{fig:families} resolves the difference from holistic assessment by role.
'''+'; '.join(role_text)+r'''. Repeated conditions share the same role definition,
so these profiles explain where the observation design helps or hinders retrieval.
Compression and tension return are particularly informative because both receive
the same spring-category prior: their interfaces supply the role-specific distinction.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{repeated_role_contrasts.pdf}
\caption{Role-specific nDCG@5 differences relative to the Full+$J$ holistic Luna control: a, embedding; b, descriptor. Points average three paired collection differences within each role; bars show their sample SD. Each collection averages the same four exemplars and six conditions. Positive values favour clause-directed evidence. Luna contrasts compare observation designs with the same model; Sol contrasts also change the observer and are descriptive.}
\label{fig:families}
\end{figure*}

'''
    main=section(main,'\\subsection{Within-category functional discrimination}',
                 '\\subsection{Source-linked explanation of a ranking change}',role_block)
    main=main.replace('Figure~\\ref{fig:evidence} links actual observations to a descriptor shaft query.',
        'Figure~\\ref{fig:evidence} links observations from the retained first Luna collection to a descriptor shaft query.')
    main=main.replace('Recorded GPT observations give Z0430 coverage 0 and Z0972 coverage 1.',
        'Observations from Luna collection 1 give Z0430 coverage 0 and Z0972 coverage 1.')
    main=main.replace('The proposed method moves Z0972 from Base Full rank 4 to rank 1;',
        'That collection moves Z0972 from Base Full rank 4 to rank 1;')
    main=main.replace('This disagreement motivates expert assessment of role scope and geometric interpretation rather than treating coverage as a suitability certificate.',
        'This disagreement identifies role scope and geometric interpretation as priorities for expert assessment.')
    completion=json.loads((STUDY/'evidence_completion.json').read_text())
    main=section(main,'\\subsection{Acquisition cost and scope of interpretation}','\\section{Conclusions}',
        DISCUSSION_SCOPE+'\n\n'+
        f"The repeated study adds {completion['new_calls']:,} API calls. Its accounting estimate is US\\${completion['estimated_new_cost_usd']:.2f}, "
        'using reported token usage and conservative reservations where usage is unavailable. '
        'Observations are cached by role and CAD identity. These collection costs are separate from the local initial-ranker timings in Supplementary Section~S5. Full response accounting appears in Section~S9.')
    conclusion=(r'''\section{Conclusions}

A knowledge-guided relational-evidence method connects requested CAD functions
to authored geometric predicates, view-linked model observations and deterministic
coverage. Measured dimensions govern candidate eligibility. Three collections per
observer evaluate acquisition variability on 1,816 CADs and 264 queries across
eleven roles. KG-directed Luna reaches nDCG@5 of '''+
      f"{val('embedding','kg_luna'):.4f} and {val('descriptor','kg_luna'):.4f}, with corresponding Sol values of "+
      f"{val('embedding','kg_sol'):.4f} and {val('descriptor','kg_sol'):.4f}. "+
      'The matched holistic-control differences are '+pair('kg_luna','role_only_hybrid')+
      r''' for Luna. Component ablation separates geometric and category-derived
signals, dimensional screening and visual acquisition. The resulting evidence is
traceable to source views and supports preliminary functional discovery within
the evaluated AI-graded collection.''')
    main=section(main,'\\section{Conclusions}','\\section*{Data and code availability}',conclusion)
    main=section(main,'\\section*{Use of generative AI}','\\par\\vspace{6pt}',DECLARATION)
    main=main.replace('complete predicate definitions, input/output hashes, retained model responses, graph observations and ranked results.',
        'complete predicate and role-only contracts, input/output hashes, all three response collections, graph observations and ranked results.')
    (PAPER/'main.tex').write_text(main,encoding='utf8')
    sup=sup.replace(OLDTITLE,TITLE)
    # S8 remains a clearly marked historical first-collection audit, preserving
    # its actual saved counts and original decision rule rather than recasting it.
    needle='\\section{KG-directed visual evidence: matched protocol}'
    if needle in sup:
        sup=sup.replace(needle,'\\section{Initial KG-directed observation collection}\n\nThis section documents the retained first Luna collection. Section~S9 gives the repeated-observer comparison used in the revised main results.')
    sup=re.sub(r' The manuscript-update criterion was fixed before collection:.*?engineering-validated benefit threshold\.',
               '',sup,flags=re.S)
    extra=(r'''\section{Repeated observers and role-only assessment}

The revised primary comparison retains three response collections per observer.
The historical Luna and Sol records form collection 1 for their respective
contract-identical KG arms; two further passes per arm and three holistic
role-only Luna passes add 14,588 requests. The gallery, queries, exemplar
identities, dimensional evidence, relevance grades, pool membership, four KG
clauses per role and scoring weights are retained. There are 18,756 role--CAD
observations across all nine collections. Local seeds 29 and 43 shuffle the
two new KG passes; seeds 17, 29 and 43 shuffle holistic passes. A round-robin
queue interleaves the seven fresh collections. These seeds are not API sampling
controls. The original KG collection order is retained without a seed claim.

\subsection{Direct role contract and serving controls}

The holistic Luna input supplies the plain purpose of each of the eleven requested
roles and the original six-view image. It omits authored geometric clauses,
triples, component-binding instructions, class probabilities, reference identities,
rankings, bounds and relevance labels. Generic observation guidance covers
uncertainty, simplified geometry and uncapped display clips. One visual support
score $J\in\{0,1/8,\ldots,1\}$ shares $H$'s possible values. Zero indicates visible
incompatibility, one strong visible support and one half unresolved support.
Intermediate values express graded support. Cited views and a short geometric
description accompany the score. The full contract is supplied with the data.

The primary control uses $(3s+J)/4$ in the same eligible pool as the proposed
$(3s+H)/4$. Score-only $J+C$ and $H+C$ additionally order that pool by visual
support alone. They omit $G,A,F$ from scoring but inherit candidate access from
the same initial branches. They are therefore screened-pool controls. Invalid
responses retain neutral support of one half; all terminal records enter the
analysis once. There is no best-of-three selection or response repair.

\subsection{Repeated analysis}

Each collection generates a full top-five evaluation. The six conditions and four
exemplars first average within each role; eleven roles then receive equal weight.
Displayed means and sample SDs use the three aggregate collection metrics, with
SD denominator two. Statistical contrasts first average the three observations
within each role and use eleven paired role effects. Exact two-sided tests enumerate
2,048 sign patterns; one Holm correction covers the four comparisons in
Table~\ref{tab:repeated_tests}. Bootstrap intervals use 20,000 role resamples,
seed 20261023. The original component and initial-observer test families remain
historical analyses. This sequential extension is conditional on the existing
AI-graded collection.

\begin{table}[htbp]
\centering\small
\resizebox{\textwidth}{!}{\input{tables/repeated_paired_tests_v7.tex}}
\caption{Four primary contrasts based on three-collection role means. Intervals are unadjusted 95\% role-bootstrap intervals. Raw $p$ values use exact two-sided sign flips; Holm correction covers these four contrasts. W/T/L counts role wins, ties and losses. The role remains the inferential unit.}
\label{tab:repeated_tests}
\end{table}

The representation checkpoint is fixed throughout. Three independently ordered
local serving passes produce identical ranking arrays and zero execution SD.
Original surface sampling and training use seed 42; neither is repeated in this
study. The supplied local audit distinguishes those stochastic training stages
from deterministic retrieval. Thus collection SD quantifies repeated API
acquisition under the fixed representation and query panel.

'''+f"Of the {completion['new_calls']:,} new terminal calls, {completion['status_counts'].get('completed',0):,} return valid structured outputs and {sum(v for key,v in completion['validity_counts'].items() if key.endswith('/fallback')):,} observation jobs use neutral fallback. "+
      f"The accounting estimate is US\\${completion['estimated_new_cost_usd']:.4f}, based on reported token usage and conservative reservations for calls without usage. "+
      'Request, schema, response, image and source hashes are retained, along with fallback accounting and per-collection metrics. An isolated saved-response replay and an independent scalar calculation verify the published values without further API calls.\n')
    sup=sup.replace('\\end{document}',extra+'\n\\end{document}')
    (PAPER/'supplement.tex').write_text(sup,encoding='utf8')
    return dict(title=TITLE,abstract_words=len(abstract.split()),primary_summary=summary,completion=completion)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--integrate',action='store_true');a=p.parse_args()
    if a.integrate:
        report=integrate();print(json.dumps({'title':report['title'],'abstract_words':report['abstract_words']},indent=2))
    else:
        prepare_sources()
        print('Revision6 archived. Revised prose will be integrated only from complete audited outcomes.')
