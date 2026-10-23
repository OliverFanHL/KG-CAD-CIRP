"""Synthetic workload and native memory instrumentation contracts."""
import os
from scripts.benchmark_audited_scalability import make_workload,memory,stratified_order
from cadkg.io import object_hash

def fixture():
    protocol={'families':[{'family_id':'first','function_id':'f1'},{'family_id':'second','function_id':'f2'}],
      'queries':[{'scenario_id':'first','reference_part_id':'r1'},{'scenario_id':'second','reference_part_id':'r2'}]}
    knowledge={'functions':[{'function_id':'f1'},{'function_id':'f2'}]}
    catalog={'parts':[{'split':'val','measurements':{a:{'value':float(i),'status':'computed'} for a in ['aabb_large_mm','aabb_mid_mm','aabb_small_mm']}} for i in range(1,21)]}
    return protocol,knowledge,catalog

def test_workload_reproducibility_and_uniqueness():
    p,k,c=fixture();a,probes=make_workload(p,k,c,100,17);b,_=make_workload(p,k,c,100,17)
    assert a==b and len(probes)==6 and len(a)==100
    signatures={object_hash({x:q[x] for x in ['function_ids','reference_part_id','constraints','category_ids']}) for q in a}
    assert len(signatures)==100

def test_stratified_nested_order():
    parts={'a':{'label':'one'},'b':{'label':'two'},'c':{'label':'one'},'d':{'label':'two'}}
    order=stratified_order(parts,list(parts),17)
    assert len(set(order))==4 and parts[order[0]]['label']!=parts[order[1]]['label']
    assert order==stratified_order(parts,list(parts),17)

def test_windows_memory_handle():
    data=memory()
    if os.name=='nt':assert data['WorkingSetSize']>0 and data['PeakWorkingSetSize']>=data['WorkingSetSize']
