"""Neutral development/test geometry views for feature-based annotation review."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from cadkg.io import read_json,read_csv,write_json,write_csv,sha256

ROOT=Path('outputs/feature_review_v5')
CATALOG=Path('outputs/intent_batch2_matrix_reproduction/bundle/catalog.json')

def select_pilot():
    ROOT.mkdir(exist_ok=True)
    catalog=read_json(CATALOG);testhash={p['file_sha256'] for p in catalog['parts'] if p['split']=='test'}
    groups=defaultdict(list)
    for p in catalog['parts']:
        if p['split']=='val' and p['file_sha256'] not in testhash:groups[p['label']].append(p)
    rng=np.random.default_rng(20261026);selected=[]
    for label,rows in sorted(groups.items()):
        rng.shuffle(rows);selected.extend(rows[:8 if label=='spring' else 4])
    rng.shuffle(selected)
    items=[dict(review_id=f'D{i:03}',part_id=p['part_id'],source_sha256=p['file_sha256'],source_path=p['absolute_path'],split='val',image=str(ROOT/'pilot'/f'D{i:03}.png')) for i,p in enumerate(selected,1)]
    write_csv(ROOT/'pilot_manifest.csv',items)
    write_json(ROOT/'pilot_freeze.json',dict(seed=20261026,selection='Four validation CADs per source class, eight springs; categories balance selection only.',test_source_hash_overlap=0,items=len(items),catalog_sha256=sha256(CATALOG),prior_annotations_consumed=False,ranking_outcomes_consumed=False,manifest_sha256=sha256(ROOT/'pilot_manifest.csv')))
    print(dict(pilot=len(items)))

def vtk_poly(triangles):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
    xyz=np.asarray(triangles.reshape(-1,3),dtype=np.float32)
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(xyz,deep=True))
    cells=vtk.vtkCellArray();cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,len(xyz)+1,3,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(np.arange(len(xyz),dtype=np.int64),deep=True))
    mesh=vtk.vtkPolyData();mesh.SetPoints(points);mesh.SetPolys(cells)
    normal=vtk.vtkPolyDataNormals();normal.SetInputData(mesh);normal.SplittingOff();normal.ConsistencyOn();normal.AutoOrientNormalsOn();normal.ComputePointNormalsOn();normal.Update()
    return normal.GetOutput()

def render(item):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    target=Path(item['image']);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.with_suffix('.json').exists() and read_json(target.with_suffix('.json')).get('render_version')==2:return item['review_id']
    if item.get('mesh_path'):
        triangles=np.load(item['mesh_path'])['triangles']
    else:
        from cadkg.geometry import surface_mesh
        if sha256(item['source_path'])!=item['source_sha256']:raise ValueError('Pilot source mismatch')
        triangles,_=surface_mesh(item['source_path'],.03)
    points=triangles.reshape(-1,3);dims=np.ptp(points,axis=0)
    # Align the thinnest disk axis, otherwise the longest body axis, to Z.
    order=np.argsort(dims);z=int(order[0] if dims[order[0]]<.5*dims[order[2]] and dims[order[1]]>.75*dims[order[2]] else order[2])
    permutation=[a for a in range(3) if a!=z]+[z]
    triangles=triangles[:,:,permutation];center=(triangles.reshape(-1,3).min(0)+triangles.reshape(-1,3).max(0))/2
    triangles=triangles-center
    mesh=vtk_poly(triangles);radius=max(float(np.max(np.linalg.norm(triangles.reshape(-1,3),axis=1))),1e-6)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1500,1000);window.SetMultiSamples(0)
    definitions=[('oblique',(3,-4,3),None),('opposite',(-3,4,2),None),('profile',(0,-1,.05),None),('axial',(0,0,1),None),('cut_X',(3,-4,2),(1,0,0)),('cut_Y',(-3,4,2),(0,1,0))]
    holders=[]
    for i,(label,direction,clipnormal) in enumerate(definitions):
        renderer=vtk.vtkRenderer();renderer.SetBackground(1,1,1);renderer.SetViewport((i%3)/3,1-(i//3+1)/2,(i%3+1)/3,1-(i//3)/2)
        mapper=vtk.vtkPolyDataMapper()
        if clipnormal:
            plane=vtk.vtkPlane();plane.SetOrigin(0,0,0);plane.SetNormal(*clipnormal)
            clip=vtk.vtkClipPolyData();clip.SetInputData(mesh);clip.SetClipFunction(plane);clip.Update();mapper.SetInputConnection(clip.GetOutputPort());holders.extend([plane,clip])
        else:mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff();actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetColor(.31,.58,.70);actor.GetProperty().SetInterpolationToPhong();actor.GetProperty().SetAmbient(.3);actor.GetProperty().SetDiffuse(.7);actor.GetProperty().EdgeVisibilityOff()
        renderer.AddActor(actor);camera=renderer.GetActiveCamera();d=np.array(direction,dtype=float);d/=np.linalg.norm(d);camera.SetPosition(*(d*radius*4));camera.SetFocalPoint(0,0,0);camera.SetViewUp(0,1,0) if label=='axial' else camera.SetViewUp(0,0,1);camera.ParallelProjectionOn();renderer.ResetCamera();camera.Zoom(.9);renderer.ResetCameraClippingRange();window.AddRenderer(renderer)
    window.Render();grab=vtk.vtkWindowToImageFilter();grab.SetInput(window);grab.ReadFrontBufferOff();grab.Update();pixels=vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(1000,1500,-1)[::-1,:,:3]
    picture=Image.fromarray(pixels);draw=ImageDraw.Draw(picture);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for i,(label,_,_) in enumerate(definitions):draw.text((i%3*500+8,i//3*500+6),item['review_id']+' / '+label,font=font,fill='black')
    picture.save(target);window.Finalize()
    write_json(target.with_suffix('.json'),dict(render_version=2,review_id=item['review_id'],part_id=item['part_id'],source_sha256=item['source_sha256'],mesh_sha256=sha256(item['mesh_path']) if item.get('mesh_path') else None,image_sha256=sha256(target),axis_permutation=permutation,views=[r[0] for r in definitions],rendering='VTK orthographic, surface normals, edges hidden; cutaways are uncapped visual clips, not new measured CAD surfaces.',triangles=len(triangles)))
    return item['review_id']

def render_manifest(path,workers=4):
    items=read_csv(path)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        tasks=[ex.submit(render,item) for item in items]
        for i,future in enumerate(as_completed(tasks),1):
            code=future.result()
            if i%20==0 or i==len(items):print(f'{code}: {i}/{len(items)}',flush=True)

def contact(items,path):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    for start in range(0,len(items),6):
        target=Path(path)/f'page_{start//6+1:03}.png';target.parent.mkdir(parents=True,exist_ok=True)
        page=Image.new('RGB',(1800,1800),'white')
        for j,item in enumerate(items[start:start+6]):
            im=Image.open(item['image']).convert('RGB');im.thumbnail((900,570));x,y=j%2*900,j//2*600
            page.paste(im,(x,y+30));ImageDraw.Draw(page).text((x+8,y),item['review_id'],font=font,fill='black')
        page.save(target)
    print('Contact pages',len(items),(len(items)+5)//6)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pilot','render','contact']);p.add_argument('--manifest');p.add_argument('--workers',type=int,default=4);p.add_argument('--output');a=p.parse_args()
    if a.action=='pilot':select_pilot()
    elif a.action=='render':render_manifest(a.manifest,a.workers)
    else:contact(read_csv(a.manifest),a.output)
