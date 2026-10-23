'use strict';
const $ = id => document.getElementById(id);
let data, state, reviewer, mode='parts', selected, selectionList=[], currentRecord={}, textDecision='', saveTimer, pending=false, editRevision=0, saving=null;
const keyOf = item => item.family_id+'|'+item.part_id;
async function api(url, payload) {
  const r=await fetch(url,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});
  const value=await r.json(); if(!r.ok) throw Error(value.error||'Request failed'); return value;
}
function status(id,message,error=false){$(id).textContent=message;$(id).classList.toggle('error',error);}
function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
function activeDecision(container,attribute,value){$(container).querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b.dataset[attribute]===value));}
function ready(){if(!reviewer||!$('attest').checked)throw Error('Open your reviewer workspace and confirm your own human review first.');}
async function start(){
  try{if(pending&&!await savePart())return;if(!$('attest').checked)throw Error('Confirm these will be your own human review decisions.');
    const id=$('reviewer').value.trim(); const result=await api('/api/bootstrap?reviewer='+encodeURIComponent(id));
    reviewer=id;data=result;state=result.state;selected=null;sessionStorage.setItem('reviewer',id);
    status('identity-status',state.source==='validation_fixture'?'Synthetic UI validation only — these are not expert annotations.':'Workspace ready.');renderList();
  }catch(e){status('identity-status',e.message,true);}
}
function renderList(){
  if(!data)return;
  const fid=$('family').value, filter=$('queue').value;
  const items=mode==='parts'?data.parts.filter(p=>p.family_id===fid):mode==='tasks'?data.tasks:data.knowledge;
  selectionList=items.filter(p=>{
    if(mode!=='parts')return true;
    const r=state?.judgments[keyOf(p)];return filter==='all'||filter==='unreviewed'&&!r||filter==='cannot_assess'&&r?.decision==='cannot_assess'||filter==='reviewed'&&!!r;
  });
  $('items').replaceChildren();
  selectionList.forEach((p,i)=>{
    const key=mode==='parts'?keyOf(p):mode==='tasks'?p.family_id:p.key;
    const record=state?.[mode==='parts'?'judgments':mode==='tasks'?'task_reviews':'knowledge_reviews'][key];
    const button=element('button',mode==='parts'?p.inspection_id:mode==='tasks'?p.label:p.class_id+' → '+p.function_label);
    if(record)button.append(element('span',record.decision.replaceAll('_',' '),'badge'));
    button.classList.toggle('selected',selected===key);button.onclick=()=>select(i);$('items').append(button);
  });
  const n=state?Object.keys(state.judgments).length:0;
  const unknown=state?Object.values(state.judgments).filter(r=>r.decision==='cannot_assess').length:0;
  $('progress').textContent=reviewer?`${n} / ${data.parts.length} CAD decisions · ${unknown} to revisit`:'Choose a reviewer ID to begin.';
  if(!selectionList.length){$('part-panel').hidden=true;$('text-panel').hidden=true;$('intro').hidden=false;$('intro').querySelector('h1').textContent='No candidates in this queue';return;}
  if(!selectionList.some(p=>(mode==='parts'?keyOf(p):mode==='tasks'?p.family_id:p.key)===selected))select(0);
}
async function select(index){
  if(pending&&!await savePart())return;
  const item=selectionList[index];if(!item)return;
  selected=mode==='parts'?keyOf(item):mode==='tasks'?item.family_id:item.key;
  $('intro').hidden=true;$('part-panel').hidden=mode!=='parts';$('text-panel').hidden=mode==='parts';
  if(mode==='parts')showPart(item);else showText(item);
  renderList();
}
function showPart(item){
  const t=data.tasks.find(t=>t.family_id===item.family_id);
  currentRecord=structuredClone(state?.judgments[selected]||{family_id:item.family_id,part_id:item.part_id,features:{},notes:'',point:null});pending=false;
  $('cad-id').textContent='Candidate '+item.inspection_id;$('task-title').textContent=t.label;$('request').textContent=t.request;
  $('rules').replaceChildren();for(const [k,v]of Object.entries(t.rules))$('rules').append(element('p',k.toUpperCase()+': '+v));
  $('features').replaceChildren();t.features.forEach(f=>{
    const wrap=element('div',undefined,'feature'),label=element('label',f.label),select=element('select');select.setAttribute('aria-label',f.label);
    [['','Choose an observation'],['present','Visible'],['absent','Visibly absent'],['unclear','Unclear / concealed']].forEach(([v,l])=>{const o=element('option',l);o.value=v;select.append(o);});
    select.value=currentRecord.features[f.id]||'';select.onchange=()=>{if(select.value)currentRecord.features[f.id]=select.value;else delete currentRecord.features[f.id];queueSave();};
    label.append(select);wrap.append(label);$('features').append(wrap);
  });
  activeDecision('decisions','decision',currentRecord.decision);$('reason').value=currentRecord.reason||'';$('notes').value=currentRecord.notes||'';
  $('views').src='/api/evidence/'+item.inspection_id+'.png';$('step-link').href='/api/evidence/'+item.inspection_id+'.step';
  status('save-status',currentRecord.reviewed_at?'Saved review loaded.':'Choose visible evidence and a decision.');
  viewer.load(item.inspection_id,currentRecord.point);
}
function showText(item){
  const kind=mode==='tasks'?'task_reviews':'knowledge_reviews',record=state?.[kind][selected];textDecision=record?.decision||'';
  $('text-kind').textContent=mode==='tasks'?'Task definition review':'Generic knowledge review';$('text-title').textContent=mode==='tasks'?item.label:item.class_id+' → '+item.function_label;
  const body=$('text-body');body.replaceChildren();
  const fields=mode==='tasks'?{Request:item.request,...item.rules}:{Scope:item.scope,Limitations:item.caveat,Status:'Generic class prior. Not a verified function of every CAD in this category. Inclusion weights are not calibrated confidence.'};
  for(const [label,value]of Object.entries(fields)){const div=element('div',undefined,'prior-field');div.append(element('strong',label+': '),element('span',value));body.append(div);}
  if(mode==='knowledge'){const a=element('a','Open supporting source');a.href=item.evidence;a.target='_blank';a.rel='noopener noreferrer';body.append(a);}
  $('text-notes').value=record?.notes||'';activeDecision('text-decisions','review',textDecision);status('text-status',record?'Saved review loaded.':'Changes are recorded for a new protocol; this frozen experiment is not edited.');
}
function queueSave(){editRevision++;pending=true;status('save-status','Unsaved changes');clearTimeout(saveTimer);saveTimer=setTimeout(()=>{if(currentRecord.decision&&currentRecord.reason)savePart();},450);}
async function savePart(){
  clearTimeout(saveTimer);if(saving){await saving;if(pending)return savePart();return true;}if(!pending)return true;
  try{ready();currentRecord.notes=$('notes').value.trim();currentRecord.reason=$('reason').value;
    const key=selected, revision=editRevision;
    saving=api('/api/save',{reviewer,human_review_attested:true,kind:'judgments',key,record:structuredClone(currentRecord)});
    const result=await saving;
    state.judgments[key]=result.record;
    if(selected===key&&revision===editRevision){currentRecord=result.record;pending=false;status('save-status','Saved automatically.');}
    // Do not re-filter or navigate while the reviewer is editing the current candidate.
    $('progress').textContent=`${Object.keys(state.judgments).length} / ${data.parts.length} CAD decisions`;
    return true;
  }catch(e){status('save-status',e.message,true);return false;}finally{saving=null;}
}
async function move(delta){if(delta>0&&!currentRecord.decision){status('save-status','Choose a decision before saving, or select another candidate to skip it.',true);return;}if(!await savePart())return;const i=selectionList.findIndex(p=>keyOf(p)===selected);select(Math.min(Math.max(i+delta,0),selectionList.length-1));}
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>{currentRecord.decision=b.dataset.decision;activeDecision('decisions','decision',currentRecord.decision);queueSave();});
document.querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>{textDecision=b.dataset.review;activeDecision('text-decisions','review',textDecision);});
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=async()=>{if(pending&&!await savePart())return;mode=b.dataset.mode;selected=null;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));$('family').disabled=mode!=='parts';$('queue').disabled=mode!=='parts';renderList();});
$('start').onclick=start;$('reason').onchange=queueSave;$('notes').oninput=queueSave;
$('save-next').onclick=()=>move(1);$('previous').onclick=()=>move(-1);
$('family').onchange=async()=>{if(pending&&!await savePart())return;selected=null;renderList();};$('queue').onchange=async()=>{if(pending&&!await savePart())return;selected=null;renderList();};
$('text-save').onclick=async()=>{try{ready();const kind=mode==='tasks'?'task_reviews':'knowledge_reviews';const r=await api('/api/save',{reviewer,human_review_attested:true,kind,key:selected,record:{decision:textDecision,notes:$('text-notes').value}});state[kind][selected]=r.record;status('text-status','Saved.');renderList();}catch(e){status('text-status',e.message,true);}};
$('export').onclick=async()=>{try{ready();if(pending&&!await savePart())return;const a=element('a');a.href='/api/export?reviewer='+encodeURIComponent(reviewer);a.download=reviewer+'_review.json';document.body.append(a);a.click();a.remove();}catch(e){status('identity-status',e.message,true);}};
window.addEventListener('beforeunload',e=>{if(pending){e.preventDefault();e.returnValue='';}});

// Offline WebGL viewer: original source triangles, no mesh decimation or remote libraries.
class Viewer{
  constructor(canvas){
    this.canvas=canvas;this.gl=canvas.getContext('webgl',{antialias:true});this.ticket=0;this.point=null;this.reset();
    if(!this.gl){status('viewer-status','3D unavailable: use the three views or download STEP.',true);return;}
    const gl=this.gl;const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    this.program=gl.createProgram();gl.attachShader(this.program,shader(gl.VERTEX_SHADER,'attribute vec3 p;attribute vec3 n;uniform mat4 m;uniform mat3 r;varying vec3 normal;varying vec3 pos;void main(){gl_Position=m*vec4(p,1.0);gl_PointSize=12.0;normal=r*n;pos=p;}'));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 normal;varying vec3 pos;uniform int axis;uniform float cut;uniform bool marker;void main(){if(axis==0&&pos.x>cut||axis==1&&pos.y>cut||axis==2&&pos.z>cut)discard;if(marker){gl_FragColor=vec4(.9,.2,.15,1.);return;}float light=.4+.6*abs(dot(normalize(normal),normalize(vec3(.3,.5,1.))));gl_FragColor=vec4(vec3(.25,.57,.67)*light,1.);}'));
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error('Viewer shader link failed');gl.useProgram(this.program);
    this.buffer=gl.createBuffer();this.position=gl.getAttribLocation(this.program,'p');this.normal=gl.getAttribLocation(this.program,'n');
    this.um=gl.getUniformLocation(this.program,'m');this.ur=gl.getUniformLocation(this.program,'r');this.ua=gl.getUniformLocation(this.program,'axis');this.uc=gl.getUniformLocation(this.program,'cut');this.umark=gl.getUniformLocation(this.program,'marker');this.markerBuffer=gl.createBuffer();
    let drag;
    canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};
    canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};if(e.shiftKey){this.pan[0]+=dx/canvas.clientWidth*3/this.zoom;this.pan[1]-=dy/canvas.clientHeight*3/this.zoom;}else{this.yaw+=dx*.01;this.pitch=Math.max(-1.5,Math.min(1.5,this.pitch+dy*.01));}this.draw();};
    canvas.onpointerup=()=>{drag=null;};canvas.onpointercancel=()=>{drag=null;};
    canvas.onwheel=e=>{e.preventDefault();this.zoom=Math.max(.15,Math.min(15,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();};
    canvas.ondblclick=e=>this.mark(e);new ResizeObserver(()=>this.draw()).observe(canvas);
  }
  reset(){this.yaw=.65;this.pitch=-.35;this.zoom=.85;this.pan=[0,0];$('cut-axis').value='-1';$('cut-position').value='1.1';if(this.gl)this.draw();}
  rotation(){const a=Math.cos(this.yaw),b=Math.sin(this.yaw),c=Math.cos(this.pitch),d=Math.sin(this.pitch);return[a,d*b,-c*b,0,c,d,b,-d*a,c*a];}
  transformed(p,r){return[r[0]*p[0]+r[3]*p[1]+r[6]*p[2],r[1]*p[0]+r[4]*p[1]+r[7]*p[2],r[2]*p[0]+r[5]*p[1]+r[8]*p[2]];}
  async load(code,point){
    const ticket=++this.ticket;this.triangles=null;this.count=0;this.point=point;this.reset();if(!this.gl)return;
    status('viewer-status','Loading original source-CAD surfaces…');
    try{const response=await fetch('/api/evidence/'+code+'.bin');if(!response.ok){const e=await response.json();throw Error(e.error);}
      const raw=new Float32Array(await response.arrayBuffer());if(ticket!==this.ticket)return;if(raw.length%9)throw Error('Invalid triangle data');
      this.triangles=raw;const packed=new Float32Array(raw.length*2);
      for(let i=0;i<raw.length;i+=9){const a=raw.slice(i,i+3),b=raw.slice(i+3,i+6),c=raw.slice(i+6,i+9);const u=b.map((x,j)=>x-a[j]),v=c.map((x,j)=>x-a[j]);let n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];const len=Math.hypot(...n)||1;n=n.map(x=>x/len);for(let j=0;j<3;j++){packed.set(raw.slice(i+j*3,i+j*3+3),(i/3+j)*6);packed.set(n,(i/3+j)*6+3);}}
      const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,packed,gl.STATIC_DRAW);this.count=raw.length/3;this.draw();
      status('viewer-status','Drag to rotate · wheel to zoom · Shift-drag to pan · double-click to mark a surface');
    }catch(e){if(ticket===this.ticket)status('viewer-status',e.message+' Use source views or STEP.',true);}
  }
  draw(){
    if(!this.gl||!this.program)return;const gl=this.gl,c=this.canvas;const w=Math.round(c.clientWidth*devicePixelRatio),h=Math.round(c.clientHeight*devicePixelRatio);if(!w||!h)return;if(c.width!==w||c.height!==h){c.width=w;c.height=h;}
    gl.viewport(0,0,w,h);gl.clearColor(.96,.98,.99,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(this.program);
    const r=this.rotation(),sx=this.zoom/(w/h),sy=this.zoom,m=[r[0]*sx,r[1]*sy,-r[2]/4,0,r[3]*sx,r[4]*sy,-r[5]/4,0,r[6]*sx,r[7]*sy,-r[8]/4,0,this.pan[0]*sx,this.pan[1]*sy,0,1];
    gl.uniformMatrix4fv(this.um,false,new Float32Array(m));gl.uniformMatrix3fv(this.ur,false,new Float32Array(r));gl.uniform1i(this.ua,+$('cut-axis').value);gl.uniform1f(this.uc,+$('cut-position').value);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.enableVertexAttribArray(this.position);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(this.normal);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,24,12);gl.uniform1i(this.umark,0);if(this.count)gl.drawArrays(gl.TRIANGLES,0,this.count);
    if(this.point){gl.bindBuffer(gl.ARRAY_BUFFER,this.markerBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([...this.point,0,0,1]),gl.DYNAMIC_DRAW);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,24,12);gl.uniform1i(this.umark,1);gl.depthFunc(gl.LEQUAL);gl.drawArrays(gl.POINTS,0,1);gl.depthFunc(gl.LESS);status('viewer-status',`Optional marker (${this.point.map(x=>x.toFixed(3)).join(', ')} in normalized source coordinates).`);}
  }
  mark(e){
    if(!this.triangles)return;const rect=this.canvas.getBoundingClientRect(),aspect=rect.width/rect.height;
    const x=((e.clientX-rect.left)/rect.width*2-1)*aspect/this.zoom-this.pan[0],y=(1-(e.clientY-rect.top)/rect.height*2)/this.zoom-this.pan[1];
    const r=this.rotation(),raw=this.triangles;let best=-Infinity,point=null;
    for(let i=0;i<raw.length;i+=9){const a=this.transformed(raw.slice(i,i+3),r),b=this.transformed(raw.slice(i+3,i+6),r),c=this.transformed(raw.slice(i+6,i+9),r);const den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(den)<1e-12)continue;
      const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den,v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den,w=1-u-v;if(Math.min(u,v,w)<0)continue;
      const p=[0,1,2].map(j=>u*raw[i+j]+v*raw[i+3+j]+w*raw[i+6+j]),axis=+$('cut-axis').value;if(axis>=0&&p[axis]>+$('cut-position').value)continue;const z=u*a[2]+v*b[2]+w*c[2];if(z>best){best=z;point=p;}}
    if(point){this.point=point;currentRecord.point=point;queueSave();this.draw();}
  }
}
const viewer=new Viewer($('viewer'));
$('reset').onclick=()=>viewer.reset();$('cut-axis').onchange=()=>viewer.draw();$('cut-position').oninput=()=>viewer.draw();$('clear-point').onclick=()=>{viewer.point=null;currentRecord.point=null;queueSave();viewer.draw();};
async function initialize(){try{data=await api('/api/bootstrap');for(const t of data.tasks){const o=element('option',t.label);o.value=t.family_id;$('family').append(o);}for(const [key,label]of Object.entries(data.reasons)){const o=element('option',label);o.value=key;$('reason').append(o);}$('reviewer').value=sessionStorage.getItem('reviewer')||'';renderList();}catch(e){status('identity-status',e.message,true);}}
initialize();
