#!/usr/bin/env python3
"""
Build an interactive 'smart assembler' (assembler_smart.html) from an RDP cache.
- same-line candidates via full row profile (not the fragile edge registration)
- per-tile OCR text as a hint + a filter box that searches ALL tiles by text
Requires: rdp_cache_reconstruct_v8.py (same folder), Pillow, numpy, and OCR
(tesserocr  or  pytesseract + tesseract-ocr).

Gebruik:
  python asm_build.py <tiles_dir_with_bmps>  [-o assembler_smart.html]
  (tiles_dir = the 'tiles/' folder produced by v8 or bmc-tools)
"""
import sys, glob, io, base64, json, argparse, importlib.util, re
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location("v8","rdp_cache_reconstruct.py")
v8=importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)

# ---- OCR backend: tesserocr (fast) or pytesseract ----
def make_ocr():
    try:
        from tesserocr import PyTessBaseAPI, PSM
        for p in ["/usr/share/tesseract-ocr/5/tessdata","/usr/share/tesseract-ocr/4.00/tessdata","/usr/share/tessdata"]:
            try:
                api=PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, path=p)
                def f(img): api.SetImage(img); return api.GetUTF8Text()
                return f
            except Exception: continue
    except Exception: pass
    try:
        import pytesseract
        def f(img): return pytesseract.image_to_string(img, config="--psm 6")
        return f
    except Exception:
        return None

def gray(arr2d):
    g=arr2d.copy()
    if g.mean()<128: g=255-g
    rng=float(g.max()-g.min()); g=(g-g.min())/(rng+1e-3)*255
    return Image.fromarray(g.astype(np.uint8))

_HTML2=r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>RDP Terminal Assembler (smart)</title>
<style>
:root{--bg:#0e0e12;--panel:#1a1a22;--line:#2a2a36;--txt:#d8d8e0;--accent:#ffd24a;--accent2:#4aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:13px system-ui,Segoe UI,Arial}
header{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:var(--panel);color:var(--txt);border:1px solid var(--line);padding:5px 10px;border-radius:6px;cursor:pointer}
button:hover{border-color:var(--accent2)}
#main{display:flex;height:calc(100vh - 46px)}
#stage{flex:1;overflow:auto;position:relative;background:linear-gradient(90deg,#15151c 1px,transparent 1px) 0 0/64px 64px,linear-gradient(#15151c 1px,transparent 1px) 0 0/64px 64px,#0a0a0e}
#canvas{position:relative;transform-origin:0 0}
.cell{position:absolute;width:64px;height:64px}.cell img{width:64px;height:64px;display:block;image-rendering:pixelated}
.cell.sel{outline:2px solid var(--accent);outline-offset:-2px;z-index:5}
.plus{position:absolute;width:64px;height:64px;border:1px dashed #3a3a4a;color:#5a5a6a;display:flex;align-items:center;justify-content:center;font-size:22px;cursor:pointer;background:rgba(74,163,255,.04)}
.plus:hover{border-color:var(--accent2);color:var(--accent2);background:rgba(74,163,255,.12)}
#side{width:340px;border-left:1px solid var(--line);overflow:auto;padding:10px;background:var(--panel)}
#side h3{margin:6px 0;font-size:12px;color:#9a9aa8;text-transform:uppercase;letter-spacing:.5px}
.cands{display:flex;flex-direction:column;gap:5px;margin-bottom:8px}
.crow{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:5px;cursor:pointer;padding:3px}
.crow:hover{border-color:var(--accent);background:#ffd24a0d}
.crow img{width:80px;height:80px;image-rendering:pixelated;flex:none}
.crow .meta{font-size:12px;color:#bfbfca;overflow:hidden}
.crow .meta b{color:var(--accent2)}
.crow .ocr{font-family:ui-monospace,Consolas,monospace;color:#8fd18f;white-space:pre-wrap;word-break:break-all}
#palette{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:50}
#palbox{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;width:84vw;height:84vh;display:flex;flex-direction:column}
#palfilter{margin-bottom:8px;padding:6px;background:#0e0e12;border:1px solid var(--line);border-radius:6px;color:var(--txt);width:260px}
#palgrid{display:flex;flex-wrap:wrap;gap:3px;overflow:auto;align-content:flex-start}
.pcell{display:flex;flex-direction:column;align-items:center;width:64px}
.pcell img{width:64px;height:64px;image-rendering:pixelated;border:1px solid var(--line);cursor:pointer}
.pcell img:hover{border-color:var(--accent)}.pcell span{font:9px monospace;color:#7a9;max-width:64px;overflow:hidden;white-space:nowrap}
.hint{color:#9a9aa8;font-size:12px}
</style></head><body>
<header><b style="color:var(--accent)">Terminal Assembler — smart</b>
<button onclick="startPick()">+ Start tile</button><button onclick="delSel()">Delete selection</button>
<button onclick="clearAll()">Clear</button><span style="flex:1"></span>
<label>zoom <input type="range" min="0.5" max="2" step="0.1" value="1" oninput="setZoom(this.value)"></label>
<button onclick="exportPNG()">Export PNG</button></header>
<div id="main"><div id="stage"><div id="canvas"></div></div>
<div id="side"><div class="hint">Click a <b>+</b> cell to get same-line candidates, ranked by whether the text continues (OCR across the seam). The green line is the recognized text — pick the one that fits. Tip: type in the filter what you expect (e.g. "ndows") to search all tiles.</div><div id="panel"></div></div></div>
<div id="palette"><div id="palbox"><h3>Choose a start tile</h3>
<input id="palfilter" placeholder="filter by recognized text… (e.g. Mailbox)" oninput="fillPal(this.value)">
<div id="palgrid"></div><div style="text-align:right;margin-top:6px"><button onclick="closePal()">Close</button></div></div></div>
<script>
const DATA=/*__DATA__*/;const T=DATA.b64,TX=DATA.txt,R=DATA.R,L=DATA.L,U=DATA.U,D=DATA.D;
const placed=new Map();let sel=null,zoom=1;const key=(x,y)=>x+","+y;
function setZoom(z){zoom=+z;document.getElementById('canvas').style.transform=`scale(${zoom})`;}
function combined(x,y){
  const map=[[-1,0,R],[1,0,L],[0,-1,D],[0,1,U]];const sc={},pr={};
  for(const[dx,dy,LST]of map){const k=key(x+dx,y+dy);
    if(placed.has(k)){const ni=placed.get(k);for(const[j,s]of(LST[ni]||[])){sc[j]=(sc[j]||0)+s;pr[j]=(pr[j]||0)+1;}}}
  let a=Object.keys(sc).map(j=>({j:+j,s:sc[j],p:pr[j]})).filter(o=>![...placed.values()].includes(o.j));
  a.sort((x,y)=>(y.p-x.p)||(y.s-x.s));return a.slice(0,60);
}
function render(){const cv=document.getElementById('canvas');cv.innerHTML='';
  if(placed.size===0){cv.style.width=cv.style.height='64px';return;}
  let xs=[...placed.keys()].map(k=>+k.split(',')[0]),ys=[...placed.keys()].map(k=>+k.split(',')[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),pad=1;
  cv.style.width=(maxx-minx+1+2*pad)*64+'px';cv.style.height=(maxy-miny+1+2*pad)*64+'px';
  const ox=minx-pad,oy=miny-pad,empt=new Set();
  for(const k of placed.keys()){const[x,y]=k.split(',').map(Number);for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const nk=key(x+dx,y+dy);if(!placed.has(nk))empt.add(nk);}}
  for(const k of empt){const[x,y]=k.split(',').map(Number);const d=document.createElement('div');d.className='plus';d.textContent='+';
    d.style.left=(x-ox)*64+'px';d.style.top=(y-oy)*64+'px';d.onclick=()=>openTarget(x,y);cv.appendChild(d);}
  for(const[k,ti]of placed){const[x,y]=k.split(',').map(Number);const d=document.createElement('div');d.className='cell'+(sel===k?' sel':'');
    d.style.left=(x-ox)*64+'px';d.style.top=(y-oy)*64+'px';const im=document.createElement('img');im.src='data:image/png;base64,'+T[ti];d.appendChild(im);
    d.onclick=()=>{sel=k;render();selPanel();};cv.appendChild(d);}}
let curTarget=null,curCands=[];
function openTarget(x,y){sel=null;curTarget=[x,y];curCands=combined(x,y);const p=document.getElementById('panel');
  p.innerHTML=`<h3>Candidates for (${x},${y})</h3>
   <input id="cfilter" placeholder="filter by text… (e.g. type ndows)" style="width:100%;padding:6px;margin-bottom:6px;background:#0e0e12;border:1px solid var(--line);border-radius:6px;color:var(--txt)" oninput="drawCands(this.value)">
   <small class="hint" id="ccount"></small><div class="cands" id="cl"></div>`;
  document.getElementById('cfilter').focus();drawCands('');render();}
function drawCands(f){const[x,y]=curTarget;f=(f||'').toLowerCase();
  let cs;
  if(f){ cs=[]; const used=new Set([...placed.values()]);
    for(let j=0;j<TX.length;j++){ if(used.has(j))continue; if((TX[j]||'').toLowerCase().includes(f)) cs.push({j:j,p:0}); }
    cs=cs.slice(0,80);
    document.getElementById('ccount').textContent=`${cs.length} tiles containing "${f}" (searched all tiles)`;
  } else { cs=curCands;
    document.getElementById('ccount').textContent=`${cs.length} same-line suggestions — or type text to search all tiles`; }
  const cl=document.getElementById('cl');cl.innerHTML='';
  if(!cs.length){cl.innerHTML='<small class="hint">No match. Clear the filter or place via + Start tile.</small>';return;}
  cs.forEach(o=>{const c=document.createElement('div');c.className='crow';
    c.innerHTML=`<img src="data:image/png;base64,${T[o.j]}"><div class="meta"><b>${o.p}× neighbor</b><div class="ocr">${(TX[o.j]||'').replace(/</g,'&lt;')}</div></div>`;
    c.onclick=()=>{placed.set(key(x,y),o.j);render();};cl.appendChild(c);});}
function selPanel(){const p=document.getElementById('panel');if(!sel){p.innerHTML='';return;}const[x,y]=sel.split(',').map(Number);
  p.innerHTML=`<h3>Tile (${x},${y})</h3><div class="ocr" style="color:#8fd18f;margin-bottom:6px">${(TX[placed.get(sel)]||'').replace(/</g,'&lt;')}</div>
   <div style="display:flex;gap:6px;flex-wrap:wrap"><button onclick="openTarget(${x+1},${y})">→ right</button><button onclick="openTarget(${x-1},${y})">← left</button>
   <button onclick="openTarget(${x},${y-1})">↑ up</button><button onclick="openTarget(${x},${y+1})">↓ down</button><button onclick="delSel()">🗑</button></div>`;}
function startPick(){fillPal('');document.getElementById('palette').style.display='flex';}
function fillPal(f){const g=document.getElementById('palgrid');g.innerHTML='';f=f.toLowerCase();
  T.forEach((b,i)=>{if(f&&!(TX[i]||'').toLowerCase().includes(f))return;const d=document.createElement('div');d.className='pcell';
    d.innerHTML=`<img src="data:image/png;base64,${b}"><span>${(TX[i]||'').slice(0,9)}</span>`;
    d.querySelector('img').onclick=()=>{if(placed.size===0)placed.set('0,0',i);else{let xs=[...placed.keys()].map(k=>+k.split(',')[0]);placed.set(key(Math.max(...xs)+2,0),i);}closePal();render();};g.appendChild(d);});}
function closePal(){document.getElementById('palette').style.display='none';}
function delSel(){if(sel){placed.delete(sel);sel=null;document.getElementById('panel').innerHTML='';render();}}
function clearAll(){placed.clear();sel=null;document.getElementById('panel').innerHTML='';render();}
function exportPNG(){if(!placed.size)return;let xs=[...placed.keys()].map(k=>+k.split(',')[0]),ys=[...placed.keys()].map(k=>+k.split(',')[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);const W=(maxx-minx+1)*64,H=(maxy-miny+1)*64;
  const cv=document.createElement('canvas');cv.width=W;cv.height=H;const ctx=cv.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);let pend=placed.size;
  for(const[k,ti]of placed){const[x,y]=k.split(',').map(Number);const im=new Image();im.onload=()=>{ctx.drawImage(im,(x-minx)*64,(y-miny)*64);if(--pend===0){const a=document.createElement('a');a.download='terminal_reconstruction.png';a.href=cv.toDataURL('image/png');a.click();}};im.src='data:image/png;base64,'+T[ti];}}
render();
</script></body></html>"""

def build(tiles_dir, out_path, pool=60):
    bmps=sorted(p for p in glob.glob(f"{tiles_dir}/*.bmp")+glob.glob(f"{tiles_dir}/*.png") if "collage" not in p)
    tiles,sizes=v8.load_tiles(bmps)
    sub=[n for n in tiles if sizes[n]==(64,64)
         and 4<v8._term_feats(tiles[n])['ink']<140
         and v8._term_feats(tiles[n])['bg'].mean()<75
         and v8._term_feats(tiles[n])['sharp']>0.018]
    sub=v8.scroll_dedup(sub,tiles,sizes); N=len(sub)
    print(f"{N} terminal tiles (after scroll dedup)")
    A=np.stack([tiles[n] for n in sub]).astype(np.float32)
    bg=np.stack([v8._term_bg(tiles[n]) for n in sub]); BL=(bg[:,2]-bg[:,0]).astype(np.float32)
    ink=np.abs(A-bg[:,None,None,:]).sum(3); FRP=ink.mean(2)
    def zn(M):
        m=M-M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s<1e-3]=1; return m/s
    FRPz=zn(FRP)
    CAND,_=v8._jigsaw_candidates(sub,tiles,sizes,k=16,h_gate=0.5,v_gate=0.2)
    def dirarr(d):
        out=[[] for _ in range(N)]
        for i,lst in CAND[d].items(): out[i]=[[int(j),round(float(s),1)] for s,j in lst]
        return out
    U=dirarr("U"); D=dirarr("D")
    ocr=make_ocr()
    TXT=[]; b64=[]
    for k in range(N):
        if ocr:
            try: TXT.append(" ".join(ocr(gray(A[k].mean(2)).resize((192,192),Image.LANCZOS)).split())[:40])
            except Exception: TXT.append("")
        else: TXT.append("")
        buf=io.BytesIO(); Image.fromarray(A[k].astype(np.uint8),"RGB").save(buf,"PNG"); b64.append(base64.b64encode(buf.getvalue()).decode())
        if k%200==0: print(f"  ocr {k}/{N}")
    if not ocr: print("WARNING: no OCR available — text hints/filter will be empty. Install tesserocr or pytesseract.")
    R=[[] for _ in range(N)]
    for i in range(N):
        corr=(FRPz[i]@FRPz.T)/64.0; bld=np.abs(BL[i]-BL)
        R[i]=[[int(j),round(float(corr[j]),3)] for j in np.argsort(-corr) if j!=i and bld[j]<=10][:pool]
    L=[[] for _ in range(N)]
    for i in range(N):
        for j,s in R[i]: L[j].append([i,s])
    for i in range(N): L[i]=sorted(L[i],key=lambda x:-x[1])[:pool]
    data=dict(b64=b64,txt=TXT,R=R,L=L,U=U,D=D)
    open(out_path,"w",encoding="utf-8").write(_HTML2.replace("/*__DATA__*/",json.dumps(data)))
    print(f"-> {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("tiles_dir"); ap.add_argument("-o","--out",default="assembler_smart.html")
    ap.add_argument("--pool",type=int,default=60)
    a=ap.parse_args(); build(a.tiles_dir,a.out,a.pool)