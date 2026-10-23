"""
PVT Studio — interactive well schematic (JavaScript / SVG).

Draws a completion-style well schematic in the browser:
formation layers and the reservoir, sea / ground and the wellhead tree,
every casing string with its cement and shoe, a liner hanger, the tubing
with animated flow, the production packer, perforation jets, the solution
node (reference depth) and the Pr datum. Hovering anywhere along the well
reads out MD, TVD, temperature, pressure (when a traverse is supplied) and
which strings are present at that depth.

Pure Python builds a JSON spec; the HTML page renders it with plain JS +
SVG (no CDN, no extra package), shown with streamlit.components.v1.html.
"""

APP_VERSION = "1.4.5"   # must match pvt_app.py (deployment check)

import json
import numpy as np

# Common casing / tubing sizes (OD inch -> label)
_SIZE_LABELS = {30.0: '30"', 20.0: '20"', 18.625: '18⅝"', 13.375: '13⅜"',
                10.75: '10¾"', 9.625: '9⅝"', 7.625: '7⅝"', 7.0: '7"',
                5.5: '5½"', 5.0: '5"', 4.5: '4½"', 3.5: '3½"',
                2.875: '2⅞"', 2.375: '2⅜"'}

_TUBING_OD_BY_ID = ((1.995, 2.375), (2.441, 2.875), (2.992, 3.5),
                    (3.958, 4.5), (4.892, 5.5), (6.184, 7.0))


def size_label(od):
    for k, v in _SIZE_LABELS.items():
        if abs(k - od) < 0.02:
            return v
    return f'{od:.3g}"'


def tubing_od_for_id(id_in):
    """Nearest API tubing OD for an inner diameter (else ID + 0.5")."""
    for i_d, o_d in _TUBING_OD_BY_ID:
        if abs(i_d - id_in) < 0.12:
            return o_d
    return id_in + 0.5


def default_casing_program(td_md, seabed_md=0.0):
    """A typical NCS-style casing programme scaled to the well's TD (MD,
    any length unit) below the seabed / ground. Returns rows: name,
    OD (in), top MD, shoe MD, cement-top MD."""
    td, sb = float(td_md), float(seabed_md)
    f = lambda a: sb + a * (td - sb)
    rows = [
        ("Conductor", 30.0, 0.0, f(0.05)),
        ("Surface casing", 20.0, 0.0, f(0.20)),
        ("Intermediate casing", 13.375, 0.0, f(0.48)),
        ("Production casing", 9.625, 0.0, f(0.86)),
        ("Production liner", 7.0, f(0.82), td),
    ]
    out = []
    for name, od, top, shoe in rows:
        cem = (max(top, sb) if od >= 20.0
               else max(top, sb, shoe - 0.35 * (shoe - top)))
        out.append({"String": name, "OD (in)": od,
                    "Top MD": round(top, 1), "Shoe MD": round(shoe, 1),
                    "Cement top MD": round(cem, 1)})
    return out


def trajectory_points(segments_len, rathole=0.0, step=None):
    """Densely sampled (md, x, tvd) from the wellhead down, from
    segments [(length, inclination_deg_from_horizontal_up)] listed
    OUTLET -> INLET (length in the display unit). The well is extended by
    `rathole` along the last segment's direction."""
    segs = [(float(L), float(t)) for L, t in segments_len if L > 0]
    if not segs:
        segs = [(1.0, 90.0)]
    if rathole > 0:
        segs = segs + [(float(rathole), segs[-1][1])]
    total = sum(L for L, _ in segs)
    step = step or max(total / 400.0, 1e-6)
    md, x, z = [0.0], [0.0], [0.0]
    for L, th in segs:
        r = np.radians(th)
        n = max(int(np.ceil(L / step)), 1)
        for _ in range(n):
            dl = L / n
            md.append(md[-1] + dl)
            x.append(x[-1] + dl * np.cos(r))
            z.append(z[-1] + dl * np.sin(r))
    return [[round(a, 3), round(b, 3), round(c, 3)] for a, b, c in zip(md, x, z)]


def build_schematic_html(spec, height=780):
    """Self-contained HTML page for the schematic. `spec` keys:
    traj [[md,x,tvd]...], unit ('m'|'ft'), strings [{name, od, top, shoe,
    cem_top}], tubing {od, id, bottom}, packer_md, perf [top, bottom],
    node {md, tvd}, datum_tvd, water_depth, air_gap, res_top_tvd,
    res_bot_tvd, fluid ('oil'|'gas'|'water'), outlet_txt, T_top, T_bot,
    T_unit, P_profile [[md, P]...] or None, P_unit, title.
    """
    data = json.dumps(spec)
    return (_HTML.replace("__SPEC__", data)
                 .replace("__HEIGHT__", str(int(height))))


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html,body{margin:0;padding:0;background:#ffffff;font-family:Inter,Segoe UI,Arial,sans-serif}
  #wrap{position:relative;width:100%;height:__HEIGHT__px}
  svg{width:100%;height:100%;display:block}
  #tip{position:absolute;pointer-events:none;background:rgba(36,55,70,.95);color:#fff;
       font-size:11px;line-height:1.45;padding:7px 9px;border-radius:6px;display:none;
       box-shadow:0 2px 8px rgba(0,0,0,.25);white-space:nowrap;z-index:5}
  #tip b{color:#ffd166}
  .lbl{font-size:10.5px;fill:#243746}
  .lblb{font-size:10.5px;fill:#243746;font-weight:600}
  .ax{font-size:10px;fill:#56697a}
  .flow{animation:flow 1.1s linear infinite}
  @keyframes flow{from{stroke-dashoffset:0}to{stroke-dashoffset:22}}
  #ctl{position:absolute;left:4px;bottom:4px;font-size:11px;color:#56697a;z-index:4}
  #ctl label{cursor:pointer;margin-left:8px}
</style></head>
<body><div id="wrap"><div id="ctl">
<label><input type="checkbox" id="cbFlow" checked> flow</label>
<label><input type="checkbox" id="cbGeo" checked> geology</label>
<label><input type="checkbox" id="cbLbl" checked> labels</label></div>
<svg id="sv" xmlns="http://www.w3.org/2000/svg"></svg><div id="tip"></div></div>
<script>
const S = __SPEC__;
const NS = "http://www.w3.org/2000/svg";
const sv = document.getElementById("sv");
const W = sv.clientWidth || 640, H = sv.clientHeight || __HEIGHT__;
sv.setAttribute("viewBox", `0 0 ${W} ${H}`);
function el(t, a, p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);(p||sv).appendChild(e);return e;}
function grp(id){return el("g",{id:id});}

// ---------- geometry & scales ----------
const T = S.traj, U = S.unit;
const tvdMax = Math.max(...T.map(p=>p[2]), S.res_bot_tvd||0, S.datum_tvd||0);
const xMin = Math.min(0, ...T.map(p=>p[1])), xMax = Math.max(0, ...T.map(p=>p[1]));
const padT = 118, padB = 28, left = 128, right = 230;
const plotW = W-left-right, plotH = H-padT-padB;
const s = plotH / (tvdMax*1.04);          // vertical scale (px per unit)
const xSpan = (xMax-xMin);
// horizontal scale: true scale when it fits, otherwise compressed
const sx = (xSpan*s > plotW*0.80) ? plotW*0.80/Math.max(xSpan,1e-9) : s;
const compressed = sx < s*0.98;
const x0 = left + (xSpan*sx < plotW*0.35 ? plotW*0.42 : plotW*0.10) - xMin*sx;
const X = x=>x0+x*sx, Y = z=>padT+z*s;
const maxOD = Math.max(...S.strings.map(c=>c.od), S.tubing.od, 1);
const k = Math.min(46/maxOD, 3.2);              // px per inch (exaggerated radially)

function interp(md){ // position + unit tangent at MD
  if (md<=T[0][0]) md=T[0][0];
  for(let i=1;i<T.length;i++){ if(T[i][0]>=md){const a=T[i-1],b=T[i],f=(md-a[0])/Math.max(b[0]-a[0],1e-12);
      const x=a[1]+f*(b[1]-a[1]), z=a[2]+f*(b[2]-a[2]); let tx=b[1]-a[1], tz=b[2]-a[2];const n=Math.hypot(tx,tz)||1;
      return {x:x,z:z,tx:tx/n,tz:tz/n};}}
  const a=T[T.length-2],b=T[T.length-1];let tx=b[1]-a[1],tz=b[2]-a[2];const n=Math.hypot(tx,tz)||1;
  return {x:b[1],z:b[2],tx:tx/n,tz:tz/n};
}
function pathD(m0,m1){ // SVG path along the well between two MDs
  const pts=[interp(m0)];
  for(const p of T){ if(p[0]>m0 && p[0]<m1) pts.push({x:p[1],z:p[2]}); }
  pts.push(interp(m1));
  return pts.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(2)+" "+Y(p.z).toFixed(2)).join(" ");
}
function at(md, off){ // screen point offset normal to the well (px)
  const p=interp(md); let ux=p.tx*sx, uz=p.tz*s; const n=Math.hypot(ux,uz)||1; ux/=n; uz/=n;
  return [X(p.x)-uz*off, Y(p.z)+ux*off, p];
}
const mdTD = T[T.length-1][0];

// ---------- defs: patterns & gradients ----------
const defs = el("defs",{});
function pat(id,w,h,bg,draw){const p=el("pattern",{id:id,width:w,height:h,patternUnits:"userSpaceOnUse"},defs);
  el("rect",{width:w,height:h,fill:bg},p); draw(p); return p;}
pat("sand",14,14,"#efe0b9",p=>{el("circle",{cx:3,cy:4,r:1.1,fill:"#c8a96a"},p);el("circle",{cx:10,cy:10,r:1.1,fill:"#c8a96a"},p);el("circle",{cx:11,cy:3,r:.7,fill:"#b89a5c"},p)});
pat("shale",22,8,"#b9c3b0",p=>{el("path",{d:"M0 4 H12 M15 7 H22",stroke:"#93a08a","stroke-width":1},p)});
pat("lime",24,12,"#c9d4dd",p=>{el("path",{d:"M0 0H24M0 6H24M6 0V6M18 6V12",stroke:"#a5b4c0","stroke-width":1},p)});
pat("res",14,14, S.fluid==="gas"?"#f3d9b1":"#e9d39a",p=>{el("circle",{cx:3,cy:4,r:1.3,fill:S.fluid==="gas"?"#d98b5f":"#8a6a22"},p);el("circle",{cx:10,cy:10,r:1.3,fill:S.fluid==="gas"?"#d98b5f":"#8a6a22"},p)});
pat("cement",8,8,"#a7a9ac",p=>{el("path",{d:"M0 8 L8 0",stroke:"#8d8f93","stroke-width":1.2},p)});
pat("sea",40,10,"#cfe6f5",p=>{el("path",{d:"M0 5 Q10 1 20 5 T40 5",stroke:"#9fcbe8",fill:"none","stroke-width":1},p)});
const fluidCol = {oil:"#6b4f1d", gas:"#e07a3f", water:"#2f7fc1"}[S.fluid]||"#6b4f1d";
const fluidLite = {oil:"#b08a3e", gas:"#f5b183", water:"#86bde6"}[S.fluid]||"#b08a3e";

// ---------- geology ----------
const gGeo = grp("geo");
const seaTop = S.water_depth>0 ? S.air_gap : 0;
const seabed = S.water_depth>0 ? S.air_gap + S.water_depth : 0;
// layer stack below surface / seabed, deterministic
let z = seabed, li = 0;
const kinds = ["shale","sand","shale","lime","shale","sand","shale"];
const rt = S.res_top_tvd, rb = S.res_bot_tvd;
while (z < tvdMax*1.04){
  let th = (0.07 + 0.05*((li*37)%5)/4) * tvdMax;
  let z2 = Math.min(z+th, tvdMax*1.04);
  if (z < rt-0.06*tvdMax && z2 > rt-0.06*tvdMax){ z2 = rt-0.06*tvdMax; }
  if (z >= rt-0.06*tvdMax && z < rt){ z2 = rt; el("rect",{x:0,y:Y(z),width:W-right+40,height:(z2-z)*s,fill:"url(#shale)"},gGeo);
      el("text",{x:6,y:Y(z)+12,class:"ax"},gGeo).textContent="Cap rock"; z=z2; li++; continue; }
  if (z >= rt && z < rb){ z2 = rb; el("rect",{x:0,y:Y(z),width:W-right+40,height:(z2-z)*s,fill:"url(#res)"},gGeo);
      const t=el("text",{x:6,y:Y((rt+rb)/2)+4,class:"lblb"},gGeo); t.textContent="Reservoir"; z=z2; li++; continue; }
  if (z < rt && z2 > rt) z2 = rt;
  el("rect",{x:0,y:Y(z),width:W-right+40,height:Math.max((z2-z)*s,0.5),fill:`url(#${kinds[li%kinds.length]})`,opacity:.85},gGeo);
  z = z2; li++;
}
el("rect",{x:0,y:0,width:W,height:Y(seabed),fill:"#ffffff"},gGeo);
if (S.water_depth>0){
  el("rect",{x:0,y:Y(seaTop),width:W-right+40,height:(seabed-seaTop)*s,fill:"url(#sea)"},gGeo);
  el("text",{x:6,y:Y(seaTop)+13,class:"ax"},gGeo).textContent="Sea";
  el("line",{x1:0,x2:W-right+40,y1:Y(seabed),y2:Y(seabed),stroke:"#7b6a4f","stroke-width":2});
  el("text",{x:6,y:Y(seabed)-4,class:"ax"}).textContent="Seabed";
} else {
  el("rect",{x:0,y:Y(0),width:W-right+40,height:5,fill:"#7aa95c"});
  el("text",{x:6,y:Y(0)-5,class:"ax"}).textContent="Ground level";
}

// ---------- well: cement, casings, liner hanger ----------
const gWell = grp("well");
const strings = [...S.strings].sort((a,b)=>b.od-a.od);
const holeExtra = 3.0; // in
strings.forEach(c=>{
  const cTop = Math.max(c.cem_top, c.od>=20 && S.water_depth>0 ? 0 : c.cem_top);
  if (c.shoe > cTop) el("path",{d:pathD(cTop, c.shoe), stroke:"url(#cement)", "stroke-width":(c.od+holeExtra)*k, fill:"none","stroke-linecap":"butt"},gWell);
});
strings.forEach(c=>{
  const wpx = Math.max(2.2, 0.06*c.od*k);
  el("path",{d:pathD(c.top, c.shoe), stroke:"#4d5b66","stroke-width":c.od*k, fill:"none","stroke-linecap":"butt"},gWell);
  el("path",{d:pathD(c.top, c.shoe), stroke:"#cfe1ea","stroke-width":c.od*k-2*wpx, fill:"none","stroke-linecap":"butt"},gWell);
  // shoe symbols
  for (const sg of [-1,1]){
    const a=at(c.shoe, sg*c.od*k/2), b=at(c.shoe, sg*(c.od*k/2+7)), cc=at(c.shoe-6/s, sg*c.od*k/2);
    el("polygon",{points:`${a[0]},${a[1]} ${b[0]},${b[1]} ${cc[0]},${cc[1]}`,fill:"#243746"},gWell);
  }
  if (c.top > 0.5){ // liner hanger
    for (const sg of [-1,1]){
      const a=at(c.top, sg*c.od*k/2), b=at(c.top+10/s, sg*(c.od*k/2+5));
      el("rect",{x:Math.min(a[0],b[0]),y:Math.min(a[1],b[1]),width:Math.abs(b[0]-a[0])+3,height:Math.abs(b[1]-a[1])+3,fill:"#243746"},gWell);
    }
  }
});
// riser / conductor above the seabed (offshore platform well)
if (S.water_depth>0){
  const od = strings.length? strings[0].od : 20;
  el("path",{d:pathD(0, Math.min(seabed, mdTD)), stroke:"#7d8b96","stroke-width":od*k*0.85, fill:"none"},gWell);
  el("path",{d:pathD(0, Math.min(seabed, mdTD)), stroke:"#dfeaf0","stroke-width":od*k*0.85-5, fill:"none"},gWell);
}
// rathole / open hole below the deepest shoe
const deepest = Math.max(...strings.map(c=>c.shoe), 0);
if (mdTD > deepest + 1e-6){
  const od = Math.min(...strings.map(c=>c.od), 8.5);
  el("path",{d:pathD(deepest, mdTD), stroke:"#8a7a5e","stroke-width":od*k, fill:"none","stroke-dasharray":"3 3"},gWell);
}

// ---------- perforations ----------
const gPerf = grp("perf");
const prodOD = strings.length ? Math.min(...strings.filter(c=>c.shoe>=S.perf[1]-1e-6).map(c=>c.od).concat([strings[strings.length-1].od])) : 7;
const nPerf = Math.max(4, Math.round((S.perf[1]-S.perf[0])*s/7));
for (let i=0;i<=nPerf;i++){
  const md = S.perf[0] + (S.perf[1]-S.perf[0])*i/nPerf;
  for (const sg of [-1,1]){
    const a=at(md, sg*(prodOD*k/2)), b=at(md, sg*(prodOD*k/2+18)), c1=at(md+2.4/s, sg*(prodOD*k/2)), c2=at(md-2.4/s, sg*(prodOD*k/2));
    el("polygon",{points:`${c1[0]},${c1[1]} ${b[0]},${b[1]} ${c2[0]},${c2[1]}`,fill:"#e4572e",opacity:.9},gPerf);
  }
}

// ---------- tubing + packer ----------
const gTub = grp("tub");
const tb = S.tubing.bottom;
el("path",{d:pathD(0, tb), stroke:"#33414c","stroke-width":S.tubing.od*k, fill:"none","stroke-linecap":"butt"},gTub);
el("path",{d:pathD(0, tb), stroke:fluidLite,"stroke-width":S.tubing.id*k, fill:"none","stroke-linecap":"butt"},gTub);
const flow = el("path",{d:pathD(0, tb), stroke:fluidCol,"stroke-width":Math.max(S.tubing.id*k*0.38,1.6), fill:"none","stroke-dasharray":"8 14",class:"flow",opacity:.9},gTub);
// fluid between tubing shoe and perforations (inflow)
el("path",{d:pathD(tb, Math.min(S.perf[1], mdTD)), stroke:fluidLite,"stroke-width":(prodOD*k)*0.72, fill:"none",opacity:.85},gTub);
// wireline entry guide
{const a=at(tb,-S.tubing.od*k/2-1), b=at(tb,S.tubing.od*k/2+1); el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"#243746","stroke-width":3},gTub);}
// packer
if (S.packer_md>0 && S.packer_md<tb){
  const pkOD = strings.filter(c=>c.top<=S.packer_md && c.shoe>=S.packer_md).reduce((m,c)=>Math.min(m,c.od), 99);
  const rIn = S.tubing.od*k/2, rOut = (pkOD<99? pkOD*k/2 - Math.max(2.2,0.06*pkOD*k) : rIn+8);
  for (const sg of [-1,1]){
    const p1=at(S.packer_md-7/s, sg*rIn), p2=at(S.packer_md-7/s, sg*rOut), p3=at(S.packer_md+7/s, sg*rOut), p4=at(S.packer_md+7/s, sg*rIn);
    el("polygon",{points:[p1,p2,p3,p4].map(p=>p[0]+","+p[1]).join(" "),fill:"#1b1b1b"},gTub);
    el("line",{x1:p1[0],y1:p1[1],x2:p3[0],y2:p3[1],stroke:"#bbb","stroke-width":1},gTub);
    el("line",{x1:p2[0],y1:p2[1],x2:p4[0],y2:p4[1],stroke:"#bbb","stroke-width":1},gTub);
  }
}

// ---------- wellhead & Christmas tree ----------
const gTree = grp("tree");
const wx = X(0), wy = Y(0);
if (S.water_depth>0){
  el("rect",{x:wx-70,y:wy-4,width:140,height:6,fill:"#56697a"},gTree);
  el("line",{x1:wx-60,y1:wy+2,x2:wx-60,y2:Y(seaTop)+2,stroke:"#56697a","stroke-width":4},gTree);
  el("line",{x1:wx+60,y1:wy+2,x2:wx+60,y2:Y(seaTop)+2,stroke:"#56697a","stroke-width":4},gTree);
  el("text",{x:wx+74,y:wy+12,class:"ax"},gTree).textContent="Platform deck";
}
const tw = 22;
el("rect",{x:wx-tw/2-6,y:wy-14,width:tw+12,height:14,fill:"#5d6d7a",rx:2},gTree);   // tubing head
el("rect",{x:wx-tw/2,y:wy-62,width:tw,height:48,fill:"#c0392b",rx:2},gTree);         // master valves body
for (const yy of [wy-54, wy-34]) el("rect",{x:wx-tw/2-5,y:yy,width:tw+10,height:10,fill:"#7b241c",rx:2},gTree);
el("rect",{x:wx-4,y:wy-82,width:8,height:20,fill:"#c0392b"},gTree);                   // swab
el("rect",{x:wx-9,y:wy-90,width:18,height:8,fill:"#7b241c",rx:2},gTree);
el("rect",{x:wx+tw/2,y:wy-48,width:52,height:9,fill:"#c0392b"},gTree);                // wing + choke
el("rect",{x:wx+tw/2+14,y:wy-53,width:9,height:19,fill:"#7b241c",rx:1},gTree);
el("polygon",{points:`${wx+tw/2+52},${wy-52} ${wx+tw/2+66},${wy-43.5} ${wx+tw/2+52},${wy-35}`,fill:"#243746"},gTree);
el("text",{x:wx+tw/2+70,y:wy-48,class:"lblb"},gTree).textContent="Outlet node";
el("text",{x:wx+tw/2+70,y:wy-35,class:"lbl"},gTree).textContent=S.outlet_txt;

// ---------- datum, node, labels ----------
const gLbl = grp("lbls");
if (S.datum_tvd!==null && S.datum_tvd!==undefined){
  el("line",{x1:0,x2:W-right+30,y1:Y(S.datum_tvd),y2:Y(S.datum_tvd),stroke:"#7a5195","stroke-width":1.3,"stroke-dasharray":"7 4"},gLbl);
  el("text",{x:6,y:Y(S.datum_tvd)+13,class:"lblb",fill:"#7a5195"},gLbl).textContent=`Pr datum · TVD ${S.datum_tvd.toFixed(0)} ${U}`;
}
const nd = interp(S.node.md);
el("circle",{cx:X(nd.x),cy:Y(nd.z),r:7.5,fill:"#ffd166",stroke:"#d62828","stroke-width":2.5},gLbl);
el("line",{x1:X(nd.x)-13,x2:X(nd.x)+13,y1:Y(nd.z),y2:Y(nd.z),stroke:"#d62828","stroke-width":1.5},gLbl);

// right-hand callouts, de-overlapped
const notes = [];
strings.forEach(c=>notes.push({md:c.shoe, off:c.od*k/2+8, t:`${c.label} ${c.name.toLowerCase()} shoe`, t2:`MD ${c.shoe.toFixed(0)} ${U}`}));
strings.filter(c=>c.top>0.5).forEach(c=>notes.push({md:c.top, off:c.od*k/2+6, t:`Liner hanger (${c.label})`, t2:`MD ${c.top.toFixed(0)} ${U}`}));
if (S.packer_md>0) notes.push({md:S.packer_md, off:S.tubing.od*k/2+12, t:"Production packer", t2:`MD ${S.packer_md.toFixed(0)} ${U}`});
notes.push({md:tb, off:S.tubing.od*k/2+4, t:`${S.tubing.label} tubing shoe`, t2:`MD ${tb.toFixed(0)} ${U}`});
notes.push({md:(S.perf[0]+S.perf[1])/2, off:prodOD*k/2+18, t:"Perforations", t2:`${S.perf[0].toFixed(0)}–${S.perf[1].toFixed(0)} ${U} MD`});
notes.push({md:S.node.md, off:9, t:"Solution node = reference depth", t2:`MD ${S.node.md.toFixed(0)} / TVD ${S.node.tvd.toFixed(0)} ${U}`, hot:true});
notes.forEach(n=>{const p=at(n.md, n.off); n.ax=p[0]; n.ay=p[1];});
notes.sort((a,b)=>a.ay-b.ay);
const lx = W-right+38; let last=-1e9;
notes.forEach(n=>{ n.ly=Math.max(n.ay, last+27); last=n.ly; });
const over = last - (H-18); if (over>0){ for(let i=notes.length-1;i>=0;i--){ const lim=(i===notes.length-1? H-18 : notes[i+1].ly-27); notes[i].ly=Math.min(notes[i].ly, lim);} }
notes.forEach(n=>{
  el("polyline",{points:`${n.ax},${n.ay} ${lx-14},${n.ly} ${lx-4},${n.ly}`,fill:"none",stroke:n.hot?"#d62828":"#7b8a96","stroke-width":n.hot?1.4:0.9},gLbl);
  el("text",{x:lx,y:n.ly-1,class:"lblb",fill:n.hot?"#d62828":"#243746"},gLbl).textContent=n.t;
  el("text",{x:lx,y:n.ly+11,class:"lbl"},gLbl).textContent=n.t2;
});

// ---------- TVD axis ----------
const gAx = grp("ax");
el("line",{x1:left-8,x2:left-8,y1:Y(0),y2:Y(tvdMax),stroke:"#8796a3"},gAx);
const step = [50,100,200,250,500,1000,2000,2500,5000].find(v=>tvdMax/v<=9)||10000;
for (let v=0; v<=tvdMax+1e-6; v+=step){
  el("line",{x1:left-12,x2:left-8,y1:Y(v),y2:Y(v),stroke:"#8796a3"},gAx);
  const t=el("text",{x:left-15,y:Y(v)+3,class:"ax","text-anchor":"end",stroke:"#ffffff","stroke-width":3,"paint-order":"stroke"},gAx); t.textContent=v.toFixed(0);
}
const at0=el("text",{x:left-56,y:Y(tvdMax/2),class:"ax",transform:`rotate(-90 ${left-56} ${Y(tvdMax/2)})`,"text-anchor":"middle"},gAx);
at0.textContent=`TVD below wellhead (${U})`;
el("text",{x:10,y:16,class:"lblb"},gAx).textContent=S.title||"Well schematic";
el("text",{x:10,y:30,class:"ax"},gAx).textContent="Radial dimensions exaggerated" + (compressed ? " · horizontal scale compressed" : "") + " · hover the well for readouts";

// ---------- hover readout ----------
const tip = document.getElementById("tip");
const hl = el("circle",{r:5,fill:"none",stroke:"#d62828","stroke-width":2,visibility:"hidden"});
function Tat(md){ return S.T_top + (S.T_bot-S.T_top)*Math.min(md/Math.max(S.node.md,1e-9),1); }
function Pat(md){ const P=S.P_profile; if(!P||!P.length) return null; if(md<=P[0][0]) return P[0][1];
  for(let i=1;i<P.length;i++) if(P[i][0]>=md){const f=(md-P[i-1][0])/Math.max(P[i][0]-P[i-1][0],1e-9); return P[i-1][1]+f*(P[i][1]-P[i-1][1]);}
  return P[P.length-1][1]; }
sv.addEventListener("mousemove", ev=>{
  const r = sv.getBoundingClientRect(); const mx=(ev.clientX-r.left)*W/r.width, my=(ev.clientY-r.top)*H/r.height;
  let best=null, bd=1e9;
  for (const p of T){ const d=Math.hypot(X(p[1])-mx, Y(p[2])-my); if(d<bd){bd=d;best=p;} }
  if (!best || bd>40){ tip.style.display="none"; hl.setAttribute("visibility","hidden"); return; }
  const md=best[0], tvd=best[2];
  const here = strings.filter(c=>c.top<=md && c.shoe>=md).map(c=>c.label).join(", ") || "open hole";
  let h = `<b>MD</b> ${md.toFixed(0)} ${U} &nbsp; <b>TVD</b> ${tvd.toFixed(0)} ${U}<br>`;
  if (md <= S.node.md) h += `<b>T</b> ${Tat(md).toFixed(1)} ${S.T_unit}`;
  const P = Pat(md); if (P!==null && md<=S.node.md) h += ` &nbsp; <b>P</b> ${P.toFixed(1)} ${S.P_unit}`;
  if (S.P_label && P!==null && md<=S.node.md) h += `<br><span style="opacity:.75">${S.P_label}</span>`;
  h += `<br><b>Casing</b> ${here}`;
  if (md <= tb) h += `<br><b>Tubing</b> ${S.tubing.label} (ID ${S.tubing.id.toFixed(3)} in)`;
  if (md >= S.perf[0] && md <= S.perf[1]) h += `<br><b style="color:#ff8c69">Perforated interval</b>`;
  tip.innerHTML = h; tip.style.display="block";
  let tx = ev.clientX-r.left+14, ty = ev.clientY-r.top+12;
  if (tx > r.width-210) tx -= 230; tip.style.left=tx+"px"; tip.style.top=ty+"px";
  hl.setAttribute("cx",X(best[1])); hl.setAttribute("cy",Y(best[2])); hl.setAttribute("visibility","visible");
});
sv.addEventListener("mouseleave", ()=>{tip.style.display="none"; hl.setAttribute("visibility","hidden");});
document.getElementById("cbFlow").onchange=e=>{flow.style.display=e.target.checked?"":"none";};
document.getElementById("cbGeo").onchange=e=>{gGeo.style.display=e.target.checked?"":"none";};
document.getElementById("cbLbl").onchange=e=>{gLbl.style.display=e.target.checked?"":"none";};
</script></body></html>
"""
