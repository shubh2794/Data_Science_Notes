/* linear-algebra-viz.js — shared D3 helpers for the Linear Algebra sub-pages.
   Loaded after ../notes.js and before each page's inline viz <script>.
   Provides: PC (palette), axes(), arrowDefs(), eigSym2(). */
const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* shared: grid + axes helper for a centered ℝ² canvas (unit = px scale) */
function axes(svg,W,H,ox,oy,unit){
  const g=svg.append("g");
  for(let gx=Math.ceil((-ox)/unit);gx*unit<=W-ox;gx++) g.append("line").attr("x1",ox+gx*unit).attr("y1",10).attr("x2",ox+gx*unit).attr("y2",H-10).attr("stroke","#1b2130");
  for(let gy=Math.floor((10-oy)/unit);oy+gy*unit<=H-10;gy++) g.append("line").attr("x1",10).attr("y1",oy+gy*unit).attr("x2",W-10).attr("y2",oy+gy*unit).attr("stroke","#1b2130");
  g.append("line").attr("x1",10).attr("y1",oy).attr("x2",W-10).attr("y2",oy).attr("stroke","#3a4150");
  g.append("line").attr("x1",ox).attr("y1",10).attr("x2",ox).attr("y2",H-10).attr("stroke","#3a4150");
  return g;
}
function arrowDefs(svg,specs){ const defs=svg.append("defs");
  specs.forEach(([id,c])=>defs.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",8).attr("refY",5)
    .attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto-start-reverse").append("path").attr("d","M0,0L10,5L0,10").attr("fill",c)); }

/* shared: eigendecomposition of a symmetric 2×2 [[p,q],[q,r]] */
function eigSym2(p,q,r){
  const tr=p+r, det=p*r-q*q, disc=Math.sqrt(Math.max(0,tr*tr-4*det));
  const l1=(tr+disc)/2, l2=(tr-disc)/2;
  function vecFor(l){ let vx,vy;
    if(Math.abs(q)>1e-9){ vx=q; vy=l-p; }
    else { vx=Math.abs(p-l)<1e-9?1:0; vy=Math.abs(p-l)<1e-9?0:1; }
    const n=Math.hypot(vx,vy)||1; return {x:vx/n,y:vy/n}; }
  return {l1,l2,v1:vecFor(l1),v2:vecFor(l2)};
}
