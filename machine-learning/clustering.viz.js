/* clustering.viz.js — extracted from clustering.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

/* Interactive k-Means (Lloyd's algorithm) */
(function(){
  const svg=d3.select("#km-svg"), W=640, H=360, pal=["#5b9cff","#ffb454","#4ade80","#f87171","#c084fc","#22d3ee"];
  let pts=[], cents=[], timer=null;
  function genData(){
    pts=[]; const k=+d3.select("#km-k").property("value");
    const blobs=Math.max(k, 3);
    for(let b=0;b<blobs;b++){ const cx=80+Math.random()*(W-160), cy=60+Math.random()*(H-120);
      for(let i=0;i<22;i++) pts.push({x:cx+(Math.random()-.5)*90,y:cy+(Math.random()-.5)*90,c:-1}); }
    initCents();
  }
  function initCents(){ const k=+d3.select("#km-k").property("value");
    cents=d3.range(k).map(i=>({x:80+Math.random()*(W-160),y:60+Math.random()*(H-120),i})); }
  function assign(){ let changed=false; pts.forEach(p=>{ let best=0,bd=1e9;
    cents.forEach(c=>{const d=Math.hypot(p.x-c.x,p.y-c.y); if(d<bd){bd=d;best=c.i;}}); if(p.c!==best){p.c=best;changed=true;} p.c=best; }); return changed; }
  function update(){ cents.forEach(c=>{ const mem=pts.filter(p=>p.c===c.i); if(mem.length){ c.x=d3.mean(mem,d=>d.x); c.y=d3.mean(mem,d=>d.y);} }); }
  const gPts=svg.append("g"), gCent=svg.append("g");
  function draw(animate){
    const p=gPts.selectAll("circle").data(pts);
    p.enter().append("circle").attr("r",4).attr("opacity",.8)
      .merge(p).attr("cx",d=>d.x).attr("cy",d=>d.y).attr("fill",d=>d.c<0?"#566":pal[d.c%pal.length]);
    p.exit().remove();
    const c=gCent.selectAll("path").data(cents);
    const sym=d3.symbol().type(d3.symbolStar).size(280);
    c.enter().append("path").attr("d",sym).attr("stroke","#0f1117").attr("stroke-width",2)
      .merge(c).transition().duration(animate?500:0).attr("transform",d=>`translate(${d.x},${d.y})`).attr("fill",d=>pal[d.i%pal.length]);
    c.exit().remove();
  }
  function inertia(){ let s=0; pts.forEach(p=>{const c=cents[p.c]||cents[0]; s+=Math.hypot(p.x-c.x,p.y-c.y)**2;}); return s; }
  function step(){ assign(); draw(false); setTimeout(()=>{ update(); draw(true); d3.select("#km-readout").html(`inertia ≈ <b>${(inertia()/1000).toFixed(1)}k</b>`);},120); }
  d3.select("#km-step").on("click",step);
  d3.select("#km-run").on("click",function(){ if(timer){clearInterval(timer);timer=null;return;} let it=0; timer=setInterval(()=>{ const ch=assign(); draw(false); setTimeout(()=>{update();draw(true);d3.select("#km-readout").html(`inertia ≈ <b>${(inertia()/1000).toFixed(1)}k</b>`);},120); if(!ch||++it>20){clearInterval(timer);timer=null;} },650); });
  d3.select("#km-reset").on("click",()=>{ if(timer){clearInterval(timer);timer=null;} genData(); draw(false); d3.select("#km-readout").text(""); });
  d3.select("#km-k").on("input",function(){ d3.select("#km-kval").text(this.value); if(timer){clearInterval(timer);timer=null;} genData(); draw(false); });
  genData(); draw(false);
})();
