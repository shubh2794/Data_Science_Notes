/* pandas.viz.js — extracted from pandas.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",cell:"#1b2130",cellb:"#2a2f3a"};

/* ───────────────────────── 07 · split-apply-combine ───────────────────────── */
(function(){
  const svg=d3.select("#gby-svg"),W=640,H=320;
  const out=document.getElementById("gby-readout");
  const keys=["A","B","C"], kcol={A:PC.accent,B:PC.a2,C:PC.good};
  // 9 rows: key + value
  const rows=[["A",4],["B",2],["A",3],["C",5],["B",6],["A",1],["C",2],["B",4],["C",3]]
    .map((r,i)=>({key:r[0],val:r[1],id:i}));
  const cw=210, rh=26, rg=6, x0=40, y0=40;
  let state=0; // 0 raw, 1 split, 2 sum

  function layout(){
    if(state===0){
      rows.forEach((r,i)=>{ r.tx=x0; r.ty=y0+i*(rh+rg); });
    } else {
      // grouped into 3 columns by key
      const cols={A:[],B:[],C:[]}; rows.forEach(r=>cols[r.key].push(r));
      keys.forEach((k,ki)=>{ cols[k].forEach((r,i)=>{ r.tx=x0+ki*cw; r.ty=y0+30+i*(rh+rg); }); });
    }
  }
  function draw(){
    layout();
    svg.selectAll("*").remove();
    if(state>=1){
      keys.forEach((k,ki)=>svg.append("text").attr("x",x0+ki*cw).attr("y",y0+14).attr("fill",kcol[k]).attr("font-size",12).attr("font-weight",700).text("key = "+k));
    } else {
      svg.append("text").attr("x",x0).attr("y",y0-12).attr("fill",PC.muted).attr("font-size",12).text("rows (key, value) — unsorted");
    }
    const g=svg.selectAll(".row").data(rows,d=>d.id).enter().append("g").attr("class","row");
    g.append("rect").attr("x",d=>d.tx).attr("y",d=>d.ty).attr("width",cw-50).attr("height",rh).attr("rx",6)
      .attr("fill",d=>kcol[d.key]).attr("fill-opacity",.18).attr("stroke",d=>kcol[d.key]);
    g.append("text").attr("x",d=>d.tx+14).attr("y",d=>d.ty+rh/2+4).attr("fill",PC.ink).attr("font-size",13).text(d=>d.key);
    g.append("text").attr("x",d=>d.tx+cw-66).attr("y",d=>d.ty+rh/2+4).attr("text-anchor","end").attr("fill",PC.ink).attr("font-size",13).text(d=>d.val);
    // animate to positions
    svg.selectAll(".row rect").transition().duration(600).attr("x",d=>d.tx).attr("y",d=>d.ty);
    svg.selectAll(".row text").transition().duration(600)
      .attr("x",function(d){return d3.select(this).attr("text-anchor")==="end"? d.tx+cw-66 : d.tx+14;}).attr("y",d=>d.ty+rh/2+4);

    if(state===2){
      const sums={}; keys.forEach(k=>sums[k]=rows.filter(r=>r.key===k).reduce((a,r)=>a+r.val,0));
      const cy=y0+30+4*(rh+rg)+24;
      keys.forEach((k,ki)=>{
        const X=x0+ki*cw;
        svg.append("rect").attr("x",X).attr("y",cy).attr("width",cw-50).attr("height",rh+6).attr("rx",6)
          .attr("fill",kcol[k]).attr("fill-opacity",.30).attr("stroke",kcol[k]).attr("stroke-width",2)
          .attr("opacity",0).transition().delay(300).duration(400).attr("opacity",1);
        svg.append("text").attr("x",X+14).attr("y",cy+rh/2+5).attr("fill",PC.ink).attr("font-size",13).attr("font-weight",700)
          .attr("opacity",0).transition().delay(300).duration(400).attr("opacity",1).text("Σ "+k);
        svg.append("text").attr("x",X+cw-66).attr("y",cy+rh/2+5).attr("text-anchor","end").attr("fill",PC.ink).attr("font-size",14).attr("font-weight",700)
          .attr("opacity",0).transition().delay(300).duration(400).attr("opacity",1).text(sums[k]);
      });
      out.innerHTML = `combine → <b style="color:${PC.accent}">A=${sums.A}</b> · <b style="color:${PC.a2}">B=${sums.B}</b> · <b style="color:${PC.good}">C=${sums.C}</b> · one row per group (aggregate shrinks)`;
    } else if(state===1){
      out.innerHTML = `split: rows partitioned by key into ${keys.length} buckets (≈ Θ(n) via hashing) · now aggregate`;
    } else {
      out.innerHTML = `<span style="color:var(--muted)">9 raw rows. Press “Split by key”.</span>`;
    }
  }
  document.getElementById("g-split").addEventListener("click",()=>{ if(state<1)state=1; draw(); });
  document.getElementById("g-sum").addEventListener("click",()=>{ state=2; draw(); });
  document.getElementById("g-reset").addEventListener("click",()=>{ state=0; draw(); });
  draw();
})();
