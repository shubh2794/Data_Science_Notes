/* knowledge-graphs.viz.js — extracted from knowledge-graphs.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#kg-svg"),W=640,H=260;
  const data=[
    {triples:[["Marie Curie","won","Nobel Prize"],["Nobel Prize","field","Physics"]],
     pos:{"Marie Curie":[130,80],"Nobel Prize":[380,80],"Physics":[500,190]}},
    {triples:[["Apple","founded by","Steve Jobs"],["Apple","located in","California"]],
     pos:{"Apple":[140,130],"Steve Jobs":[420,70],"California":[440,200]}},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const d=data[+d3.select("#kg-sel").property("value")];
    // edges
    d.triples.forEach(([h,r,t])=>{
      const a=d.pos[h], b=d.pos[t];
      svg.append("line").attr("x1",a[0]).attr("y1",a[1]).attr("x2",b[0]).attr("y2",b[1]).attr("stroke","#3a4150").attr("stroke-width",1.5).attr("marker-end","url(#kga)");
      const mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2;
      svg.append("rect").attr("x",mx-r.length*3.4-6).attr("y",my-9).attr("width",r.length*6.8+12).attr("height",17).attr("rx",8).attr("fill","#15181f").attr("stroke",C.accent).attr("stroke-opacity",.5);
      svg.append("text").attr("x",mx).attr("y",my+3).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.a2).text(r);
    });
    // arrow marker
    const defs=svg.append("defs"); defs.append("marker").attr("id","kga").attr("viewBox","0 -5 10 10").attr("refX",20).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto").append("path").attr("d","M0,-4L8,0L0,4").attr("fill","#3a4150");
    // nodes
    Object.entries(d.pos).forEach(([id,p])=>{
      const w=id.length*7.5+20;
      svg.append("rect").attr("x",p[0]-w/2).attr("y",p[1]-16).attr("width",w).attr("height",32).attr("rx",16).attr("fill","rgba(192,132,252,.16)").attr("stroke",C.accent).attr("stroke-width",1.5);
      svg.append("text").attr("x",p[0]).attr("y",p[1]+5).attr("text-anchor","middle").attr("font-size",12).attr("fill",C.ink).text(id);
    });
    d3.select("#kg-read").html(`<b>${Object.keys(d.pos).length}</b> entities · <b>${d.triples.length}</b> relations: `+d.triples.map(t=>`(${t[0]}, <span style="color:${C.a2}">${t[1]}</span>, ${t[2]})`).join("  "));
  }
  d3.select("#kg-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 05 · TransE scoring + link prediction ───────────────────────── */
(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  // entity vectors in 2-D, laid out so relations act as ~consistent translations
  const ent={
    Paris:[2.0,7.0], France:[2.4,2.0],
    Rome:[6.5,7.2], Italy:[6.9,2.2],
    Berlin:[9.6,7.4], Germany:[9.9,2.4],
    Tokyo:[5.0,8.6], Japan:[5.4,3.6],
  };
  // relations as translation vectors (approx)
  const rels={
    "capital_of":[0.4,-5.0],   // capital → country
    "located_in":[0.4,-5.0],
  };
  const heads=["Paris","Rome","Berlin","Tokyo"];
  const tails=["France","Italy","Germany","Japan"];
  const truth={Paris:"France",Rome:"Italy",Berlin:"Germany",Tokyo:"Japan"};

  const svg=d3.select("#te-svg"),W=640,H=300, plotW=380;
  const m={t:18,r:16,b:24,l:34};
  const xs=d3.scaleLinear().domain([0,12]).range([m.l,plotW-m.r]);
  const ys=d3.scaleLinear().domain([0,11]).range([H-m.b,m.t]);

  const selH=d3.select("#te-head"), selR=d3.select("#te-rel");
  selH.selectAll("option").data(heads).join("option").attr("value",d=>d).text(d=>d);
  selR.selectAll("option").data(Object.keys(rels)).join("option").attr("value",d=>d).text(d=>d);

  const gGrid=svg.append("g"), gLink=svg.append("g"), gPts=svg.append("g"), gBars=svg.append("g");
  // grid
  for(let gx=0;gx<=12;gx+=2) gGrid.append("line").attr("x1",xs(gx)).attr("x2",xs(gx)).attr("y1",ys(0)).attr("y2",ys(11)).attr("stroke",C.line).attr("stroke-opacity",.4);
  for(let gy=0;gy<=11;gy+=2) gGrid.append("line").attr("x1",xs(0)).attr("x2",xs(12)).attr("y1",ys(gy)).attr("y2",ys(gy)).attr("stroke",C.line).attr("stroke-opacity",.4);
  // arrow marker
  const defs=svg.append("defs");
  defs.append("marker").attr("id","tea").attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto").append("path").attr("d","M0,-4L9,0L0,4").attr("fill",C.a2);

  function draw(){
    const h=selH.property("value"), rel=selR.property("value");
    const hv=ent[h], rv=rels[rel];
    const target=[hv[0]+rv[0], hv[1]+rv[1]];   // h + r
    const dist=t=>Math.hypot(ent[t][0]-target[0], ent[t][1]-target[1]);
    const ranked=tails.map(t=>({t,d:dist(t),s:-dist(t)})).sort((a,b)=>a.d-b.d);
    const best=ranked[0];

    // ── plot: entities ──
    gLink.selectAll("*").remove();
    // translation arrow h -> h+r
    gLink.append("line").attr("x1",xs(hv[0])).attr("y1",ys(hv[1])).attr("x2",xs(target[0])).attr("y2",ys(target[1]))
      .attr("stroke",C.a2).attr("stroke-width",2).attr("stroke-dasharray","5 3").attr("marker-end","url(#tea)");
    // line from target to nearest tail
    gLink.append("line").attr("x1",xs(target[0])).attr("y1",ys(target[1])).attr("x2",xs(ent[best.t][0])).attr("y2",ys(ent[best.t][1]))
      .attr("stroke",C.good).attr("stroke-width",1.5).attr("stroke-opacity",.7);
    // target star (h+r)
    const star="M0,-9 L2.6,-2.6 L9,-2.6 L3.7,1.4 L5.6,8 L0,3.8 L-5.6,8 L-3.7,1.4 L-9,-2.6 L-2.6,-2.6 Z";
    gLink.append("path").attr("d",star).attr("transform",`translate(${xs(target[0])},${ys(target[1])})`).attr("fill",C.a2).attr("stroke","#0f1117").attr("stroke-width",1);
    gLink.append("text").attr("x",xs(target[0])+12).attr("y",ys(target[1])-6).attr("font-size",10).attr("fill",C.a2).text("h + r");

    const pts=gPts.selectAll("g.te").data(Object.keys(ent),d=>d);
    const en=pts.enter().append("g").attr("class","te");
    en.append("circle").attr("r",6).attr("stroke","#0f1117").attr("stroke-width",1.4);
    en.append("text").attr("font-size",10).attr("paint-order","stroke").attr("stroke","#0f1117").attr("stroke-width",3).attr("stroke-linejoin","round");
    const all=en.merge(pts);
    all.attr("transform",d=>`translate(${xs(ent[d][0])},${ys(ent[d][1])})`);
    all.select("circle").attr("fill",d=> d===h ? C.accent : d===best.t ? C.good : tails.includes(d) ? C.blue : "#3a4150");
    all.select("text").attr("x",9).attr("dy",4).attr("fill",d=> d===h||d===best.t ? C.ink : C.muted).text(d=>d);
    pts.exit().remove();

    // ── ranked bars (link prediction) ──
    const bx0=plotW+18, bw=W-bx0-90, maxD=d3.max(ranked,d=>d.d)||1;
    const by=d3.scaleBand().domain(ranked.map(d=>d.t)).range([m.t+8,H-m.b-8]).padding(0.3);
    const bx=d3.scaleLinear().domain([0,maxD*1.05]).range([0,bw]);
    svg.selectAll("text.bttl").data([0]).join("text").attr("class","bttl").attr("x",bx0).attr("y",m.t).attr("font-size",10).attr("fill",C.muted).text("candidate tails by distance ‖h+r−t‖");

    const rows=gBars.selectAll("g.br").data(ranked,d=>d.t);
    const ren=rows.enter().append("g").attr("class","br");
    ren.append("rect").attr("class","bg").attr("height",by.bandwidth()).attr("rx",3).attr("fill","#15181f");
    ren.append("rect").attr("class","fg").attr("height",by.bandwidth()).attr("rx",3);
    ren.append("text").attr("class","lab").attr("font-size",10);
    ren.append("text").attr("class","val").attr("font-size",10);
    const rall=ren.merge(rows);
    rall.attr("transform",d=>`translate(${bx0},${by(d.t)})`);
    rall.select("rect.bg").attr("width",bw);
    rall.select("rect.fg").transition().duration(400).attr("width",d=>Math.max(2,bx(d.d))).attr("fill",d=>d.t===best.t?C.good:C.blue).attr("fill-opacity",.85);
    rall.select("text.lab").attr("x",-6).attr("y",by.bandwidth()/2+3).attr("text-anchor","end").attr("fill",d=>d.t===best.t?C.good:C.muted).attr("font-weight",d=>d.t===best.t?700:400).text(d=>d.t);
    rall.select("text.val").attr("x",d=>Math.max(2,bx(d.d))+5).attr("y",by.bandwidth()/2+3).attr("fill",C.ink).text(d=>d.d.toFixed(2));
    rows.exit().remove();

    const ok = best.t===truth[h];
    d3.select("#te-read").html(`f(${h}, <span style="color:${C.a2}">${rel}</span>, t) = −‖h+r−t‖ &nbsp;·&nbsp; best <b style="color:${C.good}">${best.t}</b> &nbsp; score <b>${best.s.toFixed(2)}</b>`);
    d3.select("#te-rank").html(`predicted tail: <b style="color:${C.good}">${best.t}</b> &nbsp; ${ok?`<span style="color:${C.good}">✓ matches the known fact</span>`:`<span style="color:${C.bad}">(ground truth: ${truth[h]})</span>`}`);
  }
  selH.on("change",draw); selR.on("change",draw);
  draw();
})();
