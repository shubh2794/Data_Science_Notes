/* text-classification.viz.js — extracted from text-classification.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* ───────────────────────── 04 · sentiment confidence bars ───────────────────────── */
(function(){
  // toy corpus: each review carries word-level sentiment cues; we score 3 classes
  const reviews=[
    {t:"This movie was absolutely fantastic and a joy to watch", s:{pos:0.88,neu:0.09,neg:0.03}},
    {t:"The plot was fine but nothing special, just okay", s:{pos:0.18,neu:0.70,neg:0.12}},
    {t:"A boring, terrible waste of two hours, deeply disappointing", s:{pos:0.03,neu:0.10,neg:0.87}},
    {t:"Great acting, though the ending felt a little flat", s:{pos:0.62,neu:0.27,neg:0.11}},
    {t:"I do not recommend this; it was dull and forgettable", s:{pos:0.06,neu:0.22,neg:0.72}},
  ];
  const classes=[{k:"pos",lab:"positive",c:PC.good},{k:"neu",lab:"neutral",c:PC.a2},{k:"neg",lab:"negative",c:PC.bad}];
  const sel=d3.select("#conf-sel");
  reviews.forEach((r,i)=>sel.append("option").attr("value",i).text(r.t.slice(0,42)+(r.t.length>42?"…":"")));
  const svg=d3.select("#conf-svg"),W=640,H=240,m={t:54,r:24,b:30,l:96};
  const x=d3.scaleLinear().domain([0,1]).range([m.l,W-m.r]);
  const y=d3.scaleBand().domain(classes.map(c=>c.lab)).range([m.t,H-m.b]).padding(0.34);
  svg.append("text").attr("id","conf-quote").attr("x",m.l).attr("y",28).attr("font-size",13).attr("fill",PC.ink).attr("font-style","italic");
  const gRow=svg.append("g");
  function render(i){
    const r=reviews[i];
    svg.select("#conf-quote").text("“"+r.t+"”");
    const rows=classes.map(c=>({lab:c.lab,c:c.c,v:r.s[c.k],k:c.k}));
    const top=rows.slice().sort((a,b)=>b.v-a.v)[0];
    const g=gRow.selectAll("g.row").data(rows,d=>d.lab).join(
      enter=>{const e=enter.append("g").attr("class","row");
        e.append("text").attr("class","lbl").attr("x",m.l-10).attr("text-anchor","end").attr("font-size",12).attr("fill",PC.muted);
        e.append("rect").attr("class","track").attr("x",m.l).attr("rx",5).attr("height",y.bandwidth()).attr("fill","#222733");
        e.append("rect").attr("class","bar").attr("x",m.l).attr("rx",5).attr("height",y.bandwidth());
        e.append("text").attr("class","val").attr("font-size",12).attr("dy",y.bandwidth()/2+4).attr("fill",PC.ink);
        return e;});
    g.attr("transform",d=>`translate(0,${y(d.lab)})`);
    g.select(".lbl").attr("y",y.bandwidth()/2+4).text(d=>d.lab);
    g.select(".track").attr("width",x(1)-x(0));
    g.select(".bar").transition().duration(450).attr("width",d=>x(d.v)-x(0)).attr("fill",d=>d.lab===top.lab?d.c:d3.color(d.c).copy({opacity:0.4}));
    g.select(".val").transition().duration(450).attr("x",d=>x(d.v)+8).text(d=>(d.v*100).toFixed(0)+"%");
    d3.select("#conf-readout").html(`predicted: <b style="color:${top.c}">${top.lab}</b> (${(top.v*100).toFixed(0)}%)`);
  }
  sel.on("change",()=>render(+sel.property("value")));
  d3.select("#conf-next").on("click",()=>{const n=(+sel.property("value")+1)%reviews.length; sel.property("value",n); render(n);});
  render(0);
})();

/* ───────────────────────── 06 · threshold → confusion matrix → P/R/F1 ───────────────────────── */
(function(){
  // toy scored dataset: each point has a model score in [0,1] and a true label (1=positive)
  const N=60, pts=[];
  let seed=7; const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff;};
  for(let i=0;i<N;i++){
    const pos=rnd()<0.42;
    // positives score higher on average, with overlap
    const score=Math.max(0,Math.min(1, (pos?0.62:0.38) + (rnd()-0.5)*0.7));
    pts.push({y:pos?1:0, s:score});
  }
  const svg=d3.select("#thr-svg"),W=640,H=280;
  const tEl=document.getElementById("thr-t"), out=document.getElementById("thr-readout");
  // confusion matrix layout (right side)
  const cell=64, cx0=400, cy0=70;
  const cells=[{r:0,c:0,key:"TP",lab:"TP"},{r:0,c:1,key:"FN",lab:"FN"},{r:1,c:0,key:"FP",lab:"FP"},{r:1,c:1,key:"TN",lab:"TN"}];
  const gMat=svg.append("g");
  svg.append("text").attr("x",cx0+cell).attr("y",cy0-26).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.muted).text("predicted");
  svg.append("text").attr("x",cx0+cell*0.5).attr("y",cy0-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",PC.good).text("positive");
  svg.append("text").attr("x",cx0+cell*1.5).attr("y",cy0-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",PC.bad).text("negative");
  svg.append("text").attr("transform",`translate(${cx0-26},${cy0+cell})rotate(-90)`).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.muted).text("actual");
  // score strip (left side)
  const sx=d3.scaleLinear().domain([0,1]).range([40,360]);
  const gStrip=svg.append("g");
  svg.append("text").attr("x",40).attr("y",40).attr("font-size",11).attr("fill",PC.muted).text("model score →");
  pts.forEach(p=>{ gStrip.append("circle").attr("cx",sx(p.s)).attr("cy",p.y?120:170).attr("r",4.5)
      .attr("fill",p.y?PC.good:PC.bad).attr("fill-opacity",0.8).attr("stroke","#0f1117").attr("stroke-width",0.8); });
  svg.append("text").attr("x",26).attr("y",124).attr("text-anchor","end").attr("font-size",10).attr("fill",PC.good).text("pos");
  svg.append("text").attr("x",26).attr("y",174).attr("text-anchor","end").attr("font-size",10).attr("fill",PC.bad).text("neg");
  const thrLine=gStrip.append("line").attr("y1",100).attr("y2",195).attr("stroke",PC.accent).attr("stroke-width",2.5);
  const thrLab=gStrip.append("text").attr("y",212).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.accent).attr("font-weight",700);
  function counts(t){ let TP=0,FP=0,FN=0,TN=0;
    pts.forEach(p=>{ const pred=p.s>=t?1:0;
      if(pred&&p.y)TP++; else if(pred&&!p.y)FP++; else if(!pred&&p.y)FN++; else TN++; });
    return {TP,FP,FN,TN}; }
  function render(){
    const t=+tEl.value, c=counts(t);
    thrLine.attr("x1",sx(t)).attr("x2",sx(t)); thrLab.attr("x",sx(t)).text("τ = "+t.toFixed(2));
    const sel=gMat.selectAll("g.cell").data(cells).join(enter=>{const e=enter.append("g").attr("class","cell");
        e.append("rect").attr("width",cell).attr("height",cell).attr("rx",6).attr("stroke",PC.line);
        e.append("text").attr("class","ct").attr("x",cell/2).attr("y",cell/2-2).attr("text-anchor","middle").attr("font-size",20).attr("font-weight",700).attr("fill",PC.ink);
        e.append("text").attr("class","cl").attr("x",cell/2).attr("y",cell/2+18).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.muted);
        return e;});
    const fillFor={TP:PC.good,TN:PC.good,FP:PC.bad,FN:PC.bad};
    sel.attr("transform",d=>`translate(${cx0+d.c*cell},${cy0+d.r*cell})`);
    sel.select("rect").attr("fill",d=>d3.color(fillFor[d.key]).copy({opacity:0.18+0.5*(c[d.key]/N)}));
    sel.select(".ct").text(d=>c[d.key]);
    sel.select(".cl").text(d=>d.lab);
    const prec=c.TP/((c.TP+c.FP)||1), rec=c.TP/((c.TP+c.FN)||1), f1=2*prec*rec/((prec+rec)||1);
    out.innerHTML=`precision <b style="color:${PC.good}">${prec.toFixed(2)}</b> · recall <b style="color:${PC.a2}">${rec.toFixed(2)}</b> · F1 <b style="color:${PC.accent}">${f1.toFixed(2)}</b>`;
  }
  tEl.addEventListener("input",render); render();
})();
