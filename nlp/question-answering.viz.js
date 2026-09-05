/* question-answering.viz.js — extracted from question-answering.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

// shared toy passage + question→span mapping for sections 02 & 03
const PASSAGE=["The","Eiffel","Tower","was","completed","in","1889","by","Gustave","Eiffel","in","Paris","."];
const QUESTIONS=[
  {q:"When was the tower completed?", start:6, end:6},        // "1889"
  {q:"Who built the tower?", start:8, end:9},                 // "Gustave Eiffel"
  {q:"Where is the tower located?", start:11, end:11},        // "Paris"
];

/* ───────────────────────── 02 · span highlighting ───────────────────────── */
(function(){
  const sel=d3.select("#span-q");
  QUESTIONS.forEach((o,i)=>sel.append("option").attr("value",i).text(o.q));
  const svg=d3.select("#span-svg"),W=640;
  const g=svg.append("g").attr("transform","translate(20,70)");
  // pre-measure token widths and lay out (wrap)
  const toks=PASSAGE.map((w,i)=>({w,i}));
  function layout(){
    let x=0,y=0; const maxW=600;
    toks.forEach(t=>{ t._tw=t.w.length*8.5+18; if(x+t._tw>maxW){x=0;y+=40;} t._x=x; t._y=y; x+=t._tw+7; });
  }
  layout();
  svg.append("text").attr("x",20).attr("y",36).attr("font-size",12).attr("fill",PC.muted).text("PASSAGE");
  function render(qi){
    const q=QUESTIONS[qi];
    const node=g.selectAll("g.tok").data(toks,d=>d.i).join(enter=>{
        const e=enter.append("g").attr("class","tok");
        e.append("rect").attr("height",28).attr("rx",6);
        e.append("text").attr("y",19).attr("text-anchor","middle").attr("font-size",13);
        return e;});
    node.attr("transform",d=>`translate(${d._x},${d._y})`);
    node.select("rect").attr("width",d=>d._tw)
      .transition().duration(350)
      .attr("fill",d=>(d.i>=q.start&&d.i<=q.end)?PC.accent:"#222733")
      .attr("stroke",d=>(d.i>=q.start&&d.i<=q.end)?"#fff":PC.line);
    node.select("text").attr("x",d=>d._tw/2)
      .attr("fill",d=>(d.i>=q.start&&d.i<=q.end)?"#1a1320":PC.ink)
      .attr("font-weight",d=>(d.i>=q.start&&d.i<=q.end)?700:400).text(d=>d.w);
    const ans=PASSAGE.slice(q.start,q.end+1).join(" ");
    d3.select("#span-readout").html(`answer span [${q.start}, ${q.end}] → <b style="color:${PC.accent}">“${ans}”</b>`);
  }
  sel.on("change",()=>render(+sel.property("value")));
  render(0);
})();

/* ───────────────────────── 03 · start/end distributions ───────────────────────── */
(function(){
  const sel=d3.select("#dist-q");
  QUESTIONS.forEach((o,i)=>sel.append("option").attr("value",i).text(o.q));
  const svg=d3.select("#dist-svg"),W=640,H=260,m={t:30,r:20,b:60,l:46};
  const n=PASSAGE.length;
  const x=d3.scaleBand().domain(d3.range(n)).range([m.l,W-m.r]).padding(0.25);
  const yS=d3.scaleLinear().domain([0,1]).range([120,m.t]);     // start dist (top)
  const yE=d3.scaleLinear().domain([0,1]).range([220,140]);     // end dist (bottom)
  svg.append("text").attr("x",m.l).attr("y",22).attr("font-size",11).attr("fill",PC.accent).text("P(start)");
  svg.append("text").attr("x",m.l).attr("y",136).attr("font-size",11).attr("fill",PC.a2).text("P(end)");
  // token labels along the bottom
  svg.append("g").selectAll("text.tk").data(PASSAGE).join("text").attr("class","tk")
    .attr("x",(d,i)=>x(i)+x.bandwidth()/2).attr("y",240).attr("text-anchor","middle")
    .attr("font-size",10).attr("fill",PC.muted).attr("transform",(d,i)=>`rotate(35,${x(i)+x.bandwidth()/2},240)`).text(d=>d);
  const gS=svg.append("g"), gE=svg.append("g");
  function softmaxPeak(peak,sharp){ // build a distribution peaked at `peak`
    const raw=d3.range(n).map(i=>Math.exp(-sharp*Math.abs(i-peak)));
    const z=d3.sum(raw); return raw.map(v=>v/z);
  }
  function render(qi){
    const q=QUESTIONS[qi];
    const ps=softmaxPeak(q.start,1.1), pe=softmaxPeak(q.end,1.1);
    gS.selectAll("rect").data(ps).join("rect")
      .attr("x",(d,i)=>x(i)).attr("width",x.bandwidth()).attr("rx",2).attr("fill",PC.accent)
      .transition().duration(350).attr("y",d=>yS(d)).attr("height",d=>120-yS(d));
    gE.selectAll("rect").data(pe).join("rect")
      .attr("x",(d,i)=>x(i)).attr("width",x.bandwidth()).attr("rx",2).attr("fill",PC.a2)
      .transition().duration(350).attr("y",d=>yE(d)).attr("height",d=>220-yE(d));
    // constrained argmax: best (i,j) with j>=i maximizing ps[i]*pe[j]
    let best={i:0,j:0,score:-1};
    for(let i=0;i<n;i++)for(let j=i;j<Math.min(n,i+5);j++){ const sc=ps[i]*pe[j]; if(sc>best.score) best={i,j,score:sc}; }
    const ans=PASSAGE.slice(best.i,best.j+1).join(" ");
    d3.select("#dist-readout").html(`argmax span (j ≥ i): [${best.i}, ${best.j}] → <b style="color:${PC.good}">“${ans}”</b> · score ${best.score.toFixed(3)}`);
  }
  sel.on("change",()=>render(+sel.property("value")));
  render(0);
})();
