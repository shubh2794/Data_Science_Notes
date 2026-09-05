/* data-preprocessing.viz.js — extracted from data-preprocessing.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#22d3ee",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#pp-svg"),W=640,H=240;
  // fixed skewed-ish sample of a feature (e.g., income in $k), range ~20..180
  const raw=[];
  for(let i=0;i<200;i++){ raw.push(20 + Math.pow(Math.random(),2)*160 + (Math.random()-0.5)*8); }
  const mean=d3.mean(raw), sd=d3.deviation(raw), mn=d3.min(raw), mx=d3.max(raw);
  function transform(k){ return raw.map(x=> k==="std"? (x-mean)/sd : k==="norm"? (x-mn)/(mx-mn) : x); }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#pp-sel").property("value");
    const v=transform(k);
    const dmn=d3.min(v),dmx=d3.max(v), px=46,py=24,pw=W-92,ph=160;
    const x=d3.scaleLinear().domain([dmn,dmx]).range([px,px+pw]);
    const bins=d3.bin().domain([dmn,dmx]).thresholds(26)(v);
    const ymax=d3.max(bins,b=>b.length);
    const y=d3.scaleLinear().domain([0,ymax]).range([py+ph,py]);
    bins.forEach(b=>{ svg.append("rect").attr("x",x(b.x0)+1).attr("y",y(b.length)).attr("width",Math.max(1,x(b.x1)-x(b.x0)-1.5)).attr("height",py+ph-y(b.length)).attr("fill",C.accent).attr("fill-opacity",.8); });
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    // ticks
    [dmn,(dmn+dmx)/2,dmx].forEach(t=>{ svg.append("text").attr("x",x(t)).attr("y",py+ph+16).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(t.toFixed(k==="raw"?0:2)); });
    if(k==="std"){ svg.append("line").attr("x1",x(0)).attr("y1",py).attr("x2",x(0)).attr("y2",py+ph).attr("stroke",C.good).attr("stroke-dasharray","3 3"); svg.append("text").attr("x",x(0)).attr("y",py-6).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",9).text("mean 0"); }
    svg.append("text").attr("x",px).attr("y",16).attr("fill",C.muted).attr("font-size",10).text("count");
    const note=k==="raw"?`raw values span <b>${mn.toFixed(0)}…${mx.toFixed(0)}</b> — this feature would dominate distance/gradient`:
      k==="std"?`now mean <b>0</b>, std <b>1</b> (≈ ${dmn.toFixed(1)}…${dmx.toFixed(1)}) — comparable to other standardized features`:
      `now bounded to <b>[0, 1]</b> — comparable, bounded range`;
    d3.select("#pp-read").html(note);
  }
  d3.select("#pp-sel").on("change",draw);
  draw();
})();
