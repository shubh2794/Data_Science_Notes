/* agents.viz.js — extracted from agents.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#fb923c",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#agt-svg"),W=640;
  const steps=[
    {t:"thought",x:"I need France's population, then double it."},
    {t:"action",x:"search(\"population of France\")"},
    {t:"obs",x:"≈ 68 million"},
    {t:"thought",x:"Now compute 68,000,000 × 2."},
    {t:"action",x:"calc(68000000 * 2)"},
    {t:"obs",x:"136,000,000"},
    {t:"answer",x:"About 136 million."},
  ];
  const colors={thought:"#818cf8",action:"#fb923c",obs:"#2dd4bf",answer:"#4ade80"};
  const labels={thought:"💡 Thought",action:"⚙ Action",obs:"👁 Observation",answer:"✅ Answer"};
  function draw(){
    svg.selectAll("*").remove();
    const k=+d3.select("#agt-step").property("value");
    svg.append("text").attr("x",20).attr("y",22).attr("fill",C.muted).attr("font-size",11).text("task: “What is double the population of France?”");
    const top=38,rowH=34;
    steps.slice(0,k).forEach((s,i)=>{
      const y=top+i*rowH;
      svg.append("rect").attr("x",20).attr("y",y).attr("width",W-40).attr("height",rowH-7).attr("rx",6).attr("fill","#15181f").attr("stroke",colors[s.t]).attr("stroke-opacity",.6);
      svg.append("text").attr("x",32).attr("y",y+18).attr("font-size",10.5).attr("font-weight",600).attr("fill",colors[s.t]).text(labels[s.t]);
      svg.append("text").attr("x",140).attr("y",y+18).attr("font-size",11).attr("fill",C.ink).attr("font-family",s.t==='action'?'SF Mono, Menlo, monospace':'inherit').text(s.x);
    });
    if(k>0 && k<7) svg.append("text").attr("x",W/2).attr("y",top+k*rowH+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("↻ loop");
    d3.select("#agt-read").html(`step <b>${k}/${steps.length}</b> — loops <b>Thought → Action → Observation</b> until it can answer`);
  }
  d3.select("#agt-step").on("input",draw);
  d3.select("#agt-next").on("click",()=>{ const v=Math.min(7,+d3.select("#agt-step").property("value")+1); d3.select("#agt-step").property("value",v); draw(); });
  draw();
})();

(function(){
  const C={accent:"#fb923c",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#rel-svg"),W=640,H=290,m={t:26,r:18,b:36,l:46},NMAX=30;
  const x=d3.scaleLinear().domain([1,NMAX]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
  const line=d3.line().x(d=>x(d.n)).y(d=>y(d.s)).curve(d3.curveMonotoneX);
  function draw(){
    svg.selectAll("*").remove();
    const p=+d3.select("#rel-p").property("value")/100;
    const n=+d3.select("#rel-n").property("value");
    const retry=d3.select("#rel-retry").property("checked");
    const pr=1-(1-p)*(1-p);                       // verify + one retry
    const base=d3.range(1,NMAX+1).map(k=>({n:k,s:Math.pow(p,k)}));
    const withRetry=d3.range(1,NMAX+1).map(k=>({n:k,s:Math.pow(pr,k)}));
    // gridlines + axes
    y.ticks(5).forEach(t=>{
      svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(t)).attr("y2",y(t)).attr("stroke",C.line).attr("stroke-opacity",.7);
      svg.append("text").attr("x",m.l-8).attr("y",y(t)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(Math.round(t*100)+"%");
    });
    [1,5,10,15,20,25,30].forEach(t=>{
      svg.append("text").attr("x",x(t)).attr("y",H-m.b+16).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(t);
    });
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.muted).text("steps in the task");
    svg.append("text").attr("x",m.l).attr("y",16).attr("font-size",11).attr("fill",C.muted).text("probability the whole task succeeds");
    // curves
    if(retry) svg.append("path").datum(withRetry).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2).attr("stroke-dasharray","5 3").attr("d",line);
    svg.append("path").datum(base).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2).attr("d",line);
    // marker at n
    const sBase=Math.pow(p,n), sRetry=Math.pow(pr,n);
    svg.append("line").attr("x1",x(n)).attr("x2",x(n)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.muted).attr("stroke-opacity",.5).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",x(n)).attr("cy",y(sBase)).attr("r",4.5).attr("fill",C.accent);
    if(retry) svg.append("circle").attr("cx",x(n)).attr("cy",y(sRetry)).attr("r",4.5).attr("fill",C.good);
    // legend
    svg.append("text").attr("x",W-m.r).attr("y",m.t+2).attr("text-anchor","end").attr("font-size",10.5).attr("fill",C.accent).text("pⁿ  (no recovery)");
    if(retry) svg.append("text").attr("x",W-m.r).attr("y",m.t+18).attr("text-anchor","end").attr("font-size",10.5).attr("fill",C.good).text("(1−(1−p)²)ⁿ  (verify + retry)");
    d3.select("#rel-read").html(
      `p = <b>${(p*100).toFixed(0)}%</b> per step · <b>${n}</b> steps → task success <b>${(sBase*100).toFixed(1)}%</b>`
      + (retry?` · with verify+retry (effective p = ${(pr*100).toFixed(1)}%) → <b>${(sRetry*100).toFixed(1)}%</b>`:"")
    );
  }
  d3.select("#rel-p").on("input",draw);
  d3.select("#rel-n").on("input",draw);
  d3.select("#rel-retry").on("change",draw);
  draw();
})();
