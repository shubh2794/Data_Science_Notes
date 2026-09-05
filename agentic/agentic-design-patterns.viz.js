/* agentic-design-patterns.viz.js — extracted from agentic-design-patterns.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#fb923c",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#pat-svg"),W=640,H=240;
  function node(x,y,w,h,label,fill){
    svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",8).attr("fill",fill||"#1e222d").attr("stroke",C.line);
    svg.append("text").attr("x",x+w/2).attr("y",y+h/2+4).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.ink).text(label);
    return {x,y,w,h};
  }
  function arrow(x1,y1,x2,y2,label){
    svg.append("line").attr("x1",x1).attr("y1",y1).attr("x2",x2).attr("y2",y2).attr("stroke",C.muted).attr("stroke-width",1.5).attr("marker-end","url(#pa)");
    if(label) svg.append("text").attr("x",(x1+x2)/2).attr("y",(y1+y2)/2-6).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(label);
  }
  function ensureMarker(){
    const d=svg.append("defs");
    d.append("marker").attr("id","pa").attr("viewBox","0 -5 10 10").attr("refX",9).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto")
      .append("path").attr("d","M0,-4L8,0L0,4").attr("fill",C.muted);
  }
  const BLUE="rgba(129,140,248,.15)", ORANGE="rgba(251,146,60,.15)", GREEN="rgba(74,222,128,.15)";
  const draws={
    chaining(){
      node(18,96,104,44,"Call 1",BLUE);
      node(150,96,66,44,"gate",ORANGE);
      node(244,96,104,44,"Call 2",BLUE);
      node(376,96,104,44,"Call 3",BLUE);
      node(508,96,104,44,"Output",GREEN);
      arrow(122,118,150,118); arrow(216,118,244,118); arrow(348,118,376,118); arrow(480,118,508,118);
      arrow(183,140,183,178);
      node(112,178,142,34,"fail → retry / exit","#15181f");
      return "Prompt chaining: a fixed sequence of calls with a programmatic gate between stages — cheapest way to turn one hard task into several easy ones.";
    },
    routing(){
      node(16,96,92,44,"Input",BLUE);
      node(140,96,102,44,"Router",ORANGE);
      node(304,24,152,40,"Cheap model");
      node(304,96,152,40,"Strong model");
      node(304,168,152,40,"Specialist + tools");
      node(506,96,110,44,"Output",GREEN);
      arrow(108,118,140,118);
      arrow(242,108,304,46); arrow(242,118,304,116); arrow(242,130,304,186);
      arrow(456,46,506,110); arrow(456,116,506,118); arrow(456,186,506,128);
      return "Routing: one small classification picks the branch, so each class gets its own prompt, tools, and cost tier — plus a fallback when the router is unsure.";
    },
    parallel(){
      node(16,96,88,44,"Input",BLUE);
      node(160,20,158,40,"Branch A / sample 1");
      node(160,96,158,40,"Branch B / sample 2");
      node(160,172,158,40,"Branch C / sample 3");
      node(380,96,124,44,"Aggregate",ORANGE);
      node(534,96,86,44,"Output",GREEN);
      arrow(104,110,160,42); arrow(104,118,160,116); arrow(104,128,160,190);
      arrow(318,40,380,110); arrow(318,116,380,118); arrow(318,192,380,128);
      arrow(504,118,534,118);
      return "Parallelization: sectioning splits independent subtasks, voting samples the same task k times — latency is the slowest branch, cost is k×.";
    },
    reflection(){
      node(40,90,120,46,"Generate",BLUE);
      node(260,90,120,46,"Critique",ORANGE);
      node(480,90,120,46,"Revise",GREEN);
      arrow(160,113,260,113); arrow(380,113,480,113);
      arrow(540,90,540,60); arrow(540,60,100,60); arrow(100,60,100,90,"loop until good");
      node(240,178,160,34,"stop: score ≥ τ or k ≥ kmax","#15181f");
      return "Reflection: generate → critique → revise, looping until the output passes. Needs an explicit stop condition or it polishes forever.";
    },
    evalopt(){
      node(46,88,148,48,"Generator",BLUE);
      node(288,88,148,48,"Evaluator (rubric)",ORANGE);
      node(506,88,110,48,"Accept",GREEN);
      arrow(194,104,288,104,"draft");
      arrow(288,128,194,128,"score + critique");
      arrow(436,112,506,112,"score ≥ τ");
      node(266,182,192,34,"tests · schema · tool result","#15181f");
      arrow(362,182,362,140);
      return "Evaluator–optimizer: a separate critic scores against a rubric and returns actionable feedback. The gain comes from the critic having information the generator lacked.";
    },
    tool(){
      node(60,90,130,46,"Model (think)",BLUE);
      node(420,90,140,46,"Tool / API",ORANGE);
      arrow(190,105,420,105,"action");
      arrow(420,121,190,121,"observation");
      node(250,170,140,40,"→ answer when done","#15181f");
      return "ReAct: the model calls a tool, reads the observation, and repeats until it can answer. Reliability = select × args × use.";
    },
    planning(){
      node(40,40,150,42,"Plan (decompose)",BLUE);
      node(40,120,120,40,"Step 1");
      node(200,120,120,40,"Step 2");
      node(360,120,120,40,"Step 3");
      node(520,120,90,40,"Result",GREEN);
      arrow(115,82,115,120);
      arrow(160,140,200,140); arrow(320,140,360,140); arrow(480,140,520,140);
      arrow(420,120,420,90); arrow(420,90,190,61,"re-plan on failure");
      return "Planning: decompose the goal into an ordered plan, execute each step, and re-plan when an observation contradicts the expectation.";
    },
    multi(){
      node(250,30,150,42,"Orchestrator",ORANGE);
      node(60,130,130,40,"Worker: research");
      node(255,130,130,40,"Worker: write");
      node(450,130,130,40,"Worker: check");
      arrow(300,72,125,130); arrow(322,72,320,130); arrow(345,72,515,130);
      arrow(125,170,300,205); arrow(320,170,320,205); arrow(515,170,340,205);
      node(250,200,150,34,"Synthesise →","#15181f");
      return "Orchestrator–workers: the lead decomposes at runtime and gives each worker a clean, narrow window — context isolation is the real win, not extra brains.";
    },
  };
  function draw(){
    svg.selectAll("*").remove();
    ensureMarker();
    const k=d3.select("#pat-sel").property("value");
    const note=draws[k]();
    d3.select("#pat-read").html(note);
  }
  d3.select("#pat-sel").on("change",draw);
  draw();
})();

(function(){
  const C={accent:"#fb923c",good:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#rel-svg"),W=640,H=250,M={t:18,r:16,b:34,l:46};
  const NMAX=30;
  const x=d3.scaleLinear().domain([1,NMAX]).range([M.l,W-M.r]);
  const y=d3.scaleLinear().domain([0,1]).range([H-M.b,M.t]);
  function draw(){
    svg.selectAll("*").remove();
    const p=+d3.select("#rel-p").property("value");
    const r=+d3.select("#rel-r").property("value");
    const pv=p+(1-p)*r;                       // effective per-step rate with verify-and-retry
    // grid + axes
    d3.range(0,1.01,0.25).forEach(v=>{
      svg.append("line").attr("x1",M.l).attr("x2",W-M.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",C.line).attr("stroke-dasharray","2,3");
      svg.append("text").attr("x",M.l-8).attr("y",y(v)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(d3.format(".0%")(v));
    });
    [1,5,10,15,20,25,30].forEach(v=>{
      svg.append("text").attr("x",x(v)).attr("y",H-M.b+16).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(v);
    });
    svg.append("text").attr("x",(M.l+W-M.r)/2).attr("y",H-4).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("steps in the task (n)");
    const line=d3.line().x(d=>x(d.n)).y(d=>y(d.v)).curve(d3.curveMonotoneX);
    const mk=q=>d3.range(1,NMAX+1).map(n=>({n,v:Math.pow(q,n)}));
    svg.append("path").datum(mk(pv)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2).attr("d",line);
    svg.append("path").datum(mk(p)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2).attr("d",line);
    // legend
    const lg=[["plain loop  p = "+p.toFixed(3),C.accent],["verify + retry  p′ = "+pv.toFixed(3),C.good]];
    lg.forEach((d,i)=>{
      svg.append("rect").attr("x",W-M.r-186).attr("y",M.t+i*18).attr("width",10).attr("height",10).attr("rx",2).attr("fill",d[1]);
      svg.append("text").attr("x",W-M.r-170).attr("y",M.t+9+i*18).attr("font-size",10.5).attr("fill",C.ink).text(d[0]);
    });
    const f=d3.format(".1%");
    d3.select("#rel-read").html(
      `at <b>n = 10</b>: plain <b style="color:${C.accent}">${f(Math.pow(p,10))}</b> · verified <b style="color:${C.good}">${f(Math.pow(pv,10))}</b>` +
      ` &nbsp;·&nbsp; at <b>n = 30</b>: plain <b style="color:${C.accent}">${f(Math.pow(p,30))}</b> · verified <b style="color:${C.good}">${f(Math.pow(pv,30))}</b>`);
  }
  d3.select("#rel-p").on("input",draw);
  d3.select("#rel-r").on("input",draw);
  draw();
})();
