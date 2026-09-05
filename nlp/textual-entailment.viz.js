/* textual-entailment.viz.js — extracted from textual-entailment.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#nli-svg"),W=640;
  const premise="A man is playing a guitar on stage.";
  const ex=[
    {h:"A person is performing music.", p:[0.88,0.04,0.08], label:"entailment"},
    {h:"The stage is completely empty.", p:[0.03,0.91,0.06], label:"contradiction"},
    {h:"The musician is world-famous.", p:[0.12,0.09,0.79], label:"neutral"},
  ];
  const cls=["entailment","contradiction","neutral"], col={entailment:C.good,contradiction:C.bad,neutral:C.a2};
  function draw(){
    svg.selectAll("*").remove();
    const e=ex[+d3.select("#nli-sel").property("value")];
    svg.append("text").attr("x",20).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("premise");
    svg.append("text").attr("x",20).attr("y",44).attr("fill",C.ink).attr("font-size",13).text("“"+premise+"”");
    svg.append("text").attr("x",20).attr("y",74).attr("fill",C.muted).attr("font-size",11).text("hypothesis");
    svg.append("text").attr("x",20).attr("y",94).attr("fill",C.accent).attr("font-size",13).text("“"+e.h+"”");
    const x0=200, bw=300, top=120, rowH=30;
    cls.forEach((c,i)=>{
      const y=top+i*rowH;
      svg.append("text").attr("x",x0-12).attr("y",y+13).attr("text-anchor","end").attr("font-size",11).attr("fill",e.label===c?col[c]:C.muted).attr("font-weight",e.label===c?700:400).text(c);
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",bw).attr("height",16).attr("rx",4).attr("fill","#15181f");
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",e.p[i]*bw).attr("height",16).attr("rx",4).attr("fill",col[c]).attr("fill-opacity",e.label===c?0.95:0.5);
      svg.append("text").attr("x",x0+e.p[i]*bw+8).attr("y",y+13).attr("font-size",10).attr("fill",col[c]).text((e.p[i]*100).toFixed(0)+"%");
    });
    d3.select("#nli-read").html(`predicted: <b style="color:${col[e.label]}">${e.label}</b> (${(Math.max(...e.p)*100).toFixed(0)}% confidence)`);
  }
  d3.select("#nli-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 05 · zero-shot classification via entailment ───────────────────────── */
(function(){
  const C={accent:"#c084fc",good:"#4ade80",bad:"#f87171",a2:"#ffb454",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const labels=["sports","politics","technology","cooking","finance"];
  // plausible per-label entailment logits per example (run through a real softmax in JS)
  const examples=[
    {text:"The striker scored a hat-trick in the cup final.",          logits:[5.4,0.6,0.3,0.1,0.5]},
    {text:"The central bank raised interest rates again this quarter.", logits:[0.2,1.6,0.5,0.1,5.1]},
    {text:"The new chip doubles on-device inference throughput.",       logits:[0.3,0.4,5.2,0.2,1.1]},
    {text:"Fold the egg whites gently into the batter before baking.",  logits:[0.2,0.1,0.4,5.0,0.2]},
  ];
  const sel=d3.select("#zs-sel");
  sel.selectAll("option").data(examples).join("option").attr("value",(d,i)=>i).text(d=>"“"+d.text+"”");

  function softmax(z){ const mx=Math.max(...z); const e=z.map(v=>Math.exp(v-mx)); const s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }

  const svg=d3.select("#zs-svg"),W=640,H=240,m={t:18,r:60,b:20,l:96};
  const gBars=svg.append("g");
  svg.append("text").attr("x",m.l).attr("y",12).attr("font-size",10).attr("fill",C.muted).text("P(entailment | text, “This text is about {label}”)");

  function draw(){
    const ex=examples[+sel.property("value")];
    const probs=softmax(ex.logits);
    const rows=labels.map((l,i)=>({label:l,logit:ex.logits[i],p:probs[i]})).sort((a,b)=>b.p-a.p);
    const best=rows[0];

    const y=d3.scaleBand().domain(rows.map(d=>d.label)).range([m.t,H-m.b]).padding(0.3);
    const x=d3.scaleLinear().domain([0,1]).range([m.l,W-m.r]);

    const g=gBars.selectAll("g.zr").data(rows,d=>d.label);
    const en=g.enter().append("g").attr("class","zr");
    en.append("rect").attr("class","bg").attr("rx",4).attr("fill","#15181f");
    en.append("rect").attr("class","fg").attr("rx",4);
    en.append("text").attr("class","lab").attr("text-anchor","end").attr("font-size",12);
    en.append("text").attr("class","val").attr("font-size",11);
    const all=en.merge(g);
    all.select("rect.bg").attr("x",m.l).attr("y",d=>y(d.label)).attr("width",W-m.r-m.l).attr("height",y.bandwidth());
    all.select("rect.fg").attr("x",m.l).attr("y",d=>y(d.label)).attr("height",y.bandwidth())
      .transition().duration(450).attr("width",d=>Math.max(1,x(d.p)-m.l))
      .attr("fill",d=>d.label===best.label?C.good:C.accent).attr("fill-opacity",d=>d.label===best.label?0.95:0.55);
    all.select("text.lab").attr("x",m.l-10).attr("y",d=>y(d.label)+y.bandwidth()/2+4)
      .attr("fill",d=>d.label===best.label?C.good:C.muted).attr("font-weight",d=>d.label===best.label?700:400).text(d=>d.label);
    all.select("text.val").attr("x",d=>Math.max(m.l+2,x(d.p))+8).attr("y",d=>y(d.label)+y.bandwidth()/2+4)
      .attr("fill",d=>d.label===best.label?C.good:C.ink).text(d=>(d.p*100).toFixed(1)+"%");
    g.exit().remove();

    d3.select("#zs-pred").html(`argmax ⇒ <b style="color:${C.good}">${best.label}</b>`);
    d3.select("#zs-read").html(`predicted class: <b style="color:${C.good}">${best.label}</b> &nbsp; (${(best.p*100).toFixed(1)}% of the softmax mass over ${labels.length} candidate labels)`);
  }
  sel.on("change",draw);
  draw();
})();
