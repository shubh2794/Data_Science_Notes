/* text-summarization.viz.js — extracted from text-summarization.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* ───────────────────────── 02 · TextRank sentence graph ───────────────────────── */
(function(){
  // toy document: 6 sentences, fixed positions; weights = word-overlap similarity
  const sents=[
    {id:0,t:"Solar power is a clean source of energy.",x:150,y:80},
    {id:1,t:"Solar panels convert sunlight into electricity.",x:330,y:60},
    {id:2,t:"Clean energy reduces carbon emissions sharply.",x:500,y:120},
    {id:3,t:"Wind turbines also generate clean electricity.",x:470,y:250},
    {id:4,t:"Reducing emissions slows global climate change.",x:300,y:270},
    {id:5,t:"Many countries now invest in solar energy.",x:130,y:220},
  ];
  // symmetric similarity weights (hand-set to mimic word overlap)
  const W=[
    [0,.5,.5,.2,.1,.6],
    [.5,0,.3,.4,.1,.4],
    [.5,.3,0,.5,.6,.3],
    [.2,.4,.5,0,.3,.2],
    [.1,.1,.6,.3,0,.1],
    [.6,.4,.3,.2,.1,0],
  ];
  const n=sents.length, d=0.85;
  let score=sents.map(()=>1/n), iter=0;
  const svg=d3.select("#rank-svg");
  const gE=svg.append("g"), gN=svg.append("g");
  const kEl=document.getElementById("rank-k");
  // draw edges once (static layout)
  const edges=[];
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++) if(W[i][j]>0.25) edges.push({i,j,w:W[i][j]});
  gE.selectAll("line").data(edges).join("line")
    .attr("x1",e=>sents[e.i].x).attr("y1",e=>sents[e.i].y)
    .attr("x2",e=>sents[e.j].x).attr("y2",e=>sents[e.j].y)
    .attr("stroke",PC.line).attr("stroke-width",e=>0.6+e.w*3).attr("stroke-opacity",0.5);
  const rScale=d3.scaleLinear().domain([0.5/n,2/n]).range([10,30]);
  function step(){
    const next=sents.map((_,i)=>{
      let acc=0;
      for(let j=0;j<n;j++){ if(j===i||W[j][i]===0) continue;
        const outSum=W[j].reduce((a,b)=>a+b,0);
        acc += (W[j][i]/outSum)*score[j];
      }
      return (1-d)/n + d*acc;
    });
    const s=next.reduce((a,b)=>a+b,0); score=next.map(v=>v/s); iter++;
    render();
  }
  function render(){
    const k=+kEl.value;
    const order=score.map((v,i)=>({i,v})).sort((a,b)=>b.v-a.v);
    const chosen=new Set(order.slice(0,k).map(o=>o.i));
    const g=gN.selectAll("g.node").data(sents,s=>s.id).join(enter=>{
        const e=enter.append("g").attr("class","node").attr("transform",s=>`translate(${s.x},${s.y})`);
        e.append("circle").attr("stroke","#0f1117").attr("stroke-width",2);
        e.append("text").attr("text-anchor","middle").attr("dy",4).attr("font-size",12).attr("font-weight",700);
        e.append("title");
        return e;});
    g.select("circle").transition().duration(400)
      .attr("r",d=>rScale(score[d.id]))
      .attr("fill",d=>chosen.has(d.id)?PC.accent:"#2c3340")
      .attr("stroke",d=>chosen.has(d.id)?"#fff":PC.line);
    g.select("text").attr("fill",d=>chosen.has(d.id)?"#1a1320":PC.ink).text(d=>"S"+(d.id+1));
    g.select("title").text(d=>`S${d.id+1}: ${d.t}  ·  score ${score[d.id].toFixed(3)}`);
    d3.select("#rank-iter").text("iteration "+iter);
    const picked=order.slice(0,k).map(o=>o.i).sort((a,b)=>a-b);
    d3.select("#rank-readout").html("<b>Summary (top-"+k+"):</b> "+picked.map(i=>sents[i].t).join(" "));
  }
  document.getElementById("rank-step").addEventListener("click",step);
  document.getElementById("rank-reset").addEventListener("click",()=>{score=sents.map(()=>1/n);iter=0;render();});
  kEl.addEventListener("input",render);
  render();
})();

/* ───────────────────────── 04 · faithfulness: copy vs fabricate ───────────────────────── */
(function(){
  const source=["The","council","approved","the","new","park","on","Tuesday"];
  // extractive summary = a copied span; abstractive introduces an unsupported token ("unanimously")
  const ext=[{w:"The",src:true},{w:"council",src:true},{w:"approved",src:true},{w:"the",src:true},{w:"park",src:true}];
  const abs=[{w:"The",src:true},{w:"council",src:true},{w:"unanimously",src:false},{w:"backed",src:false},{w:"the",src:true},{w:"park",src:true}];
  const svg=d3.select("#faith-svg"),W=640;
  svg.append("text").attr("x",24).attr("y",30).attr("font-size",12).attr("fill",PC.muted).text("SOURCE");
  const gSrc=svg.append("g").attr("transform","translate(24,44)");
  let sx=0;
  source.forEach(w=>{ const tw=w.length*8+22;
    gSrc.append("rect").attr("x",sx).attr("y",0).attr("width",tw).attr("height",28).attr("rx",6).attr("fill","#222733").attr("stroke",PC.line);
    gSrc.append("text").attr("x",sx+tw/2).attr("y",19).attr("text-anchor","middle").attr("font-size",12).attr("fill",PC.ink).text(w);
    sx+=tw+8; });
  svg.append("text").attr("x",24).attr("y",118).attr("font-size",12).attr("fill",PC.muted).text("SUMMARY");
  const gSum=svg.append("g").attr("transform","translate(24,132)");
  const out=document.getElementById("faith-readout");
  function render(mode){
    const words=mode==="ext"?ext:abs;
    let x=0;
    const g=gSum.selectAll("g.tok").data(words,(d,i)=>i+"-"+d.w).join(
      enter=>{const e=enter.append("g").attr("class","tok");
        e.append("rect").attr("y",0).attr("height",28).attr("rx",6);
        e.append("text").attr("y",19).attr("text-anchor","middle").attr("font-size",12).attr("font-weight",600);
        return e;}, update=>update, exit=>exit.remove());
    g.each(function(d){ const tw=d.w.length*8+22; d._tw=tw; });
    g.attr("transform",function(d){ const t=`translate(${x},0)`; x+=d._tw+8; return t; });
    g.select("rect").attr("width",d=>d._tw)
      .attr("fill",d=>d.src?d3.color(PC.good).copy({opacity:0.25}):d3.color(PC.bad).copy({opacity:0.3}))
      .attr("stroke",d=>d.src?PC.good:PC.bad);
    g.select("text").attr("x",d=>d._tw/2).attr("fill",d=>d.src?PC.ink:PC.bad).text(d=>d.w);
    const bad=words.filter(d=>!d.src).map(d=>d.w);
    out.innerHTML=mode==="ext"
      ? `<b style="color:${PC.good}">Extractive:</b> every word traces to the source — faithful by construction.`
      : `<b style="color:${PC.bad}">Abstractive:</b> introduced "${bad.join(", ")}" — not in the source (potential hallucination).`;
  }
  document.getElementById("faith-ext").onclick=()=>render("ext");
  document.getElementById("faith-abs").onclick=()=>render("abs");
  render("ext");
})();
