/* text-preprocessing.viz.js — extracted from text-preprocessing.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#pre-svg"),W=640;
  const raw="The Quick, Brown FOXES are Running fast!!!";
  const stop=new Set(["the","are","a","an","is","of","to","and","in"]);
  function stem(w){ return w.replace(/(ing|es|ed|s)$/,"")||w; }
  function draw(){
    svg.selectAll("*").remove();
    const lc=d3.select("#pre-lc").property("checked"), pu=d3.select("#pre-punct").property("checked"),
          st=d3.select("#pre-stop").property("checked"), sm=d3.select("#pre-stem").property("checked");
    svg.append("text").attr("x",20).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("raw text");
    svg.append("text").attr("x",20).attr("y",44).attr("fill",C.ink).attr("font-size",13).attr("font-family","SF Mono, Menlo, monospace").text("“"+raw+"”");
    let txt=raw;
    if(lc) txt=txt.toLowerCase();
    if(pu) txt=txt.replace(/[^A-Za-z0-9 ]/g,"");
    let toks=txt.split(/\s+/).filter(Boolean);
    if(st) toks=toks.filter(t=>!stop.has(t.toLowerCase()));
    if(sm) toks=toks.map(stem);
    svg.append("text").attr("x",20).attr("y",84).attr("fill",C.muted).attr("font-size",11).text("→ tokens");
    let x=20, y=104;
    toks.forEach(t=>{
      const w=Math.max(28,t.length*9+16);
      if(x+w>W-20){x=20;y+=34;}
      svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",26).attr("rx",6).attr("fill","rgba(192,132,252,.15)").attr("stroke",C.accent);
      svg.append("text").attr("x",x+w/2).attr("y",y+17).attr("text-anchor","middle").attr("font-size",11.5).attr("fill",C.ink).attr("font-family","SF Mono, Menlo, monospace").text(t);
      x+=w+8;
    });
    d3.select("#pre-read").html(`<b>${toks.length}</b> tokens after preprocessing (raw had ${raw.split(/\s+/).length} words)`);
  }
  ["#pre-lc","#pre-punct","#pre-stop","#pre-stem"].forEach(s=>d3.select(s).on("change",draw));
  draw();
})();

/* ───────────────────────── 05 · TF-IDF over a toy corpus ───────────────────────── */
(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const corpus=[
    {name:"D1 · cats", text:"the cat sat on the soft mat the cat purred"},
    {name:"D2 · dogs", text:"the dog ran in the park the dog barked loud"},
    {name:"D3 · cooking", text:"the chef cooked the pasta with fresh basil and oil"},
    {name:"D4 · space", text:"the rocket reached orbit the crew watched the stars"},
  ];
  const tok=s=>s.toLowerCase().split(/\s+/).filter(Boolean);
  const docs=corpus.map(d=>tok(d.text));
  const N=docs.length;
  // document frequency per term
  const df=new Map();
  docs.forEach(toks=>{ new Set(toks).forEach(t=>df.set(t,(df.get(t)||0)+1)); });
  const idf=t=>Math.log(N/(1+(df.get(t)||0)));

  const sel=d3.select("#tf-sel");
  sel.selectAll("option").data(corpus).join("option").attr("value",(d,i)=>i).text(d=>d.name);
  d3.select("#tf-corpus").html(`corpus of <b>${N}</b> docs · idf(t) = ln( N / (1 + df) )`);

  const svg=d3.select("#tf-svg"),W=640,H=280,m={t:22,r:60,b:24,l:78};
  const gAxis=svg.append("g"), gBars=svg.append("g");
  gAxis.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text("tf-idf  =  tf · idf");

  function draw(){
    const di=+sel.property("value"), toks=docs[di];
    const tf=new Map();
    toks.forEach(t=>tf.set(t,(tf.get(t)||0)+1));
    const rows=[...tf.keys()].map(t=>({term:t, tf:tf.get(t), idf:idf(t), score:tf.get(t)*idf(t)}))
      .sort((a,b)=>b.score-a.score);

    const y=d3.scaleBand().domain(rows.map(d=>d.term)).range([m.t,H-m.b]).padding(0.28);
    const maxS=d3.max(rows,d=>Math.max(d.score,0.001));
    const x=d3.scaleLinear().domain([0,maxS*1.08]).range([m.l,W-m.r]);
    // "the" appears in every doc → idf = ln(N/(N+1)) < 0, so its tf-idf is ≤ 0: a common term
    const isCommon=d=>d.score<=0;

    const g=gBars.selectAll("g.tr").data(rows,d=>d.term);
    const en=g.enter().append("g").attr("class","tr");
    en.append("rect").attr("class","bg").attr("rx",3).attr("fill","#15181f");
    en.append("rect").attr("class","fg").attr("rx",3);
    en.append("text").attr("class","lab").attr("text-anchor","end").attr("font-size",11).attr("font-family","SF Mono, Menlo, monospace");
    en.append("text").attr("class","val").attr("font-size",10);
    const all=en.merge(g);
    all.select("rect.bg").attr("x",m.l).attr("y",d=>y(d.term)).attr("width",W-m.r-m.l).attr("height",y.bandwidth());
    all.select("rect.fg").attr("x",m.l).attr("y",d=>y(d.term)).attr("height",y.bandwidth())
      .transition().duration(450).attr("width",d=>isCommon(d)?3:Math.max(1,x(d.score)-m.l))
      .attr("fill",d=>isCommon(d)?C.bad:C.accent).attr("fill-opacity",.85);
    all.select("text.lab").attr("x",m.l-8).attr("y",d=>y(d.term)+y.bandwidth()/2+4).attr("fill",d=>isCommon(d)?C.muted:C.ink).text(d=>d.term);
    all.select("text.val").attr("x",d=>(isCommon(d)?m.l+3:Math.max(m.l+2,x(d.score)))+6).attr("y",d=>y(d.term)+y.bandwidth()/2+4)
      .attr("fill",d=>isCommon(d)?C.bad:C.good).text(d=>d.score.toFixed(2)+"  (tf "+d.tf+", idf "+d.idf.toFixed(2)+")");
    g.exit().remove();

    const top=rows[0], zero=rows.filter(d=>isCommon(d)).map(d=>d.term);
    d3.select("#tf-read").html(`most distinctive in <b>${corpus[di].name}</b>: <b style="color:${C.good}">${top.term}</b> (${top.score.toFixed(2)}) &nbsp;·&nbsp; near-zero (common): <b style="color:${C.bad}">${zero.join(", ")||"none"}</b>`);
  }
  sel.on("change",draw);
  draw();
})();
