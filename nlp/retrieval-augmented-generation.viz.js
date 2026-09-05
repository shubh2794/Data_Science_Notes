/* retrieval-augmented-generation.viz.js — extracted from retrieval-augmented-generation.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const docs=[
    {id:"D1",title:"Positional encodings",text:"Transformers add positional encodings so the model knows word order, since attention itself is permutation-invariant.",kw:["transformer","position","order","word","attention"]},
    {id:"D2",title:"Sampling & temperature",text:"Temperature divides the logits before softmax; high temperature flattens the distribution, raising randomness and creativity.",kw:["temperature","creativity","sampling","random","softmax","logits"]},
    {id:"D3",title:"Why RAG",text:"RAG grounds answers in retrieved documents, fixing stale knowledge and reducing hallucination in language models.",kw:["rag","retrieval","hallucination","knowledge","ground","problem"]},
    {id:"D4",title:"Sub-word tokenization",text:"Rare words are split into sub-word pieces from a fixed vocabulary, so there are no out-of-vocabulary tokens.",kw:["token","subword","rare","word","vocabulary"]},
    {id:"D5",title:"Embeddings",text:"Embeddings map tokens to vectors where semantic similarity becomes geometric proximity.",kw:["embedding","vector","similarity","semantic"]},
    {id:"D6",title:"Attention",text:"Self-attention lets every token weigh every other token in parallel via query-key matching.",kw:["attention","query","key","token","parallel"]},
  ];
  const queries=[
    {q:"How do transformers handle word order?",kw:["transformer","word","order","position"],ans:d=>`Transformers add **positional encodings** to token embeddings, because attention alone is order-agnostic. (from ${d})`},
    {q:"Why does temperature affect creativity?",kw:["temperature","creativity","random","sampling"],ans:d=>`**Temperature** rescales logits before softmax — higher values flatten the distribution, so sampling is more random and "creative". (from ${d})`},
    {q:"What problem does RAG solve?",kw:["rag","problem","hallucination","knowledge","retrieval"],ans:d=>`**RAG** grounds the answer in retrieved documents, fixing stale knowledge and cutting hallucination. (from ${d})`},
    {q:"How are rare words tokenized?",kw:["rare","word","token","subword","vocabulary"],ans:d=>`Rare words are broken into **sub-word pieces** from a fixed vocabulary, so nothing is out-of-vocabulary. (from ${d})`},
  ];
  const svg=d3.select("#rag-svg"),W=640,H=430;
  function score(qkw,d){ let s=0; qkw.forEach(k=>{ if(d.kw.includes(k))s+=1; d.kw.forEach(dk=>{if(dk.includes(k)||k.includes(dk))s+=0.2;}); }); return s; }
  function run(){
    svg.selectAll("*").remove();
    const qi=+d3.select("#rag-q").property("value"); const Q=queries[qi];
    const scored=docs.map(d=>({d,s:score(Q.kw,d)})).sort((a,b)=>b.s-a.s);
    const maxS=d3.max(scored,o=>o.s)||1;
    const top=scored.slice(0,2).map(o=>o.d.id);
    svg.append("rect").attr("x",20).attr("y",16).attr("width",W-40).attr("height",40).attr("rx",8).attr("fill","#1e222d").attr("stroke",C.accent).attr("opacity",0).transition().duration(300).attr("opacity",1);
    svg.append("text").attr("x",32).attr("y",41).attr("fill",C.ink).attr("font-size",13).attr("opacity",0).transition().delay(150).duration(300).attr("opacity",1).text("❓ "+Q.q);
    const y0=80, rowH=42;
    scored.forEach((o,i)=>{
      const g=svg.append("g").attr("opacity",0);
      g.transition().delay(400+i*110).duration(300).attr("opacity",1);
      const isTop=top.includes(o.d.id);
      g.append("rect").attr("x",20).attr("y",y0+i*rowH).attr("width",W-40).attr("height",rowH-8).attr("rx",7)
        .attr("fill",isTop?"rgba(192,132,252,.14)":"#15181f").attr("stroke",isTop?C.accent:C.line);
      g.append("text").attr("x",34).attr("y",y0+i*rowH+17).attr("fill",isTop?C.accent:C.ink).attr("font-size",12).attr("font-weight",600).text(`${o.d.id} · ${o.d.title}`);
      g.append("text").attr("x",34).attr("y",y0+i*rowH+30).attr("fill",C.muted).attr("font-size",10).text(o.d.text.slice(0,70)+"…");
      const bw=120;
      g.append("rect").attr("x",W-40-bw).attr("y",y0+i*rowH+10).attr("width",bw).attr("height",6).attr("rx",3).attr("fill","#2a2f3a");
      g.append("rect").attr("x",W-40-bw).attr("y",y0+i*rowH+10).attr("width",0).attr("height",6).attr("rx",3).attr("fill",isTop?C.accent:"#4b5563")
        .transition().delay(500+i*110).duration(400).attr("width",bw*(o.s/maxS));
      if(isTop) g.append("text").attr("x",W-46).attr("y",y0+i*rowH+30).attr("text-anchor","end").attr("font-size",9).attr("fill",C.accent).text("✓ retrieved");
    });
    const ay=y0+scored.length*rowH+14;
    const ans=svg.append("g").attr("opacity",0);
    ans.transition().delay(1200).duration(400).attr("opacity",1);
    ans.append("rect").attr("x",20).attr("y",ay).attr("width",W-40).attr("height",58).attr("rx",8).attr("fill","rgba(74,222,128,.1)").attr("stroke",C.good);
    ans.append("text").attr("x",32).attr("y",ay+20).attr("fill",C.good).attr("font-size",11).attr("font-weight",600).text("💬 Generated answer (grounded in retrieved docs)");
    const txt=Q.ans(top.join(" + ")).replace(/\*\*(.+?)\*\*/g,"$1");
    const words=txt.split(" "); let line="",lines=[];
    words.forEach(w=>{ if((line+w).length>78){lines.push(line);line="";} line+=w+" "; }); lines.push(line);
    lines.forEach((ln,i)=>ans.append("text").attr("x",32).attr("y",ay+38+i*15).attr("fill",C.ink).attr("font-size",12).text(ln));
  }
  d3.select("#rag-run").on("click",run);
  run();
})();
