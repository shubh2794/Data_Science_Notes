/* large-language-models.viz.js — extracted from large-language-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const contexts=[
    {ctx:"The cat sat on the", toks:[["mat",4.0],["roof",2.6],["floor",2.4],["sofa",2.0],["table",1.8],["windowsill",1.2],["throne",0.4],["algorithm",-1.5]]},
    {ctx:"To be or not to", toks:[["be",5.0],["do",1.2],["live",1.0],["exist",0.9],["fight",0.6],["sleep",0.5],["dream",0.4],["compute",-1.0]]},
    {ctx:"The capital of France is", toks:[["Paris",5.5],["located",1.5],["a",1.2],["the",1.0],["home",0.8],["Lyon",0.5],["beautiful",0.3],["London",-0.5]]},
    {ctx:"def add(a, b): return", toks:[["a + b",4.2],["a",3.6],["b",2.2],["sum(a",1.8],["result",1.4],["(a",1.0],["None",0.2],["cat",-2.0]]},
  ];
  const svg=d3.select("#llm-svg"),W=640,H=360;
  function softmax(logits,T){ const m=Math.max(...logits); const ex=logits.map(l=>Math.exp((l-m)/T)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(v=>v/s); }
  function draw(){
    svg.selectAll("*").remove();
    const ci=+d3.select("#llm-ctx").property("value");
    const T=+d3.select("#llm-temp").property("value");
    const P=+d3.select("#llm-topp").property("value");
    const data=contexts[ci];
    let toks=data.toks.map(t=>({w:t[0],logit:t[1]}));
    const probs=softmax(toks.map(t=>t.logit),T);
    toks.forEach((t,i)=>t.p=probs[i]);
    toks.sort((a,b)=>b.p-a.p);
    let cum=0; toks.forEach((t,i)=>{ t.inN = (cum < P) || i===0; cum+=t.p; });
    const left=180, top=66, barMax=W-left-72, rowH=(H-top-16)/toks.length;
    svg.append("text").attr("x",24).attr("y",26).attr("fill",C.muted).attr("font-size",12).text("context");
    svg.append("text").attr("x",24).attr("y",46).attr("fill",C.ink).attr("font-size",13).attr("font-family","SF Mono, Menlo, monospace").text("“"+data.ctx+" ▮”");
    toks.forEach((t,i)=>{
      const y=top+i*rowH;
      svg.append("text").attr("x",left-12).attr("y",y+rowH/2+1).attr("text-anchor","end").attr("font-size",12)
        .attr("fill",t.inN?C.ink:C.muted).attr("font-family","SF Mono, Menlo, monospace").text(t.w);
      svg.append("rect").attr("x",left).attr("y",y+3).attr("width",barMax).attr("height",rowH-8).attr("rx",4).attr("fill","#15181f");
      svg.append("rect").attr("x",left).attr("y",y+3).attr("width",Math.max(2,t.p*barMax)).attr("height",rowH-8).attr("rx",4)
        .attr("fill",t.inN?C.accent:"#3a4150");
      svg.append("text").attr("x",left+Math.max(2,t.p*barMax)+8).attr("y",y+rowH/2+1).attr("font-size",11)
        .attr("fill",t.inN?C.accent:C.muted).text((t.p*100).toFixed(1)+"%");
    });
    const Hent=-toks.reduce((a,t)=>a+(t.p>1e-9?t.p*Math.log2(t.p):0),0);
    const inN=toks.filter(t=>t.inN).length;
    const mood = T<0.5?" · sharp / confident" : T>1.3?" · flat / random" : "";
    d3.select("#llm-read").html(`temperature <b>${T.toFixed(2)}</b> · top-p <b>${P.toFixed(2)}</b> → nucleus keeps <b>${inN}</b> token${inN>1?"s":""} · entropy <b>${Hent.toFixed(2)}</b> bits${mood}`);
  }
  ["#llm-ctx","#llm-temp","#llm-topp"].forEach(s=>d3.select(s).on("input",draw).on("change",draw));
  draw();
})();
