/* gpt.viz.js — extracted from gpt.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#gpt-svg"),W=640;
  const prompt=["The","robot","learned","to"];
  const steps=[
    {cands:[["walk",0.40],["read",0.22],["fly",0.15],["cook",0.10]], pick:"walk"},
    {cands:[["and",0.48],["very",0.16],[",",0.12],["fast",0.09]], pick:"and"},
    {cands:[["talk",0.44],["dance",0.20],["paint",0.13]], pick:"talk"},
    {cands:[[".",0.58],["fluently",0.22],["slowly",0.10]], pick:"."},
  ];
  let gen=0;
  function draw(){
    svg.selectAll("*").remove();
    const seq=prompt.concat(steps.slice(0,gen).map(s=>s.pick));
    // sequence
    let x=20; const y=44;
    seq.forEach((t,i)=>{
      const isGen=i>=prompt.length, w=Math.max(40,t.length*9+18);
      if(x+w>W-20){x=20;}
      svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",30).attr("rx",6)
        .attr("fill",isGen?"rgba(45,212,191,.18)":"#15181f").attr("stroke",isGen?C.accent:"#2a2f3a");
      svg.append("text").attr("x",x+w/2).attr("y",y+20).attr("text-anchor","middle").attr("font-size",12).attr("fill",isGen?C.accent:C.ink).attr("font-family","SF Mono,monospace").text(t);
      x+=w+6;
    });
    svg.append("text").attr("x",20).attr("y",26).attr("fill",C.muted).attr("font-size",10).text(gen<steps.length?"prompt + "+gen+" generated token(s) — predicting the next:":"complete sequence");
    // candidate bars for next step
    if(gen<steps.length){
      const cands=steps[gen].cands, top=100, x0=160, bw=300;
      svg.append("text").attr("x",x0).attr("y",top-8).attr("fill",C.muted).attr("font-size",11).text("next-token distribution");
      cands.forEach((c,i)=>{
        const yy=top+i*28;
        svg.append("text").attr("x",x0-12).attr("y",yy+13).attr("text-anchor","end").attr("font-size",11).attr("fill",i===0?C.good:C.ink).attr("font-family","SF Mono,monospace").text(c[0]);
        svg.append("rect").attr("x",x0).attr("y",yy).attr("width",bw).attr("height",16).attr("rx",4).attr("fill","#15181f");
        svg.append("rect").attr("x",x0).attr("y",yy).attr("width",c[1]*bw).attr("height",16).attr("rx",4).attr("fill",i===0?C.good:C.accent).attr("fill-opacity",i===0?0.9:0.55);
        svg.append("text").attr("x",x0+c[1]*bw+8).attr("y",yy+13).attr("font-size",10).attr("fill",i===0?C.good:C.muted).text((c[1]*100).toFixed(0)+"%");
      });
    }
    d3.select("#gpt-read").html(gen<steps.length?`each step samples the next token (here picking <b style="color:${C.good}">${steps[gen].pick}</b>) using only the tokens to its <b>left</b>`:`generated left-to-right — <b>"${seq.slice(prompt.length).join(" ")}"</b> — each token saw only what came before it`);
  }
  d3.select("#gpt-next").on("click",()=>{ if(gen<steps.length){gen++; draw();} });
  d3.select("#gpt-reset").on("click",()=>{ gen=0; draw(); });
  draw();
})();
