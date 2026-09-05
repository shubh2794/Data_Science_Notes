/* knowledge-distillation.viz.js — extracted from knowledge-distillation.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#kd-svg"),W=640,H=260;
  const labels=["cat","dog","fox","car","ship","tree"];
  const logits=[6.0,3.2,2.6,0.4,0.1,1.0];
  function softmax(z,T){ const m=Math.max(...z); const e=z.map(v=>Math.exp((v-m)/T)); const s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }
  function draw(){
    svg.selectAll("*").remove();
    const T=+d3.select("#kd-t").property("value");
    const p=softmax(logits,T);
    const x0=70, bw=70, gap=18, base=200, maxH=150;
    svg.append("text").attr("x",24).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("teacher soft targets at T = "+T.toFixed(1));
    labels.forEach((l,i)=>{
      const x=x0+i*(bw+gap), h=p[i]*maxH*1.0;
      svg.append("rect").attr("x",x).attr("y",base-h).attr("width",bw).attr("height",Math.max(1,h)).attr("rx",4)
        .attr("fill",i===0?C.accent:"#5b9cff").attr("fill-opacity",i===0?0.9:0.7);
      svg.append("text").attr("x",x+bw/2).attr("y",base-h-6).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.ink).text((p[i]*100).toFixed(1)+"%");
      svg.append("text").attr("x",x+bw/2).attr("y",base+16).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text(l);
    });
    svg.append("line").attr("x1",x0-10).attr("y1",base).attr("x2",x0+labels.length*(bw+gap)-gap+10).attr("y2",base).attr("stroke","#3a4150");
    const ent=-p.reduce((a,v)=>a+(v>1e-9?v*Math.log2(v):0),0);
    d3.select("#kd-read").html(`T=<b>${T.toFixed(1)}</b> · entropy <b>${ent.toFixed(2)}</b> bits · ${T<=1.2?"sharp — almost one-hot, little extra signal":"softer — the small dog/fox/tree probabilities are the dark knowledge the student learns"}`);
  }
  d3.select("#kd-t").on("input",draw);
  draw();
})();
