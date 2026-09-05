/* agentic-rl.viz.js — extracted from agentic-rl.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#fb923c",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#arl-svg"),W=640,H=300, N=10;
  let p=0.3, rolls=[];
  function roll(){ rolls=d3.range(N).map(()=>Math.random()<p); draw(); }
  function draw(){
    svg.selectAll("*").remove();
    svg.append("text").attr("x",20).attr("y",22).attr("fill",C.muted).attr("font-size",11).text(N+" sampled rollouts (multi-step tool-use trajectories)");
    const top=38,rowH=18, x0=24;
    rolls.forEach((ok,i)=>{
      const y=top+i*rowH;
      const steps=4+(i%3);
      for(let s=0;s<steps;s++){
        svg.append("circle").attr("cx",x0+s*16).attr("cy",y+rowH/2).attr("r",3).attr("fill",ok?C.good:"#4b5563");
      }
      svg.append("line").attr("x1",x0).attr("y1",y+rowH/2).attr("x2",x0+(steps-1)*16).attr("y2",y+rowH/2).attr("stroke",ok?C.good:"#4b5563").attr("opacity",.4);
      svg.append("text").attr("x",x0+(steps-1)*16+14).attr("y",y+rowH/2+4).attr("font-size",11).attr("fill",ok?C.good:C.bad).text(ok?"✓ reward 1":"✗ reward 0");
    });
    // policy success-rate bar
    const by=top, bx=330, bw=170, bh=170;
    svg.append("text").attr("x",bx).attr("y",by-2).attr("fill",C.muted).attr("font-size",11).text("policy success rate");
    svg.append("rect").attr("x",bx).attr("y",by+8).attr("width",bw).attr("height",24).attr("rx",4).attr("fill","#15181f");
    svg.append("rect").attr("x",bx).attr("y",by+8).attr("width",p*bw).attr("height",24).attr("rx",4).attr("fill",C.accent);
    svg.append("text").attr("x",bx+bw+8).attr("y",by+25).attr("font-size",11).attr("fill",C.accent).text((p*100).toFixed(0)+"%");
    // simple training curve
    svg.append("text").attr("x",bx).attr("y",by+62).attr("fill",C.muted).attr("font-size",10).text("each train step reinforces ✓ rollouts");
    const succ=rolls.filter(Boolean).length;
    d3.select("#arl-read").html(`${succ}/${N} rollouts succeeded · policy success rate <b>${(p*100).toFixed(0)}%</b> ${p>0.85?"· converged ✓":"· keep training"}`);
  }
  d3.select("#arl-train").on("click",()=>{ p=Math.min(0.95, p + 0.12 + Math.random()*0.04); roll(); });
  d3.select("#arl-roll").on("click",roll);
  d3.select("#arl-reset").on("click",()=>{ p=0.3; roll(); });
  roll();
})();
