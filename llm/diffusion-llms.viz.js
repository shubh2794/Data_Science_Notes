/* diffusion-llms.viz.js — extracted from diffusion-llms.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#dl-svg"),W=640;
  const words=["The","cat","sat","on","the","warm","mat","today"];
  // diffusion reveal order (by "confidence", not L→R)
  const diffOrder=[4,0,6,1,3,7,2,5];
  let timer=null;
  function draw(){
    svg.selectAll("*").remove();
    const mode=d3.select("#dl-mode").property("value");
    const step=+d3.select("#dl-step").property("value");
    const n=words.length, cw=70, gap=6, x0=24, y=70;
    let revealed=new Set();
    if(mode==="ar"){ for(let i=0;i<Math.min(step* Math.ceil(n/6+0.5)| 0, n);i++) revealed.add(i);
      // reveal ceil(n/6) per step left to right
      revealed=new Set(); const per=Math.ceil(n/6); for(let i=0;i<Math.min(step*per,n);i++) revealed.add(i);
    } else {
      const per=Math.ceil(n/6); for(let i=0;i<Math.min(step*per,n);i++) revealed.add(diffOrder[i]);
    }
    words.forEach((w,i)=>{
      const x=x0+i*(cw+gap), on=revealed.has(i);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",cw).attr("height",34).attr("rx",6)
        .attr("fill",on?"rgba(45,212,191,.16)":"#15181f").attr("stroke",on?C.accent:"#2a2f3a");
      svg.append("text").attr("x",x+cw/2).attr("y",y+22).attr("text-anchor","middle").attr("font-size",on?12:11)
        .attr("fill",on?C.ink:C.muted).attr("font-family",on?"inherit":"SF Mono, monospace").text(on?w:"[MASK]");
      svg.append("text").attr("x",x+cw/2).attr("y",y-8).attr("text-anchor","middle").attr("font-size",8).attr("fill","#46506180").text(i+1);
    });
    const done=revealed.size, per=Math.ceil(n/6);
    svg.append("text").attr("x",x0).attr("y",36).attr("fill",C.muted).attr("font-size",11)
      .text(mode==="ar"?"autoregressive — fills strictly left → right, ~1 region/step":"diffusion — unmasks most-confident positions anywhere, in parallel");
    d3.select("#dl-read").html(`step <b>${step}</b>/6 · <b>${done}/${n}</b> tokens decoded ${mode==='diff'?`(up to ${per} in parallel per step, any order)`:`(left-to-right)`}`);
  }
  d3.select("#dl-mode").on("change",draw);
  d3.select("#dl-step").on("input",draw);
  d3.select("#dl-play").on("click",function(){
    if(timer){clearInterval(timer);timer=null;return;}
    let s=0; d3.select("#dl-step").property("value",0); draw();
    timer=setInterval(()=>{ s++; d3.select("#dl-step").property("value",s); draw(); if(s>=6){clearInterval(timer);timer=null;} },420);
  });
  draw();
})();
