/* fine-tuning-transfer-learning.viz.js — extracted from fine-tuning-transfer-learning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#ft-svg"),W=640,H=260, L=8;
  function draw(){
    svg.selectAll("*").remove();
    const n=+d3.select("#ft-n").property("value");   // number of top layers unfrozen
    const lw=46, gap=10, x0=70, y=70, lh=90;
    // layers bottom(input)→top(output) left→right; unfrozen are the top n (rightmost)
    for(let i=0;i<L;i++){
      const frozen = i < (L-n);
      const x=x0+i*(lw+gap);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",lw).attr("height",lh).attr("rx",6)
        .attr("fill",frozen?"#2a2f3a":"rgba(248,113,113,.18)").attr("stroke",frozen?"#3a4150":C.accent);
      svg.append("text").attr("x",x+lw/2).attr("y",y+lh/2+4).attr("text-anchor","middle").attr("font-size",13).attr("fill",frozen?C.muted:C.accent).text(frozen?"🔒":"🔓");
      svg.append("text").attr("x",x+lw/2).attr("y",y+lh+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("L"+(i+1));
    }
    // head
    const hx=x0+L*(lw+gap)+6;
    svg.append("rect").attr("x",hx).attr("y",y).attr("width",lw).attr("height",lh).attr("rx",6).attr("fill","rgba(255,180,84,.18)").attr("stroke",C.a2);
    svg.append("text").attr("x",hx+lw/2).attr("y",y+lh/2+4).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.a2).text("head");
    svg.append("text").attr("x",x0).attr("y",46).attr("fill",C.muted).attr("font-size",10).text("input / general features");
    svg.append("text").attr("x",hx+lw).attr("y",46).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("task-specific →");
    const trainable=n+1; // layers + head
    const mode = n===0?"feature extraction":n>=L?"full fine-tuning":"partial fine-tuning";
    d3.select("#ft-read").html(`<b>${mode}</b> · training <b>${trainable}</b>/${L+1} blocks (head + ${n} layer${n!==1?"s":""}) · ${n===0?"least data/compute, smallest shift":n>=L?"most data/compute, biggest shift, forgetting risk":"moderate"}`);
  }
  d3.select("#ft-n").on("input",draw);
  draw();
})();
