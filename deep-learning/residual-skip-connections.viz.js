/* residual-skip-connections.viz.js — extracted from residual-skip-connections.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#res-svg"),W=640,H=240;
  function draw(){
    svg.selectAll("*").remove();
    const depth=+d3.select("#res-d").property("value"), skip=d3.select("#res-skip").property("checked");
    const px=40,py=30,pw=W-80,ph=150, base=py+ph;
    const bw=pw/depth;
    for(let l=0;l<depth;l++){
      // gradient reaching layer l (0 = earliest). Without skip: decays exponentially from output.
      const back=depth-l;
      const g = skip ? (0.85+0.15*Math.exp(-0.04*back)) : Math.pow(0.82, back*0.6);
      const h=g*ph;
      const col = g>0.5?C.good:g>0.15?C.a2:C.bad;
      svg.append("rect").attr("x",px+l*bw).attr("y",base-h).attr("width",Math.max(1,bw-1.5)).attr("height",h).attr("fill",col).attr("fill-opacity",.85);
    }
    svg.append("line").attr("x1",px).attr("y1",base).attr("x2",px+pw).attr("y2",base).attr("stroke","#3a4150");
    svg.append("text").attr("x",px).attr("y",base+18).attr("fill",C.muted).attr("font-size",10).text("← early layers (input)");
    svg.append("text").attr("x",px+pw).attr("y",base+18).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("output layers →");
    svg.append("text").attr("x",px).attr("y",20).attr("fill",C.muted).attr("font-size",10).text("gradient magnitude");
    const earlyG = skip ? (0.85+0.15*Math.exp(-0.04*depth)) : Math.pow(0.82, depth*0.6);
    d3.select("#res-read").html(skip
      ? `<b style="color:#4ade80">with skips</b> · gradient reaching the first layer ≈ <b>${(earlyG*100).toFixed(0)}%</b> — all ${depth} layers train`
      : `<b style="color:#f87171">no skips</b> · gradient reaching the first layer ≈ <b>${(earlyG*100).toFixed(1)}%</b> — early layers barely learn (vanishing gradient)`);
  }
  d3.select("#res-d").on("input",draw); d3.select("#res-skip").on("change",draw);
  draw();
})();
