/* differential-privacy.viz.js — extracted from differential-privacy.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#dp-svg"),W=640,H=250;
  const TRUE=142, sens=1;
  function lapSample(b){ const u=Math.random()-0.5; return -b*Math.sign(u)*Math.log(1-2*Math.abs(u)); }
  let noisy=TRUE;
  function reseed(){ const eps=+d3.select("#dp-eps").property("value"); noisy=TRUE+lapSample(sens/eps); draw(); }
  function draw(){
    svg.selectAll("*").remove();
    const eps=+d3.select("#dp-eps").property("value");
    const b=sens/eps; // noise scale
    const px=40,py=30,pw=W-80,ph=160;
    const span=Math.max(8, 6*b);
    const lo=TRUE-span, hi=TRUE+span;
    const X=v=>px+(v-lo)/(hi-lo)*pw;
    // laplace pdf
    const xs=d3.range(lo,hi,(hi-lo)/200);
    const pdf=x=>1/(2*b)*Math.exp(-Math.abs(x-TRUE)/b);
    const ymax=pdf(TRUE);
    const Y=v=>py+ph-v/ymax*ph;
    svg.append("path").attr("d",d3.area().x(d=>X(d)).y0(py+ph).y1(d=>Y(pdf(d)))(xs)).attr("fill",C.accent).attr("fill-opacity",.18);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(pdf(d)))(xs)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    // true value
    svg.append("line").attr("x1",X(TRUE)).attr("y1",py).attr("x2",X(TRUE)).attr("y2",py+ph).attr("stroke",C.good).attr("stroke-dasharray","3 3");
    svg.append("text").attr("x",X(TRUE)).attr("y",py-8).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",10).text("true = "+TRUE);
    // sampled noisy answer
    svg.append("line").attr("x1",X(noisy)).attr("y1",py).attr("x2",X(noisy)).attr("y2",py+ph).attr("stroke",C.a2).attr("stroke-width",2);
    svg.append("circle").attr("cx",X(noisy)).attr("cy",py+12).attr("r",5).attr("fill",C.a2);
    svg.append("text").attr("x",X(noisy)).attr("y",py+ph+18).attr("text-anchor","middle").attr("fill",C.a2).attr("font-size",10).text("released ≈ "+noisy.toFixed(1));
    const lvl = eps<=0.6?"strong privacy":eps>=2.5?"weak privacy":"moderate privacy";
    d3.select("#dp-read").html(`ε = <b>${eps.toFixed(1)}</b> → noise scale <b>${b.toFixed(2)}</b> · <b>${lvl}</b> · ${eps<1?"answer is fuzzy but individuals are protected":"answer is accurate but leaks more"}`);
  }
  d3.select("#dp-eps").on("input",draw);
  d3.select("#dp-sample").on("click",reseed);
  draw();
})();
