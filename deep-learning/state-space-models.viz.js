/* state-space-models.viz.js — extracted from state-space-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#ssm-svg"),W=640,H=260;
  const nMax=16384;
  function draw(){
    svg.selectAll("*").remove();
    const n=+d3.select("#ssm-n").property("value");
    const px=50,py=24,pw=W-100,ph=190;
    const X=v=>px+(v/nMax)*pw;
    const quad=v=>(v/nMax)*(v/nMax); // normalized n²
    const lin=v=>(v/nMax);
    const Y=v=>py+ph-v*ph;
    // axes
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("y1",py).attr("x2",px).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+22).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("sequence length n →");
    svg.append("text").attr("x",px-6).attr("y",py-8).attr("fill",C.muted).attr("font-size",10).text("relative cost");
    const xs=d3.range(0,nMax+1,256);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(quad(d)))(xs)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(lin(d)))(xs)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.2);
    // current n markers
    svg.append("line").attr("x1",X(n)).attr("y1",py).attr("x2",X(n)).attr("y2",py+ph).attr("stroke",C.muted).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(n)).attr("cy",Y(quad(n))).attr("r",5).attr("fill",C.accent);
    svg.append("circle").attr("cx",X(n)).attr("cy",Y(lin(n))).attr("r",5).attr("fill",C.good);
    svg.append("text").attr("x",X(n)+8).attr("y",Y(quad(n))+4).attr("fill",C.accent).attr("font-size",10).text("attention O(n²)");
    svg.append("text").attr("x",X(n)+8).attr("y",Y(lin(n))+14).attr("fill",C.good).attr("font-size",10).text("SSM O(n)");
    const ratio=(n/512);
    d3.select("#ssm-read").html(`at n=<b>${n.toLocaleString()}</b> (${ratio.toFixed(0)}× a 512 baseline): attention cost ≈ <b>${(ratio*ratio).toFixed(0)}×</b>, SSM cost ≈ <b>${ratio.toFixed(0)}×</b> — a <b>${ratio.toFixed(0)}×</b> gap`);
  }
  d3.select("#ssm-n").on("input",draw);
  draw();
})();
