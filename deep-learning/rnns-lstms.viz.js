/* rnns-lstms.viz.js — extracted from rnns-lstms.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* Interactive: gradient magnitude back through time, λ^(t−k) */
(function(){
  const svg=d3.select("#grad-svg"), W=640, H=240;
  const m={t:24, r:22, b:36, l:54}, iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const x=d3.scaleLinear().range([0,iw]);
  const y=d3.scaleLog().range([ih,0]).clamp(true);
  const xAxisG=g.append("g").attr("transform",`translate(0,${ih})`).attr("color",C.muted);
  const yAxisG=g.append("g").attr("color",C.muted);
  const refLine=g.append("line").attr("stroke",C.muted).attr("stroke-dasharray","4 4").attr("opacity",0.7);
  const area=g.append("path").attr("fill-opacity",0.14).attr("stroke","none");
  const path=g.append("path").attr("fill","none").attr("stroke-width",2.5);
  // axis labels
  g.append("text").attr("x",iw).attr("y",ih+30).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("timestep  (loss is at t = T, right) →");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-40).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("‖ gradient ‖");

  function draw(){
    const lam=+d3.select("#grad-lam").property("value");
    const T=+d3.select("#grad-T").property("value");
    const data=d3.range(1,T+1).map(k=>({k, mag:Math.max(Math.pow(lam, T-k), 1e-12)}));
    const mags=data.map(d=>d.mag);
    x.domain([1,T]);
    y.domain([Math.min(1e-9, d3.min(mags)), Math.max(1, d3.max(mags))]);
    const col = lam < 0.98 ? C.bad : lam > 1.02 ? C.B : C.good;

    const line=d3.line().x(d=>x(d.k)).y(d=>y(d.mag));
    const ar=d3.area().x(d=>x(d.k)).y0(ih).y1(d=>y(d.mag));
    path.datum(data).attr("d",line).attr("stroke",col);
    area.datum(data).attr("d",ar).attr("fill",col);
    refLine.attr("x1",0).attr("x2",iw).attr("y1",y(1)).attr("y2",y(1));
    xAxisG.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")));
    yAxisG.call(d3.axisLeft(y).ticks(5,"~g"));

    const first=Math.pow(lam, T-1);
    const verdict = lam < 0.98 ? "vanishes" : lam > 1.02 ? "explodes" : "stays stable";
    d3.select("#grad-readout").html(
      `Gradient reaching the <b>first</b> timestep ≈ <b>${first.toExponential(2)}</b> &nbsp;→&nbsp; signal <b>${verdict}</b> over ${T} steps &nbsp;(λ = ‖Wₕ‖·σ′ = ${(+lam).toFixed(2)})`
    );
  }
  ["#grad-lam","#grad-T"].forEach(id=>d3.select(id).on("input",draw));
  draw();
})();
