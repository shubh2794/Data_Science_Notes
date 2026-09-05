/* linear-regression.viz.js — extracted from linear-regression.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---------- 1 · least-squares playground ---------- */
(function(){
  const svg = d3.select("#reg-svg"), W=640, H=340, m={t:16,r:16,b:32,l:40};
  const x = d3.scaleLinear().domain([0,10]).range([m.l, W-m.r]);
  const y = d3.scaleLinear().domain([0,10]).range([H-m.b, m.t]);
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
  const seed = () => d3.range(8).map(i => ({ x: 1+i*1.05 + (Math.random()-.5), y: 1.2 + i*0.85 + (Math.random()-.5)*1.5 }));
  let pts = seed();

  svg.append("clipPath").attr("id","reg-clip").append("rect")
     .attr("x",m.l).attr("y",m.t).attr("width",W-m.l-m.r).attr("height",H-m.t-m.b);
  const gAxis = svg.append("g");
  gAxis.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6));
  gAxis.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
  const gClip = svg.append("g").attr("clip-path","url(#reg-clip)");
  const gRes  = gClip.append("g");
  const line  = gClip.append("path").attr("fill","none").attr("stroke",C.A).attr("stroke-width",2.5);
  const gPts  = svg.append("g");
  const readout = d3.select("#reg-readout");

  function fit(){
    const n = pts.length;
    if(n < 2) return null;
    const mx = d3.mean(pts,d=>d.x), my = d3.mean(pts,d=>d.y);
    let num=0, den=0;
    pts.forEach(p=>{ num += (p.x-mx)*(p.y-my); den += (p.x-mx)**2; });
    const w = den === 0 ? 0 : num/den, b = my - w*mx;
    let ssr=0, sst=0;
    pts.forEach(p=>{ ssr += (p.y - (w*p.x+b))**2; sst += (p.y-my)**2; });
    return { w, b, mse: ssr/n, r2: sst === 0 ? NaN : 1 - ssr/sst, n };
  }

  function draw(){
    const f = fit();
    if(!f){ readout.html("Need at least two points."); gRes.selectAll("line").remove(); line.attr("d",null); }
    else {
      line.attr("d", `M${x(0)},${y(f.b)}L${x(10)},${y(f.w*10 + f.b)}`);
      gRes.selectAll("line").data(pts).join("line")
        .attr("stroke", C.bad).attr("stroke-width", 1.6).attr("stroke-dasharray","3 3").attr("opacity",.85)
        .attr("x1", d=>x(d.x)).attr("x2", d=>x(d.x))
        .attr("y1", d=>y(d.y)).attr("y2", d=>y(f.w*d.x + f.b));
      const r2 = isNaN(f.r2) ? "—" : f.r2.toFixed(3);
      readout.html(`ŷ = <b>${f.w.toFixed(2)}</b>·x + <b>${f.b.toFixed(2)}</b> &nbsp;·&nbsp; MSE <b>${f.mse.toFixed(3)}</b> &nbsp;·&nbsp; RMSE <b>${Math.sqrt(f.mse).toFixed(3)}</b> &nbsp;·&nbsp; R² <b>${r2}</b> &nbsp;·&nbsp; n = <b>${f.n}</b>`);
    }
    gPts.selectAll("circle").data(pts).join("circle")
      .attr("class","dragpt").attr("r",10)
      .attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",2)
      .style("cursor","grab")
      .attr("cx", d=>x(d.x)).attr("cy", d=>y(d.y))
      .call(d3.drag()
        .on("start", function(){ d3.select(this).style("cursor","grabbing"); })
        .on("drag",  function(ev,d){ d.x = clamp(x.invert(ev.x),0,10); d.y = clamp(y.invert(ev.y),0,10); draw(); })
        .on("end",   function(){ d3.select(this).style("cursor","grab"); }));
  }

  svg.on("click", function(ev){
    if(ev.target.tagName === "circle" || pts.length >= 40) return;
    const [px,py] = d3.pointer(ev, svg.node());
    if(px < m.l || px > W-m.r || py < m.t || py > H-m.b) return;
    pts.push({ x: clamp(x.invert(px),0,10), y: clamp(y.invert(py),0,10) });
    draw();
  });
  d3.select("#reg-reset").on("click", ()=>{ pts = seed(); draw(); });
  d3.select("#reg-outlier").on("click", ()=>{ pts.push({ x: 0.6, y: 9.4 }); draw(); });
  d3.select("#reg-example").on("click", ()=>{ pts = [{x:1,y:2},{x:2,y:3},{x:3,y:5}]; draw(); });
  draw();
})();
