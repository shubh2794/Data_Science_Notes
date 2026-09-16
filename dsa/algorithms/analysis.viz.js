/* big-o-complexity.viz.js — extracted from big-o-complexity.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#bigo-svg"), W=680, H=360;
  const m={l:54,r:130,t:20,b:36};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const funcs=[
    {key:"1",     name:"O(1)",       color:C.muted, f:n=>1},
    {key:"logn",  name:"O(log n)",   color:C.good,  f:n=>Math.log2(n)},
    {key:"n",     name:"O(n)",       color:C.A,     f:n=>n},
    {key:"nlogn", name:"O(n log n)", color:"#a78bfa",f:n=>n*Math.log2(n)},
    {key:"n2",    name:"O(n²)",      color:C.B,     f:n=>n*n},
    {key:"2n",    name:"O(2ⁿ)",      color:C.bad,   f:n=>Math.pow(2,n)}
  ];
  let logScale=false, N=32;
  const gAxis=svg.append("g"), gPlot=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const gLbl=svg.append("g");
  const x=d3.scaleLinear().range([0,iw]);

  function maxAt(n){ return Math.max.apply(null, funcs.map(fn=>fn.f(n))); }

  function render(){
    x.domain([1,N]);
    const ymax=maxAt(N);
    const y = logScale
      ? d3.scaleLog().domain([1, Math.max(2,ymax)]).range([ih,0]).clamp(true)
      : d3.scaleLinear().domain([0, ymax]).range([ih,0]);
    gAxis.selectAll("*").remove();
    // axes
    gAxis.append("g").attr("transform",`translate(${m.l},${m.t+ih})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
    gAxis.append("g").attr("transform",`translate(${m.l},${m.t})`).call(d3.axisLeft(y).ticks(5,"~s")).attr("color",C.muted);
    gAxis.append("text").attr("x",m.l+iw/2).attr("y",H-4).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("n");

    const line=d3.line().x(d=>x(d.n)).y(d=>y(Math.max(logScale?1:0, d.v)));
    const data=funcs.map(fn=>({fn, pts:d3.range(1,N+0.0001,N/120).map(n=>({n, v:fn.f(n)}))}));
    const sel=gPlot.selectAll("path").data(data,d=>d.fn.key);
    sel.enter().append("path").attr("fill","none").attr("stroke-width",2.2)
      .merge(sel).attr("stroke",d=>d.fn.color).attr("d",d=>line(d.pts));

    // end labels at n=N
    gLbl.selectAll("text").remove();
    funcs.forEach(fn=>{
      const v=fn.f(N);
      gLbl.append("text").attr("x",m.l+iw+6).attr("y",m.t+y(Math.max(logScale?1:0,v)))
        .attr("fill",fn.color).attr("font-size",11).attr("dominant-baseline","middle").text(fn.name);
    });

    const fmt=v=> v>=1e6? d3.format(".2s")(v) : Math.round(v).toLocaleString();
    d3.select("#bigo-readout").html(
      `n = <b>${N}</b> &nbsp;·&nbsp; ops: O(n)=${fmt(N)} · O(n log n)=${fmt(N*Math.log2(N))} · O(n²)=${fmt(N*N)} · O(2ⁿ)=${fmt(Math.pow(2,N))}`);
  }
  d3.select("#bigo-n").on("input",function(){ N=+this.value; render(); });
  d3.select("#bigo-scale").on("click",function(){
    logScale=!logScale;
    d3.select(this).text(logScale?"y-axis: linear":"y-axis: log");
    render();
  });
  render();
})();
