/* end-to-end-learning.viz.js — extracted from end-to-end-learning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Viz: pipeline vs end-to-end crossover ---- */
(function(){
  const C={accent:"#f87171",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#cx-svg"),W=640,H=280;
  const px=48,py=24,pw=W-96,ph=200, mmax=100;
  const X=m=>px+m/mmax*pw;
  const ymax=1, Y=v=>py+ph-Math.min(v,ymax)/ymax*ph;   // performance 0..1
  // pipeline: good with little data, plateaus (capped by hand-engineering)
  const pipe=m=>0.55*(1-Math.exp(-m/8))+0.30;          // → ~0.85 ceiling, fast
  // end-to-end: poor with little data, higher ceiling, slower to rise
  const e2e =m=>0.97*(1-Math.exp(-m/38));              // → ~0.97, needs data
  function draw(){
    svg.selectAll("*").remove();
    const m=+d3.select("#cx-m").property("value");
    // axes
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("y1",py).attr("x2",px).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("amount of labeled (input → output) data →");
    svg.append("text").attr("x",px-6).attr("y",py-9).attr("fill",C.muted).attr("font-size",10).text("performance");
    const ms=d3.range(1,mmax+0.1,1);
    const line=(f,col,dash)=>svg.append("path").attr("d",d3.line().x(X).y(d=>Y(f(d)))(ms)).attr("fill","none").attr("stroke",col).attr("stroke-width",2.4).attr("stroke-dasharray",dash||null);
    line(pipe,C.blue); line(e2e,C.accent);
    svg.append("text").attr("x",X(mmax)).attr("y",Y(pipe(mmax))+14).attr("text-anchor","end").attr("fill",C.blue).attr("font-size",10).text("hand-engineered pipeline");
    svg.append("text").attr("x",X(mmax)).attr("y",Y(e2e(mmax))-6).attr("text-anchor","end").attr("fill",C.accent).attr("font-size",10).text("end-to-end");
    // crossover point
    let cross=null; for(let x=2;x<=mmax;x++){ if(e2e(x)>=pipe(x)){cross=x;break;} }
    if(cross){ svg.append("line").attr("x1",X(cross)).attr("y1",py).attr("x2",X(cross)).attr("y2",py+ph).attr("stroke","#3a4150").attr("stroke-dasharray","2 3");
      svg.append("text").attr("x",X(cross)).attr("y",py+10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text("crossover"); }
    // marker
    svg.append("line").attr("x1",X(m)).attr("y1",py).attr("x2",X(m)).attr("y2",py+ph).attr("stroke",C.good).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(m)).attr("cy",Y(pipe(m))).attr("r",4.5).attr("fill",C.blue);
    svg.append("circle").attr("cx",X(m)).attr("cy",Y(e2e(m))).attr("r",4.5).attr("fill",C.accent);
    const better = e2e(m)>=pipe(m) ? "end-to-end" : "the pipeline";
    const col = e2e(m)>=pipe(m) ? C.accent : C.blue;
    d3.select("#cx-read").html(`data = <b>${m}</b> · pipeline <b>${(pipe(m)*100).toFixed(0)}%</b> vs end-to-end <b>${(e2e(m)*100).toFixed(0)}%</b> → favor <b style="color:${col}">${better}</b>`);
  }
  d3.select("#cx-m").on("input",draw);
  draw();
})();
