/* context-length-extension.viz.js — extracted from context-length-extension.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#cle-svg"),W=640,H=220;
  const trained=4096;
  function draw(){
    svg.selectAll("*").remove();
    const len=+d3.select("#cle-len").property("value");
    const mode=d3.select("#cle-mode").property("value");
    const px=30, pw=W-60, y=120;
    // trained range track
    svg.append("rect").attr("x",px).attr("y",y-14).attr("width",pw).attr("height",28).attr("rx",5).attr("fill","#15181f").attr("stroke","#2a2f3a");
    // trained sub-range highlighted (it maps to full track width = trained positions)
    svg.append("text").attr("x",px).attr("y",y-24).attr("fill",C.muted).attr("font-size",10).text("trained range 0 … "+trained.toLocaleString());
    svg.append("text").attr("x",px+pw).attr("y",y-24).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("position →");
    const scale=len/trained;
    // sample positions across target length
    const N=18;
    let outOfRange=0;
    for(let i=0;i<N;i++){
      const pos=i/(N-1)*len;
      let mapped = mode==="interp" ? pos/scale : pos;     // interp squeezes back into trained range
      const within = mapped<=trained;
      if(!within) outOfRange++;
      const x=px + Math.min(mapped,trained*1.0)/trained*pw;
      const xover = px + mapped/trained*pw; // actual (may exceed track)
      const drawX = mode==="interp" ? x : Math.min(xover, px+pw+40);
      svg.append("circle").attr("cx",Math.min(drawX,px+pw+30)).attr("cy",y).attr("r",5)
        .attr("fill",within?C.accent:C.bad).attr("fill-opacity",.9);
    }
    // capacity line
    svg.append("line").attr("x1",px+pw).attr("y1",y-22).attr("x2",px+pw).attr("y2",y+22).attr("stroke",C.muted).attr("stroke-dasharray","2 3");
    if(mode==="extrap"){
      svg.append("text").attr("x",px+pw+8).attr("y",y+4).attr("fill",C.bad).attr("font-size",10).text("⚠ unseen");
    }
    const note = mode==="interp"
      ? `interpolation: every position ÷ ${scale.toFixed(1)} → all <b>${len.toLocaleString()}</b> positions fit inside the trained 0…${trained.toLocaleString()} range ✓`
      : `extrapolation: positions run to <b>${len.toLocaleString()}</b>, <b style="color:#f87171">${outOfRange}/${18}</b> beyond the trained range — out-of-distribution ✗`;
    d3.select("#cle-read").html(`target <b>${len.toLocaleString()}</b> (${(len/trained).toFixed(1)}× trained) · ${note}`);
  }
  d3.select("#cle-len").on("input",draw);
  d3.select("#cle-mode").on("change",draw);
  draw();
})();
