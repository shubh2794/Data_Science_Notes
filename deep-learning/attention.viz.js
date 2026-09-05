/* attention.viz.js — extracted from attention.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#gqa-svg"), W=640;
  const NH=8;
  function draw(){
    const G=+d3.select("#gqa-g").property("value");
    svg.selectAll("*").remove();
    const pad=70, topY=95, botY=215;
    const qx = i => pad + i*(W-2*pad)/(NH-1);
    const groupOf = i => Math.min(G-1, Math.floor(i*G/NH));
    const kvx = g => G===1 ? W/2 : pad + g*(W-2*pad)/(G-1);
    const color = d3.scaleSequential(d3.interpolateTurbo).domain([-0.6, G-0.4]);
    svg.append("text").attr("x",W/2).attr("y",34).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12.5).attr("font-weight",600)
      .text(`${NH} query heads  →  ${G} key/value head${G>1?"s":""}`);
    svg.append("text").attr("x",pad-14).attr("y",topY+4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("Q");
    svg.append("text").attr("x",pad-14).attr("y",botY+4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("KV");
    for(let i=0;i<NH;i++){
      const g=groupOf(i);
      svg.append("line").attr("x1",qx(i)).attr("y1",topY).attr("x2",kvx(g)).attr("y2",botY)
        .attr("stroke",color(g)).attr("stroke-width",1.5).attr("stroke-opacity",.45);
    }
    for(let g=0;g<G;g++){
      svg.append("rect").attr("x",kvx(g)-24).attr("y",botY-15).attr("width",48).attr("height",30).attr("rx",7)
        .attr("fill",color(g)).attr("stroke","#0f1117").attr("stroke-width",1.5);
      svg.append("text").attr("x",kvx(g)).attr("y",botY+4).attr("text-anchor","middle").attr("font-size",10).attr("fill","#06101f").attr("font-weight",700).text("KV"+(g+1));
    }
    for(let i=0;i<NH;i++){
      const g=groupOf(i);
      svg.append("circle").attr("cx",qx(i)).attr("cy",topY).attr("r",13).attr("fill",color(g)).attr("stroke","#0f1117").attr("stroke-width",1.5);
      svg.append("text").attr("x",qx(i)).attr("y",topY+4).attr("text-anchor","middle").attr("font-size",9.5).attr("fill","#06101f").attr("font-weight",700).text("Q"+(i+1));
    }
    const label = G===1?"MQA · Multi-Query":G===NH?"MHA · Multi-Head":"GQA · Grouped-Query";
    const rel = Math.round(G/NH*100);
    const fold = NH/G; const foldStr = Number.isInteger(fold)?fold.toFixed(0):fold.toFixed(1);
    d3.select("#gqa-read").html(`<b>${label}</b> &nbsp;·&nbsp; KV-cache ∝ ${G} (vs ${NH} for MHA) &nbsp;→&nbsp; <b>${rel}%</b> of full cache &nbsp;·&nbsp; <b>${foldStr}×</b> smaller`);
  }
  d3.select("#gqa-g").on("input",draw);
  draw();
})();
