/* vision-transformers.viz.js — extracted from vision-transformers.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#patch-svg"), W=640, H=320;
  const boxX=20, boxY=20, boxS=260;          // image canvas
  function draw(){
    const P=+d3.select("#p-patch").property("value");
    const res=+d3.select("#p-res").property("value");
    const D=+d3.select("#p-dim").property("value");
    const perSide=Math.max(1,Math.round(res/P));   // patches per row/col
    const N=perSide*perSide;
    const patchVec=P*P*3;                            // flattened patch length
    svg.selectAll("*").remove();

    // ---- left: patch grid over a faux image ----
    svg.append("text").attr("x",boxX).attr("y",14).attr("fill",C.muted).attr("font-size",11)
       .text(`${res}×${res} image · ${P}×${P} patches`);
    const cell=boxS/perSide;
    const g=svg.append("g");
    for(let i=0;i<perSide;i++)for(let j=0;j<perSide;j++){
      const t=(i+j)/(2*Math.max(1,perSide-1));      // diagonal gradient = stand-in image
      g.append("rect").attr("x",boxX+j*cell).attr("y",boxY+i*cell)
        .attr("width",Math.max(0.5,cell-1.4)).attr("height",Math.max(0.5,cell-1.4))
        .attr("rx",cell>10?2:0)
        .attr("fill",d3.interpolateViridis(t)).attr("opacity",0.92);
    }
    svg.append("rect").attr("x",boxX).attr("y",boxY).attr("width",boxS).attr("height",boxS)
       .attr("fill","none").attr("stroke",C.line).attr("rx",4);

    // ---- right: token sequence schematic ----
    const rx=boxX+boxS+60;
    svg.append("text").attr("x",rx).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("token sequence");
    const tokW=26, tokH=18, gap=6, cols=6, startY=boxY+6;
    const labels=["[CLS]"];
    const show=Math.min(N, cols*4-1);
    for(let k=0;k<show;k++) labels.push("p"+(k+1));
    const trailing = N>show;
    labels.forEach((lab,idx)=>{
      const r=Math.floor(idx/cols), c=idx%cols;
      const x=rx+c*(tokW+gap), y=startY+r*(tokH+gap);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",tokW).attr("height",tokH).attr("rx",3)
        .attr("fill",idx===0?C.B:C.A).attr("opacity",idx===0?0.95:0.55);
      svg.append("text").attr("x",x+tokW/2).attr("y",y+tokH/2+3).attr("text-anchor","middle")
        .attr("font-size",8).attr("fill","#0b0e14").attr("pointer-events","none").text(lab);
    });
    if(trailing){
      const idx=labels.length, r=Math.floor(idx/cols), c=idx%cols;
      svg.append("text").attr("x",rx+c*(tokW+gap)).attr("y",startY+r*(tokH+gap)+tokH/2+3)
        .attr("font-size",11).attr("fill",C.muted).text("… +"+(N-show));
    }
    // each token -> D-dim arrow note
    svg.append("text").attr("x",rx).attr("y",startY+5*(tokH+gap)+14)
       .attr("fill",C.muted).attr("font-size",10).text(`each → ℝ^${patchVec} → embed → ℝ^${D}`);

    // ---- bottom: cost bars ----
    const baseN=(224*224)/(16*16);                  // reference: ViT-B/16 = 196
    const baseCost=baseN*baseN;
    const cost=N*N;                                  // attention ∝ N²
    const ratio=cost/baseCost;
    const barY=boxY+boxS+18, barX=boxX, barMaxW=W-2*boxX-10;
    const frac=Math.min(1, Math.log10(ratio+0.0001)/2 + 0.5); // log-scaled fill, centered on 1×
    svg.append("text").attr("x",barX).attr("y",barY-6).attr("fill",C.muted).attr("font-size",10)
       .text("attention cost ∝ N²  (relative to ViT-B/16, N=196)");
    svg.append("rect").attr("x",barX).attr("y",barY).attr("width",barMaxW).attr("height",14).attr("rx",7)
       .attr("fill",C.line).attr("opacity",0.5);
    svg.append("rect").attr("x",barX).attr("y",barY).attr("width",Math.max(4,barMaxW*frac)).attr("height",14).attr("rx",7)
       .attr("fill",ratio>2?C.bad:(ratio<0.6?C.good:C.A));
    svg.append("text").attr("x",barX+barMaxW).attr("y",barY+11).attr("text-anchor","end")
       .attr("font-size",10).attr("fill",C.ink).text(ratio.toFixed(2)+"×");

    d3.select("#patch-readout").html(
      `<b>P=${P}</b>, image ${res}×${res} → grid <b>${perSide}×${perSide}</b> → `+
      `<b>N=${N}</b> patch tokens (+1 [CLS] = ${N+1}). `+
      `Each patch is a ${P}·${P}·3 = <b>${patchVec}</b>-vector projected to D=${D}. `+
      `Attention cost ≈ <b>${ratio.toFixed(2)}×</b> a ViT-B/16.`
    );
  }
  d3.select("#p-patch").on("input",draw);
  d3.select("#p-res").on("input",draw);
  d3.select("#p-dim").on("input",draw);
  draw();
})();
