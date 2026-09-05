/* vision-language-models.viz.js — extracted from vision-language-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#vlm-svg"),W=640,H=260;
  function draw(){
    svg.selectAll("*").remove();
    const p=+d3.select("#vlm-p").property("value");      // patch size index (bigger = larger patches = fewer)
    const grid=[6,4,3,2,2,1][p-1] ? null : null;
    const n=[ [12],[8],[6],[4],[3],[2] ][p-1][0];        // patches per side
    const img=20, side=180, cell=side/n;
    // draw image as a simple gradient grid
    const col=d3.scaleSequential(d3.interpolateCividis).domain([0,n*1.4]);
    for(let r=0;r<n;r++) for(let c=0;c<n;c++){
      svg.append("rect").attr("x",img+c*cell).attr("y",40+r*cell).attr("width",cell-1.5).attr("height",cell-1.5).attr("rx",2)
        .attr("fill",col((r+c)+ (Math.sin(r*0.9)+Math.cos(c*0.8))*1.5)).attr("opacity",.92);
    }
    svg.append("text").attr("x",img+side/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("image · "+n+"×"+n+" patches");
    // arrow
    svg.append("text").attr("x",img+side+26).attr("y",40+side/2).attr("font-size",18).attr("fill",C.muted).text("→");
    // tokens column feeding LLM
    const tot=n*n, tx=img+side+58, maxShow=Math.min(tot,24);
    svg.append("text").attr("x",tx).attr("y",30).attr("fill",C.accent).attr("font-size",10).text(tot+" visual tokens");
    const cols=4, ts=16;
    for(let i=0;i<maxShow;i++){
      const r=Math.floor(i/cols), c=i%cols;
      svg.append("rect").attr("x",tx+c*(ts+4)).attr("y",40+r*(ts+4)).attr("width",ts).attr("height",ts).attr("rx",3).attr("fill",C.accent).attr("fill-opacity",.85);
    }
    if(tot>maxShow) svg.append("text").attr("x",tx).attr("y",40+Math.ceil(maxShow/cols)*(ts+4)+12).attr("font-size",10).attr("fill",C.muted).text("… +"+(tot-maxShow)+" more");
    // LLM box
    const lx=tx+cols*(ts+4)+30;
    svg.append("rect").attr("x",lx).attr("y",70).attr("width",70).attr("height",110).attr("rx",8).attr("fill","#1e222d").attr("stroke",C.line);
    svg.append("text").attr("x",lx+35).attr("y",128).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",11).text("LLM");
    svg.append("text").attr("x",lx-8).attr("y",60).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text("+ text tokens");
    d3.select("#vlm-read").html(`${n}×${n} patch grid → <b>${tot} visual tokens</b> · halving patch size ≈ <b>4×</b> the tokens (cost scales with resolution²)`);
  }
  d3.select("#vlm-p").on("input",draw);
  draw();
})();
