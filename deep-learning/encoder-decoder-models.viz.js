/* encoder-decoder-models.viz.js — extracted from encoder-decoder-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#enc-svg"),W=640;
  const toks=["The","cat","sat","on","mat"];
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#enc-sel").property("value");
    const n=toks.length, cell=42, top=56, left=120;
    const isCross = k==="cross";
    const colTok = toks, rowTok = toks;
    const colLabel = isCross? "source (encoder) tokens" : "key tokens";
    const rowLabel = isCross? "target (decoder) tokens" : "query tokens";
    svg.append("text").attr("x",left+n*cell/2).attr("y",24).attr("text-anchor","middle").attr("fill",C.blue).attr("font-size",11).text(colLabel+" →");
    svg.append("text").attr("x",18).attr("y",top+n*cell/2).attr("fill",C.a2).attr("font-size",11).attr("transform",`rotate(-90,26,${top+n*cell/2})`).text(rowLabel);
    colTok.forEach((t,j)=>svg.append("text").attr("x",left+j*cell+cell/2).attr("y",top-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(t));
    rowTok.forEach((t,i)=>{
      svg.append("text").attr("x",left-8).attr("y",top+i*cell+cell/2+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(t);
      colTok.forEach((_,j)=>{
        let allow;
        if(k==="enc") allow=true;
        else if(k==="dec") allow=j<=i;
        else allow=true; // cross: decoder token attends to all source
        svg.append("rect").attr("x",left+j*cell).attr("y",top+i*cell).attr("width",cell-3).attr("height",cell-3).attr("rx",3)
          .attr("fill",allow?C.accent:"#15181f").attr("fill-opacity",allow?0.82:1).attr("stroke",allow?"none":"#2a2f3a");
      });
    });
    const notes={enc:"Encoder: every token attends to all tokens (bidirectional) — full understanding of the input.",
      dec:"Decoder: each token attends only to earlier tokens (lower-triangular causal mask) — no peeking ahead.",
      cross:"Cross-attention: every decoder (target) token may attend to all encoder (source) tokens."};
    d3.select("#enc-read").innerHTML?0:0;
    document.getElementById("enc-read").innerHTML=notes[k];
  }
  d3.select("#enc-sel").on("change",draw);
  draw();
})();
