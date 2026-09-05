/* distributed-training.viz.js — extracted from distributed-training.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#dt-svg"),W=640,H=250, G=4;
  const layerCols=["#5b9cff","#4ade80","#fbbf24","#f87171"];
  function gpuBox(i){ const x=40+i*150; return {x,y:60,w:120,h:130}; }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#dt-sel").property("value");
    let note="";
    for(let i=0;i<G;i++){
      const g=gpuBox(i);
      svg.append("rect").attr("x",g.x).attr("y",g.y).attr("width",g.w).attr("height",g.h).attr("rx",8).attr("fill","#14171f").attr("stroke",C.line);
      svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("GPU "+(i+1));
      if(k==="data"){
        // full model (all 4 layer colors) on each, different batch shard
        for(let l=0;l<4;l++) svg.append("rect").attr("x",g.x+12).attr("y",g.y+12+l*26).attr("width",g.w-24).attr("height",20).attr("rx",3).attr("fill",layerCols[l]).attr("fill-opacity",.8);
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.a2).text("batch "+(i+1)+"/4");
      } else if(k==="tensor"){
        // each layer split: each GPU holds a vertical slice of every layer
        for(let l=0;l<4;l++) svg.append("rect").attr("x",g.x+12).attr("y",g.y+12+l*26).attr("width",g.w-24).attr("height",20).attr("rx",3).attr("fill",layerCols[l]).attr("fill-opacity",.35+0.15*0).attr("stroke",layerCols[l]).attr("stroke-dasharray","2 2");
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("¼ of each layer");
      } else {
        // pipeline: GPU i holds layer i only
        svg.append("rect").attr("x",g.x+12).attr("y",g.y+50).attr("width",g.w-24).attr("height",30).attr("rx",4).attr("fill",layerCols[i]).attr("fill-opacity",.85);
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+70).attr("text-anchor","middle").attr("font-size",10).attr("fill","#0f1117").attr("font-weight",600).text("layer "+(i+1));
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("stage "+(i+1));
      }
    }
    // arrows between GPUs
    for(let i=0;i<G-1;i++){
      const g=gpuBox(i);
      svg.append("text").attr("x",g.x+g.w+15).attr("y",g.y+g.h/2+4).attr("text-anchor","middle").attr("font-size",14).attr("fill",C.muted).text(k==="pipeline"?"→":"↔");
    }
    if(k==="data"){ svg.append("text").attr("x",W/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("full model on every GPU · gradients all-reduced ↔");
      note="Data parallel: replicate the model, split the batch, average gradients — for throughput."; }
    else if(k==="tensor"){ svg.append("text").attr("x",W/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("each layer's matrix sliced across GPUs · combine each layer ↔");
      note="Tensor parallel: split each layer across GPUs — for a model too big per device (heavy comms)."; }
    else { svg.append("text").attr("x",W/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("different layers on different GPUs · micro-batches flow →");
      note="Pipeline parallel: stage layers across GPUs like an assembly line (mind the bubble)."; }
    d3.select("#dt-read").html(note);
  }
  d3.select("#dt-sel").on("change",draw);
  draw();
})();
