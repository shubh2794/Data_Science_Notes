/* context-engineering.viz.js — extracted from context-engineering.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#ctx-svg"),W=640,H=240;
  const comps=[
    {name:"System",       color:"#5b9cff", tok:()=>250},
    {name:"Instructions", color:"#818cf8", tok:()=>200},
    {name:"Few-shot",     color:"#c084fc", tok:()=> +d3.select("#ctx-few").property("value")*350},
    {name:"Retrieved",    color:"#2dd4bf", tok:()=> +d3.select("#ctx-docs").property("value")*320},
    {name:"History",      color:"#a3e635", tok:()=> +d3.select("#ctx-hist").property("value")*110},
    {name:"Query",        color:"#ffb454", tok:()=>90},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const cap=+d3.select("#ctx-cap").property("value");
    const left=24, right=24, trackW=W-left-right, top=64, barH=46;
    const scale=trackW/cap;
    const segs=comps.map(c=>({name:c.name,color:c.color,tok:c.tok()}));
    const total=d3.sum(segs,s=>s.tok);
    // capacity track
    svg.append("rect").attr("x",left).attr("y",top).attr("width",trackW).attr("height",barH).attr("rx",6).attr("fill","#11141b").attr("stroke",C.line);
    // segments
    let x=left;
    segs.forEach(s=>{
      const w=s.tok*scale; if(w<=0) return;
      const drawW=Math.max(0,Math.min(x+w,left+trackW)-x);
      if(drawW>0){
        svg.append("rect").attr("x",x).attr("y",top).attr("width",drawW).attr("height",barH).attr("fill",s.color).attr("fill-opacity",.85).attr("stroke","#0f1117");
        if(drawW>34) svg.append("text").attr("x",x+drawW/2).attr("y",top+barH/2+4).attr("text-anchor","middle").attr("font-size",10).attr("fill","#06101f").attr("font-weight",700).text(s.name);
      }
      x+=w;
    });
    // overflow region
    if(x>left+trackW){
      svg.append("rect").attr("x",left+trackW).attr("y",top).attr("width",Math.min(x,W-2)-(left+trackW)).attr("height",barH)
        .attr("fill","rgba(248,113,113,.18)").attr("stroke",C.bad).attr("stroke-dasharray","3 3");
    }
    // capacity line
    svg.append("line").attr("x1",left+trackW).attr("x2",left+trackW).attr("y1",top-10).attr("y2",top+barH+12).attr("stroke",C.bad).attr("stroke-width",1.5);
    svg.append("text").attr("x",left+trackW).attr("y",top-15).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10).text("window limit");
    // "edges vs middle" guide
    svg.append("text").attr("x",left+6).attr("y",top+barH+26).attr("font-size",9.5).attr("fill",C.muted).text("◀ models attend most to the edges");
    svg.append("text").attr("x",left+trackW-6).attr("y",top+barH+26).attr("text-anchor","end").attr("font-size",9.5).attr("fill",C.muted).text("edges ▶");
    // legend with tokens
    let lx=left, ly=H-26;
    segs.forEach(s=>{
      svg.append("rect").attr("x",lx).attr("y",ly-9).attr("width",10).attr("height",10).attr("rx",2).attr("fill",s.color);
      const t=svg.append("text").attr("x",lx+15).attr("y",ly).attr("font-size",10).attr("fill",C.muted).text(`${s.name} ${s.tok}`);
      lx += 15 + (s.name.length*6.2 + (""+s.tok).length*6.5 + 22);
    });
    const over=Math.max(0,total-cap);
    const pct=Math.round(total/cap*100);
    d3.select("#ctx-read").html(`used <b>${total.toLocaleString()}</b> / ${cap.toLocaleString()} tokens (<b>${pct}%</b>) ${over>0?`· <span style="color:#f87171">⚠ ${over.toLocaleString()} tokens truncated</span>`:"· ✓ fits"}`);
  }
  ["#ctx-cap","#ctx-docs","#ctx-hist","#ctx-few"].forEach(s=>d3.select(s).on("input",draw));
  draw();
})();
