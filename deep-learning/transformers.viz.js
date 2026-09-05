/* transformers.viz.js — extracted from transformers.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#attn-svg"), W=640;
  let tokens=[], scores=[];
  function softmaxRow(row,temp){ const m=Math.max(...row); const ex=row.map(v=>Math.exp((v-m)*temp)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(v=>v/s); }
  function reroll(){
    const sent=d3.select("#attn-sent").property("value");
    tokens=sent.split(" ");
    const n=tokens.length;
    scores=d3.range(n).map((_,i)=>d3.range(n).map((__,j)=>{
      let base=Math.random()*0.6;
      if(i===j) base+=0.5;
      if(Math.abs(i-j)===1) base+=0.4;
      if(tokens[i].toLowerCase()===tokens[j].toLowerCase()) base+=0.7;
      return base;
    }));
    draw();
  }
  function draw(){
    svg.selectAll("*").remove();
    const n=tokens.length, temp=+d3.select("#attn-temp").property("value");
    const top=50, left=110, cell=Math.min(46,(W-left-30)/n);
    const weights=scores.map(r=>softmaxRow(r,temp));
    const col=d3.interpolateBlues;
    tokens.forEach((t,j)=>svg.append("text").attr("x",left+j*cell+cell/2).attr("y",top-10).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text(t));
    svg.append("text").attr("x",left+n*cell/2).attr("y",18).attr("text-anchor","middle").attr("fill",C.A).attr("font-size",11).text("Keys (attended to) →");
    tokens.forEach((t,i)=>{
      svg.append("text").attr("x",left-10).attr("y",top+i*cell+cell/2+4).attr("text-anchor","end").attr("font-size",11).attr("fill",C.muted).attr("class",`rowlbl r${i}`).text(t);
      tokens.forEach((_,j)=>{
        svg.append("rect").attr("x",left+j*cell).attr("y",top+i*cell).attr("width",cell-2).attr("height",cell-2).attr("rx",3)
          .attr("fill",col(weights[i][j])).attr("class",`cell r${i}`)
          .on("mouseover",()=>{ svg.selectAll("rect").attr("opacity",0.25); svg.selectAll(`.cell.r${i}`).attr("opacity",1); svg.selectAll(".rowlbl").attr("fill",C.muted); svg.select(`.rowlbl.r${i}`).attr("fill",C.B); })
          .on("mouseout",()=>{ svg.selectAll("rect").attr("opacity",1); svg.selectAll(".rowlbl").attr("fill",C.muted); });
        svg.append("text").attr("x",left+j*cell+cell/2-1).attr("y",top+i*cell+cell/2+3).attr("text-anchor","middle").attr("font-size",9)
          .attr("fill",weights[i][j]>0.5?"#fff":"#8aa").attr("pointer-events","none").text(weights[i][j].toFixed(2));
      });
    });
    svg.append("text").attr("x",10).attr("y",top+n*cell/2).attr("fill",C.B).attr("font-size",11).attr("transform",`rotate(-90,18,${top+n*cell/2})`).text("Queries");
  }
  d3.select("#attn-sent").on("change",reroll);
  d3.select("#attn-temp").on("input",draw);
  d3.select("#attn-reroll").on("click",reroll);
  reroll();
})();
