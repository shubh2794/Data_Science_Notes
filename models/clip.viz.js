/* clip.viz.js — extracted from clip.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#e879f9",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#clip-svg"),W=640,H=320;
  const items=[{e:"🐱",t:"a cat"},{e:"🐶",t:"a dog"},{e:"🚗",t:"a car"},{e:"🌳",t:"a tree"},{e:"🍕",t:"pizza"}];
  const noise=items.map((_,i)=>items.map((_,j)=>(Math.sin(i*3.1+j*1.7)*0.5+0.5)*0.18));
  function draw(){
    svg.selectAll("*").remove();
    const t=+d3.select("#clip-t").property("value");
    const n=items.length, cell=46, left=130, top=70;
    // column (text) headers
    items.forEach((it,j)=>svg.append("text").attr("x",left+j*cell+cell/2).attr("y",top-12).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(it.t));
    svg.append("text").attr("x",left+n*cell/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",11).text("text encodings →");
    svg.append("text").attr("x",26).attr("y",top+n*cell/2).attr("fill","#5b9cff").attr("font-size",11).attr("transform",`rotate(-90,30,${top+n*cell/2})`).text("image encodings");
    let correct=0;
    items.forEach((it,i)=>{
      svg.append("text").attr("x",left-14).attr("y",top+i*cell+cell/2+8).attr("text-anchor","end").attr("font-size",20).text(it.e);
      // similarity row
      let best=-1,bj=0;
      const row=items.map((_,j)=>{
        const base=noise[i][j];
        const s=Math.max(0.02,Math.min(1, base + (i===j ? 0.30+0.62*t : 0.18-0.16*t)));
        if(s>best){best=s;bj=j;}
        return s;
      });
      if(bj===i) correct++;
      items.forEach((_,j)=>{
        const s=row[j], on=i===j;
        svg.append("rect").attr("x",left+j*cell).attr("y",top+i*cell).attr("width",cell-3).attr("height",cell-3).attr("rx",4)
          .attr("fill",d3.interpolateMagma(0.08+s*0.85)).attr("stroke",on?C.good:"none").attr("stroke-width",on?2:0);
        svg.append("text").attr("x",left+j*cell+cell/2-1).attr("y",top+i*cell+cell/2+3).attr("text-anchor","middle").attr("font-size",9)
          .attr("fill",s>0.5?"#fff":"#aaa").attr("pointer-events","none").text(s.toFixed(2));
      });
    });
    const acc=Math.round(correct/n*100);
    d3.select("#clip-read").html(`training <b>${Math.round(t*100)}%</b> · the <b style="color:${C.good}">diagonal</b> (matched pairs) sharpens · batch match accuracy <b>${acc}%</b> ${acc===100?"✓":""}`);
  }
  d3.select("#clip-t").on("input",draw);
  draw();
})();
