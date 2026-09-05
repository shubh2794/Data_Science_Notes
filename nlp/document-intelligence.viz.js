/* document-intelligence.viz.js — extracted from document-intelligence.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#doc-svg"),W=640,H=300;
  const TYPES={title:"#5b9cff",para:"#9aa3b2",table:"#4ade80",figure:"#fb923c"};
  const regions=[
    {t:"title", x:60,y:24,w:300,h:24, lbl:"Title"},
    {t:"para",  x:60,y:60,w:230,h:70, lbl:"Paragraph"},
    {t:"figure",x:310,y:60,w:160,h:70, lbl:"Figure"},
    {t:"para",  x:60,y:142,w:410,h:46, lbl:"Paragraph"},
    {t:"table", x:60,y:200,w:410,h:74, lbl:"Table"},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const show=d3.select("#doc-show").property("checked");
    // page
    svg.append("rect").attr("x",40).attr("y",12).attr("width",450).attr("height",276).attr("rx",4).attr("fill","#10131a").attr("stroke","#2a2f3a");
    // faux content (text lines)
    regions.forEach(r=>{
      if(r.t==="figure"){ svg.append("rect").attr("x",r.x+6).attr("y",r.y+6).attr("width",r.w-12).attr("height",r.h-12).attr("rx",3).attr("fill","#1a1f29");
        svg.append("text").attr("x",r.x+r.w/2).attr("y",r.y+r.h/2+4).attr("text-anchor","middle").attr("font-size",16).attr("fill","#3a4150").text("🖼"); return; }
      const lines=r.t==="title"?1:Math.max(2,Math.floor(r.h/14));
      for(let i=0;i<lines;i++){
        const lw=(r.t==="title")?r.w*0.7:r.w*(0.7+Math.random()*0.28);
        svg.append("rect").attr("x",r.x+4).attr("y",r.y+6+i*14).attr("width",lw).attr("height",r.t==="title"?12:6).attr("rx",2).attr("fill",r.t==="title"?"#3a4458":"#262b35");
      }
      if(r.t==="table"){ for(let c=1;c<4;c++) svg.append("line").attr("x1",r.x+c*r.w/4).attr("y1",r.y).attr("x2",r.x+c*r.w/4).attr("y2",r.y+r.h).attr("stroke","#222836"); }
    });
    // detected overlays
    if(show){
      regions.forEach(r=>{
        const col=TYPES[r.t];
        svg.append("rect").attr("x",r.x).attr("y",r.y).attr("width",r.w).attr("height",r.h).attr("rx",3).attr("fill","none").attr("stroke",col).attr("stroke-width",1.6).attr("stroke-dasharray","4 2");
        svg.append("rect").attr("x",r.x).attr("y",r.y-13).attr("width",r.lbl.length*6.5+10).attr("height",13).attr("rx",2).attr("fill",col);
        svg.append("text").attr("x",r.x+5).attr("y",r.y-3).attr("font-size",9).attr("fill","#0f1117").attr("font-weight",700).text(r.lbl);
      });
    }
    // legend
    let lx=510; const ly=40;
    svg.append("text").attr("x",lx).attr("y",ly-14).attr("fill",C.muted).attr("font-size",10).text("region types");
    Object.entries(TYPES).forEach(([k,c],i)=>{ const y=ly+i*22;
      svg.append("rect").attr("x",lx).attr("y",y-9).attr("width",10).attr("height",10).attr("rx",2).attr("fill",c);
      svg.append("text").attr("x",lx+15).attr("y",y).attr("font-size",10).attr("fill",C.muted).text(k); });
    d3.select("#doc-read").html(show?`detected <b>${regions.length}</b> regions across <b>${new Set(regions.map(r=>r.t)).size}</b> types — structure recovered`:"raw page — just pixels/text until layout analysis runs");
  }
  d3.select("#doc-show").on("change",draw);
  draw();
})();
