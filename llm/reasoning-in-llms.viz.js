/* reasoning-in-llms.viz.js — extracted from reasoning-in-llms.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#rsn-svg"),W=640,H=380;
  const correct="18";
  const dist=[{a:"18",w:0.55},{a:"16",w:0.20},{a:"21",w:0.15},{a:"12",w:0.10}];
  function sample(){ const r=Math.random(); let c=0; for(const o of dist){ c+=o.w; if(r<=c) return o.a; } return dist[dist.length-1].a; }
  let paths=[];
  function resample(){ paths=d3.range(+d3.select("#rsn-n").property("value")).map(sample); draw(); }
  function draw(){
    svg.selectAll("*").remove();
    const n=paths.length;
    svg.append("text").attr("x",20).attr("y",22).attr("fill",C.muted).attr("font-size",12).text("problem");
    svg.append("text").attr("x",20).attr("y",41).attr("fill",C.ink).attr("font-size",12).attr("font-family","SF Mono, Menlo, monospace").text("multi-step word problem … (true answer: "+correct+")");
    const top=58, rowH=Math.min(22,(H-150)/Math.max(n,1));
    paths.forEach((a,i)=>{
      const y=top+i*rowH, ok=a===correct, steps=3+(i%3);
      svg.append("line").attr("x1",40).attr("y1",y+rowH/2).attr("x2",40+(steps-1)*15).attr("y2",y+rowH/2).attr("stroke",ok?C.accent:"#4b5563").attr("stroke-width",1).attr("opacity",.5);
      for(let s=0;s<steps;s++) svg.append("circle").attr("cx",40+s*15).attr("cy",y+rowH/2).attr("r",3).attr("fill",ok?C.accent:"#4b5563");
      svg.append("text").attr("x",40+(steps-1)*15+14).attr("y",y+rowH/2+4).attr("font-size",11).attr("fill",C.muted).text("→");
      svg.append("text").attr("x",40+(steps-1)*15+30).attr("y",y+rowH/2+4).attr("font-size",11).attr("font-weight",600).attr("fill",ok?C.good:C.bad).text("$"+a);
    });
    const tally={}; paths.forEach(a=>tally[a]=(tally[a]||0)+1);
    const entries=Object.entries(tally).sort((a,b)=>b[1]-a[1]);
    const tx=380, ty=72, bw=170, maxv=d3.max(entries,e=>e[1])||1;
    svg.append("text").attr("x",tx).attr("y",ty-16).attr("fill",C.muted).attr("font-size",11).text("vote tally");
    entries.forEach((e,i)=>{
      const y=ty+i*30;
      svg.append("text").attr("x",tx).attr("y",y+12).attr("font-size",12).attr("font-family","SF Mono, monospace").attr("fill",e[0]===correct?C.good:C.muted).text("$"+e[0]);
      svg.append("rect").attr("x",tx+44).attr("y",y).attr("width",bw).attr("height",16).attr("rx",3).attr("fill","#15181f");
      svg.append("rect").attr("x",tx+44).attr("y",y).attr("width",e[1]/maxv*bw).attr("height",16).attr("rx",3).attr("fill",i===0?C.accent:"#3a4150");
      svg.append("text").attr("x",tx+44+e[1]/maxv*bw+8).attr("y",y+12).attr("font-size",10).attr("fill",C.muted).text(e[1]);
    });
    const maj=entries[0][0];
    d3.select("#rsn-read").html(`${n} reasoning path${n>1?"s":""} → majority vote = <b style="color:${maj===correct?'#4ade80':'#f87171'}">$${maj}</b> ${maj===correct?"✓ correct":"✗ wrong — sample more"}`);
  }
  d3.select("#rsn-n").on("input",resample);
  d3.select("#rsn-resample").on("click",resample);
  resample();
})();
