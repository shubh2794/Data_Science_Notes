/* agent-skills.viz.js — extracted from agent-skills.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#fb923c",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#sk-svg"),W=640,H=270;
  const library=["search_flights","compare_prices","book_flight","send_email","fetch_emails","summarize","rank_priority","web_search","read_page","extract_facts","write_report"];
  const goals=[
    {name:"Book a trip", chain:["search_flights","compare_prices","book_flight","send_email"]},
    {name:"Summarize my inbox", chain:["fetch_emails","summarize","rank_priority"]},
    {name:"Research a topic", chain:["web_search","read_page","extract_facts","write_report"]},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const g=goals[+d3.select("#sk-goal").property("value")];
    const used=new Set(g.chain);
    // library palette
    svg.append("text").attr("x",20).attr("y",22).attr("fill",C.muted).attr("font-size",11).text("skill library");
    const cols=4, bw=140, bh=26, gx=20, gy=34, gapx=12, gapy=10;
    library.forEach((s,i)=>{
      const r=Math.floor(i/cols), c=i%cols, x=gx+c*(bw+gapx), y=gy+r*(bh+gapy), on=used.has(s);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",bw).attr("height",bh).attr("rx",6)
        .attr("fill",on?"rgba(251,146,60,.16)":"#15181f").attr("stroke",on?C.accent:"#2a2f3a");
      svg.append("text").attr("x",x+bw/2).attr("y",y+17).attr("text-anchor","middle").attr("font-size",10.5)
        .attr("fill",on?C.accent:C.muted).attr("font-family","SF Mono, Menlo, monospace").text(s);
    });
    // composed chain
    const cy=200;
    svg.append("text").attr("x",20).attr("y",cy-8).attr("fill",C.ink).attr("font-size",11).attr("font-weight",600).text("composed for: "+g.name);
    let x=20; const cw=120, ch=30;
    g.chain.forEach((s,i)=>{
      svg.append("rect").attr("x",x).attr("y",cy).attr("width",cw).attr("height",ch).attr("rx",6).attr("fill",C.accent).attr("fill-opacity",.85);
      svg.append("text").attr("x",x+cw/2).attr("y",cy+19).attr("text-anchor","middle").attr("font-size",9.5).attr("fill","#1a1207").attr("font-weight",600).text(s);
      x+=cw;
      if(i<g.chain.length-1){ svg.append("text").attr("x",x+6).attr("y",cy+20).attr("font-size",14).attr("fill",C.muted).text("→"); x+=22; }
    });
    d3.select("#sk-read").html(`“${g.name}” → <b>${g.chain.length}</b> skills composed from a library of <b>${library.length}</b>`);
  }
  d3.select("#sk-goal").on("change",draw);
  draw();
})();
