/* data-quality-governance.viz.js — extracted from data-quality-governance.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#dq-svg"),W=640,H=230;
  const cols=["id","name","age","email"];
  const rawRows=[
    ["1","Ada Lovelace","36","ada@example.com",null],
    ["2","Alan Turing","","alan@example.com","missing"],
    ["1","Ada Lovelace","36","ada@example.com","dup"],
    ["3","Grace Hopper","999","grace@example.com","outlier"],
    ["4","Katherine J.","41","katherine@real.com","pii"],
  ];
  const cleanRows=[
    ["1","Ada Lovelace","36","a***@example.com",null],
    ["2","Alan Turing","37*","a***@example.com","fixed"],
    ["3","Grace Hopper","41*","g***@example.com","fixed"],
    ["4","Katherine J.","41","k***@real.com","fixed"],
  ];
  function draw(){
    svg.selectAll("*").remove();
    const clean=d3.select("#dq-clean").property("checked");
    const rows=clean?cleanRows:rawRows;
    const x0=30, cw=[50,170,70,200], rh=30, top=46;
    cols.forEach((c,j)=>{ const x=x0+cw.slice(0,j).reduce((a,b)=>a+b,0); svg.append("text").attr("x",x+6).attr("y",36).attr("font-size",11).attr("fill",C.muted).attr("font-weight",600).text(c); });
    rows.forEach((r,i)=>{
      const y=top+i*rh; const issue=r[4];
      const col = clean? (issue?C.good:"#2a2f3a") : (issue==="missing"?C.a2:issue==="dup"?C.bad:issue==="outlier"?C.a2:issue==="pii"?"#c084fc":"#2a2f3a");
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",cw.reduce((a,b)=>a+b,0)).attr("height",rh-4).attr("rx",4)
        .attr("fill",issue?(clean?"rgba(74,222,128,.08)":"rgba(248,113,113,.07)"):"#15181f").attr("stroke",col).attr("stroke-opacity",.7);
      cols.forEach((c,j)=>{ const x=x0+cw.slice(0,j).reduce((a,b)=>a+b,0); const val=r[j]===""?"∅":r[j];
        svg.append("text").attr("x",x+6).attr("y",y+19).attr("font-size",11).attr("fill",r[j]===""?C.a2:C.ink).attr("font-family","SF Mono, monospace").text(val); });
      if(issue && !clean) svg.append("text").attr("x",x0+cw.reduce((a,b)=>a+b,0)+8).attr("y",y+19).attr("font-size",10).attr("fill",col).text(issue==="pii"?"PII":issue);
    });
    d3.select("#dq-read").html(clean?`cleaned: <b>impute</b> missing · <b>cap</b> outliers · <b>dedup</b> rows · <b>mask</b> PII → ${cleanRows.length} trustworthy rows`:`<span style="color:#f87171">${rawRows.filter(r=>r[4]).length} issues</span>: a missing age, a duplicate, an outlier (age 999), and unmasked emails (PII)`);
  }
  d3.select("#dq-clean").on("change",draw);
  draw();
})();
