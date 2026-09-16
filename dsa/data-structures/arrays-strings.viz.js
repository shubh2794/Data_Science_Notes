/* arrays-strings.viz.js — extracted from arrays-strings.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#tp-svg"), W=680;
  const arr=[1,3,4,6,8,10,13,15];
  const target=14;
  const n=arr.length, cw=64, gap=10, x0=(W-(n*cw+(n-1)*gap))/2, y=40, h=58;
  let lo,hi,done;
  const cells=svg.selectAll("g.cell").data(arr).enter().append("g")
    .attr("transform",(d,i)=>`translate(${x0+i*(cw+gap)},${y})`);
  cells.append("rect").attr("class","box").attr("width",cw).attr("height",h).attr("rx",8)
    .attr("fill","#1a2030").attr("stroke",C.line).attr("stroke-width",1.5);
  cells.append("text").attr("x",cw/2).attr("y",h/2+6).attr("text-anchor","middle")
    .attr("fill",C.ink).attr("font-size",18).attr("font-weight",600).text(d=>d);
  cells.append("text").attr("class","ptr").attr("x",cw/2).attr("y",h+20).attr("text-anchor","middle")
    .attr("font-size",12).attr("font-weight",700).text("");

  function paint(){
    cells.select("rect.box").attr("fill","#1a2030").attr("stroke",C.line).attr("stroke-width",1.5);
    cells.select("text.ptr").text("").attr("fill",C.muted);
    if(done){
      [lo,hi].forEach(k=>cells.filter((d,i)=>i===k).select("rect.box").attr("fill","#173a26").attr("stroke",C.good).attr("stroke-width",2.5));
      return;
    }
    [lo,hi].forEach(k=>cells.filter((d,i)=>i===k).select("rect.box").attr("stroke",C.B).attr("stroke-width",2.5).attr("fill","#3a2e16"));
    cells.filter((d,i)=>i===lo).select("text.ptr").text("lo ↑").attr("fill",C.B);
    cells.filter((d,i)=>i===hi).select("text.ptr").text("hi ↑").attr("fill",C.B);
  }
  function reset(){ lo=0; hi=n-1; done=false; paint();
    d3.select("#tp-readout").html(`target = <b>${target}</b> · lo=arr[0]=${arr[0]}, hi=arr[${n-1}]=${arr[n-1]}`); }
  function step(){
    if(done||lo>=hi) return;
    const s=arr[lo]+arr[hi];
    if(s===target){ done=true; paint(); d3.select("#tp-readout").html(`arr[${lo}]+arr[${hi}] = ${arr[lo]}+${arr[hi]} = <b>${target}</b> ✓ found`); return; }
    if(s<target){ d3.select("#tp-readout").html(`${arr[lo]}+${arr[hi]} = ${s} &lt; ${target} → move lo right`); lo++; }
    else { d3.select("#tp-readout").html(`${arr[lo]}+${arr[hi]} = ${s} &gt; ${target} → move hi left`); hi--; }
    if(lo>=hi){ done=true; }
    paint();
  }
  d3.select("#tp-step").on("click",step);
  d3.select("#tp-reset").on("click",reset);
  reset();
})();
