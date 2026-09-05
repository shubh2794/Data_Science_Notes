/* naive-bayes.viz.js — extracted from naive-bayes.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

/* Interactive Naive Bayes posterior */
(function(){
  const svg=d3.select("#nb-svg"), W=640, H=170;
  const x=d3.scaleLinear().domain([0,1]).range([20,W-20]);
  svg.append("rect").attr("x",20).attr("y",60).attr("width",W-40).attr("height",30).attr("rx",6).attr("fill","#1e222d").attr("stroke",C.line);
  const bar=svg.append("rect").attr("x",20).attr("y",60).attr("height",30).attr("rx",6).attr("fill",C.bad);
  const mark=svg.append("line").attr("y1",50).attr("y2",100).attr("stroke",C.ink).attr("stroke-dasharray","4 4");
  svg.append("text").attr("x",20).attr("y",120).attr("fill",C.muted).attr("font-size",11).text("P(ham | \"free\")");
  const lbl=svg.append("text").attr("y",120).attr("fill",C.B).attr("font-size",11).attr("text-anchor","end").attr("x",W-20).text("P(spam | \"free\")");
  function calc(){
    const ps=+d3.select("#nb-prior").property("value");
    const l1=+d3.select("#nb-l1").property("value"); // P(free|spam)
    const l2=+d3.select("#nb-l2").property("value"); // P(free|ham)
    const num=ps*l1, den=ps*l1+(1-ps)*l2;
    const post=den===0?0:num/den;
    bar.transition().duration(200).attr("width",(W-40)*post).attr("fill",post>0.5?C.B:C.bad);
    mark.attr("x1",x(0.5)).attr("x2",x(0.5));
    d3.select("#nb-readout").html(`Posterior P(spam | "free") = <b>${(post*100).toFixed(1)}%</b> &nbsp;→&nbsp; classify as <b>${post>0.5?"SPAM":"HAM"}</b>`);
  }
  ["#nb-prior","#nb-l1","#nb-l2"].forEach(id=>d3.select(id).on("input",calc));
  calc();
})();
