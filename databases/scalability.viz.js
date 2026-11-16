/* scalability.viz.js — interactive figures for scalability.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: Amdahl vs Universal Scalability Law ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",grid:"#2a2f3a"};
  const svg=d3.select("#usl-svg"); if(svg.empty()) return;
  const W=640,H=280,m={l:52,r:20,t:16,b:38};
  const NMAX=128;
  const usl=(n,s,k)=>n/(1+s*(n-1)+k*n*(n-1));
  function draw(){
    const s=+d3.select("#usl-s").property("value");
    const k=+d3.select("#usl-k").property("value");
    const n=+d3.select("#usl-n").property("value");
    d3.select("#usl-s-v").text(s.toFixed(3));
    d3.select("#usl-k-v").text(k.toFixed(4));
    d3.select("#usl-n-v").text(n);
    svg.selectAll("*").remove();
    const ns=d3.range(1,NMAX+1);
    const amd=ns.map(x=>usl(x,s,0)), u=ns.map(x=>usl(x,s,k));
    const ymax=Math.max(8,Math.min(NMAX,d3.max(amd)*1.1));
    const X=d3.scaleLinear().domain([1,NMAX]).range([m.l,W-m.r]);
    const Y=d3.scaleLinear().domain([0,ymax]).range([H-m.b,m.t]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(X).ticks(8));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y).ticks(6));
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("nodes N");
    svg.append("text").attr("transform",`translate(14,${(m.t+H-m.b)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("relative throughput C(N)");
    // clip
    svg.append("clipPath").attr("id","usl-clip").append("rect").attr("x",m.l).attr("y",m.t).attr("width",W-m.l-m.r).attr("height",H-m.t-m.b);
    const g=svg.append("g").attr("clip-path","url(#usl-clip)");
    const line=d3.line().x((d,i)=>X(i+1)).y(d=>Y(d));
    g.append("path").attr("d",line(ns)).attr("fill","none").attr("stroke",C.muted).attr("stroke-dasharray","4 4").attr("stroke-width",1.4);
    g.append("path").attr("d",line(amd)).attr("fill","none").attr("stroke",C.node).attr("stroke-width",2.2);
    g.append("path").attr("d",line(u)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.6);
    // Amdahl ceiling
    if(s>0 && 1/s<ymax){
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",Y(1/s)).attr("y2",Y(1/s)).attr("stroke",C.node).attr("stroke-opacity",.45).attr("stroke-dasharray","2 4");
      g.append("text").attr("x",W-m.r-4).attr("y",Y(1/s)-4).attr("text-anchor","end").attr("fill",C.node).attr("font-size",10).text("Amdahl ceiling 1/σ = "+(1/s).toFixed(1));
    }
    // USL peak
    let peakTxt="no peak (κ = 0): throughput only saturates";
    if(k>0){
      const nstar=Math.sqrt((1-s)/k), cpk=usl(nstar,s,k);
      if(nstar<=NMAX){
        g.append("line").attr("x1",X(nstar)).attr("x2",X(nstar)).attr("y1",Y(0)).attr("y2",Y(cpk)).attr("stroke",C.good).attr("stroke-dasharray","3 3");
        g.append("circle").attr("cx",X(nstar)).attr("cy",Y(cpk)).attr("r",4.5).attr("fill",C.good);
        g.append("text").attr("x",X(nstar)+6).attr("y",Y(cpk)-8).attr("fill",C.good).attr("font-size",10).text("peak N* ≈ "+nstar.toFixed(0));
      }
      peakTxt=`USL peak at <b>N* ≈ ${nstar.toFixed(1)}</b> with C(N*) ≈ <b>${cpk.toFixed(1)}×</b>`;
    }
    // current N marker
    const cn=usl(n,s,k);
    svg.append("line").attr("x1",X(n)).attr("x2",X(n)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.q).attr("stroke-opacity",.7);
    svg.append("circle").attr("cx",X(n)).attr("cy",Y(Math.min(cn,ymax))).attr("r",5).attr("fill",C.q);
    const eff=cn/n*100;
    const prev=n>1?usl(n-1,s,k):0, marg=cn-prev;
    const msg = marg<0 ? `<span style="color:${C.bad}">retrograde: adding node ${n} <b>lowers</b> throughput by ${(-marg).toFixed(2)}</span>`
              : `adding node ${n} adds <b>${marg.toFixed(2)}</b> nodes' worth of capacity`;
    d3.select("#usl-read").html(`N = <b>${n}</b> → C(N) = <b>${cn.toFixed(1)}×</b> (Amdahl ${usl(n,s,0).toFixed(1)}×) · efficiency <b>${eff.toFixed(0)}%</b> · ${msg}<br>${peakTxt}`);
  }
  d3.selectAll("#usl-s,#usl-k,#usl-n").on("input",draw);
  draw();
})();

/* ---- Viz 2: fan-out tail amplification ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2"};
  const svg=d3.select("#tl-svg"); if(svg.empty()) return;
  const W=640,H=220,m={l:52,r:20,t:14,b:36};
  function draw(){
    const n=+d3.select("#tl-n").property("value");
    const q=+d3.select("#tl-q").property("value");
    const hedge=d3.select("#tl-h").property("checked");
    d3.select("#tl-n-v").text(n);
    d3.select("#tl-q-v").text((q*100).toFixed(1)+"%");
    // hedging: a leaf is slow only if both the original and the backup are slow (independence assumed)
    const qe=hedge?q*q:q;
    const f=(nn,qq)=>1-Math.pow(1-qq,nn);
    svg.selectAll("*").remove();
    const X=d3.scaleLinear().domain([1,200]).range([m.l,W-m.r]);
    const Y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(X).ticks(10));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y).ticks(5).tickFormat(d3.format(".0%")));
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-4).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("fan-out n (leaves per request)");
    const ns=d3.range(1,201);
    const line=d3.line().x(d=>X(d));
    svg.append("path").attr("d",line.y(d=>Y(f(d,q)))(ns)).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2.4).attr("stroke-opacity",hedge?.35:1);
    if(hedge) svg.append("path").attr("d",line.y(d=>Y(f(d,qe)))(ns)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.4);
    svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",Y(0.5)).attr("y2",Y(0.5)).attr("stroke",C.muted).attr("stroke-dasharray","2 4").attr("stroke-opacity",.6);
    svg.append("text").attr("x",W-m.r-4).attr("y",Y(0.5)-4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("half of all requests");
    const p=f(n,qe);
    svg.append("line").attr("x1",X(n)).attr("x2",X(n)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.q).attr("stroke-opacity",.7);
    svg.append("circle").attr("cx",X(n)).attr("cy",Y(p)).attr("r",5).attr("fill",hedge?C.good:C.bad);
    // which percentile of the request distribution equals the leaf p(1-q)
    d3.select("#tl-read").html(`fan-out <b>${n}</b>, leaf slow probability <b>${(q*100).toFixed(1)}%</b> → <b style="color:${hedge?C.good:C.bad}">${(p*100).toFixed(1)}%</b> of requests wait on a slow leaf`
      + (hedge?` (idealised hedging with an independent backup: effective leaf q ≈ q² = ${(qe*100).toFixed(3)}%; sending the backup only after the p95 deadline costs about 5% extra leaf load)`:` · without fan-out it would be ${(q*100).toFixed(1)}%`));
  }
  d3.selectAll("#tl-n,#tl-q,#tl-h").on("input",draw).on("change",draw);
  draw();
})();
