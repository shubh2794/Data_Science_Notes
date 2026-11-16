/* knowledge-distillation.viz.js — extracted from knowledge-distillation.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#kd-svg"),W=640,H=260;
  const labels=["cat","dog","fox","car","ship","tree"];
  const logits=[6.0,3.2,2.6,0.4,0.1,1.0];
  function softmax(z,T){ const m=Math.max(...z); const e=z.map(v=>Math.exp((v-m)/T)); const s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }
  function draw(){
    svg.selectAll("*").remove();
    const T=+d3.select("#kd-t").property("value");
    const p=softmax(logits,T);
    const x0=70, bw=70, gap=18, base=200, maxH=150;
    svg.append("text").attr("x",24).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("teacher soft targets at T = "+T.toFixed(1));
    labels.forEach((l,i)=>{
      const x=x0+i*(bw+gap), h=p[i]*maxH*1.0;
      svg.append("rect").attr("x",x).attr("y",base-h).attr("width",bw).attr("height",Math.max(1,h)).attr("rx",4)
        .attr("fill",i===0?C.accent:"#5b9cff").attr("fill-opacity",i===0?0.9:0.7);
      svg.append("text").attr("x",x+bw/2).attr("y",base-h-6).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.ink).text((p[i]*100).toFixed(1)+"%");
      svg.append("text").attr("x",x+bw/2).attr("y",base+16).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text(l);
    });
    svg.append("line").attr("x1",x0-10).attr("y1",base).attr("x2",x0+labels.length*(bw+gap)-gap+10).attr("y2",base).attr("stroke","#3a4150");
    const ent=-p.reduce((a,v)=>a+(v>1e-9?v*Math.log2(v):0),0);
    d3.select("#kd-read").html(`T=<b>${T.toFixed(1)}</b> · entropy <b>${ent.toFixed(2)}</b> bits · ${T<=1.2?"sharp — almost one-hot, little extra signal":"softer — the small dog/fox/tree probabilities are the dark knowledge the student learns"}`);
  }
  d3.select("#kd-t").on("input",draw);
  draw();
})();

/* §04 — exact soft-target gradient T²·∂C/∂zᵢ = T·(qᵢ − pᵢ) against the
   high-temperature logit-matching approximation (zᵢ − vᵢ)/N.
   Fixed zero-mean teacher (v) and student (z) logits; "spread" scales both. */
(function(){
  const svg=d3.select("#kdg-svg"); if(svg.empty()) return;
  const C={teach:"#5b9cff",exact:"#f87171",approx:"#ffb454",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const W=640,H=280,m={l:56,r:16,t:30,b:40};
  const labels=["cat","dog","fox","car","ship","tree"];
  const v0=[3.0,1.2,0.6,-1.8,-2.4,-0.6];   // teacher logits, sum 0
  const z0=[2.0,1.6,-0.2,-1.0,-1.6,-0.8];  // student logits, sum 0
  const N=labels.length;
  function softmax(z,T){const mx=Math.max(...z);const e=z.map(x=>Math.exp((x-mx)/T));const s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);}
  const x=d3.scaleBand().domain(labels).range([m.l,W-m.r]).padding(0.28);
  const gAxisY=svg.append("g").attr("transform",`translate(${m.l},0)`);
  svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickSize(0))
     .call(g=>g.select(".domain").remove()).selectAll("text").attr("fill",C.muted).attr("font-size",11);
  svg.append("text").attr("x",m.l).attr("y",16).attr("fill",C.muted).attr("font-size",11)
     .text("per-logit gradient on the student · solid = exact T²·∂C/∂z · outlined = (z − v)/N");
  const zero=svg.append("line").attr("stroke","#3a4150");
  const gBars=svg.append("g");
  function draw(){
    const T=+d3.select("#kdg-t").property("value"), s=+d3.select("#kdg-s").property("value");
    const v=v0.map(a=>a*s), z=z0.map(a=>a*s);
    const p=softmax(v,T), q=softmax(z,T);
    const exact=q.map((qi,i)=>T*(qi-p[i]));
    const approx=z.map((zi,i)=>(zi-v[i])/N);
    const lim=Math.max(0.25,d3.max(exact.concat(approx),d=>Math.abs(d)))*1.1;
    const y=d3.scaleLinear().domain([-lim,lim]).range([H-m.b,m.t]);
    gAxisY.call(d3.axisLeft(y).ticks(5)).call(g=>g.selectAll("text").attr("fill",C.muted))
          .call(g=>g.selectAll("line,path").attr("stroke",C.line));
    zero.attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(0)).attr("y2",y(0));
    const bw=x.bandwidth()/2;
    const rows=labels.map((l,i)=>({l,e:exact[i],a:approx[i]}));
    const g=gBars.selectAll("g.c").data(rows).join(en=>{const gg=en.append("g").attr("class","c");
      gg.append("rect").attr("class","e");gg.append("rect").attr("class","a");return gg;});
    g.select("rect.e").attr("x",d=>x(d.l)).attr("width",bw-2).attr("y",d=>Math.min(y(0),y(d.e)))
      .attr("height",d=>Math.max(1,Math.abs(y(d.e)-y(0)))).attr("fill",C.exact).attr("fill-opacity",0.85).attr("rx",2);
    g.select("rect.a").attr("x",d=>x(d.l)+bw).attr("width",bw-2).attr("y",d=>Math.min(y(0),y(d.a)))
      .attr("height",d=>Math.max(1,Math.abs(y(d.a)-y(0)))).attr("fill","none").attr("stroke",C.approx).attr("stroke-width",2).attr("rx",2);
    const num=d3.sum(rows,d=>(d.e-d.a)**2), den=d3.sum(rows,d=>d.a**2);
    const rel=Math.sqrt(num/den);
    const tail=Math.abs(exact[4])/Math.max(1e-9,Math.abs(approx[4]));
    d3.select("#kdg-read").html(`T = <b>${T.toFixed(1)}</b> · relative gap between exact and logit-matching gradients <b>${(rel*100).toFixed(1)}%</b> · on the most negative teacher logit ("ship") the exact gradient is <b>${(tail*100).toFixed(0)}%</b> of the approximation — ${rel<0.1?"high-T regime: distillation ≈ logit matching":"low-T regime: very negative logits are largely ignored"}`);
  }
  d3.selectAll("#kdg-t,#kdg-s").on("input",draw);
  draw();
})();
