/* ddpm.viz.js — interactive figures for deep-learning/diffusion-models/ddpm.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-wt  per-timestep weight of the true bound vs L_simple's flat weight (linear schedule)
     #viz-rd  reported rate–distortion along the reverse chain (CIFAR-10 test set, Table 4) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* the schedule of the original study: T = 1000, beta linear from 1e-4 (t = 1) to 0.02 (t = T) */
  const T=1000, beta=[NaN], alpha=[NaN], abar=[1];
  for(let t=1;t<=T;t++){const b=1e-4+(t-1)/(T-1)*(0.02-1e-4);beta[t]=b;alpha[t]=1-b;abar[t]=abar[t-1]*(1-b);}
  const sig2=(t,mode)=>mode==="beta"?beta[t]:beta[t]*(1-abar[t-1])/(1-abar[t]);
  /* weight on ‖ε − ε_θ‖² inside L_(t−1):  β_t² / (2 σ_t² α_t (1 − ᾱ_t)),  defined for t = 2..T */
  const weight=(t,mode)=>beta[t]*beta[t]/(2*sig2(t,mode)*alpha[t]*(1-abar[t]));

  /* ═══════════════ 1. true-bound weight vs L_simple ═══════════════ */
  (function(){
    const svg=d3.select("#wt-svg"); if(svg.empty()) return;
    const W=760,H=360,m={l:62,r:24,t:34,b:50};
    const TS=d3.range(2,T+1);
    const norm={};
    ["beta","tilde"].forEach(mode=>{
      const w=TS.map(t=>weight(t,mode)), mean=d3.mean(w), tot=d3.sum(w);
      const n=w.map(v=>v/mean), cum=[]; let c=0; n.forEach((v,i)=>{c+=v;cum[i]=c/(TS.length);});
      let half=TS[TS.length-1]; for(let i=0;i<cum.length;i++){if(cum[i]>=0.5){half=TS[i];break;}}
      let mi=0; n.forEach((v,i)=>{if(v<n[mi])mi=i;});
      norm[mode]={n,cum,half,minT:TS[mi],minV:n[mi],tot};
    });
    const y=d3.scaleLog().domain([0.3,100]).range([H-m.b,m.t]);
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0.3,0.5,1,2,5,10,20,50,100]).tickFormat(d=>d+"×")).attr("color",C.muted);
    svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11)
      .text("weight on ‖ε − ε_θ‖² at step t, each scheme normalised to the same total over t = 2…1000 (log scale)");
    const xlab=svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11);
    const shade=svg.append("rect").attr("y",m.t).attr("height",H-m.b-m.t).attr("fill",C.accent).attr("opacity",0.08);
    const flat=svg.append("line").attr("stroke",C.a2).attr("stroke-width",2);
    const flatTxt=svg.append("text").attr("fill",C.a2).attr("font-size",11).attr("text-anchor","end");
    const other=svg.append("path").attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.2).attr("stroke-dasharray","4 4");
    const main=svg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2);
    const mainTxt=svg.append("text").attr("fill",C.accent).attr("font-size",11);
    const kLine=svg.append("line").attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.ink).attr("stroke-dasharray","3 3");
    const kDot=svg.append("circle").attr("r",4.5).attr("fill",C.accent);
    const halfLine=svg.append("line").attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.violet).attr("stroke-dasharray","2 5");
    const halfTxt=svg.append("text").attr("fill",C.violet).attr("font-size",10.5);

    function draw(){
      const mode=d3.select("#wt-var").property("value"), xs=d3.select("#wt-x").property("value");
      const k=+d3.select("#wt-k").property("value");
      d3.select("#wt-kv").text(k);
      const x=(xs==="log"?d3.scaleLog():d3.scaleLinear()).domain([2,T]).range([m.l,W-m.r]);
      gx.call(xs==="log"?d3.axisBottom(x).tickValues([2,5,10,20,50,100,200,500,1000]).tickFormat(d3.format("d")):d3.axisBottom(x).ticks(10)).attr("color",C.muted);
      xlab.text("timestep t ("+(xs==="log"?"log axis":"linear axis")+"; t = 1 is the decoder term L₀, not plotted)");
      const nm=norm[mode], no=norm[mode==="beta"?"tilde":"beta"];
      const line=d3.line().x((d,i)=>x(TS[i])).y(d=>y(Math.max(0.3,Math.min(100,d))));
      main.attr("d",line(nm.n)); other.attr("d",line(no.n));
      flat.attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(1)).attr("y2",y(1));
      flatTxt.attr("x",W-m.r-4).attr("y",y(1)-6).text("L_simple: flat, 1×");
      mainTxt.attr("x",x(2)+8).attr("y",y(Math.min(100,nm.n[0]))+4).text("true bound, σ² = "+(mode==="beta"?"β":"β̃")+"  (dashed: σ² = "+(mode==="beta"?"β̃":"β")+")");
      shade.attr("x",m.l).attr("width",Math.max(0,x(k)-m.l));
      kLine.attr("x1",x(k)).attr("x2",x(k));
      kDot.attr("cx",x(k)).attr("cy",y(Math.max(0.3,Math.min(100,nm.n[k-2]))));
      halfLine.attr("x1",x(nm.half)).attr("x2",x(nm.half));
      halfTxt.attr("x",x(nm.half)+4).attr("y",H-m.b-8).text("half of the bound's weight lies left of t = "+nm.half);
      const shareB=100*nm.cum[k-2], shareS=100*(k-1)/(T-1), wk=nm.n[k-2];
      d3.select("#wt-readout").html(
        `σ² = <b>${mode==="beta"?"β":"β̃"}</b> · at t = <b>${k}</b> the true bound weights the noise error <b>${wk.toFixed(2)}×</b> as heavily as L_simple does`+
        (wk>1?` — L_simple <b>down-weights</b> this step by ${wk.toFixed(1)}×`:` — L_simple <b>up-weights</b> this step by ${(1/wk).toFixed(2)}×`)+
        `<br>share of total weight on t = 2…${k}: true bound <b>${shareB.toFixed(1)}%</b> vs L_simple <b>${shareS.toFixed(1)}%</b>`+
        ` · bound weight at t = 2 is ${nm.n[0].toFixed(0)}× flat; minimum ${nm.minV.toFixed(2)}× at t = ${nm.minT}; at t = 1000 it is ${nm.n[nm.n.length-1].toFixed(2)}×`);
    }
    ["#wt-var","#wt-x","#wt-k"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    draw();
  })();

  /* ═══════════════ 2. rate–distortion along the reverse chain (Table 4) ═══════════════ */
  (function(){
    const svg=d3.select("#rd-svg"); if(svg.empty()) return;
    /* reverse process time (T − t + 1), rate in bits/dim, distortion RMSE on [0, 255] — as reported */
    const ROWS=[[100,0.00000,67.60125],[200,0.00013,60.97170],[300,0.00081,54.18826],[400,0.00282,46.12765],[500,0.00716,38.03236],
                [600,0.01507,30.80948],[700,0.02866,24.43656],[800,0.05415,18.47482],[900,0.11994,12.02277],[1000,1.77581,0.95136]];
    const RATE_END=1.77581, L0_BITS=1.97;
    const W=760,H=340,m={l:62,r:30,t:34,b:50};
    const y=d3.scaleLinear().domain([0,72]).range([H-m.b,m.t]);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)).attr("color",C.muted);
    svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("distortion: RMSE of x̂₀ against x₀, on the 0–255 pixel scale");
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`);
    const xlab=svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11);
    const path=svg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    const dots=svg.append("g"), labs=svg.append("g");
    const guideV=svg.append("line").attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("y1",m.t).attr("y2",H-m.b);
    const guideH=svg.append("line").attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("x1",m.l);

    function draw(){
      const i=+d3.select("#rd-i").property("value"), view=d3.select("#rd-x").property("value");
      const row=ROWS[i];
      d3.select("#rd-iv").text(row[0]);
      const x=view==="rate"?d3.scaleLinear().domain([0,1.85]).range([m.l,W-m.r]):d3.scaleLinear().domain([0,1000]).range([m.l,W-m.r]);
      const X=r=>view==="rate"?x(r[1]):x(r[0]);
      gx.call(d3.axisBottom(x).ticks(view==="rate"?9:10)).attr("color",C.muted);
      xlab.text(view==="rate"?"rate: cumulative bits/dim received so far":"reverse process time T − t + 1 (steps of sampling completed)");
      path.attr("d",d3.line().x(X).y(r=>y(r[2]))(ROWS));
      dots.selectAll("circle").data(ROWS).join("circle").attr("cx",X).attr("cy",r=>y(r[2])).attr("r",(r,j)=>j===i?6.5:3.8)
        .attr("fill",(r,j)=>j===i?C.a2:(j<=i?C.accent:C.muted)).style("cursor","pointer")
        .on("click",(e,r)=>{d3.select("#rd-i").property("value",ROWS.indexOf(r));draw();});
      labs.selectAll("text").data(view==="step"?ROWS:ROWS.filter((r,j)=>j===0||j>=8)).join("text")
        .attr("x",r=>X(r)+(view==="rate"&&r[0]<1000?8:-6)).attr("y",r=>y(r[2])-9).attr("text-anchor",r=>view==="rate"&&r[0]<1000?"start":"end")
        .attr("fill",C.muted).attr("font-size",10).text(r=>r[0]);
      guideV.attr("x1",X(row)).attr("x2",X(row));
      guideH.attr("x2",X(row)).attr("y1",y(row[2])).attr("y2",y(row[2]));
      const t=T-row[0]+1, ab=abar[t];
      const prev=i>0?ROWS[i-1]:null;
      d3.select("#rd-readout").html(
        `after <b>${row[0]}</b> reverse steps (x<sub>t</sub> with t = ${t}): rate <b>${row[1].toFixed(5)}</b> bits/dim · RMSE <b>${row[2].toFixed(2)}</b> on 0–255`+
        ` · <b>${(100*row[1]/RATE_END).toFixed(1)}%</b> of the final ${RATE_END} bits/dim received`+
        (prev?` · the last 100 steps added ${(row[1]-prev[1]).toFixed(5)} bits/dim and cut RMSE by ${(prev[2]-row[2]).toFixed(2)}`:"")+
        `<br>schedule at that t (own arithmetic): ᾱ<sub>t</sub> = ${ab.toExponential(2)}, so x<sub>t</sub> keeps √ᾱ<sub>t</sub> = ${Math.sqrt(ab).toFixed(3)} of the signal`+
        (row[0]===1000?` · the decoder L₀ then costs a further ${L0_BITS} bits/dim: ${RATE_END.toFixed(2)} + ${L0_BITS} ≈ 3.75 bits/dim, the lossless total`:""));
    }
    ["#rd-i","#rd-x"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    draw();
  })();
})();
