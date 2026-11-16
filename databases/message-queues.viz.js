/* message-queues.viz.js — interactive figures for message-queues.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: consumer-group partition assignment ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#222833"};
  const svg=d3.select("#cg-svg"); if(svg.empty()) return;
  const W=640,H=260;
  const TRAFFIC=1200, CAPACITY=320;           // msg/s produced · msg/s one consumer can handle
  const PAL=["#5b9cff","#4ade80","#c084fc","#22d3ee","#f472b6","#facc15","#fb923c","#a3e635","#38bdf8","#e879f9"];
  let dead=new Set();
  function draw(){
    const P=+d3.select("#cg-p").property("value"), Cn=+d3.select("#cg-c").property("value"), hot=d3.select("#cg-hot").property("checked");
    d3.select("#cg-p-v").text(P); d3.select("#cg-c-v").text(Cn);
    dead=new Set([...dead].filter(i=>i<Cn));
    const live=d3.range(Cn).filter(i=>!dead.has(i));
    // range assignor: contiguous blocks, first (P mod n) consumers get one extra
    const owner=new Array(P).fill(-1);
    if(live.length){ const base=Math.floor(P/live.length), extra=P%live.length; let p=0;
      live.forEach((c,idx)=>{ const k=base+(idx<extra?1:0); for(let j=0;j<k;j++) owner[p++]=c; }); }
    const share=d3.range(P).map(p=> hot ? (p===0?0.4:0)+0.6/P : 1/P);
    const load=d3.range(Cn).map(c=>d3.sum(d3.range(P).filter(p=>owner[p]===c),p=>share[p]*TRAFFIC));
    svg.selectAll("*").remove();
    const prow=Math.min(20,(H-40)/P), crow=Math.min(24,(H-40)/Cn);
    const px=40, cx=300, top=28;
    svg.append("text").attr("x",px).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("topic partitions");
    svg.append("text").attr("x",cx).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("consumers (click to crash)");
    svg.append("text").attr("x",450).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("load vs capacity "+CAPACITY+" msg/s");
    const py=p=>top+p*prow+prow/2, cy=c=>top+c*crow+crow/2;
    d3.range(P).forEach(p=>{
      if(owner[p]>=0) svg.append("path").attr("d",`M${px+90},${py(p)} C${px+170},${py(p)} ${cx-80},${cy(owner[p])} ${cx},${cy(owner[p])}`)
        .attr("fill","none").attr("stroke",PAL[owner[p]]).attr("stroke-opacity",.6).attr("stroke-width",1.4+share[p]*8);
      svg.append("rect").attr("x",px).attr("y",py(p)-prow/2+2).attr("width",90).attr("height",prow-4).attr("rx",3)
        .attr("fill",owner[p]>=0?PAL[owner[p]]:C.panel).attr("fill-opacity",owner[p]>=0?.25:1).attr("stroke",hot&&p===0?C.q:C.line);
      svg.append("text").attr("x",px+6).attr("y",py(p)+4).attr("fill",C.ink).attr("font-size",10).text("P"+p+"  "+Math.round(share[p]*TRAFFIC)+"/s"+(hot&&p===0?"  hot":""));
    });
    d3.range(Cn).forEach(c=>{
      const isDead=dead.has(c), idle=!isDead && load[c]===0;
      const g=svg.append("g").style("cursor","pointer").on("click",()=>{ if(dead.has(c)) dead.delete(c); else dead.add(c); draw(); });
      g.append("rect").attr("x",cx).attr("y",cy(c)-crow/2+2).attr("width",120).attr("height",crow-4).attr("rx",4)
        .attr("fill",isDead?"#2a1416":C.panel).attr("stroke",isDead?C.bad:PAL[c]);
      g.append("text").attr("x",cx+8).attr("y",cy(c)+4).attr("fill",isDead?C.bad:C.ink).attr("font-size",10)
        .text("consumer "+(c+1)+(isDead?"  ✗ crashed":idle?"  (idle)":""));
      if(!isDead){
        const bw=150, w=Math.min(load[c]/CAPACITY,1.6)/1.6*bw;
        svg.append("rect").attr("x",450).attr("y",cy(c)-crow/2+5).attr("width",bw).attr("height",crow-10).attr("rx",3).attr("fill",C.panel).attr("stroke",C.line);
        svg.append("rect").attr("x",450).attr("y",cy(c)-crow/2+5).attr("width",w).attr("height",crow-10).attr("rx",3).attr("fill",load[c]>CAPACITY?C.bad:C.good).attr("fill-opacity",.85);
        svg.append("line").attr("x1",450+bw/1.6).attr("x2",450+bw/1.6).attr("y1",cy(c)-crow/2+3).attr("y2",cy(c)+crow/2-3).attr("stroke",C.ink).attr("stroke-dasharray","2 2");
        svg.append("text").attr("x",450+bw+6).attr("y",cy(c)+4).attr("fill",load[c]>CAPACITY?C.bad:C.muted).attr("font-size",10).text(Math.round(load[c]));
      }
    });
    const idle=live.filter(c=>load[c]===0).length, over=live.filter(c=>load[c]>CAPACITY);
    const lagRate=d3.sum(over,c=>load[c]-CAPACITY);
    let msg=`${live.length} live consumer${live.length!==1?"s":""} on ${P} partition${P!==1?"s":""} · idle <b>${idle}</b>`;
    if(!live.length) msg+=` · <span style="color:${C.bad}">no consumers — lag grows at ${TRAFFIC} msg/s</span>`;
    else if(over.length) msg+=` · <span style="color:${C.bad}">${over.length} overloaded consumer${over.length>1?"s":""}: lag grows by ≈ <b>${Math.round(lagRate)}</b> msg/s</span>`;
    else msg+=` · <span style="color:${C.good}">group keeps up</span>`;
    if(idle) msg+=` — consumers beyond the partition count do nothing`;
    if(hot && over.length) msg+=` — adding consumers cannot split partition P0; fix the key`;
    d3.select("#cg-read").html(msg);
  }
  d3.selectAll("#cg-p,#cg-c").on("input",draw);
  d3.select("#cg-hot").on("change",draw);
  d3.select("#cg-reset").on("click",()=>{ dead.clear(); draw(); });
  draw();
})();

/* ---- Viz 2: M/M/1 knee + simulated backlog ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#mm-svg"); if(svg.empty()) return;
  const W=640,H=320,m={l:56,r:20};
  const HORIZON=60;
  let seed=2024;
  function rng(s){ return function(){ s|=0; s=s+0x6D2B79F5|0; let t=Math.imul(s^s>>>15,1|s);
    t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  function simulate(lam,mu){
    const r=rng(seed); let t=0,L=0; const pts=[[0,0]]; let area=0;
    while(t<HORIZON){
      const rate=lam+(L>0?mu:0); const dt=-Math.log(1-r())/rate;
      const tn=Math.min(HORIZON,t+dt); area+=L*(tn-t); t=tn; if(t>=HORIZON) break;
      if(r()<lam/rate) L++; else L--;
      pts.push([t,L]);
    }
    pts.push([HORIZON,L]);
    return {pts, mean:area/HORIZON, end:L};
  }
  function draw(){
    const lam=+d3.select("#mm-l").property("value"), mu=+d3.select("#mm-m").property("value");
    d3.select("#mm-l-v").text(lam); d3.select("#mm-m-v").text(mu);
    const rho=lam/mu, stable=rho<1;
    svg.selectAll("*").remove();
    // top: W(ρ) in ms
    const t0=14,t1=140;
    const X=d3.scaleLinear().domain([0,1]).range([m.l,W-m.r]);
    const wmax=1000*20/mu;
    const Y=d3.scaleLinear().domain([0,wmax]).range([t1,t0]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${t1})`).call(d3.axisBottom(X).ticks(10));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y).ticks(4).tickFormat(d=>d+" ms"));
    svg.append("text").attr("x",W-m.r).attr("y",t1+28).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("utilisation ρ = λ/μ");
    const rs=d3.range(0,0.995,0.005).filter(x=>1000/(mu*(1-x))<=wmax);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(1000/(mu*(1-d))))(rs)).attr("fill","none").attr("stroke",C.node).attr("stroke-width",2.4);
    svg.append("line").attr("x1",X(0)).attr("x2",X(1)).attr("y1",Y(1000/mu)).attr("y2",Y(1000/mu)).attr("stroke",C.muted).attr("stroke-dasharray","2 4");
    svg.append("text").attr("x",X(0.02)).attr("y",Y(1000/mu)-4).attr("fill",C.muted).attr("font-size",10).text("service time 1/μ = "+(1000/mu).toFixed(1)+" ms");
    const Wms=stable?1000/(mu-lam):Infinity;
    if(stable && Wms<=wmax){
      svg.append("circle").attr("cx",X(rho)).attr("cy",Y(Wms)).attr("r",5.5).attr("fill",rho>0.85?C.bad:rho>0.7?C.q:C.good);
    } else {
      svg.append("text").attr("x",X(Math.min(rho,1))-4).attr("y",t0+12).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",11).text(stable?"off the chart ↑":"ρ ≥ 1: unstable");
    }
    svg.append("line").attr("x1",X(Math.min(rho,1))).attr("x2",X(Math.min(rho,1))).attr("y1",t0).attr("y2",t1).attr("stroke",C.accent).attr("stroke-dasharray","3 3");
    // bottom: simulated backlog
    const b0=176,b1=296;
    const sim=simulate(lam,mu);
    const ymax=Math.max(10,d3.max(sim.pts,d=>d[1]));
    const X2=d3.scaleLinear().domain([0,HORIZON]).range([m.l,W-m.r]);
    const Y2=d3.scaleLinear().domain([0,ymax]).nice().range([b1,b0]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${b1})`).call(d3.axisBottom(X2).ticks(12).tickFormat(d=>d+"s"));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y2).ticks(4));
    svg.append("text").attr("x",m.l+4).attr("y",b0-4).attr("fill",C.muted).attr("font-size",10).text("simulated messages in system over "+HORIZON+" s");
    svg.append("path").attr("d",d3.line().x(d=>X2(d[0])).y(d=>Y2(d[1])).curve(d3.curveStepAfter)(sim.pts)).attr("fill","none").attr("stroke",stable?C.good:C.bad).attr("stroke-width",1.3);
    if(stable) svg.append("line").attr("x1",X2(0)).attr("x2",X2(HORIZON)).attr("y1",Y2(rho/(1-rho))).attr("y2",Y2(rho/(1-rho))).attr("stroke",C.q).attr("stroke-dasharray","4 3");
    d3.select("#mm-read").html(stable
      ? `ρ = <b>${rho.toFixed(2)}</b> · theory: L = <b>${(rho/(1-rho)).toFixed(2)}</b> messages, W = <b>${Wms.toFixed(1)} ms</b>, W<sub>q</sub> = <b>${(rho*1000/(mu-lam)).toFixed(1)} ms</b> (${(rho/(1-rho)).toFixed(1)}× the service time) · simulated mean L = <b>${sim.mean.toFixed(2)}</b> (dashed line = theory; short runs at high ρ wander far from it)`
      : `<span style="color:${C.bad}">ρ = ${rho.toFixed(2)} ≥ 1 — arrivals outpace service; backlog grows by about ${(lam-mu)} msg/s forever (${sim.end} queued after ${HORIZON} s). Add consumers or shed load.</span>`);
  }
  d3.selectAll("#mm-l,#mm-m").on("input",draw);
  d3.select("#mm-rs").on("click",()=>{ seed=(seed*16807+11)%2147483647; draw(); });
  draw();
})();
