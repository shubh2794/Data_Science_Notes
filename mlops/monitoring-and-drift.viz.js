/* monitoring-and-drift.viz.js — extracted from monitoring-and-drift.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Primary viz: interactive drift detector ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#dr-svg"),W=640,H=240,m={t:16,r:16,b:30,l:42};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);

  // shared bins over a fixed domain
  const lo=-6, hi=6, NB=40, EPS=1e-6;
  const edges=d3.range(NB+1).map(i=>lo+(hi-lo)*i/NB);
  const centers=d3.range(NB).map(i=>(edges[i]+edges[i+1])/2);
  const bw=(hi-lo)/NB;

  // reference: standard normal N(0,1)
  function pdf(x,mu,sd){return Math.exp(-0.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI));}
  function binProbs(mu,sd){
    // integrate pdf over each bin (midpoint rule * width), then normalize, floor with eps
    let raw=centers.map(c=>pdf(c,mu,sd)*bw);
    let s=d3.sum(raw)||1;
    let p=raw.map(v=>Math.max(v/s,EPS));
    let s2=d3.sum(p);
    return p.map(v=>v/s2);
  }
  const Q=binProbs(0,1); // reference, fixed

  function psi(P,Q){return d3.sum(P.map((p,i)=>(p-Q[i])*Math.log(p/Q[i])));}
  function kl(P,Q){return d3.sum(P.map((p,i)=>p*Math.log(p/Q[i])));}
  function jsDist(P,Q){
    const M=P.map((p,i)=>0.5*(p+Q[i]));
    const jsd=0.5*kl(P,M)+0.5*kl(Q,M); // nats
    return Math.sqrt(Math.max(jsd,0));
  }
  function ksD(P,Q){
    let cp=0,cq=0,mx=0;
    for(let i=0;i<P.length;i++){cp+=P[i];cq+=Q[i];mx=Math.max(mx,Math.abs(cp-cq));}
    return mx;
  }
  // approximate KS two-sample p-value via asymptotic Kolmogorov dist
  function ksP(D,n){
    const en=Math.sqrt(n/2); // both samples size n -> sqrt(n*n/(2n))=sqrt(n/2)
    const t=(en+0.12+0.11/en)*D;
    let s=0; for(let k=1;k<=100;k++){s+=2*Math.pow(-1,k-1)*Math.exp(-2*k*k*t*t);}
    return Math.min(Math.max(s,0),1);
  }

  const x=d3.scaleLinear().domain([lo,hi]).range([0,iw]);
  const y=d3.scaleLinear().range([ih,0]);
  const xAxis=g.append("g").attr("transform",`translate(0,${ih})`);
  const gRef=g.append("g"), gCur=g.append("g");
  g.append("text").attr("x",iw/2).attr("y",ih+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("feature value");
  // legend
  const leg=g.append("g");
  leg.append("rect").attr("x",iw-150).attr("y",2).attr("width",10).attr("height",10).attr("fill","rgba(154,163,178,.45)").attr("stroke",C.muted);
  leg.append("text").attr("x",iw-135).attr("y",11).attr("fill",C.muted).attr("font-size",10).text("reference");
  leg.append("rect").attr("x",iw-70).attr("y",2).attr("width",10).attr("height",10).attr("fill","rgba(163,230,53,.4)").attr("stroke",C.accent);
  leg.append("text").attr("x",iw-55).attr("y",11).attr("fill",C.muted).attr("font-size",10).text("current");

  function badge(v,amber,red){
    const col = v>=red?C.bad : v>=amber?C.a2 : C.good;
    return `<b style="color:${col}">${v.toFixed(3)}</b>`;
  }

  function draw(){
    const dmu=+d3.select("#dr-mu").property("value");
    const sdr=+d3.select("#dr-sd").property("value");
    const n=+d3.select("#dr-n").property("value");
    d3.select("#dr-mu-v").text(dmu.toFixed(2)+"σ");
    d3.select("#dr-sd-v").text("×"+sdr.toFixed(2));

    const P=binProbs(dmu,sdr);
    const _psi=psi(P,Q), _kl=kl(P,Q), _js=jsDist(P,Q), _ks=ksD(P,Q), _ksp=ksP(_ks,n);

    y.domain([0, d3.max([d3.max(P),d3.max(Q)])*1.12]);
    xAxis.call(d3.axisBottom(x).ticks(7));
    svg.selectAll(".domain,.tick line").attr("stroke",C.line);
    svg.selectAll(".tick text").attr("fill",C.muted).attr("font-size",9);

    const bars=(sel,probs,fill,stroke)=>{
      const r=sel.selectAll("rect").data(probs);
      r.enter().append("rect").merge(r)
        .attr("x",(d,i)=>x(edges[i]))
        .attr("width",x(edges[1])-x(edges[0])-1)
        .attr("y",d=>y(d)).attr("height",d=>ih-y(d))
        .attr("fill",fill).attr("stroke",stroke).attr("stroke-opacity",.6);
      r.exit().remove();
    };
    bars(gRef,Q,"rgba(154,163,178,.30)","rgba(154,163,178,.6)");
    bars(gCur,P,"rgba(163,230,53,.32)",C.accent);

    // KS verdict: at large n, even tiny D becomes "significant" (p<0.05)
    const ksSignif = _ksp<0.05;
    const ksCol = ksSignif ? C.bad : C.good;

    // overall drift state driven by PSI
    let state,scol;
    if(_psi>=0.25){state="DRIFT DETECTED";scol=C.bad;}
    else if(_psi>=0.1){state="WATCH — moderate shift";scol=C.a2;}
    else {state="STABLE";scol=C.good;}
    d3.select("#dr-state").html(`<span style="color:${scol}">● ${state}</span> <span class="lbl" style="font-weight:400">(PSI threshold)</span>`);

    d3.select("#dr-read").html(
      `PSI ${badge(_psi,0.1,0.25)} · `+
      `KL ${badge(_kl,0.1,0.3)} · `+
      `JS dist ${badge(_js,0.1,0.2)} · `+
      `KS-D <b style="color:${_ks>=0.1?C.a2:C.good}">${_ks.toFixed(3)}</b> `+
      `(p ${_ksp<0.001?"&lt;0.001":_ksp.toFixed(3)}, n=${n.toLocaleString()}: <b style="color:${ksCol}">${ksSignif?"REJECT — drift":"no rejection"}</b>)`+
      `<br><span class="lbl">KS-D is identical at both n, but the p-value verdict flips with sample size — flip n to 100k at a small Δμ to see KS scream while PSI stays calm.</span>`
    );
  }
  ["#dr-mu","#dr-sd","#dr-n"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
  draw();
})();

/* ---- Secondary viz: streaming Page-Hinkley / DDM warning->drift ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#st-svg"),W=640,H=220,m={t:16,r:16,b:30,l:46};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const N=120;

  // synthetic error stream: low error then a drift ramp after t=60
  function genStream(){
    const out=[];
    for(let t=0;t<N;t++){
      let base = t<60 ? 0.08 : 0.08 + Math.min(0.30,(t-60)*0.012);
      let e = base + (Math.random()-0.5)*0.05;
      out.push(Math.max(0.01,e));
    }
    return out;
  }
  let stream=genStream();

  const x=d3.scaleLinear().domain([0,N-1]).range([0,iw]);
  const y=d3.scaleLinear().domain([0,0.5]).range([ih,0]);
  const xAxis=g.append("g").attr("transform",`translate(0,${ih})`);
  const yAxis=g.append("g");
  g.append("text").attr("x",iw/2).attr("y",ih+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("time (samples)");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-34).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("error rate pₜ");

  const warnBand=g.append("rect").attr("fill","rgba(255,180,84,.10)");
  const driftBand=g.append("rect").attr("fill","rgba(248,113,113,.10)");
  const linePath=g.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",1.6);
  const warnLine=g.append("line").attr("stroke",C.a2).attr("stroke-dasharray","4 3").attr("stroke-width",1);
  const driftLine=g.append("line").attr("stroke",C.bad).attr("stroke-dasharray","4 3").attr("stroke-width",1);
  const cursor=g.append("circle").attr("r",4).attr("fill",C.ink).attr("stroke","#15181f").attr("stroke-width",1.5).style("display","none");
  const warnMark=g.append("circle").attr("r",5).attr("fill",C.a2).style("display","none");
  const driftMark=g.append("circle").attr("r",5).attr("fill",C.bad).style("display","none");

  xAxis.call(d3.axisBottom(x).ticks(6));
  yAxis.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  svg.selectAll(".domain,.tick line").attr("stroke",C.line);
  svg.selectAll(".tick text").attr("fill",C.muted).attr("font-size",9);

  // DDM-style running min of (p + s); warning at +2s, drift at +3s
  let timer=null;
  function reset(){
    if(timer){timer.stop();timer=null;}
    stream=genStream();
    linePath.attr("d",null);
    cursor.style("display","none"); warnMark.style("display","none"); driftMark.style("display","none");
    warnBand.attr("width",0); driftBand.attr("width",0);
    d3.select("#st-read").html(`<span class="lbl">DDM watches the online error rate; warning at p_min+2·s_min, drift at p_min+3·s_min.</span>`);
  }
  function run(){
    reset();
    let i=2, pmin=Infinity, smin=0, warnedAt=null, driftAt=null;
    const ln=d3.line().x(d=>x(d.t)).y(d=>y(d.p));
    const pts=[];
    timer=d3.interval(()=>{
      const p=stream[i];
      const s=Math.sqrt(Math.max(p*(1-p)/(i+1),1e-9));
      if(p+s < pmin+smin){pmin=p; smin=s;}
      const warnTh=pmin+2*smin, driftTh=pmin+3*smin;
      pts.push({t:i,p});
      linePath.attr("d",ln(pts));
      cursor.style("display",null).attr("cx",x(i)).attr("cy",y(p));
      warnLine.attr("x1",0).attr("x2",iw).attr("y1",y(warnTh)).attr("y2",y(warnTh));
      driftLine.attr("x1",0).attr("x2",iw).attr("y1",y(driftTh)).attr("y2",y(driftTh));

      if(warnedAt===null && p+s>=warnTh && p>0.12){
        warnedAt=i; warnMark.style("display",null).attr("cx",x(i)).attr("cy",y(p));
        warnBand.attr("x",x(i)).attr("y",0).attr("width",Math.max(0,x(N-1)-x(i))).attr("height",ih);
      }
      if(driftAt===null && p+s>=driftTh && p>0.15){
        driftAt=i; driftMark.style("display",null).attr("cx",x(i)).attr("cy",y(p));
        driftBand.attr("x",x(i)).attr("y",0).attr("width",Math.max(0,x(N-1)-x(i))).attr("height",ih);
        timer.stop();
        d3.select("#st-read").html(
          `<b style="color:${C.a2}">⚠ warning at t=${warnedAt}</b> → `+
          `<b style="color:${C.bad}">● drift confirmed at t=${driftAt}</b> `+
          `<span class="lbl">— the two-stage sequence: error climbs into the 2σ warning band, then the 3σ drift band fires the alarm.</span>`
        );
      }
      i++;
      if(i>=N){timer.stop();}
    },45);
  }
  d3.select("#st-run").on("click",run);
  d3.select("#st-reset").on("click",reset);
  reset();
})();
