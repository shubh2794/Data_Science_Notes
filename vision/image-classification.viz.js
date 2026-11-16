/* image-classification.viz.js — the two figures on vision/image-classification.html.
   Loaded after ../data.js → ../notes.js. Self-contained: needs only d3 from ../vendor.

   1. #sm-svg  — softmax, cross-entropy and label smoothing. Five draggable logits, a
                 temperature and a smoothing ε; shows p, the target q, the loss and
                 the gradient p − q that backpropagation actually sends.
   2. #cal-svg — calibration. A seeded synthetic 10-class problem whose model logits
                 are a scaled copy of the true ones; a temperature slider moves the
                 reliability diagram, ECE and NLL while accuracy stays fixed. */

(function(){
  const C={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel2:"#1e222d",violet:"#a78bfa"};

  function softmax(z,T){
    const m=Math.max(...z.map(v=>v/T));
    const e=z.map(v=>Math.exp(v/T-m));
    const s=e.reduce((a,b)=>a+b,0);
    return e.map(v=>v/s);
  }
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);
    t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  function randn(r){let u=0,v=0;while(u===0)u=r();while(v===0)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  function styleAxis(sel){sel.selectAll(".domain,.tick line").attr("stroke",C.line);
    sel.selectAll(".tick text").attr("fill",C.muted).attr("font-size",9.5);}

  /* ═══ 1 · softmax / cross-entropy / label smoothing ═══════════════════════ */
  (function(){
    const svg=d3.select("#sm-svg"); if(svg.empty()) return;
    const W=760,H=330;
    const names=["cat","dog","fox","car","truck"], K=names.length;
    const z=[2.0,1.2,0.3,-1.0,-1.6];
    const panels=[{x:44,w:200,title:"logits z (drag a bar)"},
                  {x:290,w:220,title:"p = softmax(z/T) vs target q"},
                  {x:556,w:190,title:"gradient ∂L/∂z = (p − q)/T"}];
    const top=36, ph=230;
    const gs=panels.map(p=>{
      const g=svg.append("g").attr("transform",`translate(${p.x},${top})`);
      g.append("text").attr("x",0).attr("y",-14).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text(p.title);
      return g;});

    const xb=d3.scaleBand().domain(names).range([0,panels[0].w]).padding(0.25);
    const yz=d3.scaleLinear().domain([-6,6]).range([ph,0]);
    const xp=d3.scaleBand().domain(names).range([0,panels[1].w]).padding(0.18);
    const yp=d3.scaleLinear().domain([0,1]).range([ph,0]);
    const xg=d3.scaleBand().domain(names).range([0,panels[2].w]).padding(0.25);
    const yg=d3.scaleLinear().domain([-1,1]).range([ph,0]);

    gs[0].append("g").call(d3.axisLeft(yz).ticks(6)).call(styleAxis);
    gs[0].append("line").attr("x1",0).attr("x2",panels[0].w).attr("y1",yz(0)).attr("y2",yz(0)).attr("stroke",C.line);
    gs[1].append("g").call(d3.axisLeft(yp).ticks(5)).call(styleAxis);
    gs[2].append("g").call(d3.axisLeft(yg).ticks(5)).call(styleAxis);
    gs[2].append("line").attr("x1",0).attr("x2",panels[2].w).attr("y1",yg(0)).attr("y2",yg(0)).attr("stroke",C.line);
    [[gs[0],xb],[gs[1],xp],[gs[2],xg]].forEach(([g,x])=>{
      g.append("g").attr("transform",`translate(0,${ph})`).call(d3.axisBottom(x).tickSize(0)).call(styleAxis)
        .selectAll(".tick text").attr("dy","1.1em");});
    const lg=svg.append("g").attr("transform",`translate(${panels[1].x+4},${top+ph+38})`);
    [["p (model)",C.accent],["q (target)",C.a2]].forEach((d,i)=>{
      lg.append("rect").attr("x",i*110).attr("y",-9).attr("width",10).attr("height",10).attr("fill",d[1]);
      lg.append("text").attr("x",i*110+15).attr("y",0).attr("fill",C.muted).attr("font-size",11).text(d[0]);});

    const zBars=gs[0].append("g"), pBars=gs[1].append("g"), gBars=gs[2].append("g");
    const ro=d3.select("#sm-readout");

    function draw(){
      const T=+d3.select("#sm-T").property("value");
      const eps=+d3.select("#sm-eps").property("value");
      const y=+d3.select("#sm-y").property("value");
      d3.select("#sm-Tv").text(T.toFixed(2)); d3.select("#sm-epsv").text(eps.toFixed(2));
      const p=softmax(z,T);
      const q=names.map((_,k)=>(k===y?1-eps:0)+eps/K);
      const L=-q.reduce((a,qk,k)=>a+qk*Math.log(p[k]),0);
      const Hq=-q.reduce((a,qk)=>a+(qk>0?qk*Math.log(qk):0),0);
      const grad=p.map((pk,k)=>(pk-q[k])/T);
      const pred=p.indexOf(Math.max(...p));

      const zb=zBars.selectAll("g.zb").data(z.map((v,k)=>({v,k}))).join(enter=>{
        const gg=enter.append("g").attr("class","zb dragpt").style("cursor","ns-resize");
        gg.append("rect").attr("class","bar"); gg.append("rect").attr("class","hit").attr("fill","transparent");
        gg.append("text").attr("class","val").attr("text-anchor","middle").attr("font-size",10);
        return gg;});
      zb.select("rect.bar").attr("x",d=>xb(names[d.k])).attr("width",xb.bandwidth())
        .attr("y",d=>Math.min(yz(d.v),yz(0))).attr("height",d=>Math.abs(yz(d.v)-yz(0)))
        .attr("fill",d=>d.k===y?C.a2:C.accent).attr("opacity",.85);
      zb.select("rect.hit").attr("x",d=>xb(names[d.k])-4).attr("width",xb.bandwidth()+8).attr("y",0).attr("height",ph);
      zb.select("text.val").attr("x",d=>xb(names[d.k])+xb.bandwidth()/2)
        .attr("y",d=>d.v>=0?yz(d.v)-4:yz(d.v)+12).attr("fill",C.ink).text(d=>d.v.toFixed(1));
      zb.call(d3.drag().on("drag",function(ev,d){
        z[d.k]=Math.max(-6,Math.min(6,Math.round(yz.invert(ev.y)*10)/10)); draw();}));

      const pd=names.flatMap((n,k)=>[{k,s:"p",v:p[k]},{k,s:"q",v:q[k]}]);
      const bw=xp.bandwidth()/2;
      pBars.selectAll("rect").data(pd).join("rect")
        .attr("x",d=>xp(names[d.k])+(d.s==="p"?0:bw)).attr("width",bw-1)
        .attr("y",d=>yp(d.v)).attr("height",d=>ph-yp(d.v))
        .attr("fill",d=>d.s==="p"?C.accent:C.a2).attr("opacity",d=>d.s==="p"?.9:.75);
      pBars.selectAll("text").data(p).join("text").attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.ink)
        .attr("x",(d,k)=>xp(names[k])+bw/2).attr("y",d=>yp(d)-4).text(d=>d.toFixed(2));

      gBars.selectAll("rect").data(grad).join("rect")
        .attr("x",(d,k)=>xg(names[k])).attr("width",xg.bandwidth())
        .attr("y",d=>Math.min(yg(d),yg(0))).attr("height",d=>Math.abs(yg(d)-yg(0)))
        .attr("fill",d=>d<0?C.good:C.bad).attr("opacity",.85);
      gBars.selectAll("text").data(grad).join("text").attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.ink)
        .attr("x",(d,k)=>xg(names[k])+xg.bandwidth()/2).attr("y",d=>d>=0?yg(d)-4:yg(d)+12).text(d=>d.toFixed(2));

      const gap=eps>0?Math.log((1-eps+eps/K)/(eps/K)):Infinity;
      ro.html(`true class <b>${names[y]}</b> · predicted <b>${names[pred]}</b> (${(100*p[pred]).toFixed(1)}%) · `+
        `loss L = −∑ q·log p = <b>${L.toFixed(3)}</b> · floor H(q) = ${Hq.toFixed(3)} · excess KL(q‖p) = <b>${(L-Hq).toFixed(3)}</b><br>`+
        `loss-minimising logit gap (true − each other): <b>${isFinite(gap)?gap.toFixed(2):"∞ — with ε = 0 the loss only reaches its floor as the gap → ∞"}</b>`);
    }
    d3.selectAll("#sm-T,#sm-eps,#sm-y").on("input",draw);
    d3.select("#sm-reset").on("click",()=>{[2.0,1.2,0.3,-1.0,-1.6].forEach((v,i)=>z[i]=v);draw();});
    draw();
  })();

  /* ═══ 2 · calibration: reliability diagram under temperature scaling ══════ */
  (function(){
    const svg=d3.select("#cal-svg"); if(svg.empty()) return;
    const N=3000, K=10;
    // one fixed synthetic world: true logits z, label y ~ softmax(z)
    const r=mulberry32(20241117);
    const Z=[], Y=[];
    for(let i=0;i<N;i++){
      const sc=1+5*r();
      const zi=Array.from({length:K},()=>sc*randn(r));
      const pi=softmax(zi,1); let u=r(), k=0, c=pi[0]; while(u>c&&k<K-1){k++;c+=pi[k];}
      Z.push(zi); Y.push(k);
    }
    function evaluate(s,T,B){
      const bins=Array.from({length:B},()=>({n:0,acc:0,conf:0}));
      let nll=0, acc=0, mconf=0;
      for(let i=0;i<N;i++){
        const p=softmax(Z[i].map(v=>s*v),T);
        let j=0; for(let k=1;k<K;k++) if(p[k]>p[j]) j=k;
        const cf=p[j], ok=(j===Y[i])?1:0;
        const b=Math.min(B-1,Math.floor(cf*B));
        bins[b].n++; bins[b].acc+=ok; bins[b].conf+=cf;
        nll-=Math.log(Math.max(p[Y[i]],1e-12)); acc+=ok; mconf+=cf;
      }
      let ece=0; bins.forEach(b=>{ if(b.n){ ece+=b.n/N*Math.abs(b.acc/b.n-b.conf/b.n); b.acc/=b.n; b.conf/=b.n; }});
      return {bins,ece,nll:nll/N,acc:acc/N,mconf:mconf/N};
    }

    const W=760,H=340, m={t:30,l:46};
    const S=250;
    const g1=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
    const x=d3.scaleLinear().domain([0,1]).range([0,S]), y=d3.scaleLinear().domain([0,1]).range([S,0]);
    g1.append("g").attr("transform",`translate(0,${S})`).call(d3.axisBottom(x).ticks(5)).call(styleAxis);
    g1.append("g").call(d3.axisLeft(y).ticks(5)).call(styleAxis);
    g1.append("text").attr("x",S/2).attr("y",S+30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("confidence (max p) — bin");
    g1.append("text").attr("transform",`translate(-34,${S/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("accuracy in bin");
    g1.append("text").attr("x",0).attr("y",-12).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("reliability diagram");
    const barsG=g1.append("g"), gapG=g1.append("g");
    g1.append("line").attr("x1",x(0)).attr("y1",y(0)).attr("x2",x(1)).attr("y2",y(1)).attr("stroke",C.muted).attr("stroke-dasharray","4 3");

    const g2=svg.append("g").attr("transform",`translate(${m.l+S+90},${m.t})`);
    const CW=330, CH=S;
    const Ts=d3.range(0,61).map(i=>0.4*Math.pow(6/0.4,i/60));
    const xt=d3.scaleLog().domain([0.4,6]).range([0,CW]);
    const ye=d3.scaleLinear().range([CH,0]);
    const axE=g2.append("g"), axT=g2.append("g").attr("transform",`translate(0,${CH})`);
    g2.append("text").attr("x",CW/2).attr("y",CH+30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("temperature T (log scale)");
    g2.append("text").attr("x",0).attr("y",-12).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("ECE (amber) and NLL (blue, right scale) against T");
    const yn=d3.scaleLinear().range([CH,0]);
    const axN=g2.append("g").attr("transform",`translate(${CW},0)`);
    const pE=g2.append("path").attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2);
    const pN=g2.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    const cur=g2.append("line").attr("y1",0).attr("y2",CH).attr("stroke",C.ink).attr("stroke-dasharray","3 3");
    const dotE=g2.append("circle").attr("r",4).attr("fill",C.a2), dotN=g2.append("circle").attr("r",4).attr("fill",C.accent);
    const ro=d3.select("#cal-readout");
    let curveKey="", curve=null;

    function draw(){
      const s=+d3.select("#cal-model").property("value");
      const B=+d3.select("#cal-bins").property("value");
      const T=+d3.select("#cal-T").property("value");
      d3.select("#cal-Tv").text(T.toFixed(2));
      const key=s+"|"+B;
      if(key!==curveKey){
        curve=Ts.map(t=>{const e=evaluate(s,t,B);return {t,ece:e.ece,nll:e.nll};});
        curveKey=key;
        ye.domain([0,d3.max(curve,d=>d.ece)*1.08]); yn.domain([d3.min(curve,d=>d.nll)*0.95,Math.min(d3.max(curve,d=>d.nll),6)]);
        axE.call(d3.axisLeft(ye).ticks(5)).call(styleAxis);
        axN.call(d3.axisRight(yn).ticks(5)).call(styleAxis);
        axT.call(d3.axisBottom(xt).tickValues([0.5,1,2,3,4,6]).tickFormat(d3.format("~g"))).call(styleAxis);
        pE.attr("d",d3.line().x(d=>xt(d.t)).y(d=>ye(d.ece))(curve));
        pN.attr("d",d3.line().x(d=>xt(d.t)).y(d=>yn(Math.min(d.nll,yn.domain()[1])))(curve));
      }
      const e=evaluate(s,T,B);
      const w=S/B;
      barsG.selectAll("rect").data(e.bins).join("rect")
        .attr("x",(d,i)=>i*w+1).attr("width",w-2)
        .attr("y",d=>d.n?y(d.acc):S).attr("height",d=>d.n?S-y(d.acc):0)
        .attr("fill",C.accent).attr("opacity",d=>d.n?0.35+0.6*Math.min(1,d.n/(N/B)):0);
      gapG.selectAll("rect").data(e.bins).join("rect")
        .attr("x",(d,i)=>i*w+1).attr("width",w-2)
        .attr("y",d=>d.n?y(Math.max(d.acc,d.conf)):S).attr("height",d=>d.n?Math.abs(y(d.acc)-y(d.conf)):0)
        .attr("fill",d=>d.conf>d.acc?C.bad:C.good).attr("opacity",.45);
      cur.attr("x1",xt(T)).attr("x2",xt(T));
      dotE.attr("cx",xt(T)).attr("cy",ye(Math.min(e.ece,ye.domain()[1])));
      dotN.attr("cx",xt(T)).attr("cy",yn(Math.min(e.nll,yn.domain()[1])));
      const bestN=curve.reduce((a,b)=>b.nll<a.nll?b:a), bestE=curve.reduce((a,b)=>b.ece<a.ece?b:a);
      ro.html(`accuracy <b>${(100*e.acc).toFixed(1)}%</b> (identical at every T) · mean confidence <b>${(100*e.mconf).toFixed(1)}%</b> · `+
        `ECE <b>${(100*e.ece).toFixed(2)}%</b> · NLL <b>${e.nll.toFixed(3)}</b><br>`+
        `on this grid: NLL is lowest at T ≈ <b>${bestN.t.toFixed(2)}</b>, ECE at T ≈ <b>${bestE.t.toFixed(2)}</b> · red = over-confident bins, green = under-confident`);
    }
    d3.selectAll("#cal-T,#cal-model,#cal-bins").on("input",draw);
    draw();
  })();
})();
