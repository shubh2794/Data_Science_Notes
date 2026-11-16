/* load-balancing.viz.js — interactive figures for load-balancing.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: policy simulation (discrete-time, seeded) ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#222833"};
  const svg=d3.select("#lb-svg"); if(svg.empty()) return;
  const W=640,H=270;
  const SPEED=[1,1,1,1,1,0.5,0.5,2];          // relative service rates; total capacity = 8
  const CAP=d3.sum(SPEED), DT=0.05;
  const POL=[["rr","round robin"],["wrr","weighted RR"],["rand","random"],["lc","least conn."],["p2c","P2C"]];
  function rng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed);
    t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  function poisson(mean,r){ const L=Math.exp(-mean); let k=0,p=1; do{ k++; p*=r(); }while(p>L); return k-1; }
  function makeState(policy,seed){
    return {policy, r:rng(seed), t:0, Q:SPEED.map(()=>[]), lat:[], rr:0, cw:SPEED.map(()=>0)};
  }
  function pick(S){
    const n=SPEED.length, r=S.r;
    switch(S.policy){
      case "rr": { const i=S.rr; S.rr=(S.rr+1)%n; return i; }
      case "wrr": { // smooth weighted round robin
        let best=0; for(let i=0;i<n;i++){ S.cw[i]+=SPEED[i]; if(S.cw[i]>S.cw[best]) best=i; }
        S.cw[best]-=CAP; return best; }
      case "rand": return Math.floor(r()*n);
      case "lc": { let m=Infinity, c=[]; for(let i=0;i<n;i++){ const l=S.Q[i].length; if(l<m){m=l;c=[i];} else if(l===m) c.push(i); }
        return c[Math.floor(r()*c.length)]; }
      case "p2c": { const a=Math.floor(r()*n); let b=Math.floor(r()*(n-1)); if(b>=a) b++;
        return S.Q[a].length<=S.Q[b].length?a:b; }
    }
  }
  function tick(S,lam,keep){
    const k=poisson(lam*DT,S.r);
    for(let j=0;j<k;j++){ S.Q[pick(S)].push(S.t); }
    for(let i=0;i<SPEED.length;i++){
      if(S.Q[i].length && S.r()<SPEED[i]*DT){ const a=S.Q[i].shift(); S.lat.push(S.t-a); if(keep && S.lat.length>keep) S.lat.shift(); }
    }
    S.t+=DT;
  }
  function pct(arr,p){ if(!arr.length) return 0; const s=arr.slice().sort((a,b)=>a-b); return s[Math.min(s.length-1,Math.floor(p*s.length))]; }
  function offline(rho){
    const lam=rho*CAP;
    return POL.map(([id])=>{
      const S=makeState(id,12345);
      for(let i=0;i<30000;i++) tick(S,lam,0);
      const backlog=d3.sum(S.Q,q=>q.length);
      // unfinished requests count toward the tail: include their current age
      const ages=[]; S.Q.forEach(q=>q.forEach(a=>ages.push(S.t-a)));
      const all=S.lat.concat(ages);
      return {id, p99:pct(all,0.99), backlog, unstable: backlog>400};
    });
  }
  let S, OFF, timer=null, running=true;
  function rho(){ return +d3.select("#lb-rho").property("value"); }
  function pol(){ return d3.select("#lb-pol").property("value"); }
  function reset(){ S=makeState(pol(),777); OFF=offline(rho()); draw(); }
  function draw(){
    svg.selectAll("*").remove();
    d3.select("#lb-rho-v").text(rho().toFixed(2));
    // left: server queues
    const px=40, pw=380, base=220, top=24, cap=40;
    svg.append("text").attr("x",px).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("live queue length per server (cap "+cap+")");
    const bw=pw/SPEED.length;
    SPEED.forEach((sp,i)=>{
      const len=S.Q[i].length, h=Math.min(len,cap)/cap*(base-top);
      const col= len>=cap?C.bad: len>12?C.q:C.good;
      svg.append("rect").attr("x",px+i*bw+6).attr("y",top).attr("width",bw-12).attr("height",base-top).attr("rx",4).attr("fill",C.panel).attr("stroke",C.line);
      svg.append("rect").attr("x",px+i*bw+6).attr("y",base-h).attr("width",bw-12).attr("height",h).attr("rx",3).attr("fill",col).attr("fill-opacity",.85);
      svg.append("text").attr("x",px+i*bw+bw/2).attr("y",base-h-4).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(len>=cap?len+"!":len);
      svg.append("text").attr("x",px+i*bw+bw/2).attr("y",base+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("S"+(i+1));
      svg.append("text").attr("x",px+i*bw+bw/2).attr("y",base+27).attr("text-anchor","middle").attr("fill",sp<1?C.bad:sp>1?C.accent:C.muted).attr("font-size",10).text(sp+"×");
    });
    svg.append("text").attr("x",px).attr("y",H-6).attr("fill",C.muted).attr("font-size",10).text("service speed (1× = baseline) — total capacity 8 units");
    // right: offline p99 comparison
    const rx=455, rw=170, rtop=30;
    svg.append("text").attr("x",rx).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("p99 latency, same trace");
    const finite=OFF.filter(o=>!o.unstable).map(o=>o.p99);
    const vmax=Math.max(5,(d3.max(finite)||5)*1.15);
    const X=d3.scaleLinear().domain([0,vmax]).range([0,rw-60]);
    OFF.forEach((o,i)=>{
      const y=rtop+i*38, lab=POL[i][1], cur=o.id===pol();
      svg.append("text").attr("x",rx).attr("y",y+10).attr("fill",cur?C.ink:C.muted).attr("font-size",10).attr("font-weight",cur?700:400).text(lab);
      const w=o.unstable?rw-60:X(o.p99);
      svg.append("rect").attr("x",rx).attr("y",y+15).attr("width",Math.max(2,w)).attr("height",12).attr("rx",3).attr("fill",o.unstable?C.bad:(cur?C.accent:C.node)).attr("fill-opacity",cur?1:.7);
      svg.append("text").attr("x",rx+w+5).attr("y",y+25).attr("fill",o.unstable?C.bad:C.ink).attr("font-size",10).text(o.unstable?"unstable":o.p99.toFixed(1));
    });
    svg.append("text").attr("x",rx).attr("y",H-6).attr("fill",C.muted).attr("font-size",10).text("units = mean service time at 1×");
    const live=S.lat.slice(-2000), backlog=d3.sum(S.Q,q=>q.length);
    d3.select("#lb-read").html(`<b>${POL.find(p=>p[0]===pol())[1]}</b> at ρ = <b>${rho().toFixed(2)}</b> · live p50 <b>${pct(live,0.5).toFixed(1)}</b> · live p99 <b>${pct(live,0.99).toFixed(1)}</b> · requests queued <b>${backlog}</b>`
      + (backlog>300?` <span style="color:${C.bad}">— slow servers are overloaded; their queues grow without bound</span>`:""));
  }
  function loop(){ const lam=rho()*CAP; for(let i=0;i<12;i++) tick(S,lam,4000); draw(); }
  function start(){ if(timer) timer.stop(); timer=d3.interval(loop,60); running=true; d3.select("#lb-play").text("⏸ pause"); }
  function stop(){ if(timer) timer.stop(); timer=null; running=false; d3.select("#lb-play").text("▶ play"); }
  d3.select("#lb-play").on("click",()=>running?stop():start());
  d3.select("#lb-reset").on("click",reset);
  d3.select("#lb-pol").on("change",()=>{ S=makeState(pol(),777); draw(); });
  d3.select("#lb-rho").on("input",()=>{ d3.select("#lb-rho-v").text(rho().toFixed(2)); })
    .on("change",reset);
  reset(); start();
})();

/* ---- Viz 2: consistent-hash ring ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#rg-svg"); if(svg.empty()) return;
  const W=640,H=320, cx=165, cy=162, R=128, K=240;
  const PAL=["#5b9cff","#4ade80","#c084fc","#22d3ee","#f472b6","#facc15","#fb923c","#a3e635","#38bdf8","#e879f9"];
  function h32(str){ let h=0x811c9dc5; for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,0x01000193); }
    h^=h>>>16; h=Math.imul(h,0x85ebca6b); h^=h>>>13; h=Math.imul(h,0xc2b2ae35); h^=h>>>16; return (h>>>0)/4294967296; }
  const KEYS=d3.range(K).map(j=>({j, pos:h32("key-"+j), id:(h32("id-"+j)*1e9)>>>0}));
  function ring(n,v){ const pts=[]; for(let i=0;i<n;i++) for(let r=0;r<v;r++) pts.push({node:i,pos:h32("node-"+i+"#"+r)});
    return pts.sort((a,b)=>a.pos-b.pos); }
  function owners(n,v){ const pts=ring(n,v);
    return KEYS.map(k=>{ let lo=0,hi=pts.length; while(lo<hi){ const m=(lo+hi)>>1; if(pts[m].pos<k.pos) lo=m+1; else hi=m; } return pts[lo%pts.length].node; }); }
  let prev=null, prevN=null, prevV=null;
  function draw(){
    const n=+d3.select("#rg-n").property("value"), v=+d3.select("#rg-v").property("value");
    d3.select("#rg-n-v").text(n); d3.select("#rg-v-v").text(v);
    const own=owners(n,v), pts=ring(n,v);
    const moved = (prev && prevV===v && prevN!==n) ? own.map((o,i)=>o!==prev[i]) : own.map(()=>false);
    const nMoved=moved.filter(Boolean).length;
    const modMoved = (prev && prevV===v && prevN!==n) ? KEYS.filter(k=>k.id%n!==k.id%prevN).length : 0;
    svg.selectAll("*").remove();
    const ang=p=>p*2*Math.PI-Math.PI/2;
    svg.append("circle").attr("cx",cx).attr("cy",cy).attr("r",R).attr("fill","none").attr("stroke",C.line).attr("stroke-width",2);
    // node points (virtual nodes) as radial ticks
    pts.forEach(p=>{ const a=ang(p.pos);
      svg.append("line").attr("x1",cx+(R-12)*Math.cos(a)).attr("y1",cy+(R-12)*Math.sin(a)).attr("x2",cx+(R+12)*Math.cos(a)).attr("y2",cy+(R+12)*Math.sin(a))
        .attr("stroke",PAL[p.node]).attr("stroke-width",2.4); });
    // keys
    KEYS.forEach((k,i)=>{ const a=ang(k.pos), rr=R+22+(i%3)*5;
      svg.append("circle").attr("cx",cx+rr*Math.cos(a)).attr("cy",cy+rr*Math.sin(a)).attr("r",moved[i]?3.4:2.2)
        .attr("fill",moved[i]?C.q:PAL[own[i]]).attr("fill-opacity",moved[i]?1:.75); });
    svg.append("text").attr("x",cx).attr("y",cy-4).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12).text(n+" nodes × "+v+" vnodes");
    svg.append("text").attr("x",cx).attr("y",cy+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(K+" keys · ticks = vnodes");
    // load bars
    const counts=d3.range(n).map(i=>own.filter(o=>o===i).length), ideal=K/n;
    const bx=360, bw=250, top=26, rowH=Math.min(26,250/n);
    const X=d3.scaleLinear().domain([0,Math.max(d3.max(counts),ideal)*1.1]).range([0,bw-60]);
    svg.append("text").attr("x",bx).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("keys per node (dashed = ideal K/N)");
    counts.forEach((c,i)=>{ const y=top+i*rowH;
      svg.append("text").attr("x",bx).attr("y",y+rowH*0.65).attr("fill",PAL[i]).attr("font-size",10).text("node "+(i+1));
      svg.append("rect").attr("x",bx+50).attr("y",y+3).attr("width",X(c)).attr("height",rowH-7).attr("rx",3).attr("fill",PAL[i]).attr("fill-opacity",.8);
      svg.append("text").attr("x",bx+54+X(c)).attr("y",y+rowH*0.65).attr("fill",C.ink).attr("font-size",10).text(c); });
    svg.append("line").attr("x1",bx+50+X(ideal)).attr("x2",bx+50+X(ideal)).attr("y1",top).attr("y2",top+n*rowH).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("stroke-opacity",.6);
    const imb=d3.max(counts)/ideal;
    let msg=`max/avg load = <b>${imb.toFixed(2)}</b> with ${v} vnode${v>1?"s":""} per node`;
    if(prev && prevV===v && prevN!==n){
      const ideal2=Math.round(K*Math.abs(n-prevN)/Math.max(n,prevN));
      msg+=` · ${prevN} → ${n} nodes: consistent hashing moved <b style="color:${C.q}">${nMoved}</b> keys (ideal ≈ ${ideal2}); <code>hash mod N</code> would move <b style="color:${C.bad}">${modMoved}</b> of ${K}`;
    } else msg+=` · now drag the node slider to see which keys move`;
    d3.select("#rg-read").html(msg);
  }
  d3.select("#rg-n").on("input",function(){
    // remember the configuration before this change so the move count is relative to it
    const n=+this.value, v=+d3.select("#rg-v").property("value");
    draw(); prev=owners(n,v); prevN=n; prevV=v;
  });
  d3.select("#rg-v").on("input",function(){ prev=null; draw(); const n=+d3.select("#rg-n").property("value"); prev=owners(n,+this.value); prevN=n; prevV=+this.value; });
  // initial
  (function init(){ const n=+d3.select("#rg-n").property("value"), v=+d3.select("#rg-v").property("value"); draw(); prev=owners(n,v); prevN=n; prevV=v; })();
})();
