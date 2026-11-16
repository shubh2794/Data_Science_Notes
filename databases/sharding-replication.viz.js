/* sharding-replication.viz.js — visualizations for sharding-replication.html.
   1. consistent-hash ring with virtual nodes  2. replication lag vs read-after-write. */

const SR_C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",warn:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016"};

/* ---- Viz 1: consistent-hash ring ---- */
(function(){
  const C=SR_C, svg=d3.select("#ring-svg"), W=640, H=280, NKEYS=4000;
  const PAL=d3.schemeTableau10.concat(["#e6e9ef"]);
  function h32(str){ let h=0x811c9dc5; for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,0x01000193); }
    h^=h>>>16; h=Math.imul(h,0x85ebca6b); h^=h>>>13; h=Math.imul(h,0xc2b2ae35); h^=h>>>16; return h>>>0; }
  const KEYS=d3.range(NKEYS).map(i=>h32("key:"+i));
  const KEYPOS=KEYS.map(k=>k/4294967296);
  function tokens(N,V){ const t=[]; for(let n=0;n<N;n++) for(let j=0;j<V;j++) t.push({pos:h32("node-"+n+"#vn"+j)/4294967296,node:n}); return t.sort((a,b)=>a.pos-b.pos); }
  function owner(pos,T){ let lo=0,hi=T.length; while(lo<hi){ const m=(lo+hi)>>1; if(T[m].pos<pos) lo=m+1; else hi=m; } return T[lo%T.length].node; }
  function drawRing(T,cx,cy,r0,r1,hl){
    const arc=d3.arc().innerRadius(r0).outerRadius(r1);
    T.forEach((t,i)=>{
      const prev=i===0?T[T.length-1].pos-1:T[i-1].pos;
      svg.append("path").attr("transform",`translate(${cx},${cy})`)
        .attr("d",arc({startAngle:prev*2*Math.PI,endAngle:t.pos*2*Math.PI}))
        .attr("fill",PAL[t.node]).attr("opacity",hl===undefined||t.node===hl?0.92:0.28)
        .attr("stroke",hl!==undefined&&t.node===hl?C.ink:"#0f1117").attr("stroke-width",hl!==undefined&&t.node===hl?1.2:0.5);
    });
  }
  function draw(){
    const N=+d3.select("#ring-n").property("value"), V=+d3.select("#ring-v").property("value"), add=d3.select("#ring-add").property("checked");
    const T=tokens(N,V), T2=tokens(N+1,V);
    const own=KEYPOS.map(p=>owner(p,T)), own2=KEYPOS.map(p=>owner(p,T2));
    const moved=own.reduce((a,o,i)=>a+(o!==own2[i]?1:0),0)/NKEYS;
    const modMoved=KEYS.reduce((a,k)=>a+((k%N)!==(k%(N+1))?1:0),0)/NKEYS;
    const shown=add?own2:own, NN=add?N+1:N;
    const load=new Array(NN).fill(0); shown.forEach(o=>load[o]++);
    const mean=NKEYS/NN, sd=Math.sqrt(d3.mean(load,l=>(l-mean)*(l-mean))), cv=sd/mean, mx=d3.max(load)/mean;
    svg.selectAll("*").remove();
    const cx=140, cy=140;
    drawRing(T,cx,cy,92,114);
    if(add){ drawRing(T2,cx,cy,64,86,N);
      svg.append("text").attr("x",cx).attr("y",cy-4).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("outer: before");
      svg.append("text").attr("x",cx).attr("y",cy+10).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",9.5).text("inner: after +node "+(N+1));
    } else {
      svg.append("text").attr("x",cx).attr("y",cy-2).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(N*V+" tokens");
      svg.append("text").attr("x",cx).attr("y",cy+12).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("key → next token clockwise");
    }
    // load bars
    const bx=300, by=24, bw=300, bh=200, slot=bw/NN, maxL=Math.max(d3.max(load),mean*1.6);
    const Y=v=>by+bh-v/maxL*bh;
    svg.append("text").attr("x",bx).attr("y",14).attr("fill",C.muted).attr("font-size",10).text("keys per node"+(add?" after adding node "+(N+1):""));
    load.forEach((l,i)=>{
      svg.append("rect").attr("x",bx+i*slot+3).attr("y",Y(l)).attr("width",slot-6).attr("height",by+bh-Y(l)).attr("rx",2).attr("fill",PAL[i])
        .attr("stroke",add&&i===N?C.ink:"none");
      svg.append("text").attr("x",bx+i*slot+slot/2).attr("y",by+bh+12).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(i+1);
    });
    svg.append("line").attr("x1",bx).attr("x2",bx+bw).attr("y1",Y(mean)).attr("y2",Y(mean)).attr("stroke",C.ink).attr("stroke-dasharray","4 3").attr("stroke-opacity",.7);
    svg.append("text").attr("x",bx+bw+4).attr("y",Y(mean)+3).attr("fill",C.ink).attr("font-size",9).text("ideal");
    svg.append("line").attr("x1",bx).attr("x2",bx+bw).attr("y1",by+bh).attr("y2",by+bh).attr("stroke","#3a4150");
    svg.append("text").attr("x",bx+bw/2).attr("y",by+bh+26).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("node");
    d3.select("#ring-read").html(`N=${NN}, V=${V} · busiest node <b>${mx.toFixed(2)}×</b> the mean · load CV <b>${cv.toFixed(2)}</b> (≈ 1/√V = ${(1/Math.sqrt(V)).toFixed(2)}) · `+
      `adding node ${N+1} moves <b style="color:${C.good}">${(moved*100).toFixed(1)}%</b> of keys on the ring (ideal ${(100/(N+1)).toFixed(1)}%) vs <b style="color:${C.bad}">${(modMoved*100).toFixed(1)}%</b> with h mod N`);
  }
  d3.selectAll("#ring-n,#ring-v,#ring-add").on("input change",draw);
  draw();
})();

/* ---- Viz 2: replication lag ---- */
(function(){
  const C=SR_C, svg=d3.select("#lag-svg"), W=640, H=240;
  const T0=1500, TMAX=5000, STEP=100, FACT=[0.5,1,2.5], FCOL=[C.accent,C.warn,"#c084fc"];
  const px=50, py=14, pw=470, ph=180;
  function draw(){
    const L=+d3.select("#lag-l").property("value"), d=+d3.select("#lag-d").property("value"), route=d3.select("#lag-route").property("value");
    const lags=FACT.map(f=>f*L), vStar=Math.floor(T0/STEP), tr=T0+d;
    const vmax=Math.floor(TMAX/STEP);
    const X=t=>px+t/TMAX*pw, Y=v=>py+ph-v/vmax*ph;
    const ver=(t,lag)=>Math.max(0,Math.floor((t-lag)/STEP));
    svg.selectAll("*").remove();
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",py+ph).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("x2",px).attr("y1",py).attr("y2",py+ph).attr("stroke","#3a4150");
    [0,1000,2000,3000,4000,5000].forEach(t=>svg.append("text").attr("x",X(t)).attr("y",py+ph+13).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(t/1000+"s"));
    svg.append("text").attr("x",14).attr("y",py+ph/2).attr("fill",C.muted).attr("font-size",10).attr("text-anchor","middle").attr("transform",`rotate(-90,14,${py+ph/2})`).text("applied log position");
    const ts=d3.range(0,TMAX+1,20);
    const line=lag=>d3.line().curve(d3.curveStepAfter).x(t=>X(t)).y(t=>Y(ver(t,lag)))(ts);
    svg.append("path").attr("d",line(0)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.2);
    lags.forEach((lag,i)=>{
      svg.append("path").attr("d",line(lag)).attr("fill","none").attr("stroke",FCOL[i]).attr("stroke-width",1.6).attr("stroke-opacity",.9);
    });
    // end labels, spaced so they never overlap
    const lab=[{t:"leader",c:C.good,y:Y(ver(TMAX,0))+4}].concat(lags.map((lag,i)=>({t:`f${i+1} (${Math.round(lag)}ms)`,c:FCOL[i],y:Y(ver(TMAX,lag))+4})));
    for(let i=1;i<lab.length;i++) if(lab[i].y-lab[i-1].y<12) lab[i].y=lab[i-1].y+12;
    lab.forEach(l=>svg.append("text").attr("x",px+pw+6).attr("y",l.y).attr("fill",l.c).attr("font-size",9.5).text(l.t));
    // client write and read
    svg.append("line").attr("x1",X(T0)).attr("x2",X(T0)).attr("y1",py).attr("y2",py+ph).attr("stroke",C.good).attr("stroke-dasharray","3 3");
    svg.append("text").attr("x",X(T0)).attr("y",py+8).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",9.5).text("your write");
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",Y(vStar)).attr("y2",Y(vStar)).attr("stroke",C.muted).attr("stroke-dasharray","2 4").attr("stroke-opacity",.6);
    if(tr<=TMAX){
      svg.append("line").attr("x1",X(tr)).attr("x2",X(tr)).attr("y1",py).attr("y2",py+ph).attr("stroke",C.ink).attr("stroke-dasharray","3 3");
      svg.append("text").attr("x",X(tr)).attr("y",py+20).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",9.5).text("your read");
      lags.forEach((lag,i)=>{ const fresh=ver(tr,lag)>=vStar;
        svg.append("circle").attr("cx",X(tr)).attr("cy",Y(ver(tr,lag))).attr("r",5).attr("fill",fresh?C.good:C.bad).attr("stroke",FCOL[i]).attr("stroke-width",1.5); });
    }
    const stale=lags.map(l=>l>d), nStale=stale.filter(Boolean).length;
    let msg;
    if(route==="any") msg=`random follower → P(stale read) = <b style="color:${nStale?C.bad:C.good}">${nStale}/3</b>`+
      (nStale>0&&nStale<3?` · <span style="color:${C.warn}">monotonic-read violation possible: a fresh follower then a stale one makes your write vanish</span>`:"");
    else if(route==="ryw") msg=`read carries your write's log position → <b style="color:${C.good}">never stale</b> · ${nStale}/3 followers must wait up to <b>${Math.round(Math.max(0,d3.max(lags)-d))} ms</b> or redirect to the leader`;
    else msg=`pinned to f2 (hash of user id) → `+(stale[1]?`<b style="color:${C.bad}">stale</b> until ${Math.round(lags[1])} ms after the write`:`<b style="color:${C.good}">fresh</b>`)+` · but time never goes backwards for you (monotonic reads)`;
    d3.select("#lag-read").html(`read ${d} ms after the write · ${msg}`);
  }
  d3.selectAll("#lag-l,#lag-d,#lag-route").on("input change",draw);
  draw();
})();
