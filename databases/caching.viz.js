/* caching.viz.js — interactive figures for caching.html.
   Loaded after data.js / notes.js. */

function caRng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

/* ---- Viz 1: eviction-policy hit-ratio curves ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#ca-svg"); if(svg.empty()) return;
  const W=640,H=280,m={l:52,r:20,t:16,b:38};
  const N=1000, T=12000, SIZES=[1,2,3,5,7,10,15,20,25,30,40,50];
  const T_HIT=0.5, T_MISS=20;
  function trace(kind,alpha){
    const r=caRng(4242);
    const w=d3.range(1,N+1).map(i=>Math.pow(i,-alpha)), tot=d3.sum(w);
    const cdf=[]; let acc=0; w.forEach(x=>{acc+=x/tot; cdf.push(acc);});
    const draw=()=>{ const u=r(); let lo=0,hi=N-1; while(lo<hi){const md=(lo+hi)>>1; if(cdf[md]<u) lo=md+1; else hi=md;} return lo; };
    // two popularity mappings: rank -> item id
    const perm1=d3.range(N), perm2=d3.range(N);
    for(let i=N-1;i>0;i--){ const j=Math.floor(r()*(i+1)); const tmp=perm2[i]; perm2[i]=perm2[j]; perm2[j]=tmp; }
    const out=[]; let cold=N;
    while(out.length<T){
      const i=out.length;
      if(kind==="scan" && i>0 && i%1500===0){ for(let s=0;s<250 && out.length<T;s++) out.push(cold++); continue; }
      const rank=draw();
      out.push(kind==="shift" && i>=T/2 ? perm2[rank] : perm1[rank]);
    }
    return out;
  }
  function simFIFO(tr,c){ const S=new Set(), Qa=[]; let h=0, head=0;
    for(const x of tr){ if(S.has(x)){h++;continue;} if(S.size>=c){ S.delete(Qa[head++]); } S.add(x); Qa.push(x); } return h/tr.length; }
  function simLRU(tr,c){ const M=new Map(); let h=0;
    for(const x of tr){ if(M.has(x)){h++; M.delete(x); M.set(x,1); continue;} if(M.size>=c){ M.delete(M.keys().next().value); } M.set(x,1); } return h/tr.length; }
  function simLFU(tr,c){ const M=new Map(); let h=0, t=0;
    for(const x of tr){ t++; const e=M.get(x); if(e){h++; e.f++; e.l=t; continue;}
      if(M.size>=c){ let bk=null,bf=Infinity,bl=Infinity; for(const [k,v] of M){ if(v.f<bf||(v.f===bf&&v.l<bl)){bf=v.f;bl=v.l;bk=k;} } M.delete(bk); }
      M.set(x,{f:1,l:t}); } return h/tr.length; }
  function simOPT(tr,c){
    const n=tr.length, nxt=new Array(n), last=new Map();
    for(let i=n-1;i>=0;i--){ nxt[i]=last.has(tr[i])?last.get(tr[i]):Infinity; last.set(tr[i],i); }
    const inC=new Map(), heap=[]; let h=0;
    const push=(e)=>{ heap.push(e); let i=heap.length-1; while(i>0){ const p=(i-1)>>1; if(heap[p][0]>=heap[i][0]) break; [heap[p],heap[i]]=[heap[i],heap[p]]; i=p; } };
    const pop=()=>{ const top=heap[0], end=heap.pop(); if(heap.length){ heap[0]=end; let i=0; for(;;){ const l=2*i+1,r=l+1; let b=i;
      if(l<heap.length&&heap[l][0]>heap[b][0]) b=l; if(r<heap.length&&heap[r][0]>heap[b][0]) b=r; if(b===i) break; [heap[b],heap[i]]=[heap[i],heap[b]]; i=b; } } return top; };
    for(let i=0;i<n;i++){ const x=tr[i];
      if(inC.has(x)){ h++; inC.set(x,nxt[i]); push([nxt[i],x]); continue; }
      if(inC.size>=c){ for(;;){ const [nu,k]=pop(); if(inC.has(k) && inC.get(k)===nu){ inC.delete(k); break; } } }
      inC.set(x,nxt[i]); push([nxt[i],x]); }
    return h/n; }
  const POLS=[["FIFO",simFIFO,C.muted],["LRU",simLRU,C.node],["LFU",simLFU,C.good],["OPT",simOPT,C.q]];
  let TR=null, CURVES=null;
  function recompute(){
    const kind=d3.select("#ca-w").property("value"), a=+d3.select("#ca-a").property("value");
    TR=trace(kind,a);
    CURVES=POLS.map(([name,f])=>SIZES.map(s=>({s, h:f(TR,s*N/100)})));
  }
  function draw(){
    const a=+d3.select("#ca-a").property("value"), cs=+d3.select("#ca-c").property("value");
    d3.select("#ca-a-v").text(a.toFixed(2)); d3.select("#ca-c-v").text(cs+"% ("+cs*N/100+" items)");
    svg.selectAll("*").remove();
    const X=d3.scaleLinear().domain([0,50]).range([m.l,W-m.r]);
    const Y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(X).ticks(10).tickFormat(d=>d+"%"));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y).ticks(5).tickFormat(d3.format(".0%")));
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-5).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("cache size (% of catalogue)");
    svg.append("text").attr("transform",`translate(14,${(m.t+H-m.b)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("hit ratio");
    const line=d3.line().x(d=>X(d.s)).y(d=>Y(d.h)).curve(d3.curveMonotoneX);
    CURVES.forEach((cv,i)=>{ svg.append("path").attr("d",line(cv)).attr("fill","none").attr("stroke",POLS[i][2]).attr("stroke-width",POLS[i][0]==="OPT"?1.8:2.4).attr("stroke-dasharray",POLS[i][0]==="OPT"?"5 4":null); });
    svg.append("line").attr("x1",X(cs)).attr("x2",X(cs)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.accent).attr("stroke-dasharray","3 3");
    const c=cs*N/100;
    const hs=POLS.map(([name,f])=>({name, h:f(TR,c)}));
    hs.forEach((o,i)=>svg.append("circle").attr("cx",X(cs)).attr("cy",Y(o.h)).attr("r",4).attr("fill",POLS[i][2]));
    const teff=h=>h*T_HIT+(1-h)*(T_HIT+T_MISS);
    const best=hs.slice(0,3).reduce((a,b)=>b.h>a.h?b:a);
    d3.select("#ca-read").html(hs.map((o,i)=>`<span style="color:${POLS[i][2]}">${o.name}</span> <b>${(o.h*100).toFixed(1)}%</b>`).join(" · ")
      + `<br>best practical policy here: <b>${best.name}</b> → effective latency <b>${teff(best.h).toFixed(2)} ms</b> (t<sub>hit</sub> ${T_HIT} ms, t<sub>miss</sub> ${T_MISS} ms) · origin sees <b>${((1-best.h)*100).toFixed(1)}%</b> of requests`);
  }
  d3.select("#ca-w").on("change",()=>{recompute();draw();});
  d3.select("#ca-a").on("input",()=>d3.select("#ca-a-v").text((+d3.select("#ca-a").property("value")).toFixed(2))).on("change",()=>{recompute();draw();});
  d3.select("#ca-c").on("input",draw);
  recompute(); draw();
})();

/* ---- Viz 2: stampede on hot-key expiry ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#hd-svg"); if(svg.empty()) return;
  const W=640,H=210,m={l:44,r:16,t:16,b:32};
  const TTL=10, HORIZON=30, BUCKET=0.25, BETA=1;
  function sim(strategy,R,delta){
    const r=caRng(99);
    const calls=[]; let E=TTL, waited=0, inflight=[], total=0;
    let t=0;
    const finish=(now)=>{ while(inflight.length && inflight[0]<=now){ const c=inflight.shift(); if(c+TTL>E) E=c+TTL; } };
    const issue=(now)=>{ inflight.push(now+delta); calls.push(now); };
    while(true){
      t+=-Math.log(1-r())/R; if(t>HORIZON) break; total++;
      finish(t);
      if(t<E){
        if(strategy==="xf" && t - delta*BETA*Math.log(r()) >= E){ issue(t); waited++; }
        continue;
      }
      // expired
      if(strategy==="none"){ issue(t); waited++; }
      else { if(!inflight.length) issue(t); waited++; }
    }
    return {calls, waited, total};
  }
  function draw(){
    const s=d3.select("#hd-s").property("value"), R=+d3.select("#hd-r").property("value"), dms=+d3.select("#hd-d").property("value");
    d3.select("#hd-r-v").text(R); d3.select("#hd-d-v").text(dms);
    const res=sim(s,R,dms/1000);
    const nb=Math.round(HORIZON/BUCKET), bins=new Array(nb).fill(0);
    res.calls.forEach(c=>{ bins[Math.min(nb-1,Math.floor(c/BUCKET))]++; });
    const peak=d3.max(bins);
    svg.selectAll("*").remove();
    const X=d3.scaleLinear().domain([0,HORIZON]).range([m.l,W-m.r]);
    const Y=d3.scaleLinear().domain([0,Math.max(4,peak)]).nice().range([H-m.b,m.t]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(X).ticks(15).tickFormat(d=>d+"s"));
    svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(Y).ticks(4));
    [TTL,2*TTL].forEach(e=>svg.append("line").attr("x1",X(e)).attr("x2",X(e)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.muted).attr("stroke-dasharray","2 4").attr("stroke-opacity",.5));
    const bw=X(BUCKET)-X(0);
    const col=s==="none"?C.bad:s==="sf"?C.q:C.good;
    bins.forEach((b,i)=>{ if(!b) return; svg.append("rect").attr("x",X(i*BUCKET)+1).attr("y",Y(b)).attr("width",Math.max(1,bw-2)).attr("height",Y(0)-Y(b)).attr("fill",col);
      if(b===peak || (s!=="none" && b>0)) svg.append("text").attr("x",X(i*BUCKET)+bw/2).attr("y",Y(b)-3).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",9).text(b); });
    svg.append("text").attr("x",m.l+4).attr("y",m.t+4).attr("fill",C.muted).attr("font-size",10).text("origin calls per 250 ms (dotted lines = nominal expiries)");
    const label={none:"naive cache-aside",sf:"single-flight",xf:"probabilistic early refresh"}[s];
    d3.select("#hd-read").html(`<b>${label}</b> · ${res.total.toLocaleString()} requests in ${HORIZON} s · origin calls <b style="color:${col}">${res.calls.length}</b> · peak <b>${peak}</b> per 250 ms · requests that waited on a recompute <b>${res.waited.toLocaleString()}</b>`
      + (s==="none"?` — each expiry costs ≈ rate × δ = ${Math.round(R*dms/1000)} calls`:s==="sf"?` — one call per expiry, but everyone arriving during δ still waits`:` — refreshes happen before expiry; only the refreshing readers wait (pair with a lock to get exactly one)`));
  }
  d3.select("#hd-s").on("change",draw);
  d3.selectAll("#hd-r,#hd-d").on("input",draw);
  draw();
})();
