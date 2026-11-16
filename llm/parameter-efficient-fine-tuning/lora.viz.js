/* lora.viz.js — interactive figures for llm/parameter-efficient-fine-tuning/lora.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-lr  a rank-r adapter learning a toy target update by gradient descent
     #viz-mg  merged vs unmerged adapters for a mixed-task batch
     #viz-ra  reported rank ablation and fixed-budget matrix choice (175B model) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>n.toLocaleString("en-US");

  /* ═══════════════ 1. low-rank update learner ═══════════════ */
  (function(){
    const svg=d3.select("#lr-svg"); if(svg.empty()) return;
    const N=32, ETA=0.2, MAXSTEP=2400, PER_FRAME=12, RHO=0.72, WIDE=12288;
    let seedT=7, seedA=11;
    function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
    function orth(r){const Q=[];for(let i=0;i<N;i++){const v=Array.from({length:N},()=>gauss(r));
      for(const q of Q){const d=dot(q,v);for(let t=0;t<N;t++)v[t]-=d*q[t];}
      const nv=Math.sqrt(dot(v,v));Q.push(v.map(x=>x/nv));}return Q;}

    let U,V,sig,T,Tn,B,A,r,step,hist,timer=null;

    function buildTarget(){
      const k=+d3.select("#lr-k").property("value"), noise=+d3.select("#lr-n").property("value");
      const R=rng(seedT); U=orth(R); V=orth(R);
      // sorted descending so that index i really is the i-th singular value (a large noise
      // tail can exceed the smallest signal values; Eckart–Young needs the true order)
      sig=d3.range(N).map(i=>i<k?Math.pow(RHO,i):noise*Math.pow(0.97,i-k)).sort((a,b)=>b-a);
      T=d3.range(N).map(a=>d3.range(N).map(b=>{let s=0;for(let i=0;i<N;i++)s+=sig[i]*U[i][a]*V[i][b];return s;}));
      Tn=Math.sqrt(d3.sum(sig,s=>s*s));
    }
    function initAdapter(){
      r=+d3.select("#lr-r").property("value");
      const R=rng(seedA+97*r);
      B=d3.range(N).map(()=>new Float64Array(r));                         // B = 0
      A=d3.range(r).map(()=>Float64Array.from({length:N},()=>gauss(R)/Math.sqrt(N))); // A ~ Gaussian
      step=0; hist=[{s:0,e:1}];
    }
    function product(){const P=d3.range(N).map(()=>new Float64Array(N));
      for(let a=0;a<N;a++)for(let j=0;j<r;j++){const bj=B[a][j];if(bj===0)continue;const Aj=A[j],Pa=P[a];for(let b=0;b<N;b++)Pa[b]+=bj*Aj[b];}
      return P;}
    function relErr(P){let s=0;for(let a=0;a<N;a++)for(let b=0;b<N;b++){const d=P[a][b]-T[a][b];s+=d*d;}return Math.sqrt(s)/Tn;}
    function floorAt(rr){let s=0;for(let i=rr;i<N;i++)s+=sig[i]*sig[i];return Math.sqrt(s)/Tn;}
    function gdStep(){
      const P=product();
      const G=P.map((row,a)=>row.map((p,b)=>p-T[a][b]));                 // G = BA − T
      const gB=d3.range(N).map(a=>{const g=new Float64Array(r);for(let j=0;j<r;j++)g[j]=dot(G[a],A[j]);return g;}); // G Aᵀ
      const gA=d3.range(r).map(j=>{const g=new Float64Array(N);for(let a=0;a<N;a++){const bj=B[a][j];if(bj===0)continue;const Ga=G[a];for(let b=0;b<N;b++)g[b]+=bj*Ga[b];}return g;}); // Bᵀ G
      for(let a=0;a<N;a++)for(let j=0;j<r;j++)B[a][j]-=ETA*gB[a][j];
      for(let j=0;j<r;j++)for(let b=0;b<N;b++)A[j][b]-=ETA*gA[j][b];
      step++;
    }

    // ── layout ──
    const W=760,H=440, hm=104, hmY=36, hmX=[14,132,250];
    const color=v=>{const t=Math.max(-1,Math.min(1,v));return t>=0?d3.interpolateRgb("#1b1f27",C.accent)(t):d3.interpolateRgb("#1b1f27",C.a2)(-t);};
    const cell=hm/N;
    const maps=["target ΔW*","adapter BA","residual"].map((lab,m)=>{
      const g=svg.append("g").attr("transform",`translate(${hmX[m]},${hmY})`);
      g.append("text").attr("x",hm/2).attr("y",-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text(lab);
      const rects=[];for(let a=0;a<N;a++)for(let b=0;b<N;b++)rects.push(g.append("rect").attr("x",b*cell).attr("y",a*cell).attr("width",cell+0.2).attr("height",cell+0.2));
      g.append("rect").attr("width",hm).attr("height",hm).attr("fill","none").attr("stroke",C.line);
      return rects;});
    svg.append("text").attr("x",14).attr("y",hmY+hm+18).attr("fill",C.muted).attr("font-size",10.5).text("teal = positive, amber = negative; shared scale");

    // loss curve
    const lx0=400,lx1=740,ly0=30,ly1=180;
    const xs=d3.scaleLinear().domain([0,MAXSTEP]).range([lx0,lx1]);
    const ys=d3.scaleLog().domain([1e-3,1.2]).range([ly1,ly0]).clamp(true);
    svg.append("g").attr("transform",`translate(0,${ly1})`).call(d3.axisBottom(xs).ticks(5)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${lx0},0)`).call(d3.axisLeft(ys).ticks(4,"~g")).attr("color",C.muted);
    svg.append("text").attr("x",lx0).attr("y",ly0-10).attr("fill",C.muted).attr("font-size",11).text("relative error ‖BA − ΔW*‖ / ‖ΔW*‖ vs step (log)");
    const floorLine=svg.append("line").attr("x1",lx0).attr("x2",lx1).attr("stroke",C.bad).attr("stroke-dasharray","5 4");
    const floorTxt=svg.append("text").attr("x",lx1).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10.5);
    const path=svg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);

    // spectrum bars
    const bx0=44,bx1=740,by0=250,by1=400;
    const bxs=d3.scaleBand().domain(d3.range(N)).range([bx0,bx1]).padding(0.18);
    const bys=d3.scaleLinear().domain([0,1.05]).range([by1,by0]);
    svg.append("g").attr("transform",`translate(${bx0},0)`).call(d3.axisLeft(bys).ticks(4)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(0,${by1})`).call(d3.axisBottom(bxs).tickValues([0,3,7,11,15,19,23,27,31]).tickFormat(i=>"σ"+(i+1))).attr("color",C.muted);
    svg.append("text").attr("x",bx0).attr("y",by0-14).attr("fill",C.muted).attr("font-size",11).text("per singular direction of ΔW*: target σᵢ (outline) and learned uᵢᵀ(BA)vᵢ (filled)");
    const outl=svg.append("g"), fill=svg.append("g");
    const rRule=svg.append("line").attr("y1",by0-4).attr("y2",by1).attr("stroke",C.ink).attr("stroke-dasharray","3 3");
    const rTxt=svg.append("text").attr("y",by0+6).attr("fill",C.ink).attr("font-size",10.5);

    function render(){
      const P=product(), e=relErr(P), fl=floorAt(r);
      let m=0;for(let a=0;a<N;a++)for(let b=0;b<N;b++)m=Math.max(m,Math.abs(T[a][b]));
      let i=0;for(let a=0;a<N;a++)for(let b=0;b<N;b++,i++){
        maps[0][i].attr("fill",color(T[a][b]/m));maps[1][i].attr("fill",color(P[a][b]/m));maps[2][i].attr("fill",color((T[a][b]-P[a][b])/m));}
      floorLine.attr("y1",ys(Math.max(fl,1e-3))).attr("y2",ys(Math.max(fl,1e-3)));
      floorTxt.attr("y",ys(Math.max(fl,1e-3))-5).text("Eckart–Young floor at r = "+r+": "+fl.toFixed(4));
      path.attr("d",d3.line().x(d=>xs(d.s)).y(d=>ys(Math.max(d.e,1e-3)))(hist));
      const learned=d3.range(N).map(k=>{let s=0;for(let a=0;a<N;a++)for(let b=0;b<N;b++)s+=U[k][a]*P[a][b]*V[k][b];return s;});
      outl.selectAll("rect").data(sig).join("rect").attr("x",(d,k)=>bxs(k)).attr("width",bxs.bandwidth())
        .attr("y",d=>bys(d)).attr("height",d=>by1-bys(d)).attr("fill","none").attr("stroke",(d,k)=>k<r?C.accent:C.muted);
      fill.selectAll("rect").data(learned).join("rect").attr("x",(d,k)=>bxs(k)+1.5).attr("width",Math.max(1,bxs.bandwidth()-3))
        .attr("y",d=>bys(Math.max(0,Math.min(1.05,d)))).attr("height",d=>by1-bys(Math.max(0,Math.min(1.05,d)))).attr("fill",(d,k)=>k<r?C.accent:C.a2).attr("opacity",0.85);
      const rx=bxs(r-1)+bxs.bandwidth()+bxs.step()*bxs.paddingInner()/2;
      rRule.attr("x1",rx).attr("x2",rx); rTxt.attr("x",rx+4).text("r = "+r);
      const k=+d3.select("#lr-k").property("value");
      const full=N*N, lo=2*N*r, wf=WIDE*WIDE, wl=2*WIDE*r;
      const gap=e-fl;
      d3.select("#lr-readout").html(
        `step <b>${step}</b> · relative error <b>${e.toFixed(4)}</b> · floor at r = ${r}: <b>${fl.toFixed(4)}</b> · gap <b>${gap.toExponential(1)}</b>`+
        (r<k?` · r is below the target's intrinsic rank (${k}), so directions ${r+1}–${k} cannot be learned`:` · r ≥ intrinsic rank (${k}); the floor is only the noise tail`)+
        `<br>trainable: toy <b>${fmt(lo)}</b> of ${fmt(full)} (${(100*lo/full).toFixed(1)}%) · at d = k = ${fmt(WIDE)}: <b>${fmt(wl)}</b> of ${fmt(wf)} (${(100*wl/wf).toFixed(3)}%)`+
        (step===0?" · B = 0, so BA = 0 and the adapter starts as a no-op":""));
    }
    function stop(){if(timer){timer.stop();timer=null;}d3.select("#lr-play").text(step>=MAXSTEP?"done":"train");}
    function play(){
      if(timer){stop();return;}
      if(step>=MAXSTEP)initAdapter();
      d3.select("#lr-play").text("pause");
      timer=d3.timer(()=>{for(let i=0;i<PER_FRAME&&step<MAXSTEP;i++){gdStep();}
        hist.push({s:step,e:relErr(product())});render();if(step>=MAXSTEP)stop();});
    }
    function syncLabels(){d3.select("#lr-kv").text(d3.select("#lr-k").property("value"));
      d3.select("#lr-nv").text((+d3.select("#lr-n").property("value")).toFixed(2));
      d3.select("#lr-rv").text(d3.select("#lr-r").property("value"));}
    d3.select("#lr-k").on("input",()=>{stop();syncLabels();buildTarget();initAdapter();render();});
    d3.select("#lr-n").on("input",()=>{stop();syncLabels();buildTarget();initAdapter();render();});
    d3.select("#lr-r").on("input",()=>{stop();syncLabels();initAdapter();render();});
    d3.select("#lr-play").on("click",play);
    d3.select("#lr-reset").on("click",()=>{stop();seedA+=1;initAdapter();render();});
    d3.select("#lr-new").on("click",()=>{stop();seedT+=1;buildTarget();initAdapter();render();});
    syncLabels();buildTarget();initAdapter();render();
  })();

  /* ═══════════════ 2. merged vs unmerged, mixed-task batch ═══════════════ */
  (function(){
    const svg=d3.select("#mg-svg"); if(svg.empty()) return;
    const TC=[C.accent,C.a2,C.violet,C.blue], NREQ=8;
    function draw(){
      svg.selectAll("*").remove();
      const mode=d3.select("#mg-m").property("value"), t=+d3.select("#mg-t").property("value");
      const r=+d3.select("#mg-r").property("value"), d=+d3.select("#mg-d").property("value");
      d3.select("#mg-tv").text(t);
      const task=d3.range(NREQ).map(i=>i%t);
      const chipX=30, chipW=92, chipH=26, gap=8, y0=24;
      svg.append("text").attr("x",chipX).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("batch of requests");
      const midY=y0+NREQ*(chipH+gap)/2-gap/2;
      const baseX=330, baseW=150, baseH=150, baseY=midY-baseH/2;
      task.forEach((k,i)=>{
        const y=y0+i*(chipH+gap), active=mode==="unmerged"||k===0;
        svg.append("rect").attr("x",chipX).attr("y",y).attr("width",chipW).attr("height",chipH).attr("rx",5)
          .attr("fill",active?TC[k]:"#2a2f3a").attr("opacity",active?0.9:0.6);
        svg.append("text").attr("x",chipX+chipW/2).attr("y",y+17).attr("text-anchor","middle").attr("font-size",11)
          .attr("fill",active?"#0b0d12":C.muted).text("req "+(i+1)+" · task "+(k+1));
        const tx=mode==="unmerged"?baseX:baseX, ty=mode==="unmerged"?baseY+20+i*(baseH-40)/(NREQ-1):midY;
        svg.append("path").attr("d",`M${chipX+chipW},${y+chipH/2} C${chipX+chipW+90},${y+chipH/2} ${tx-90},${ty} ${tx},${ty}`)
          .attr("fill","none").attr("stroke",active?TC[k]:C.line).attr("stroke-width",active?1.6:1).attr("stroke-dasharray",active?null:"4 4").attr("opacity",active?0.8:0.7);
      });
      if(mode==="merged"){
        svg.append("rect").attr("x",baseX).attr("y",baseY).attr("width",baseW).attr("height",baseH).attr("rx",6).attr("fill","#1d2a2a").attr("stroke",TC[0]);
        svg.append("text").attr("x",baseX+baseW/2).attr("y",baseY+baseH/2-6).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",13).text("W = W₀ + B₁A₁");
        svg.append("text").attr("x",baseX+baseW/2).attr("y",baseY+baseH/2+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("one dense "+d+"×"+d+" matrix");
        const qx=540;
        svg.append("text").attr("x",qx).attr("y",30).attr("fill",C.muted).attr("font-size",11).text("forward passes needed");
        d3.range(t).forEach(k=>{
          const y=46+k*56;
          svg.append("rect").attr("x",qx).attr("y",y).attr("width",190).attr("height",42).attr("rx",5).attr("fill",C.panel).attr("stroke",TC[k]);
          svg.append("text").attr("x",qx+10).attr("y",y+17).attr("fill",TC[k]).attr("font-size",11.5).text("pass "+(k+1)+": task "+(k+1)+" requests");
          svg.append("text").attr("x",qx+10).attr("y",y+33).attr("fill",C.muted).attr("font-size",10.5).text(k===0?"weights already merged":"first: W ← W − B"+k+"A"+k+" + B"+(k+1)+"A"+(k+1));
        });
      }else{
        svg.append("rect").attr("x",baseX).attr("y",baseY).attr("width",baseW).attr("height",baseH).attr("rx",6).attr("fill","#22262f").attr("stroke",C.muted);
        svg.append("text").attr("x",baseX+baseW/2).attr("y",baseY+baseH/2-6).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",13).text("W₀ (shared, frozen)");
        svg.append("text").attr("x",baseX+baseW/2).attr("y",baseY+baseH/2+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("every request, one pass");
        const ax=baseX+baseW+40;
        svg.append("text").attr("x",ax).attr("y",30).attr("fill",C.muted).attr("font-size",11).text("+ per-request adapter Bₜ·Aₜ (rank "+r+")");
        d3.range(t).forEach(k=>{
          const y=46+k*56, w=Math.max(6,Math.min(28,r/64*28));
          svg.append("rect").attr("x",ax).attr("y",y).attr("width",w).attr("height",42).attr("rx",2).attr("fill",TC[k]);
          svg.append("rect").attr("x",ax+w+6).attr("y",y+21-w/2).attr("width",60).attr("height",w).attr("rx",2).attr("fill",TC[k]);
          svg.append("text").attr("x",ax+w+74).attr("y",y+25).attr("fill",TC[k]).attr("font-size",11.5).text("B"+(k+1)+" A"+(k+1)+" → task "+(k+1));
        });
      }
      const extra=r*(d+d), base=d*d, pct=100*extra/base, remerge=2*d*d*r;
      if(mode==="merged"){
        d3.select("#mg-readout").html(`merged: <b>${t}</b> forward pass${t>1?"es":""} for this batch` +
          (t>1?` and <b>${t-1}</b> re-merge${t>2?"s":""}, each ≈ <b>${fmt(remerge)}</b> MACs per adapted matrix (two rank-${r} products, d·k·r each)`:" — and zero extra cost per token")+
          ` · per-token cost: exactly the base model's <b>${fmt(base)}</b> MACs per matrix`);
      }else{
        d3.select("#mg-readout").html(`unmerged: <b>1</b> forward pass for all ${t} task${t>1?"s":""} · extra per token per adapted matrix: r(d + k) = <b>${fmt(extra)}</b> MACs vs ${fmt(base)} for W₀x, i.e. <b>+${pct.toFixed(2)}%</b> of the matmul`+
          ` · no weights are rewritten, so tasks can change from one request to the next`);
      }
    }
    ["#mg-m","#mg-t","#mg-r","#mg-d"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    draw();
  })();

  /* ═══════════════ 3. reported rank ablation & fixed budget ═══════════════ */
  (function(){
    const svg=d3.select("#ra-svg"); if(svg.empty()) return;
    const RANKS=[1,2,4,8,64], LAYERS=96, DM=12288;
    const SETS=[{key:"q",lab:"Wq",n:1,col:C.a2},{key:"qv",lab:"Wq, Wv",n:2,col:C.accent},{key:"all",lab:"Wq, Wk, Wv, Wo",n:4,col:C.violet}];
    const RANK={wikisql:{q:[68.8,69.6,70.5,70.4,70.0],qv:[73.4,73.3,73.7,73.8,73.5],all:[74.1,73.7,74.0,74.0,73.9]},
                mnli:{q:[90.7,90.9,91.1,90.7,90.7],qv:[91.3,91.4,91.3,91.6,91.4],all:[91.2,91.7,91.7,91.5,91.4]}};
    const FT={wikisql:73.8,mnli:89.5};
    const BUD=[["Wq",8,1,70.4,91.0],["Wk",8,1,70.0,90.8],["Wv",8,1,73.0,91.0],["Wo",8,1,73.2,91.3],["Wq, Wk",4,2,71.4,91.3],["Wq, Wv",4,2,73.7,91.3],["Wq, Wk, Wv, Wo",2,4,73.7,91.7]];
    const params=(nm,r)=>nm*LAYERS*2*DM*r;
    const pM=p=>(p/1e6).toFixed(1)+"M";
    const W=760,H=360,m={l:150,r:30,t:30,b:44};
    function draw(){
      svg.selectAll("*").remove();
      const view=d3.select("#ra-v").property("value"), ds=d3.select("#ra-d").property("value"), showFT=d3.select("#ra-ft").property("checked");
      const dsName=ds==="wikisql"?"WikiSQL":"MultiNLI";
      if(view==="rank"){
        const ml=70;
        const vals=[].concat(...SETS.map(s=>RANK[ds][s.key])).concat(showFT?[FT[ds]]:[]);
        const y=d3.scaleLinear().domain([d3.min(vals)-0.4,d3.max(vals)+0.4]).range([H-m.b,m.t]).nice();
        const x=d3.scalePoint().domain(RANKS.map(String)).range([ml+20,W-200]).padding(0.3);
        svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x)).attr("color",C.muted);
        svg.append("g").attr("transform",`translate(${ml},0)`).call(d3.axisLeft(y).ticks(6)).attr("color",C.muted);
        svg.append("text").attr("x",(ml+W-200)/2).attr("y",H-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("LoRA rank r (categorical spacing)");
        svg.append("text").attr("x",ml).attr("y",18).attr("fill",C.muted).attr("font-size",11).text(dsName+" validation accuracy (%) — axis zoomed, does not start at 0");
        if(showFT){
          svg.append("line").attr("x1",ml).attr("x2",W-200).attr("y1",y(FT[ds])).attr("y2",y(FT[ds])).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
          svg.append("text").attr("x",W-196).attr("y",y(FT[ds])+4).attr("fill",C.bad).attr("font-size",10.5).text("full fine-tune "+FT[ds]+(ds==="mnli"?" (MNLI-m)":""));
        }
        SETS.forEach((s,si)=>{
          const v=RANK[ds][s.key];
          svg.append("path").attr("d",d3.line().x((d,i)=>x(String(RANKS[i]))).y(d=>y(d))(v)).attr("fill","none").attr("stroke",s.col).attr("stroke-width",2);
          v.forEach((d,i)=>{
            svg.append("circle").attr("cx",x(String(RANKS[i]))).attr("cy",y(d)).attr("r",4.5).attr("fill",s.col).style("cursor","pointer")
              .on("mouseenter",()=>hover(s,RANKS[i],d,ds)).append("title").text(`${s.lab}, r = ${RANKS[i]}: ${d}`);
          });
          const ly=48+si*20;
          svg.append("rect").attr("x",W-190).attr("y",ly+(showFT?40:0)-9).attr("width",12).attr("height",3).attr("fill",s.col);
          svg.append("text").attr("x",W-172).attr("y",ly+(showFT?40:0)-4).attr("fill",s.col).attr("font-size",11).text(s.lab);
        });
        summaryRank(ds,dsName);
      }else{
        const idx=ds==="wikisql"?3:4;
        const vals=BUD.map(b=>b[idx]).concat(showFT?[FT[ds]]:[]);
        const x=d3.scaleLinear().domain([d3.min(vals)-1,d3.max(vals)+0.5]).range([m.l,W-m.r-60]).nice();
        const y=d3.scaleBand().domain(BUD.map(b=>b[0])).range([m.t,H-m.b]).padding(0.25);
        svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
        svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickFormat(k=>{const b=BUD.find(z=>z[0]===k);return k+"  r="+b[1];})).attr("color",C.muted);
        svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11).text(dsName+" accuracy (%) at ≈18M trainable params — axis zoomed, does not start at 0");
        const best=d3.max(BUD,b=>b[idx]);
        BUD.forEach(b=>{
          const col=b[idx]===best?C.accent:(b[0]==="Wq"||b[0]==="Wk")?C.a2:C.muted;
          svg.append("rect").attr("x",m.l).attr("y",y(b[0])).attr("height",y.bandwidth()).attr("width",Math.max(1,x(b[idx])-m.l)).attr("fill",col).attr("opacity",0.85).style("cursor","pointer")
            .on("mouseenter",()=>hoverB(b,idx,ds)).append("title").text(`${b[0]}, r = ${b[1]}: ${b[idx]}`);
          svg.append("text").attr("x",x(b[idx])+6).attr("y",y(b[0])+y.bandwidth()/2+4).attr("fill",C.ink).attr("font-size",11).text(b[idx].toFixed(1));
        });
        if(showFT){
          svg.append("line").attr("x1",x(FT[ds])).attr("x2",x(FT[ds])).attr("y1",m.t-4).attr("y2",H-m.b).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
          svg.append("text").attr("x",x(FT[ds])+4).attr("y",m.t-6).attr("fill",C.bad).attr("font-size",10.5).text("full FT "+FT[ds]);
        }
        const worst=d3.min(BUD,b=>b[idx]);
        const bestL=BUD.filter(b=>b[idx]===best).map(b=>b[0]+" (r="+b[1]+")").join(" and ");
        const worstL=BUD.filter(b=>b[idx]===worst).map(b=>b[0]+" (r="+b[1]+")").join(", ");
        d3.select("#ra-readout").html(`${dsName}, same ≈18M budget (one matrix at r=8 = two at r=4 = four at r=2 = ${pM(params(1,8))} params): best <b>${bestL}</b> at <b>${best}</b>; worst <b>${worstL}</b> at <b>${worst}</b> — a spread of <b>${(best-worst).toFixed(1)}</b> points from allocation alone`);
      }
    }
    function summaryRank(ds,dsName){
      const parts=SETS.map(s=>{const v=RANK[ds][s.key];return `${s.lab}: ${d3.min(v)}–${d3.max(v)} (spread <b>${(d3.max(v)-d3.min(v)).toFixed(1)}</b>)`;});
      d3.select("#ra-readout").html(`${dsName}, r = 1 → 64 (a 64× change in parameters) · `+parts.join(" · ")+
        ` · full fine-tune ${FT[ds]}<br>hover a point for its parameter count`);
    }
    function hover(s,r,v,ds){
      const p=params(s.n,r);
      d3.select("#ra-readout").html(`<b>${s.lab}</b> at <b>r = ${r}</b>: ${v}% · trainable = ${s.n} matrix${s.n>1?"es":""} × ${LAYERS} layers × 2·${fmt(DM)}·${r} = <b>${pM(p)}</b> params (${(p/175e9*100).toFixed(4)}% of 175B)`);
    }
    function hoverB(b,idx,ds){
      const p=params(b[2],b[1]);
      d3.select("#ra-readout").html(`<b>${b[0]}</b> at r = ${b[1]}: ${b[idx]}% · ${b[2]} matrix${b[2]>1?"es":""} × ${LAYERS} layers × 2·${fmt(DM)}·${b[1]} = <b>${pM(p)}</b> params`);
    }
    ["#ra-v","#ra-d","#ra-ft"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();
})();
