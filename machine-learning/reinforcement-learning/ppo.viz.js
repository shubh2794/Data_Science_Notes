/* ppo.viz.js — interactive figures for machine-learning/reinforcement-learning/ppo.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-clip  one term of L^CLIP against the ratio r, for A > 0 and A < 0 (draggable)
     #viz-toy   a four-armed softmax bandit: K epochs on one batch with L^PG, L^CPI, L^CLIP,
                and a many-iteration, 40-seed comparison — all computed live
     #viz-abl   the reported MuJoCo surrogate ablation (normalised scores) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const f2=x=>(x>=0?"":"−")+Math.abs(x).toFixed(2);
  const f3=x=>(x>=0?"":"−")+Math.abs(x).toFixed(3);

  /* ═══════════════ 1. L^CLIP vs r ═══════════════ */
  (function(){
    const svg=d3.select("#cl-svg"); if(svg.empty()) return;
    const PX=[60,436], PW=290, T=44, B=282;
    const x=d3.scaleLinear().domain([0,2]).range([0,PW]);
    const ys=[d3.scaleLinear().domain([0,2.05]).range([B,T]), d3.scaleLinear().domain([-2.05,0]).range([B,T])];
    const SIGN=[1,-1];
    const clip=(r,e)=>Math.max(1-e,Math.min(1+e,r));
    const term=(r,e,A)=>Math.min(r*A,clip(r,e)*A);
    let rCur=1.3;

    const panels=SIGN.map((A,k)=>{
      const g=svg.append("g").attr("transform",`translate(${PX[k]},0)`);
      const y=ys[k];
      g.append("text").attr("x",0).attr("y",20).attr("fill",A>0?C.good:C.bad).attr("font-size",12.5).attr("font-weight",600)
        .text(A>0?"Â > 0  (a good action)":"Â < 0  (a bad action)");
      const shade=g.append("rect").attr("y",T).attr("height",B-T).attr("fill",C.bad).attr("opacity",0.10);
      const shadeT=g.append("text").attr("y",T+14).attr("fill",C.bad).attr("font-size",10.5).attr("text-anchor","middle").text("zero gradient");
      g.append("g").attr("transform",`translate(0,${B})`).call(d3.axisBottom(x).ticks(5)).attr("color",C.muted);
      g.append("g").call(d3.axisLeft(y).ticks(5)).attr("color",C.muted);
      g.append("text").attr("x",PW/2).attr("y",B+32).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("ratio r = π_θ / π_θold");
      const guides=[0,1,2].map(i=>g.append("line").attr("y1",T).attr("y2",B).attr("stroke",i===1?C.muted:C.ink).attr("stroke-dasharray",i===1?"2 3":"4 3").attr("opacity",0.7));
      const gl=[0,1,2].map(i=>g.append("text").attr("y",T-4).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10));
      const pUn=g.append("path").attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.4).attr("stroke-dasharray","6 4");
      const pCc=g.append("path").attr("fill","none").attr("stroke",C.a2).attr("stroke-width",1.3).attr("opacity",0.9);
      const pMin=g.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",3);
      const pen=g.append("line").attr("stroke",C.bad).attr("stroke-width",2);
      const mUn=g.append("circle").attr("r",4.5).attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.5);
      const mMin=g.append("circle").attr("r",6).attr("fill",C.accent).attr("stroke","#0b0d12").attr("stroke-width",1.5);
      const ov=g.append("rect").attr("x",0).attr("y",T).attr("width",PW).attr("height",B-T).attr("fill","transparent").style("cursor","ew-resize");
      const drag=ev=>{const [mx]=d3.pointer(ev,g.node());rCur=Math.max(0,Math.min(2,Math.round(x.invert(mx)*100)/100));
        d3.select("#cl-r").property("value",rCur);render();};
      ov.on("pointerdown",ev=>{drag(ev);}).on("pointermove",ev=>{if(ev.buttons)drag(ev);});
      return {A,y,shade,shadeT,guides,gl,pUn,pCc,pMin,pen,mUn,mMin};
    });
    // legend
    const lg=svg.append("g").attr("transform","translate(0,338)");
    [["min(…) = the L^CLIP term",C.accent,null,3,60],["unclipped r·Â",C.muted,"6 4",1.4,250],["clip(r)·Â alone",C.a2,null,1.3,390]].forEach(d=>{
      lg.append("line").attr("x1",d[4]).attr("x2",d[4]+24).attr("y1",0).attr("y2",0).attr("stroke",d[1]).attr("stroke-width",d[3]).attr("stroke-dasharray",d[2]);
      lg.append("text").attr("x",d[4]+29).attr("y",4).attr("fill",C.muted).attr("font-size",10.5).text(d[0]);
    });
    lg.append("line").attr("x1",540).attr("x2",540).attr("y1",-7).attr("y2",7).attr("stroke",C.bad).attr("stroke-width",2);
    lg.append("text").attr("x",547).attr("y",4).attr("fill",C.bad).attr("font-size",10.5).text("penalty = unclipped − L^CLIP term");

    function render(){
      const e=+d3.select("#cl-eps").property("value"), sUn=d3.select("#cl-un").property("checked"), sCc=d3.select("#cl-cc").property("checked");
      d3.select("#cl-epsv").text(e.toFixed(2)); d3.select("#cl-rv").text(rCur.toFixed(2));
      const rs=d3.range(0,2.0001,0.01);
      const parts=[];
      panels.forEach(P=>{
        const y=P.y, A=P.A;
        const z0=A>0?1+e:0, z1=A>0?2:1-e;
        P.shade.attr("x",x(z0)).attr("width",Math.max(0,x(z1)-x(z0)));
        P.shadeT.attr("x",(x(z0)+x(z1))/2);
        [1-e,1,1+e].forEach((v,i)=>{P.guides[i].attr("x1",x(v)).attr("x2",x(v));P.gl[i].attr("x",x(v)).text(i===0?"1−ε":i===1?"1":"1+ε");});
        P.pUn.attr("d",d3.line().x(r=>x(r)).y(r=>y(r*A))(rs)).style("display",sUn?null:"none");
        P.pCc.attr("d",d3.line().x(r=>x(r)).y(r=>y(clip(r,e)*A))(rs)).style("display",sCc?null:"none");
        P.pMin.attr("d",d3.line().x(r=>x(r)).y(r=>y(term(r,e,A)))(rs));
        const v=term(rCur,e,A), u=rCur*A, gap=u-v;
        P.mMin.attr("cx",x(rCur)).attr("cy",y(v));
        P.mUn.attr("cx",x(rCur)).attr("cy",y(u)).style("display",sUn?null:"none");
        P.pen.attr("x1",x(rCur)).attr("x2",x(rCur)).attr("y1",y(v)).attr("y2",y(u)).style("display",gap>1e-9?null:"none");
        const active=A>0?rCur<=1+e:rCur>=1-e;
        let why;
        if(A>0) why=active?(rCur<1-e?"the good action became LESS likely — not clipped, the gradient pushes it back up":"inside the trust region on the helpful side — keep pushing up")
                          :"already more than (1+ε)× as likely — no further incentive, this sample is silent";
        else why=active?(rCur>1+e?"the bad action became MORE likely — not clipped, the full penalty pushes it back down":"not yet suppressed by (1−ε) — keep pushing down")
                       :"already less than (1−ε)× as likely — no further incentive, this sample is silent";
        parts.push(`<b style="color:${A>0?C.good:C.bad}">Â = ${A>0?"+1":"−1"}</b> at r = ${rCur.toFixed(2)}: term <b>${f2(v)}</b>, unclipped ${f2(u)}, penalty <b>${gap.toFixed(2)}</b>, ∂term/∂r = <b>${active?(A>0?"+1":"−1"):"0"}</b> — ${why}`);
      });
      d3.select("#cl-readout").html(parts.join("<br>")+`<br>ε = ${e.toFixed(2)}: the band is [${(1-e).toFixed(2)}, ${(1+e).toFixed(2)}]; at r = 1 (the start of every update) both terms equal the unclipped r·Â and carry the plain policy gradient`);
    }
    d3.select("#cl-eps").on("input",render);
    d3.select("#cl-r").on("input",function(){rCur=+this.value;render();});
    d3.select("#cl-un").on("change",render); d3.select("#cl-cc").on("change",render);
    render();
  })();

  /* ═══════════════ 2. bandit toy ═══════════════ */
  (function(){
    const svg=d3.select("#ty-svg"); if(svg.empty()) return;
    const MU=[0.3,0.5,0.8,0.4], NA=4, BEST=2, SEEDS=40, ITERS=40;
    const MET=[{k:"pg",lab:"L^PG",col:C.a2},{k:"cpi",lab:"L^CPI",col:C.violet},{k:"clip",lab:"L^CLIP",col:C.accent}];
    const ARMC=[C.blue,C.muted,C.good,C.bad];
    let seed=3;
    function rng(s){s=s>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    function softmax(z){const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e[0]+e[1]+e[2]+e[3];return e.map(v=>v/s);}
    const KL=(p,q)=>{let s=0;for(let i=0;i<NA;i++)if(p[i]>0)s+=p[i]*Math.log(p[i]/Math.max(q[i],1e-300));return s;};
    const Jof=p=>p[0]*MU[0]+p[1]*MU[1]+p[2]*MU[2]+p[3]*MU[3];
    function sample(R,p0,n,sig){const a=[],rw=[];
      for(let i=0;i<n;i++){let u=R(),k=0,c=p0[0];while(u>c&&k<NA-1){k++;c+=p0[k];}a.push(k);rw.push(MU[k]+sig*gauss(R));}
      let b=0;for(const v of rw)b+=v;b/=n;return {a,rw,A:rw.map(v=>v-b)};}
    // K epochs of full-batch ascent from logits z0 on batch (a, A) collected under p0
    function optimise(m,z0,p0,bt,K,eta,eps,keep){
      const z=z0.slice(), n=bt.a.length, hist=keep?[]:null; let act=n;
      for(let e=0;e<=K;e++){
        const p=softmax(z);
        if(keep)hist.push({p,kl:KL(p0,p),J:Jof(p),act});
        if(e===K)break;
        const g=[0,0,0,0]; act=0;
        for(let i=0;i<n;i++){const ai=bt.a[i],Ai=bt.A[i],r=p[ai]/p0[ai];let w;
          if(m==="pg")w=Ai; else if(m==="cpi")w=Ai*r;
          else{const on=Ai>0?r<=1+eps:Ai<0?r>=1-eps:false;if(on)act++;w=on?Ai*r:0;}
          if(w===0)continue;
          for(let j=0;j<NA;j++)g[j]+=w*((j===ai?1:0)-p[j])/n;}
        for(let j=0;j<NA;j++)z[j]+=eta*g[j];
      }
      return keep?hist:z;
    }

    const W=760, L={x0:60,x1:350,y0:34,y1:196}, R_={x0:440,x1:740,y0:34,y1:196}, Bt={x0:60,x1:740,y0:256,y1:414};
    const symTicks=mx=>[0,0.01,0.03,0.1,0.3,1,3,10,30,100].filter(v=>v<=mx*1.05);

    function axes(xs,ys,box,xl,yl,ticks){
      svg.append("g").attr("transform",`translate(0,${box.y1})`).call(d3.axisBottom(xs).ticks(6)).attr("color",C.muted);
      const ax=d3.axisLeft(ys); if(ticks)ax.tickValues(ticks).tickFormat(d3.format("~g")); else ax.ticks(5);
      svg.append("g").attr("transform",`translate(${box.x0},0)`).call(ax).attr("color",C.muted);
      svg.append("text").attr("x",box.x0).attr("y",box.y0-12).attr("fill",C.muted).attr("font-size",11).text(yl);
      svg.append("text").attr("x",(box.x0+box.x1)/2).attr("y",box.y1+30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text(xl);
    }
    function legend(){
      MET.forEach((m,i)=>{const lx=380+i*92;
        svg.append("rect").attr("x",lx).attr("y",460).attr("width",14).attr("height",3).attr("fill",m.col);
        svg.append("text").attr("x",lx+19).attr("y",465).attr("fill",m.col).attr("font-size",11).text(m.lab);});
    }

    function read(){
      const v=id=>+d3.select(id).property("value");
      const P={mode:d3.select("#ty-m").property("value"),K:v("#ty-k"),eta:v("#ty-eta"),eps:v("#ty-eps"),n:v("#ty-n"),sig:v("#ty-s")};
      d3.select("#ty-kv").text(P.K); d3.select("#ty-etav").text(P.eta.toFixed(2)); d3.select("#ty-epsv").text(P.eps.toFixed(2));
      d3.select("#ty-nv").text(P.n); d3.select("#ty-sv").text(P.sig.toFixed(1));
      return P;
    }

    function drawBatch(P){
      const p0=[0.25,0.25,0.25,0.25], z0=[0,0,0,0];
      const bt=sample(rng(seed*2654435761),p0,P.n,P.sig);
      const H={}; MET.forEach(m=>{H[m.k]=optimise(m.k,z0,p0,bt,P.K,P.eta,P.eps,true);});
      const xs=d3.scaleLinear().domain([0,P.K]).range([L.x0,L.x1]);
      const mx=Math.max(0.05,d3.max(MET,m=>d3.max(H[m.k],h=>h.kl)));
      const yk=d3.scaleSymlog().constant(0.01).domain([0,mx]).range([L.y1,L.y0]);
      axes(xs,yk,L,"epoch on the same batch","KL[π_θold, π_θ] vs epoch (symmetric-log axis)",symTicks(mx));
      const xr=d3.scaleLinear().domain([0,P.K]).range([R_.x0,R_.x1]);
      const yj=d3.scaleLinear().domain([0.25,0.85]).range([R_.y1,R_.y0]);
      axes(xr,yj,R_,"epoch on the same batch","true return J(π_θ) = ∑ π(a)·μₐ vs epoch");
      [[0.5,"start 0.50","6 4",C.muted],[0.8,"best arm 0.80","2 3",C.good]].forEach(d=>{
        svg.append("line").attr("x1",R_.x0).attr("x2",R_.x1).attr("y1",yj(d[0])).attr("y2",yj(d[0])).attr("stroke",d[3]).attr("stroke-dasharray",d[2]);
        svg.append("text").attr("x",R_.x1).attr("y",yj(d[0])-4).attr("text-anchor","end").attr("fill",d[3]).attr("font-size",10).text(d[1]);});
      MET.forEach(m=>{
        svg.append("path").attr("d",d3.line().x((h,i)=>xs(i)).y(h=>yk(h.kl))(H[m.k])).attr("fill","none").attr("stroke",m.col).attr("stroke-width",2);
        svg.append("path").attr("d",d3.line().x((h,i)=>xr(i)).y(h=>yj(h.J))(H[m.k])).attr("fill","none").attr("stroke",m.col).attr("stroke-width",2);
      });
      // ratio bars at epoch K, log scale centred on 1
      const yr=d3.scaleLog().domain([0.05,4.2]).range([Bt.y1,Bt.y0]).clamp(true);
      const xa=d3.scaleBand().domain(d3.range(NA)).range([Bt.x0,Bt.x1]).padding(0.28);
      const xm=d3.scaleBand().domain(MET.map(m=>m.k)).range([0,xa.bandwidth()]).padding(0.12);
      svg.append("rect").attr("x",Bt.x0).attr("width",Bt.x1-Bt.x0).attr("y",yr(1+P.eps)).attr("height",yr(1-P.eps)-yr(1+P.eps)).attr("fill",C.accent).attr("opacity",0.10);
      svg.append("g").attr("transform",`translate(${Bt.x0},0)`).call(d3.axisLeft(yr).tickValues([0.05,0.1,0.25,0.5,1,2,4]).tickFormat(d3.format("~g"))).attr("color",C.muted);
      svg.append("line").attr("x1",Bt.x0).attr("x2",Bt.x1).attr("y1",yr(1)).attr("y2",yr(1)).attr("stroke",C.muted);
      svg.append("text").attr("x",Bt.x0).attr("y",Bt.y0-12).attr("fill",C.muted).attr("font-size",11).text(`ratio π_θ(a)/π_θold(a) per arm after K = ${P.K} epochs (log axis); shaded band = [1−ε, 1+ε]`);
      const stats=d3.range(NA).map(k=>{const idx=bt.a.map((v,i)=>v===k?i:-1).filter(i=>i>=0);
        return {n:idx.length,mA:idx.length?d3.mean(idx,i=>bt.A[i]):0,mR:idx.length?d3.mean(idx,i=>bt.rw[i]):NaN};});
      d3.range(NA).forEach(k=>{
        MET.forEach(m=>{const r=H[m.k][P.K].p[k]/p0[k], x0=xa(k)+xm(m.k);
          svg.append("rect").attr("x",x0).attr("width",xm.bandwidth()).attr("y",Math.min(yr(r),yr(1))).attr("height",Math.max(1,Math.abs(yr(r)-yr(1))))
            .attr("fill",m.col).attr("opacity",0.85).append("title").text(`${m.lab}, arm ${k+1}: ratio ${r.toFixed(3)}, π = ${H[m.k][P.K].p[k].toFixed(3)}`);});
        const s=stats[k];
        svg.append("text").attr("x",xa(k)+xa.bandwidth()/2).attr("y",Bt.y1+16).attr("text-anchor","middle").attr("fill",k===BEST?C.good:C.ink).attr("font-size",11)
          .text(`arm ${k+1} · μ = ${MU[k].toFixed(1)}${k===BEST?" (best)":""}`);
        svg.append("text").attr("x",xa(k)+xa.bandwidth()/2).attr("y",Bt.y1+31).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5)
          .text(`n = ${s.n} · mean Â = ${s.n?f2(s.mA):"—"}`);
      });
      legend();
      const lucky=d3.range(NA).filter(k=>stats[k].n>0).sort((a,b)=>stats[b].mA-stats[a].mA)[0];
      const parts=MET.map(m=>{const h=H[m.k][P.K];const rmax=d3.max(d3.range(NA),k=>h.p[k]/p0[k]);
        return `<b style="color:${m.col}">${m.lab}</b>: KL <b>${f3(h.kl)}</b>, J <b>${h.J.toFixed(3)}</b>, max ratio ${rmax.toFixed(2)}`+(m.k==="clip"?`, samples still carrying gradient in the last epoch <b>${h.act}</b> of ${P.n}`:"");});
      d3.select("#ty-readout").html(parts.join(" · ")+
        `<br>in this batch arm <b>${lucky+1}</b> has the highest mean advantage`+(lucky===BEST?" — the true best arm, so committing to it happens to pay off":` — but the true best is arm ${BEST+1}, so the unclipped surrogates are fitting noise`)+
        ` · press "new batch" to redraw the ${P.n} pulls`);
    }

    function drawIter(P){
      const agg={}; MET.forEach(m=>{agg[m.k]={J:new Float64Array(ITERS+1),kl:new Float64Array(ITERS),fin:[0,0,0,0],stuck:0};});
      for(let s=0;s<SEEDS;s++){
        MET.forEach(m=>{
          const R=rng((seed*7919+s*104729)>>>0); let z=[0,0,0,0]; const A=agg[m.k];
          A.J[0]+=Jof(softmax(z))/SEEDS;
          for(let it=0;it<ITERS;it++){const p0=softmax(z);const bt=sample(R,p0,P.n,P.sig);
            z=optimise(m.k,z,p0,bt,P.K,P.eta,P.eps,false);const p=softmax(z);
            A.J[it+1]+=Jof(p)/SEEDS; A.kl[it]+=KL(p0,p)/SEEDS;}
          const p=softmax(z); let am=0;for(let k=1;k<NA;k++)if(p[k]>p[am])am=k; A.fin[am]++; if(p[BEST]<0.5)A.stuck++;
        });
      }
      const xs=d3.scaleLinear().domain([0,ITERS]).range([L.x0,L.x1]);
      const yj=d3.scaleLinear().domain([0.25,0.85]).range([L.y1,L.y0]);
      axes(xs,yj,L,"iteration (fresh batch each time)",`mean true return over ${SEEDS} seeds`);
      [[0.8,"best arm 0.80","2 3",C.good],[0.5,"uniform 0.50","6 4",C.muted]].forEach(d=>{
        svg.append("line").attr("x1",L.x0).attr("x2",L.x1).attr("y1",yj(d[0])).attr("y2",yj(d[0])).attr("stroke",d[3]).attr("stroke-dasharray",d[2]);
        svg.append("text").attr("x",L.x1).attr("y",yj(d[0])-4).attr("text-anchor","end").attr("fill",d[3]).attr("font-size",10).text(d[1]);});
      const xr=d3.scaleLinear().domain([1,ITERS]).range([R_.x0,R_.x1]);
      const mx=Math.max(0.05,d3.max(MET,m=>d3.max(agg[m.k].kl)));
      const yk=d3.scaleSymlog().constant(0.01).domain([0,mx]).range([R_.y1,R_.y0]);
      axes(xr,yk,R_,"iteration","mean KL[π_θold, π_θ] of each update (symmetric-log)",symTicks(mx));
      MET.forEach(m=>{
        svg.append("path").attr("d",d3.line().x((v,i)=>xs(i)).y(v=>yj(v))(Array.from(agg[m.k].J))).attr("fill","none").attr("stroke",m.col).attr("stroke-width",2);
        svg.append("path").attr("d",d3.line().x((v,i)=>xr(i+1)).y(v=>yk(v))(Array.from(agg[m.k].kl))).attr("fill","none").attr("stroke",m.col).attr("stroke-width",2);
      });
      // stacked shares of the final dominant arm
      const xb=d3.scaleLinear().domain([0,SEEDS]).range([Bt.x0+80,Bt.x1-10]);
      const yb=d3.scaleBand().domain(MET.map(m=>m.k)).range([Bt.y0+6,Bt.y1-10]).padding(0.3);
      svg.append("text").attr("x",Bt.x0).attr("y",Bt.y0-12).attr("fill",C.muted).attr("font-size",11).text(`after ${ITERS} iterations: which arm each seed's policy puts most mass on (green = the best arm)`);
      svg.append("g").attr("transform",`translate(0,${Bt.y1-10})`).call(d3.axisBottom(xb).ticks(8)).attr("color",C.muted);
      MET.forEach(m=>{
        svg.append("text").attr("x",Bt.x0+70).attr("y",yb(m.k)+yb.bandwidth()/2+4).attr("text-anchor","end").attr("fill",m.col).attr("font-size",11.5).text(m.lab);
        let acc=0; agg[m.k].fin.forEach((c,k)=>{if(!c)return;
          svg.append("rect").attr("x",xb(acc)).attr("width",xb(acc+c)-xb(acc)).attr("y",yb(m.k)).attr("height",yb.bandwidth()).attr("fill",ARMC[k]).attr("opacity",0.85)
            .append("title").text(`${m.lab}: ${c} of ${SEEDS} seeds end on arm ${k+1} (μ = ${MU[k]})`);
          if(xb(acc+c)-xb(acc)>34)svg.append("text").attr("x",(xb(acc)+xb(acc+c))/2).attr("y",yb(m.k)+yb.bandwidth()/2+4).attr("text-anchor","middle").attr("fill","#0b0d12").attr("font-size",10.5).text("arm "+(k+1));
          acc+=c;});
      });
      legend();
      d3.select("#ty-readout").html(MET.map(m=>{const A=agg[m.k];
        return `<b style="color:${m.col}">${m.lab}</b>: final mean J <b>${A.J[ITERS].toFixed(3)}</b>, seeds with π(best) &lt; 0.5 at the end <b>${A.stuck}</b> of ${SEEDS}, mean KL per update ${f3(d3.mean(A.kl))}`;}).join("<br>")+
        `<br>K = ${P.K} epochs per batch of n = ${P.n}, η = ${P.eta.toFixed(2)}, ε = ${P.eps.toFixed(2)} · each seed starts uniform; the three methods see the same first batch`);
    }

    function draw(){svg.selectAll("*").remove();const P=read();if(P.mode==="batch")drawBatch(P);else drawIter(P);}
    ["#ty-m","#ty-k","#ty-eta","#ty-eps","#ty-n","#ty-s"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    d3.select("#ty-new").on("click",()=>{seed+=1;draw();});
    draw();
  })();

  /* ═══════════════ 3. surrogate ablation ═══════════════ */
  (function(){
    const svg=d3.select("#ab-svg"); if(svg.empty()) return;
    const FAM={none:{lab:"no constraint",col:C.bad},clip:{lab:"clipping",col:C.accent},akl:{lab:"adaptive KL",col:C.violet},fkl:{lab:"fixed KL",col:C.a2}};
    const D=[["none","no clipping or penalty",-0.39],["clip","clip, ε = 0.1",0.76],["clip","clip, ε = 0.2",0.82],["clip","clip, ε = 0.3",0.70],
      ["akl","adaptive KL, d_targ = 0.003",0.68],["akl","adaptive KL, d_targ = 0.01",0.74],["akl","adaptive KL, d_targ = 0.03",0.71],
      ["fkl","fixed KL, β = 0.3",0.62],["fkl","fixed KL, β = 1",0.71],["fkl","fixed KL, β = 3",0.72],["fkl","fixed KL, β = 10",0.69]];
    const W=760,H=360,m={l:200,r:40,t:26,b:40};
    function draw(){
      svg.selectAll("*").remove();
      const ord=d3.select("#ab-o").property("value");
      const rows=ord==="score"?D.slice().sort((a,b)=>b[2]-a[2]):D;
      const x=d3.scaleLinear().domain([-0.5,1]).range([m.l,W-m.r]);
      const y=d3.scaleBand().domain(rows.map(d=>d[1])).range([m.t,H-m.b]).padding(0.22);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickSize(0)).attr("color",C.muted).select(".domain").remove();
      svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("average normalised score over 21 runs (7 tasks × 3 seeds)");
      [[0,"random policy = 0"],[1,"best result = 1"]].forEach(d=>{
        svg.append("line").attr("x1",x(d[0])).attr("x2",x(d[0])).attr("y1",m.t-6).attr("y2",H-m.b).attr("stroke",C.muted).attr("stroke-dasharray","4 3");
        svg.append("text").attr("x",x(d[0])+(d[0]?-4:4)).attr("y",m.t-10).attr("text-anchor",d[0]?"end":"start").attr("fill",C.muted).attr("font-size",10).text(d[1]);});
      const best=d3.max(D,d=>d[2]);
      rows.forEach(d=>{
        const v=d[2], f=FAM[d[0]];
        svg.append("rect").attr("x",Math.min(x(0),x(v))).attr("width",Math.abs(x(v)-x(0))).attr("y",y(d[1])).attr("height",y.bandwidth())
          .attr("fill",f.col).attr("opacity",v===best?1:0.72).attr("stroke",v===best?C.ink:null).style("cursor","pointer")
          .on("mouseenter",()=>hover(d)).append("title").text(`${d[1]}: ${v}`);
        svg.append("text").attr("x",v>=0?x(v)+5:x(v)-5).attr("y",y(d[1])+y.bandwidth()/2+4).attr("text-anchor",v>=0?"start":"end").attr("fill",C.ink).attr("font-size",11).text(f2(v));
      });
      summary();
    }
    function famBest(k){return d3.max(D.filter(d=>d[0]===k),d=>d[2]);}
    function summary(){
      d3.select("#ab-readout").html(`best per family — clipping <b>${famBest("clip").toFixed(2)}</b> (ε = 0.2) · adaptive KL <b>${famBest("akl").toFixed(2)}</b> (d_targ = 0.01) · fixed KL <b>${famBest("fkl").toFixed(2)}</b> (β = 3) · no constraint <b>−0.39</b> · spread inside each constrained family ≤ 0.12, gap between none and any constraint ≥ 1.0`);
    }
    function hover(d){
      const f=FAM[d[0]], gap=(0.82-d[2]).toFixed(2);
      d3.select("#ab-readout").html(`<b style="color:${f.col}">${d[1]}</b> (${f.lab}): <b>${f2(d[2])}</b>`+(d[2]===0.82?" — the best setting":` · ${gap} below clipping at ε = 0.2`)+
        (d[0]==="none"?" · negative because on HalfCheetah the final policy was worse than random":""));
    }
    d3.select("#ab-o").on("change",draw);
    draw();
  })();
})();
