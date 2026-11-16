/* xgboost.viz.js — interactive figures for machine-learning/decision-trees/xgboost.html.
   Loaded after data.js / notes.js. Three independent IIFEs, all computed live:
     #viz-sg  split-gain explorer: gᵢ, hᵢ from the current prediction, Gain(threshold), λ, γ, boosting rounds
     #viz-qs  weighted vs unweighted quantile candidates when hᵢ is skewed (logistic loss)
     #viz-md  sparsity-aware split: missing → left vs missing → right vs mean imputation */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
  function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  const sig=z=>1/(1+Math.exp(-z));
  const f=(v,d)=>{const t=v.toFixed(d===undefined?2:d);return /^-0(\.0+)?$/.test(t)?t.slice(1):t;};
  const val=id=>d3.select(id).property("value");
  const score=(G,H,lam)=>G*G/(H+lam);   // (Σg)² / (Σh + λ)

  /* ═══════════════ 1. split-gain explorer ═══════════════ */
  (function(){
    const svg=d3.select("#sg-svg"); if(svg.empty()) return;
    const N=26, R=rng(5);
    const X=d3.range(N).map(()=>0.3+9.4*R()).sort((a,b)=>a-b);
    const nz=X.map(()=>gauss(R)), uu=X.map(()=>R());
    const Y={sq:X.map((x,i)=>(x<3.5?-1.5:x<7?2:0.5)+0.6*nz[i]),
             lg:X.map((x,i)=>uu[i]<(x<3.5?0.15:x<7?0.85:0.4)?1:0)};
    const THR=d3.range(N-1).map(k=>(X[k]+X[k+1])/2);
    let pred=new Float64Array(N), rounds=0, sel=-1, last="", drag=false;
    const xs=d3.scaleLinear().domain([0,10]).range([60,740]);

    function grads(loss){
      const g=new Float64Array(N),h=new Float64Array(N),y=Y[loss];
      for(let i=0;i<N;i++){
        if(loss==="sq"){g[i]=pred[i]-y[i];h[i]=1;}
        else{const p=sig(pred[i]);g[i]=p-y[i];h[i]=p*(1-p);}
      }
      return {g,h,y};
    }
    function compute(){
      const loss=val("#sg-loss"), lam=+val("#sg-lam"), gam=+val("#sg-gam");
      const {g,h,y}=grads(loss);
      const Gt=d3.sum(g), Ht=d3.sum(h);
      let GL=0,HL=0; const rows=[];
      for(let k=0;k<N-1;k++){
        GL+=g[k];HL+=h[k];const GR=Gt-GL,HR=Ht-HL;
        rows.push({k,GL,HL,GR,HR,raw:0.5*(score(GL,HL,lam)+score(GR,HR,lam)-score(Gt,Ht,lam))});
      }
      let best=0; rows.forEach((r,i)=>{if(r.raw>rows[best].raw)best=i;});
      return {loss,lam,gam,g,h,y,Gt,Ht,rows,best};
    }
    function trainLoss(loss){
      const y=Y[loss]; let s=0;
      for(let i=0;i<N;i++){
        if(loss==="sq"){const d=y[i]-pred[i];s+=0.5*d*d;}
        else{const p=Math.min(1-1e-12,Math.max(1e-12,sig(pred[i])));s-=y[i]*Math.log(p)+(1-y[i])*Math.log(1-p);}
      }
      return s;
    }
    function shown(loss,v){return loss==="sq"?v:sig(v);}

    function render(){
      svg.selectAll("*").remove();
      const c=compute(), eta=+val("#sg-eta");
      d3.select("#sg-lamv").text(f(c.lam,1)); d3.select("#sg-gamv").text(f(c.gam,1)); d3.select("#sg-etav").text(f(eta,2));
      const k=sel<0?c.best:sel, r=c.rows[k];
      const wL=-r.GL/(r.HL+c.lam), wR=-r.GR/(r.HR+c.lam), gain=r.raw-c.gam;

      // ── panel A: data and current prediction ──
      const ya=c.loss==="sq"?d3.scaleLinear().domain([-4,4.5]).range([160,34]):d3.scaleLinear().domain([-0.08,1.08]).range([160,34]);
      svg.append("g").attr("transform","translate(0,160)").call(d3.axisBottom(xs).ticks(10)).attr("color",C.muted);
      svg.append("g").attr("transform","translate(60,0)").call(d3.axisLeft(ya).ticks(5)).attr("color",C.muted);
      svg.append("text").attr("x",60).attr("y",20).attr("fill",C.muted).attr("font-size",11)
        .text(c.loss==="sq"?"targets yᵢ and current prediction ŷ⁽ᵗ⁻¹⁾ (teal); dashed amber = after adding η·f at the selected split":"labels yᵢ ∈ {0, 1} and current p = σ(ŷ⁽ᵗ⁻¹⁾) (teal); dashed amber = after adding η·f at the selected split");
      const tx=xs(THR[k]);
      svg.append("line").attr("x1",tx).attr("x2",tx).attr("y1",30).attr("y2",160).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("opacity",0.7);
      const line=d3.line().x(d=>xs(d[0])).y(d=>ya(d[1])).curve(d3.curveStep);
      const cur=d3.range(N).map(j=>[X[j],shown(c.loss,pred[j])]);
      cur.unshift([0,cur[0][1]]); cur.push([10,cur[N][1]]);
      const doSplit=c.rows[c.best].raw-c.gam>0;
      const nxt=d3.range(N).map(j=>[X[j],shown(c.loss,pred[j]+eta*(j<=k?wL:wR))]);
      nxt.unshift([0,nxt[0][1]]); nxt.push([10,nxt[N][1]]);
      svg.append("path").attr("d",line(nxt)).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",1.6).attr("stroke-dasharray","5 4");
      svg.append("path").attr("d",line(cur)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
      svg.append("g").selectAll("circle").data(d3.range(N)).join("circle")
        .attr("cx",j=>xs(X[j])).attr("cy",j=>ya(c.loss==="sq"?c.y[j]:c.y[j]+(j%2?0.025:-0.025)))
        .attr("r",4).attr("fill",j=>j<=k?C.violet:C.blue).attr("opacity",0.9);

      // ── panel B: gradients and hessians ──
      const gmax=Math.max(0.5,d3.max(c.g,v=>Math.abs(v)));
      const yg=d3.scaleLinear().domain([-gmax,gmax]).range([268,196]);
      const hmax=c.loss==="sq"?1:0.25;
      const yh=d3.scaleLinear().domain([0,hmax]).range([312,282]);
      svg.append("text").attr("x",60).attr("y",188).attr("fill",C.muted).attr("font-size",11)
        .text("gᵢ = ∂l/∂ŷ (stems; violet = left of the split, blue = right)   ·   hᵢ = ∂²l/∂ŷ² (bars, scale 0–"+hmax+")");
      svg.append("line").attr("x1",60).attr("x2",740).attr("y1",yg(0)).attr("y2",yg(0)).attr("stroke",C.line);
      svg.append("line").attr("x1",60).attr("x2",740).attr("y1",312).attr("y2",312).attr("stroke",C.line);
      svg.append("text").attr("x",54).attr("y",yg(0)+4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("g");
      svg.append("text").attr("x",54).attr("y",304).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("h");
      d3.range(N).forEach(j=>{
        const col=j<=k?C.violet:C.blue;
        svg.append("line").attr("x1",xs(X[j])).attr("x2",xs(X[j])).attr("y1",yg(0)).attr("y2",yg(c.g[j])).attr("stroke",col).attr("stroke-width",2);
        svg.append("circle").attr("cx",xs(X[j])).attr("cy",yg(c.g[j])).attr("r",2.5).attr("fill",col);
        svg.append("rect").attr("x",xs(X[j])-3).attr("width",6).attr("y",yh(c.h[j])).attr("height",312-yh(c.h[j])).attr("fill",col).attr("opacity",0.7);
      });
      svg.append("line").attr("x1",tx).attr("x2",tx).attr("y1",192).attr("y2",312).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("opacity",0.5);

      // ── panel C: gain curve vs threshold ──
      const raws=c.rows.map(q=>q.raw);
      const yc=d3.scaleLinear().domain([Math.min(0,d3.min(raws)),Math.max(d3.max(raws),c.gam,0.01)*1.08]).range([440,346]).nice();
      svg.append("g").attr("transform","translate(0,440)").call(d3.axisBottom(xs).ticks(10)).attr("color",C.muted);
      svg.append("g").attr("transform","translate(60,0)").call(d3.axisLeft(yc).ticks(4)).attr("color",C.muted);
      svg.append("text").attr("x",60).attr("y",336).attr("fill",C.muted).attr("font-size",11)
        .text("½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] for every threshold — a split is taken only where it clears γ (red)");
      svg.append("text").attr("x",400).attr("y",466).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("threshold on x (click or drag in any panel to choose one)");
      svg.append("path").attr("d",d3.line().x(q=>xs(THR[q.k])).y(q=>yc(q.raw))(c.rows)).attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.2);
      svg.append("g").selectAll("circle").data(c.rows).join("circle").attr("cx",q=>xs(THR[q.k])).attr("cy",q=>yc(q.raw))
        .attr("r",q=>q.k===k?5.5:3.2).attr("fill",q=>q.raw>c.gam?C.good:C.muted).attr("stroke",q=>q.k===k?C.ink:"none");
      svg.append("line").attr("x1",60).attr("x2",740).attr("y1",yc(c.gam)).attr("y2",yc(c.gam)).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
      svg.append("text").attr("x",736).attr("y",yc(c.gam)-5).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10.5).text("γ = "+f(c.gam,1));
      const bx=xs(THR[c.best]), byv=yc(c.rows[c.best].raw);
      svg.append("path").attr("d",`M${bx},${byv-9} l-5,-8 h10 z`).attr("fill",C.accent);
      svg.append("line").attr("x1",tx).attr("x2",tx).attr("y1",342).attr("y2",440).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("opacity",0.5);

      const nL=k+1, nR=N-nL;
      const hr=c.loss==="lg"?` · hᵢ ranges ${f(d3.min(c.h),3)}–${f(d3.max(c.h),3)}, so leaves are weighted by confidence, not by count`:" · squared loss: every hᵢ = 1, so H is just the leaf's count";
      d3.select("#sg-readout").html(
        `selected split x &lt; <b>${f(THR[k])}</b>${sel<0?" (the best one)":""} · left n=${nL}: G_L=${f(r.GL)}, H_L=${f(r.HL)} → w_L* = −G_L/(H_L+λ) = <b>${f(wL,3)}</b>`+
        ` · right n=${nR}: G_R=${f(r.GR)}, H_R=${f(r.HR)} → w_R* = <b>${f(wR,3)}</b>`+
        `<br>gain = ${f(r.raw,3)} − γ = <b style="color:${gain>0?C.good:C.bad}">${f(gain,3)}</b>`+
        ` · best threshold ${f(THR[c.best])} with gain ${f(c.rows[c.best].raw-c.gam,3)} → `+
        (doSplit?"<b>this round splits</b>":`<b>no split this round</b>: γ exceeds every gain, so the tree is one leaf with w* = −G/(H+λ) = ${f(-c.Gt/(c.Ht+c.lam),3)}`)+hr+
        `<br>round <b>${rounds}</b> · training loss <b>${f(trainLoss(c.loss),3)}</b>`+(last?` · ${last}`:""));
    }

    function boost(){
      const c=compute(), eta=+val("#sg-eta"), b=c.rows[c.best];
      const before=trainLoss(c.loss);
      if(b.raw-c.gam>0){
        const wL=-b.GL/(b.HL+c.lam), wR=-b.GR/(b.HR+c.lam);
        for(let i=0;i<N;i++)pred[i]+=eta*(i<=b.k?wL:wR);
        last=`round ${rounds+1} split at x &lt; ${f(THR[b.k])}`;
      }else{
        const w=-c.Gt/(c.Ht+c.lam);
        for(let i=0;i<N;i++)pred[i]+=eta*w;
        last=`round ${rounds+1} was a single leaf (no split cleared γ)`;
      }
      rounds++; sel=-1;
      last+=`; loss ${f(before,3)} → ${f(trainLoss(c.loss),3)}`;
      render();
    }
    function reset(){pred=new Float64Array(N);rounds=0;sel=-1;last="";render();}
    function pick(ev){
      const [px,py]=d3.pointer(ev,svg.node());
      if(px<50||py>470)return;
      let bk=0,bd=1e9;THR.forEach((t,i)=>{const d=Math.abs(xs(t)-px);if(d<bd){bd=d;bk=i;}});
      if(bk!==sel){sel=bk;render();}
    }
    svg.style("cursor","crosshair").style("touch-action","none")
      .on("pointerdown",ev=>{drag=true;pick(ev);})
      .on("pointermove",ev=>{if(drag)pick(ev);})
      .on("pointerup pointerleave",()=>{drag=false;});
    ["#sg-lam","#sg-gam","#sg-eta"].forEach(id=>d3.select(id).on("input",render));
    d3.select("#sg-loss").on("change",reset);
    d3.select("#sg-boost").on("click",boost);
    d3.select("#sg-best").on("click",()=>{sel=-1;render();});
    d3.select("#sg-reset").on("click",reset);
    render();
  })();

  /* ═══════════════ 2. weighted vs unweighted quantile candidates ═══════════════ */
  (function(){
    const svg=d3.select("#qs-svg"); if(svg.empty()) return;
    const N=800, R=rng(21), C0=1.5, CT=1.9, KT=4, LAM=1;
    const X=d3.range(N).map(()=>gauss(R)).sort((a,b)=>a-b);
    // soft labels yᵢ = the true probability, from a boundary the model places slightly wrong.
    // (Hard 0/1 draws let a single rare positive in a near-zero-h region dominate the toy.)
    const Y=X.map(x=>sig(KT*(x-CT)));
    const xs=d3.scaleLinear().domain([-3.5,3.5]).range([60,740]).clamp(true);

    function candidates(w,eps){
      const W=d3.sum(w); let cum=0,next=eps; const out=[X[0]];
      for(let i=0;i<N;i++){
        if(cum/W>=next){if(X[i]>out[out.length-1])out.push(X[i]);while(cum/W>=next)next+=eps;}
        cum+=w[i];
      }
      if(X[N-1]>out[out.length-1])out.push(X[N-1]);
      return out;
    }
    function draw(){
      svg.selectAll("*").remove();
      const k=+val("#qs-k"), eps=+val("#qs-eps");
      d3.select("#qs-kv").text(f(k,1)); d3.select("#qs-epsv").text(f(eps,2)+" (≈"+Math.round(1/eps)+" candidates)");
      const p=X.map(x=>sig(k*(x-C0)));
      const g=X.map((x,i)=>p[i]-Y[i]), h=p.map(q=>q*(1-q));
      const PG=[0],PH=[0]; for(let i=0;i<N;i++){PG.push(PG[i]+g[i]);PH.push(PH[i]+h[i]);}
      const Gt=PG[N],Ht=PH[N];
      const gainAt=idx=>(idx<=0||idx>=N)?0:0.5*(score(PG[idx],PH[idx],LAM)+score(Gt-PG[idx],Ht-PH[idx],LAM)-score(Gt,Ht,LAM));
      let ex={g:0,x:X[0]}; for(let i=1;i<N;i++){if(X[i]>X[i-1]){const v=gainAt(i);if(v>ex.g)ex={g:v,x:(X[i]+X[i-1])/2};}}
      const best=cs=>{let b={g:0,x:cs[0]};cs.forEach(s=>{const v=gainAt(d3.bisectLeft(X,s));if(v>b.g)b={g:v,x:s};});return b;};
      const cu=candidates(X.map(()=>1),eps), cw=candidates(h,eps);
      const bu=best(cu), bw=best(cw);

      // ── top: share of rows vs share of Σh per bin ──
      const bins=d3.bin().domain([-3.5,3.5]).thresholds(d3.range(-3.5,3.5,0.175))(X);
      const hBin=bins.map(b=>{let s=0;for(let i=0;i<N;i++)if(X[i]>=b.x0&&X[i]<b.x1)s+=h[i];return s/Ht;});
      const cBin=bins.map(b=>b.length/N);
      const yt=d3.scaleLinear().domain([0,Math.max(d3.max(hBin),d3.max(cBin))*1.08]).range([180,34]);
      svg.append("text").attr("x",60).attr("y",20).attr("fill",C.muted).attr("font-size",11).text("share of rows per bin (grey) vs share of total hessian Σhᵢ per bin (teal outline); dashed = current model p = σ(k·(x − 1.5))");
      svg.append("g").attr("transform","translate(60,0)").call(d3.axisLeft(yt).ticks(4,"~%")).attr("color",C.muted);
      bins.forEach((b,i)=>{
        const x0=xs(b.x0)+0.5, wd=Math.max(1,xs(b.x1)-xs(b.x0)-1);
        svg.append("rect").attr("x",x0).attr("width",wd).attr("y",yt(cBin[i])).attr("height",180-yt(cBin[i])).attr("fill",C.muted).attr("opacity",0.35);
        svg.append("rect").attr("x",x0).attr("width",wd).attr("y",yt(hBin[i])).attr("height",180-yt(hBin[i])).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",1.3);
      });
      const yp=d3.scaleLinear().domain([0,1]).range([180,34]);
      svg.append("path").attr("d",d3.line().x(x=>xs(x)).y(x=>yp(sig(k*(x-C0))))(d3.range(-3.5,3.51,0.05))).attr("fill","none").attr("stroke",C.ink).attr("stroke-dasharray","4 4").attr("opacity",0.6);
      svg.append("line").attr("x1",60).attr("x2",740).attr("y1",180).attr("y2",180).attr("stroke",C.line);

      // ── candidate rows ──
      const row=(cs,y,col,lab,bb)=>{
        svg.append("text").attr("x",54).attr("y",y+4).attr("text-anchor","end").attr("fill",col).attr("font-size",10.5).text(lab);
        cs.forEach(s=>svg.append("line").attr("x1",xs(s)).attr("x2",xs(s)).attr("y1",y-7).attr("y2",y+7).attr("stroke",col).attr("stroke-width",1.4));
        svg.append("circle").attr("cx",xs(bb.x)).attr("cy",y).attr("r",5).attr("fill","none").attr("stroke",C.ink).attr("stroke-width",1.5);
      };
      row(cu,204,C.a2,"by count",bu);
      row(cw,226,C.accent,"by h",bw);
      svg.append("line").attr("x1",xs(ex.x)).attr("x2",xs(ex.x)).attr("y1",30).attr("y2",236).attr("stroke",C.bad).attr("stroke-width",1.2).attr("stroke-dasharray","2 3");
      svg.append("text").attr("x",xs(ex.x)+4).attr("y",42).attr("fill",C.bad).attr("font-size",10.5).text("exact best split");

      // ── bottom: rank functions ──
      const yr=d3.scaleLinear().domain([0,1]).range([420,262]);
      svg.append("text").attr("x",60).attr("y",254).attr("fill",C.muted).attr("font-size",11).text("rank functions r(z): unweighted (amber) and h-weighted rₖ(z) (teal); candidates sit where r crosses multiples of ε");
      svg.append("g").attr("transform","translate(0,420)").call(d3.axisBottom(xs).ticks(7)).attr("color",C.muted);
      svg.append("g").attr("transform","translate(60,0)").call(d3.axisLeft(yr).ticks(5)).attr("color",C.muted);
      for(let e=eps;e<1;e+=eps)svg.append("line").attr("x1",60).attr("x2",740).attr("y1",yr(e)).attr("y2",yr(e)).attr("stroke",C.line).attr("stroke-width",0.6);
      const cumU=[],cumW=[];let a=0,b=0;for(let i=0;i<N;i++){cumU.push([X[i],a/N]);cumW.push([X[i],b/Ht]);a+=1;b+=h[i];}
      const ln=d3.line().x(d=>xs(d[0])).y(d=>yr(d[1]));
      svg.append("path").attr("d",ln(cumU)).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",1.8);
      svg.append("path").attr("d",ln(cumW)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",1.8);
      svg.append("text").attr("x",400).attr("y",448).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("feature value x (800 rows, x ~ N(0, 1); soft labels from a boundary at 1.9 that the model places at 1.5)");

      const inBand=cs=>cs.filter(s=>{const q=sig(k*(s-C0));return q>0.1&&q<0.9;}).length;
      const pct=v=>ex.g>0?(100*v/ex.g).toFixed(1)+"%":"—";
      d3.select("#qs-readout").html(
        `best gain (λ = 1, γ = 0): exact greedy over all ${N-1} thresholds <b>${f(ex.g,3)}</b> · `+
        `h-weighted candidates (${cw.length}) <b>${f(bw.g,3)}</b> = <b>${pct(bw.g)}</b> of exact · `+
        `count quantiles (${cu.length}) <b>${f(bu.g,3)}</b> = <b>${pct(bu.g)}</b> of exact`+
        `<br>candidates inside the uncertain band 0.1 &lt; p &lt; 0.9: by h <b>${inBand(cw)}</b> of ${cw.length}, by count <b>${inBand(cu)}</b> of ${cu.length}`+
        ` · rows in that band hold <b>${(100*d3.sum(h.filter((v,i)=>p[i]>0.1&&p[i]<0.9))/Ht).toFixed(0)}%</b> of Σh but only <b>${(100*p.filter(q=>q>0.1&&q<0.9).length/N).toFixed(0)}%</b> of the rows`);
    }
    ["#qs-k","#qs-eps"].forEach(id=>d3.select(id).on("input",draw));
    draw();
  })();

  /* ═══════════════ 3. sparsity-aware default direction ═══════════════ */
  (function(){
    const svg=d3.select("#md-svg"); if(svg.empty()) return;
    const N=30;
    let seed=33, X,Y,U;
    function sample(){
      const R=rng(seed);
      X=d3.range(N).map(()=>0.2+9.6*R()).sort((a,b)=>a-b);
      Y=X.map(x=>(x<5?-1:1.5)+0.5*gauss(R));
      U=X.map(()=>R());
    }
    const xs=d3.scaleLinear().domain([0,10]).range([130,740]);
    let sel=-1, drag=false;

    function compute(){
      const frac=+val("#md-f"), pat=val("#md-p"), lam=+val("#md-lam");
      const med=d3.median(Y);
      const miss=X.map((x,i)=>pat==="random"?U[i]<frac:(Y[i]>med&&U[i]<2*frac));
      const g=Y.map(y=>-y), h=Y.map(()=>1);          // squared loss at ŷ = 0
      const obs=d3.range(N).filter(i=>!miss[i]), mis=d3.range(N).filter(i=>miss[i]);
      const GM=d3.sum(mis,i=>g[i]), HM=mis.length;
      const Gt=d3.sum(g), Ht=N, base=score(Gt,Ht,lam);
      const rows=[]; let GL=0,HL=0;
      for(let j=0;j<obs.length-1;j++){
        GL+=g[obs[j]];HL+=1;
        const t=(X[obs[j]]+X[obs[j+1]])/2;
        const gl=(a,b)=>0.5*(score(a,b,lam)+score(Gt-a,Ht-b,lam)-base);
        rows.push({t,j,nL:HL,left:gl(GL+GM,HL+HM),right:gl(GL,HL)});
      }
      // mean imputation: every missing row placed at the observed mean
      const xbar=obs.length?d3.mean(obs,i=>X[i]):5;
      const ord=d3.range(N).map(i=>({v:miss[i]?xbar:X[i],g:g[i]})).sort((a,b)=>a.v-b.v);
      const imp=[]; let Ga=0,Ha=0;
      for(let j=0;j<N-1;j++){Ga+=ord[j].g;Ha+=1;if(ord[j+1].v>ord[j].v)imp.push({t:(ord[j].v+ord[j+1].v)/2,gain:0.5*(score(Ga,Ha,lam)+score(Gt-Ga,Ht-Ha,lam)-base)});}
      return {frac,pat,lam,miss,obs,mis,GM,HM,rows,imp,xbar};
    }
    function draw(){
      svg.selectAll("*").remove();
      const c=compute();
      d3.select("#md-fv").text(Math.round(100*c.frac)+"%"); d3.select("#md-lamv").text(f(c.lam,1));
      if(c.rows.length===0){d3.select("#md-readout").text("too few observed values to split — lower the missing fraction");return;}
      if(sel>=c.rows.length)sel=-1;
      const bestOf=key=>{let b=0;c.rows.forEach((r,i)=>{if(r[key]>c.rows[b][key])b=i;});return b;};
      const bL=bestOf("left"), bR=bestOf("right");
      const bestIdx=c.rows[bL].left>=c.rows[bR].right?bL:bR;
      const k=sel<0?bestIdx:sel, r=c.rows[k];
      let bi=0; c.imp.forEach((q,i)=>{if(q.gain>c.imp[bi].gain)bi=i;});

      // ── top: data, with the missing rows in their own column ──
      const ya=d3.scaleLinear().domain([-3,3.5]).range([170,34]);
      svg.append("text").attr("x",20).attr("y",20).attr("fill",C.muted).attr("font-size",11).text("targets y (squared loss, ŷ = 0, so gᵢ = −yᵢ and hᵢ = 1) — rows with x missing are collected on the left");
      svg.append("rect").attr("x",20).attr("y",34).attr("width",80).attr("height",136).attr("fill",C.panel).attr("stroke",C.line);
      svg.append("text").attr("x",60).attr("y",186).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("x missing ("+c.mis.length+")");
      svg.append("g").attr("transform","translate(0,170)").call(d3.axisBottom(xs).ticks(10)).attr("color",C.muted);
      svg.append("g").attr("transform","translate(130,0)").call(d3.axisLeft(ya).ticks(5)).attr("color",C.muted);
      svg.append("line").attr("x1",xs(c.xbar)).attr("x2",xs(c.xbar)).attr("y1",34).attr("y2",170).attr("stroke",C.a2).attr("stroke-dasharray","2 3").attr("opacity",0.8);
      svg.append("text").attr("x",xs(c.xbar)+4).attr("y",44).attr("fill",C.a2).attr("font-size",10.5).text("mean-impute lands here");
      const tx=xs(r.t);
      svg.append("line").attr("x1",tx).attr("x2",tx).attr("y1",30).attr("y2",170).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("opacity",0.7);
      const dirLeft=r.left>=r.right;
      c.obs.forEach((i,j)=>svg.append("circle").attr("cx",xs(X[i])).attr("cy",ya(Y[i])).attr("r",4).attr("fill",j<=r.j?C.violet:C.blue));
      c.mis.forEach((i,m)=>svg.append("circle").attr("cx",36+(m%4)*16).attr("cy",ya(Y[i])).attr("r",4).attr("fill",dirLeft?C.violet:C.blue).attr("stroke",C.ink).attr("stroke-width",0.8));

      // ── bottom: gain curves ──
      const all=c.rows.map(q=>q.left).concat(c.rows.map(q=>q.right)).concat(c.imp.map(q=>q.gain));
      const yc=d3.scaleLinear().domain([0,d3.max(all)*1.08||1]).range([410,236]).nice();
      svg.append("text").attr("x",20).attr("y",222).attr("fill",C.muted).attr("font-size",11).text("gain vs threshold: missing → left (violet), missing → right (blue), mean-imputed then split (amber, dotted)");
      svg.append("g").attr("transform","translate(0,410)").call(d3.axisBottom(xs).ticks(10)).attr("color",C.muted);
      svg.append("g").attr("transform","translate(130,0)").call(d3.axisLeft(yc).ticks(5)).attr("color",C.muted);
      const ln=key=>d3.line().x(q=>xs(q.t)).y(q=>yc(q[key]));
      svg.append("path").attr("d",ln("gain")(c.imp)).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",1.6).attr("stroke-dasharray","2 3");
      svg.append("path").attr("d",ln("left")(c.rows)).attr("fill","none").attr("stroke",C.violet).attr("stroke-width",2);
      svg.append("path").attr("d",ln("right")(c.rows)).attr("fill","none").attr("stroke",C.blue).attr("stroke-width",2);
      svg.append("line").attr("x1",tx).attr("x2",tx).attr("y1",232).attr("y2",410).attr("stroke",C.ink).attr("stroke-dasharray","3 3").attr("opacity",0.6);
      svg.append("circle").attr("cx",tx).attr("cy",yc(r.left)).attr("r",4.5).attr("fill",C.violet).attr("stroke",C.ink);
      svg.append("circle").attr("cx",tx).attr("cy",yc(r.right)).attr("r",4.5).attr("fill",C.blue).attr("stroke",C.ink);
      svg.append("text").attr("x",435).attr("y",438).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("threshold on the observed x (click or drag to choose)");

      const bestSparse=Math.max(c.rows[bL].left,c.rows[bR].right), bestImp=c.imp.length?c.imp[bi].gain:0;
      d3.select("#md-readout").html(
        `split x &lt; <b>${f(r.t)}</b>${sel<0?" (the best one)":""}: missing → left gain <b>${f(r.left,3)}</b> · missing → right gain <b>${f(r.right,3)}</b> → learned default direction <b>${c.mis.length?(dirLeft?"left":"right"):"either (no missing rows)"}</b>`+
        `<br>best over all thresholds: sparsity-aware <b>${f(bestSparse,3)}</b> (default ${c.rows[bL].left>=c.rows[bR].right?"left":"right"}) · mean-imputed <b>${f(bestImp,3)}</b>`+
        (c.mis.length?` · the scan visited only <b>${c.obs.length}</b> observed values, twice; missing rows entered only as one block (G_miss = ${f(c.GM)}, H_miss = ${c.HM})`:""));
    }
    function pick(ev){
      const [px,py]=d3.pointer(ev,svg.node()); if(px<120)return;
      const c=compute(); let bk=0,bd=1e9;
      c.rows.forEach((q,i)=>{const d=Math.abs(xs(q.t)-px);if(d<bd){bd=d;bk=i;}});
      if(bk!==sel){sel=bk;draw();}
    }
    svg.style("cursor","crosshair").style("touch-action","none")
      .on("pointerdown",ev=>{drag=true;pick(ev);})
      .on("pointermove",ev=>{if(drag)pick(ev);})
      .on("pointerup pointerleave",()=>{drag=false;});
    d3.select("#md-f").on("input",()=>{sel=-1;draw();});
    d3.select("#md-lam").on("input",draw);
    d3.select("#md-p").on("change",()=>{sel=-1;draw();});
    d3.select("#md-new").on("click",()=>{seed+=1;sel=-1;sample();draw();});
    sample(); draw();
  })();
})();
