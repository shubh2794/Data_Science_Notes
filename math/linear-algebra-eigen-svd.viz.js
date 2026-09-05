/* linear-algebra-eigen-svd.viz.js — extracted from linear-algebra-eigen-svd.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch15 · eigenvector hunt ───────────────────────── */
(function(){
  const svg=d3.select("#eig-svg"),W=640,H=360,ox=W/2,oy=H/2,U=46;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["ev",PC.accent],["eav",PC.a2]]);
  const A=[[2,0.8],[0.8,1.3]];               // symmetric → real, orthogonal eigvecs
  const E=eigSym2(A[0][0],A[0][1],A[1][1]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  [E.v1,E.v2].forEach(v=>{ svg.append("line").attr("x1",px({x:-v.x*5,y:-v.y*5})).attr("y1",py({x:-v.x*5,y:-v.y*5}))
    .attr("x2",px({x:v.x*5,y:v.y*5})).attr("y2",py({x:v.x*5,y:v.y*5})).attr("stroke",PC.good).attr("stroke-opacity",.3).attr("stroke-dasharray","5 5"); });
  let v={x:1.6,y:0.4};
  const lAv=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#eav)");
  const lv=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#ev)");
  const out=document.getElementById("eig-readout");
  const hv=svg.append("circle").attr("class","dragpt").attr("r",10).attr("fill",PC.accent).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
  hv.call(d3.drag().on("drag",e=>{ v.x=(e.x-ox)/U; v.y=(oy-e.y)/U; update(); }));
  function update(){
    const Av={x:A[0][0]*v.x+A[0][1]*v.y, y:A[1][0]*v.x+A[1][1]*v.y};
    lv.attr("x1",ox).attr("y1",oy).attr("x2",px(v)).attr("y2",py(v));
    lAv.attr("x1",ox).attr("y1",oy).attr("x2",px(Av)).attr("y2",py(Av));
    hv.attr("cx",px(v)).attr("cy",py(v));
    const nv=Math.hypot(v.x,v.y), nAv=Math.hypot(Av.x,Av.y);
    const cos=(v.x*Av.x+v.y*Av.y)/((nv*nAv)||1e-9), aligned=Math.abs(Math.abs(cos)-1)<0.01;
    const lam=(Av.x*v.x+Av.y*v.y)/((nv*nv)||1e-9);
    out.innerHTML = aligned
      ? `<b style="color:${PC.good}">Eigenvector!</b> Av ∥ v &nbsp;·&nbsp; eigenvalue λ = <b style="color:${PC.good}">${lam.toFixed(2)}</b>`
      : `v=(${v.x.toFixed(2)}, ${v.y.toFixed(2)}) &nbsp;·&nbsp; Av=(${Av.x.toFixed(2)}, ${Av.y.toFixed(2)}) &nbsp;·&nbsp; angle between = <b>${(Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI).toFixed(0)}°</b> &nbsp;<span style="color:${PC.muted}">(eigenvalues ${E.l1.toFixed(2)}, ${E.l2.toFixed(2)})</span>`;
  }
  update();
})();
/* ───────────────────────── Ch16 · SVD circle → ellipse ───────────────────────── */
(function(){
  const svg=d3.select("#svd-svg"),W=640,H=360,ox=W/2,oy=H/2,U=70;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["s1","#4ade80"],["s2","#4ade80"]]);
  const circle=svg.append("path").attr("fill","none").attr("stroke",PC.accent).attr("stroke-opacity",.5).attr("stroke-dasharray","4 3");
  const ellipse=svg.append("path").attr("fill",PC.good).attr("fill-opacity",.12).attr("stroke",PC.good).attr("stroke-width",1.5);
  const ax1=svg.append("line").attr("stroke",PC.good).attr("stroke-width",2.5).attr("marker-end","url(#s1)");
  const ax2=svg.append("line").attr("stroke",PC.good).attr("stroke-width",2).attr("stroke-opacity",.7).attr("marker-end","url(#s2)");
  const out=document.getElementById("svd-readout");
  const ids=["sv-a","sv-b","sv-c","sv-d"];
  function update(){
    const [a,b,c,d]=ids.map(id=>+document.getElementById(id).value), A=[[a,b],[c,d]];
    const samp=d3.range(0,2*Math.PI+0.01,0.1);
    const circPts=samp.map(t=>[ox+Math.cos(t)*U, oy-Math.sin(t)*U]);
    const ellPts=samp.map(t=>{ const x=Math.cos(t), y=Math.sin(t); return [ox+(a*x+b*y)*U, oy-(c*x+d*y)*U]; });
    const path=p=>"M"+p.map(q=>q.join(",")).join("L")+"Z";
    circle.attr("d",path(circPts)); ellipse.attr("d",path(ellPts));
    // singular values & left singular vectors via eig of AᵀA
    const p=a*a+c*c, q=a*b+c*d, r=b*b+d*d, E=eigSym2(p,q,r);
    const s1=Math.sqrt(Math.max(0,E.l1)), s2=Math.sqrt(Math.max(0,E.l2));
    const u1=s1>1e-6?{x:(a*E.v1.x+b*E.v1.y)/s1,y:(c*E.v1.x+d*E.v1.y)/s1}:{x:1,y:0};
    const u2=s2>1e-6?{x:(a*E.v2.x+b*E.v2.y)/s2,y:(c*E.v2.x+d*E.v2.y)/s2}:{x:0,y:1};
    ax1.attr("x1",ox).attr("y1",oy).attr("x2",ox+u1.x*s1*U).attr("y2",oy-u1.y*s1*U);
    ax2.attr("x1",ox).attr("y1",oy).attr("x2",ox+u2.x*s2*U).attr("y2",oy-u2.y*s2*U);
    const cond=s2>1e-6?(s1/s2):Infinity;
    out.innerHTML=`σ₁ = <b style="color:${PC.good}">${s1.toFixed(2)}</b> &nbsp; σ₂ = <b style="color:${PC.good}">${s2.toFixed(2)}</b> &nbsp;·&nbsp; condition number σ₁/σ₂ = <b>${isFinite(cond)?cond.toFixed(2):"∞ (singular)"}</b>`;
  }
  ids.forEach(id=>document.getElementById(id).addEventListener("input",update)); update();
})();
/* ───────────────────────── Ch16 · low-rank approximation (Eckart–Young) ───────────────────────── */
(function(){
  const svg=d3.select("#lr-svg"); if(svg.empty()) return;
  const W=640,H=360,N=32,SZ=160,CELL=SZ/N;
  const PX=[40,240,440], PY=44;
  const SX=40, SW=560, SY1=338, SH=104;

  /* ---- one-sided Jacobi SVD of a square N×N matrix (row-major) ---- */
  function svdSquare(Ain,n){
    const U=Float64Array.from(Ain), V=new Float64Array(n*n);
    for(let i=0;i<n;i++) V[i*n+i]=1;
    for(let sweep=0;sweep<40;sweep++){
      let off=0;
      for(let p=0;p<n-1;p++) for(let q=p+1;q<n;q++){
        let al=0,be=0,ga=0;
        for(let i=0;i<n;i++){ const a=U[i*n+p],b=U[i*n+q]; al+=a*a; be+=b*b; ga+=a*b; }
        off+=ga*ga;
        if(ga===0||Math.abs(ga)<1e-14*Math.sqrt(al*be)) continue;
        const z=(be-al)/(2*ga);
        const t=Math.sign(z||1)/(Math.abs(z)+Math.sqrt(1+z*z));
        const c=1/Math.sqrt(1+t*t), s=c*t;
        for(let i=0;i<n;i++){
          const a=U[i*n+p],b=U[i*n+q]; U[i*n+p]=c*a-s*b; U[i*n+q]=s*a+c*b;
          const x=V[i*n+p],y=V[i*n+q]; V[i*n+p]=c*x-s*y; V[i*n+q]=s*x+c*y;
        }
      }
      if(off<1e-26) break;
    }
    const sig=new Float64Array(n);
    for(let j=0;j<n;j++){ let s2=0; for(let i=0;i<n;i++) s2+=U[i*n+j]*U[i*n+j]; sig[j]=Math.sqrt(s2); }
    for(let j=0;j<n;j++){ const s=sig[j];
      for(let i=0;i<n;i++) U[i*n+j] = s>1e-13 ? U[i*n+j]/s : 0; }
    const ord=Array.from({length:n},(_,i)=>i).sort((a,b)=>sig[b]-sig[a]);
    const Us=new Float64Array(n*n), Vs=new Float64Array(n*n), ss=new Float64Array(n);
    ord.forEach((src,dst)=>{ ss[dst]=sig[src];
      for(let i=0;i<n;i++){ Us[i*n+dst]=U[i*n+src]; Vs[i*n+dst]=V[i*n+src]; } });
    return {U:Us,s:ss,V:Vs};
  }

  /* ---- deterministic pseudo-random source matrices ---- */
  function makeMatrix(kind){
    let seed=20240904; const rnd=()=>{ seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff; };
    const g=()=>{ let s=0; for(let i=0;i<6;i++) s+=rnd(); return s-3; };
    const A=new Float64Array(N*N);
    if(kind==="scene"){
      for(let i=0;i<N;i++) for(let j=0;j<N;j++){
        let v=0.55*Math.sin((j+1)*0.45)+0.45*Math.cos((i+1)*0.30)+0.40*((i+j)/(2*N));
        if(i>6&&i<17&&j>4&&j<13) v+=1.3;                    // a rectangle: rank 1
        if(Math.hypot(i-22,j-21)<6.5) v+=1.2;               // a disc: genuinely high rank
        if(Math.abs(i-j)<2) v-=0.8;                         // a diagonal band: high rank too
        A[i*N+j]=v+0.05*g();
      }
    } else if(kind==="lowrank"){
      const a=[],b=[];
      for(let l=0;l<3;l++){ const av=[],bv=[];
        for(let i=0;i<N;i++) av.push(Math.sin((i+1)*(0.2+0.25*l)+l));
        for(let j=0;j<N;j++) bv.push(Math.cos((j+1)*(0.15+0.3*l)-l));
        a.push(av); b.push(bv); }
      for(let i=0;i<N;i++) for(let j=0;j<N;j++){
        let v=0; for(let l=0;l<3;l++) v+=(3-l)*a[l][i]*b[l][j];
        A[i*N+j]=v+0.25*g();
      }
    } else if(kind==="smooth"){
      for(let i=0;i<N;i++) for(let j=0;j<N;j++)
        A[i*N+j]=Math.exp(-(((i-10)*(i-10))+((j-11)*(j-11)))/70)
                +0.8*Math.exp(-(((i-22)*(i-22))+((j-21)*(j-21)))/40);
    } else {
      for(let i=0;i<N;i++) for(let j=0;j<N;j++) A[i*N+j]=g();
    }
    return A;
  }

  /* ---- static chrome ---- */
  const gPanels=svg.append("g"), gSpec=svg.append("g"), gLine=svg.append("g");
  const titles=["original A","rank-k approximation Ãₖ","residual |A − Ãₖ|"];
  svg.append("g").selectAll("text").data(titles).join("text")
    .attr("x",(d,i)=>PX[i]+SZ/2).attr("y",PY-8).attr("text-anchor","middle")
    .attr("font-size",11.5).attr("fill",PC.muted).text(d=>d);
  svg.append("text").attr("x",SX).attr("y",SY1-SH-14).attr("font-size",11.5).attr("fill",PC.muted)
     .text("singular values σᵢ (bars) · cumulative share of Σσᵢ² (line)");
  svg.append("line").attr("x1",SX).attr("y1",SY1).attr("x2",SX+SW).attr("y2",SY1).attr("stroke","#3a4150");
  svg.append("g").selectAll("text").data([1,8,16,24,32]).join("text")
    .attr("x",d=>SX+(d-0.5)*(SW/N)).attr("y",SY1+13).attr("text-anchor","middle")
    .attr("font-size",9.5).attr("fill","#7b8494").text(d=>d);

  const cells=[]; for(let i=0;i<N;i++) for(let j=0;j<N;j++) cells.push([i,j]);
  const layers=[0,1,2].map(p=>gPanels.append("g").selectAll("rect").data(cells).join("rect")
    .attr("x",d=>PX[p]+d[1]*CELL).attr("y",d=>PY+d[0]*CELL)
    .attr("width",CELL+0.4).attr("height",CELL+0.4).attr("shape-rendering","crispEdges"));
  [0,1,2].forEach(p=>gPanels.append("rect").attr("x",PX[p]-0.5).attr("y",PY-0.5)
    .attr("width",SZ+1).attr("height",SZ+1).attr("fill","none").attr("stroke",PC.line));

  const out=document.getElementById("lr-readout");
  const elK=document.getElementById("lr-k"), elSrc=document.getElementById("lr-src");
  const btnMode=document.getElementById("lr-mode"), btn90=document.getElementById("lr-90");
  let worst=false, A, S, vmin, vmax, total;

  function load(kind){
    A=makeMatrix(kind); S=svdSquare(A,N);
    vmin=Infinity; vmax=-Infinity;
    for(let t=0;t<N*N;t++){ if(A[t]<vmin) vmin=A[t]; if(A[t]>vmax) vmax=A[t]; }
    total=0; for(let i=0;i<N;i++) total+=S.s[i]*S.s[i];
  }

  function render(){
    const k=+elK.value;
    const keep=worst ? d3.range(N-k,N) : d3.range(0,k);
    const kept=new Set(keep);
    const R=new Float64Array(N*N);
    for(const l of keep){ const s=S.s[l];
      if(s<1e-12) continue;
      for(let i=0;i<N;i++){ const ui=s*S.U[i*N+l];
        for(let j=0;j<N;j++) R[i*N+j]+=ui*S.V[j*N+l]; } }

    const span=(vmax-vmin)||1;
    const cVal=d3.scaleLinear().domain([vmin,(vmin+vmax)/2,vmax])
      .range(["#12203a",PC.accent,PC.a2]).clamp(true);
    const cRes=d3.scaleLinear().domain([0,span*0.5]).range(["#141821",PC.bad]).clamp(true);

    layers[0].attr("fill",d=>cVal(A[d[0]*N+d[1]]));
    layers[1].attr("fill",d=>cVal(R[d[0]*N+d[1]]));
    layers[2].attr("fill",d=>cRes(Math.abs(A[d[0]*N+d[1]]-R[d[0]*N+d[1]])));

    /* measured vs predicted error */
    let sse=0, dmax=0;
    for(let t=0;t<N*N;t++){ const d=A[t]-R[t]; sse+=d*d; if(Math.abs(d)>dmax) dmax=Math.abs(d); }
    const measF=Math.sqrt(sse);
    let dropped=0; for(let i=0;i<N;i++) if(!kept.has(i)) dropped+=S.s[i]*S.s[i];
    const predF=Math.sqrt(dropped);
    let optDropped=0; for(let i=k;i<N;i++) optDropped+=S.s[i]*S.s[i];
    const optF=Math.sqrt(optDropped);
    const nextSig=k<N?S.s[k]:0;
    const energy=100*(1-optDropped/total);

    /* spectrum */
    const bw=SW/N, sMax=S.s[0]||1;
    gSpec.selectAll("rect").data(d3.range(N)).join("rect")
      .attr("x",i=>SX+i*bw+2).attr("width",Math.max(2,bw-4))
      .attr("y",i=>SY1-(S.s[i]/sMax)*SH).attr("height",i=>Math.max(0.6,(S.s[i]/sMax)*SH))
      .attr("fill",i=>kept.has(i)?PC.good:"#3a4150");
    let run=0; const cum=d3.range(N).map(i=>{ run+=S.s[i]*S.s[i]; return run/total; });
    gLine.selectAll("path").data([cum]).join("path")
      .attr("fill","none").attr("stroke",PC.a2).attr("stroke-width",1.4).attr("stroke-opacity",.75)
      .attr("d",d3.line().x((d,i)=>SX+(i+0.5)*bw).y(d=>SY1-d*SH));

    const store=k*(2*N+1), full=N*N;
    out.innerHTML =
      `k = <b>${k}</b> of ${N} &nbsp;·&nbsp; layers kept: <b style="color:${PC.good}">${worst?`the ${k} smallest σ`:`top ${k}`}</b>`
      + ` &nbsp;·&nbsp; energy Σ_(i≤k)σᵢ² / Σσᵢ² = <b>${energy.toFixed(2)}%</b>`
      + `<br>‖A − Ãₖ‖_F measured = <b>${measF.toFixed(4)}</b> &nbsp;·&nbsp; predicted √(Σ_(dropped)σᵢ²) = <b>${predF.toFixed(4)}</b>`
      + ` &nbsp;<span style="color:${Math.abs(measF-predF)<1e-6*(1+measF)?PC.good:PC.bad}">${Math.abs(measF-predF)<1e-6*(1+measF)?"✓ Eckart–Young holds":"mismatch"}</span>`
      + `<br>largest single residual entry = ${dmax.toFixed(3)} &nbsp;·&nbsp; best possible spectral error σ_(k+1) = <b>${nextSig.toFixed(4)}</b>`
      + (worst ? `<br><b style="color:${PC.bad}">keeping the wrong k layers costs ${measF.toFixed(3)} instead of the optimal ${optF.toFixed(3)}</b>`
                 + ` — Eckart–Young says no rank-${k} matrix at all beats ${optF.toFixed(3)}`
               : ` &nbsp;·&nbsp; storage ${store} numbers vs ${full} (${(100*store/full).toFixed(0)}%)`);
  }

  elK.addEventListener("input",render);
  elSrc.addEventListener("change",()=>{ load(elSrc.value); render(); });
  btnMode.addEventListener("click",()=>{ worst=!worst;
    btnMode.textContent=worst?"Compare: top k layers":"Compare: worst k layers"; render(); });
  btn90.addEventListener("click",()=>{
    let run=0,k=N; for(let i=0;i<N;i++){ run+=S.s[i]*S.s[i]; if(run/total>=0.9){ k=i+1; break; } }
    elK.value=k; render(); });
  load("scene"); render();
})();
