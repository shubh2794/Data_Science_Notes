/* neural-networks.viz.js — extracted from neural-networks.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── 01 · feed-forward net, shapes & forward pass ───────────────────────── */
(function(){
  const svg=d3.select("#nn-svg"), W=640, H=320;
  const g=svg.append("g");
  const BATCHES=[1,8,32,128];
  const SUP=["¹","²","³","⁴"];
  const out=document.getElementById("nn-readout");
  let layers=[], busy=false;

  function shapeChain(){
    const B=BATCHES[+d3.select("#nn-batch").property("value")];
    let params=0;
    for(let i=1;i<layers.length;i++) params += layers[i-1]*layers[i] + layers[i];
    const names=layers.map((d,i)=> i===0 ? "X" : i===layers.length-1 ? "ŷ" : "h"+SUP[i-1]);
    out.innerHTML = layers.map((d,i)=>`${names[i]}: ${B}×${d}`).join(" → ") + ` &nbsp;·&nbsp; <b>${params}</b> parameters`;
  }

  function build(){
    const L=+d3.select("#nn-layers").property("value"), wid=+d3.select("#nn-width").property("value");
    const B=BATCHES[+d3.select("#nn-batch").property("value")];
    d3.select("#nn-lval").text(L); d3.select("#nn-wval").text(wid); d3.select("#nn-bval").text(B);
    g.selectAll("*").remove();
    layers=[3, ...Array(L).fill(wid), 2];
    const xstep=(W-80)/(layers.length-1);
    const nodes=[];
    layers.forEach((cnt,li)=>{ const yStep=(H-56)/(cnt+1); for(let i=0;i<cnt;i++) nodes.push({x:40+li*xstep, y:20+yStep*(i+1), li, i}); });
    for(let li=0;li<layers.length-1;li++){ const a=nodes.filter(n=>n.li===li), b=nodes.filter(n=>n.li===li+1);
      a.forEach(s=>b.forEach(t=>g.append("line").attr("class",`edge l${li}`).attr("x1",s.x).attr("y1",s.y).attr("x2",t.x).attr("y2",t.y).attr("stroke",C.line).attr("stroke-width",1))); }
    g.selectAll("circle").data(nodes).enter().append("circle").attr("class",d=>`node l${d.li}`).attr("cx",d=>d.x).attr("cy",d=>d.y).attr("r",11)
      .attr("fill",d=> d.li===0?C.A : d.li===layers.length-1?C.B : "#262b37").attr("stroke","#0f1117").attr("stroke-width",2);
    layers.forEach((cnt,li)=>{
      const cx=40+li*xstep;
      g.append("text").attr("x",cx).attr("y",H-18).attr("fill",C.muted).attr("font-size",10).attr("text-anchor","middle")
        .text(li===0?"input":li===layers.length-1?"output":"hidden "+li);
      g.append("text").attr("x",cx).attr("y",H-5).attr("fill",li===0?C.A:li===layers.length-1?C.B:C.muted).attr("font-size",10).attr("text-anchor","middle")
        .text(`${B}×${cnt}`);
    });
    shapeChain();
    return layers.length;
  }

  function fire(nLayers){
    if(busy) return; busy=true;
    for(let li=0;li<nLayers-1;li++){
      setTimeout(()=>{
        g.selectAll(`.edge.l${li}`).attr("stroke",C.good).attr("stroke-width",2).transition().duration(450).attr("stroke",C.line).attr("stroke-width",1);
        g.selectAll(`.node.l${li+1}`).transition().duration(200).attr("fill",C.good).attr("r",13).transition().duration(400).attr("r",11)
          .attr("fill", li+1===nLayers-1?C.B:"#262b37");
        const B=BATCHES[+d3.select("#nn-batch").property("value")];
        const inName = li===0 ? "X" : "h"+SUP[li-1];
        const opName = li+1===nLayers-1 ? "ŷ = "+inName+"W<sup>out</sup> + b" : "h"+SUP[li]+" = φ( "+inName+"W"+SUP[li]+" + b )";
        out.innerHTML = `<b>${opName}</b> &nbsp; (${B}×${layers[li]})·(${layers[li]}×${layers[li+1]}) → <b>${B}×${layers[li+1]}</b>`;
      }, li*520);
    }
    setTimeout(()=>{ busy=false; shapeChain(); }, (nLayers-1)*520+700);
  }

  let nL=build();
  d3.select("#nn-layers").on("input",()=>nL=build());
  d3.select("#nn-width").on("input",()=>nL=build());
  d3.select("#nn-batch").on("input",()=>nL=build());
  d3.select("#nn-fire").on("click",()=>fire(nL));
})();

/* ───────────────────────── 02 · activation explorer with derivative ───────────────────────── */
(function(){
  const svg=d3.select("#act-svg"), W=640, H=330, m={t:18,r:16,b:28,l:44};
  const sig=z=>1/(1+Math.exp(-z));
  const FN={
    sigmoid:{f:sig, d:z=>sig(z)*(1-sig(z)), label:"σ(z) = 1/(1 + e⁻ᶻ)", note:"range (0,1) · slope caps at 0.25 · not zero-centred"},
    tanh:{f:z=>Math.tanh(z), d:z=>1-Math.tanh(z)**2, label:"tanh(z)", note:"range (−1,1) · zero-centred · still saturates"},
    relu:{f:z=>Math.max(0,z), d:z=>z>0?1:0, label:"max(0, z)", note:"slope 1 on the right, 0 on the left → dead units possible"},
    leaky:{f:z=>z>0?z:0.1*z, d:z=>z>0?1:0.1, label:"z if z>0 else 0.1z", note:"keeps a trickle of gradient alive for z < 0"},
    gelu:{f:z=>0.5*z*(1+Math.tanh(Math.sqrt(2/Math.PI)*(z+0.044715*z*z*z))), d:null, label:"z·Φ(z)", note:"smooth · small negative lobe · transformer default"},
    silu:{f:z=>z*sig(z), d:null, label:"z·σ(z)", note:"smooth · self-gated · ≈ ReLU for large |z|"}
  };
  Object.values(FN).forEach(o=>{ if(!o.d){ const h=1e-4; o.d=z=>(o.f(z+h)-o.f(z-h))/(2*h); } });

  const x=d3.scaleLinear().domain([-6,6]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([-1.6,3.2]).range([H-m.b,m.t]);
  svg.append("defs").append("clipPath").attr("id","act-clip").append("rect")
     .attr("x",m.l).attr("y",m.t).attr("width",W-m.r-m.l).attr("height",H-m.b-m.t);
  const gGrid=svg.append("g"), gBand=svg.append("g"),
        gGhost=svg.append("g").attr("clip-path","url(#act-clip)"),
        gCurve=svg.append("g").attr("clip-path","url(#act-clip)"),
        gMark=svg.append("g").attr("clip-path","url(#act-clip)"), gLeg=svg.append("g");
  d3.range(-6,7,2).forEach(v=>{ gGrid.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke","#1b2130");
    gGrid.append("text").attr("x",x(v)).attr("y",H-10).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(v); });
  d3.range(-1,4,1).forEach(v=>{ gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke","#1b2130");
    gGrid.append("text").attr("x",m.l-8).attr("y",y(v)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(v); });
  gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(0)).attr("y2",y(0)).attr("stroke","#3a4150");
  gGrid.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke","#3a4150");

  const samples=d3.range(-6,6.001,0.05);
  const line=d3.line().x(d=>x(d[0])).y(d=>y(Math.max(-1.6,Math.min(3.2,d[1]))));
  Object.entries(FN).forEach(([k,o])=>gGhost.append("path").attr("class","ghost g-"+k)
    .attr("d",line(samples.map(z=>[z,o.f(z)]))).attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1).attr("stroke-opacity",.16));

  const pCurve=gCurve.append("path").attr("fill","none").attr("stroke",C.A).attr("stroke-width",2.5);
  const pDeriv=gCurve.append("path").attr("fill","none").attr("stroke",C.B).attr("stroke-width",2).attr("stroke-dasharray","5 4");
  const vline=gMark.append("line").attr("stroke",C.muted).attr("stroke-width",1).attr("stroke-dasharray","3 3");
  const tangent=gMark.append("line").attr("stroke",C.good).attr("stroke-width",1.8).attr("stroke-opacity",.9);
  const dotF=gMark.append("circle").attr("r",5.5).attr("fill",C.A).attr("stroke","#0f1117").attr("stroke-width",2);
  const dotD=gMark.append("circle").attr("r",4.5).attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",2);
  [["φ(z)",C.A,0],["φ′(z)",C.B,74],["tangent",C.good,148]].forEach(([t,c,dx])=>{
    gLeg.append("line").attr("x1",W-232+dx).attr("x2",W-214+dx).attr("y1",m.t+4).attr("y2",m.t+4).attr("stroke",c).attr("stroke-width",2.5);
    gLeg.append("text").attr("x",W-210+dx).attr("y",m.t+8).attr("font-size",10).attr("fill",C.muted).text(t);
  });

  const selEl=document.getElementById("act-fn"), xEl=document.getElementById("act-x"),
        xVal=document.getElementById("act-xval"), dEl=document.getElementById("act-deriv"),
        out=document.getElementById("act-readout");

  function render(){
    const key=selEl.value, o=FN[key], z0=+xEl.value, showD=dEl.checked;
    xVal.textContent=z0.toFixed(2);
    pCurve.attr("d",line(samples.map(z=>[z,o.f(z)])));
    pDeriv.attr("d",line(samples.map(z=>[z,o.d(z)]))).attr("display",showD?null:"none");
    gGhost.selectAll("path").attr("stroke-opacity",function(){ return d3.select(this).classed("g-"+key)?0:0.16; });
    gBand.selectAll("rect").remove();
    const step=x(0.05)-x(0);
    samples.forEach(z=>{ if(Math.abs(o.d(z))<0.1)
      gBand.append("rect").attr("x",x(z)).attr("y",m.t).attr("width",step+0.6).attr("height",H-m.b-m.t).attr("fill",C.bad).attr("fill-opacity",.07); });
    const fv=o.f(z0), dv=o.d(z0);
    vline.attr("x1",x(z0)).attr("x2",x(z0)).attr("y1",m.t).attr("y2",H-m.b);
    dotF.attr("cx",x(z0)).attr("cy",y(Math.max(-1.6,Math.min(3.2,fv))));
    dotD.attr("cx",x(z0)).attr("cy",y(dv)).attr("display",showD?null:"none");
    const h=1.3;
    tangent.attr("x1",x(z0-h)).attr("y1",y(fv-h*dv)).attr("x2",x(z0+h)).attr("y2",y(fv+h*dv));
    let verdict, col;
    if(Math.abs(dv)<0.02){ verdict="gradient ≈ 0 — this unit is saturated / dead here"; col=C.bad; }
    else if(Math.abs(dv)<0.25){ verdict="weak gradient — shrinks the signal every layer"; col=C.B; }
    else { verdict="healthy gradient — passes back near full strength"; col=C.good; }
    out.innerHTML=`<b>${o.label}</b> &nbsp;·&nbsp; φ(${z0.toFixed(2)}) = <b>${fv.toFixed(3)}</b> &nbsp;·&nbsp; φ′(${z0.toFixed(2)}) = <b style="color:${col}">${dv.toFixed(3)}</b>`
      + `<br>${verdict} &nbsp;·&nbsp; ${o.note}`;
  }
  selEl.addEventListener("change",render); xEl.addEventListener("input",render); dEl.addEventListener("change",render);
  render();
})();

/* ───────────────────────── 04 · universal approximation with K hinges ───────────────────────── */
(function(){
  const svg=d3.select("#ua-svg"), W=640, H=330, m={t:18,r:16,b:28,l:44};
  const TARGETS={
    sine:{f:x=>Math.sin(1.5*x), name:"sin(1.5x)"},
    bump:{f:x=>Math.exp(-1.2*x*x), name:"exp(−1.2x²)"},
    step:{f:x=>Math.tanh(4*x), name:"tanh(4x)"}
  };
  const XMIN=-3, XMAX=3, M=241;
  const xs=d3.range(M).map(i=>XMIN+(XMAX-XMIN)*i/(M-1));
  const x=d3.scaleLinear().domain([XMIN,XMAX]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([-1.6,1.6]).range([H-m.b,m.t]);

  svg.append("defs").append("clipPath").attr("id","ua-clip").append("rect")
     .attr("x",m.l).attr("y",m.t).attr("width",W-m.r-m.l).attr("height",H-m.b-m.t);
  const gGrid=svg.append("g"),
        gParts=svg.append("g").attr("clip-path","url(#ua-clip)"),
        gCurve=svg.append("g").attr("clip-path","url(#ua-clip)"),
        gKnots=svg.append("g"), gLeg=svg.append("g");
  d3.range(XMIN,XMAX+0.001,1).forEach(v=>{ gGrid.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke","#1b2130");
    gGrid.append("text").attr("x",x(v)).attr("y",H-10).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(v); });
  d3.range(-1.5,1.6,0.5).forEach(v=>{ gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke","#1b2130");
    gGrid.append("text").attr("x",m.l-8).attr("y",y(v)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(v.toFixed(1)); });
  gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(0)).attr("y2",y(0)).attr("stroke","#3a4150");
  [["target",C.muted,0],["network fit",C.A,64]].forEach(([t,c,dx])=>{
    gLeg.append("line").attr("x1",W-190+dx).attr("x2",W-172+dx).attr("y1",m.t+4).attr("y2",m.t+4).attr("stroke",c).attr("stroke-width",2.5);
    gLeg.append("text").attr("x",W-168+dx).attr("y",m.t+8).attr("font-size",10).attr("fill",C.muted).text(t);
  });

  const line=d3.line().x(d=>x(d[0])).y(d=>y(Math.max(-1.6,Math.min(1.6,d[1]))));
  const pTarget=gCurve.append("path").attr("fill","none").attr("stroke",C.muted).attr("stroke-width",2).attr("stroke-opacity",.75);
  const pFit=gCurve.append("path").attr("fill","none").attr("stroke",C.A).attr("stroke-width",2.5);

  // small dense solver: Gaussian elimination with partial pivoting
  function solve(A,b){
    const n=b.length;
    for(let i=0;i<n;i++){
      let piv=i;
      for(let r=i+1;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[piv][i])) piv=r;
      if(piv!==i){ const t=A[i]; A[i]=A[piv]; A[piv]=t; const tb=b[i]; b[i]=b[piv]; b[piv]=tb; }
      const d=A[i][i]; if(Math.abs(d)<1e-12) continue;
      for(let r=i+1;r<n;r++){ const f=A[r][i]/d; if(!f) continue;
        for(let c=i;c<n;c++) A[r][c]-=f*A[i][c]; b[r]-=f*b[i]; }
    }
    const out=new Array(n).fill(0);
    for(let i=n-1;i>=0;i--){ let s=b[i];
      for(let c=i+1;c<n;c++) s-=A[i][c]*out[c];
      out[i]=Math.abs(A[i][i])<1e-12?0:s/A[i][i]; }
    return out;
  }

  const unitsEl=document.getElementById("ua-units"), fnEl=document.getElementById("ua-fn"),
        partsEl=document.getElementById("ua-parts"), kEl=document.getElementById("ua-k"),
        out=document.getElementById("ua-readout");

  function render(){
    const K=+unitsEl.value, tgt=TARGETS[fnEl.value];
    kEl.textContent=K;
    // hidden layer: fixed kinks on an even grid → features [1, x, ReLU(x − tₖ)]
    const knots=d3.range(K).map(k=>XMIN+(XMAX-XMIN)*(k+1)/(K+1));
    const feat=xv=>[1,xv,...knots.map(t=>Math.max(0,xv-t))];
    const P=K+2;
    const A=Array.from({length:P},()=>new Array(P).fill(0)), rhs=new Array(P).fill(0);
    xs.forEach(xv=>{ const f=feat(xv), t=tgt.f(xv);
      for(let i=0;i<P;i++){ rhs[i]+=f[i]*t; for(let j=0;j<P;j++) A[i][j]+=f[i]*f[j]; } });
    for(let i=0;i<P;i++) A[i][i]+=1e-7;                       // tiny ridge for conditioning
    const w=solve(A,rhs);
    const fit=xv=>{ const f=feat(xv); let s=0; for(let i=0;i<P;i++) s+=w[i]*f[i]; return s; };

    pTarget.attr("d",line(xs.map(v=>[v,tgt.f(v)])));
    pFit.attr("d",line(xs.map(v=>[v,fit(v)])));

    gParts.selectAll("*").remove();
    if(partsEl.checked){
      knots.forEach((t,k)=>gParts.append("path")
        .attr("d",line(xs.map(v=>[v, w[k+2]*Math.max(0,v-t)])))
        .attr("fill","none").attr("stroke",C.B).attr("stroke-width",1).attr("stroke-opacity",.45));
      gParts.append("path").attr("d",line(xs.map(v=>[v, w[0]+w[1]*v])))
        .attr("fill","none").attr("stroke",C.good).attr("stroke-width",1).attr("stroke-opacity",.45);
    }
    gKnots.selectAll("*").remove();
    knots.forEach(t=>gKnots.append("circle").attr("cx",x(t)).attr("cy",y(0)).attr("r",3)
      .attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",1));

    let se=0, mx=0;
    xs.forEach(v=>{ const e=fit(v)-tgt.f(v); se+=e*e; mx=Math.max(mx,Math.abs(e)); });
    const rmse=Math.sqrt(se/xs.length);
    const col = rmse<0.02 ? C.good : rmse<0.08 ? C.B : C.bad;
    out.innerHTML=`target <b>${tgt.name}</b> &nbsp;·&nbsp; <b>${K}</b> hidden unit${K>1?"s":""} (${3*K+1} parameters in the equivalent net)`
      + ` &nbsp;·&nbsp; RMSE <b style="color:${col}">${rmse.toFixed(4)}</b> &nbsp;·&nbsp; max error <b>${mx.toFixed(3)}</b>`
      + `<br>kinks are fixed on a grid and the output layer is solved exactly — training would learn the kink positions too.`;
  }
  unitsEl.addEventListener("input",render); fnEl.addEventListener("change",render); partsEl.addEventListener("change",render);
  render();
})();

/* ───────────────────────── 09 · gradient flow through depth ───────────────────────── */
(function(){
  const svg=d3.select("#gf-svg"), W=640, H=300, m={t:20,r:16,b:34,l:52};
  // illustrative per-layer factor: expected |φ′| contribution × weight-scale gain
  const SLOPE={sigmoid:0.25, tanh:0.72, relu:0.71};       // ReLU: ~half the units pass, √0.5 in norm
  const GAIN ={small:0.5, xavier:1.0, he:1.41, large:1.6};
  const ACTNAME={sigmoid:"sigmoid", tanh:"tanh", relu:"ReLU"};
  const INITNAME={small:"gain 0.5", xavier:"Xavier", he:"He", large:"gain 1.6"};

  const gGrid=svg.append("g"), gBars=svg.append("g"), gAx=svg.append("g");
  const actEl=document.getElementById("gf-act"), initEl=document.getElementById("gf-init"),
        depthEl=document.getElementById("gf-depth"), dVal=document.getElementById("gf-dval"),
        resEl=document.getElementById("gf-res"), out=document.getElementById("gf-readout");

  function render(){
    const L=+depthEl.value, act=actEl.value, ini=initEl.value, res=resEl.checked;
    dVal.textContent=L;
    const f = res ? 1.0 : SLOPE[act]*GAIN[ini];
    // relative gradient reaching layer ℓ, counting from the output (layer L) backwards
    const vals=d3.range(1,L+1).map(l=>Math.pow(f, L-l));
    const logs=vals.map(v=>Math.log10(Math.max(v,1e-30)));
    const lo=Math.min(-1,d3.min(logs)), hi=Math.max(1,d3.max(logs));
    const x=d3.scaleLinear().domain([0.5,L+0.5]).range([m.l,W-m.r]);
    const y=d3.scaleLinear().domain([lo,hi]).range([H-m.b,m.t]);

    gGrid.selectAll("*").remove(); gAx.selectAll("*").remove(); gBars.selectAll("*").remove();
    const ticks=y.ticks(6);
    ticks.forEach(t=>{ gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(t)).attr("y2",y(t)).attr("stroke","#1b2130");
      gGrid.append("text").attr("x",m.l-8).attr("y",y(t)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted)
        .text(t===0?"1":"10"+String(Math.round(t)).replace(/-/g,"⁻").replace(/\d/g,d=>"⁰¹²³⁴⁵⁶⁷⁸⁹"[d])); });
    const zeroY=y(Math.max(lo,Math.min(hi,0)));
    gGrid.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",zeroY).attr("y2",zeroY).attr("stroke","#3a4150");

    const bw=Math.max(2,(W-m.r-m.l)/L-2);
    vals.forEach((v,i)=>{
      const l=i+1, lg=logs[i];
      const col = Math.abs(lg)<2 ? C.good : Math.abs(lg)<6 ? C.B : C.bad;
      const top=Math.min(y(lg),zeroY), h=Math.abs(y(lg)-zeroY);
      gBars.append("rect").attr("x",x(l)-bw/2).attr("y",top).attr("width",bw).attr("height",Math.max(1,h))
        .attr("fill",col).attr("fill-opacity",.75).attr("stroke",col).attr("stroke-opacity",.9)
        .append("title").text(`layer ${l}: ${v.toExponential(2)}× the output-layer gradient`);
    });
    gAx.append("text").attr("x",m.l).attr("y",H-10).attr("font-size",10).attr("fill",C.muted).text("layer 1 (input side)");
    gAx.append("text").attr("x",W-m.r).attr("y",H-10).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text("layer "+L+" (loss side)");

    const first=vals[0];
    let verdict, col;
    if(res){ verdict="the identity path keeps the per-layer factor at ≈1 — depth costs nothing"; col=C.good; }
    else if(first<1e-6){ verdict="vanishing — the first layers receive essentially no signal"; col=C.bad; }
    else if(first>1e6){ verdict="exploding — expect NaNs within a few steps"; col=C.bad; }
    else if(first<1e-2||first>1e2){ verdict="drifting — trainable, but the layers learn at very different rates"; col=C.B; }
    else { verdict="healthy — gradients stay within a couple of orders of magnitude"; col=C.good; }
    out.innerHTML=`${ACTNAME[act]} + ${INITNAME[ini]}${res?" + residuals":""} &nbsp;·&nbsp; per-layer factor <b>${f.toFixed(2)}</b>`
      + ` &nbsp;·&nbsp; gradient at layer 1 = <b style="color:${col}">${first.toExponential(2)}×</b> that at layer ${L}`
      + `<br>${verdict}`;
  }
  actEl.addEventListener("change",render); initEl.addEventListener("change",render);
  depthEl.addEventListener("input",render); resEl.addEventListener("change",render);
  render();
})();
