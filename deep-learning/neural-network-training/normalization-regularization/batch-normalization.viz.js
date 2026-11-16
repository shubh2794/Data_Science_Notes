/* batch-normalization.viz.js — interactive figures for
   deep-learning/neural-network-training/normalization-regularization/batch-normalization.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-ics  a live simulation of the MNIST-style experiment: two 3×100 sigmoid nets, with and
               without BN, trained side by side; held-out accuracy and the 15/50/85th percentiles
               of one sigmoid input over training
     #viz-sc   scale invariance: BN(Wu) = BN((aW)u), measured by real forward/backward passes
     #viz-res  the reported single-network ImageNet numbers (steps to 72.2%, max accuracy, LR) */

(function(){
  "use strict";
  const C={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#171a23",violet:"#a78bfa",teal:"#2dd4bf"};
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
  function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  const txt=(g,x,y,s,o)=>{const t=g.append("text").attr("x",x).attr("y",y).attr("fill",(o&&o.fill)||C.muted).attr("font-size",(o&&o.size)||11).text(s);if(o&&o.anchor)t.attr("text-anchor",o.anchor);return t;};

  /* ═══════════════ 1. the activations-over-time experiment, simulated ═══════════════ */
  (function(){
    const svg=d3.select("#ics-svg"); if(svg.empty()) return;
    const D=49,K=10,HID=100,M=60,EPS=1e-5,FLIP=0.25,NPROBE=200,PROBE_EVERY=25,MAXSTEP=1500,UNIT=0;

    // synthetic data: ten 7×7 binary prototypes, each pixel flipped with probability FLIP
    const RP=rng(3), P=[];for(let c=0;c<K;c++){const p=new Float64Array(D);for(let i=0;i<D;i++)p[i]=RP()<0.35?1:0;P.push(p);}
    function sample(R,n){const X=new Float64Array(n*D),Y=new Int32Array(n);
      for(let s=0;s<n;s++){const c=Math.floor(R()*K);Y[s]=c;for(let i=0;i<D;i++){const f=R()<FLIP;X[s*D+i]=f?1-P[c][i]:P[c][i];}}return {X,Y};}
    const probe=sample(rng(99),NPROBE);

    function makeNet(bn,std){const R=rng(1);const sz=[D,HID,HID,HID,K];const L=[];   // same seed: both nets start from identical W
      for(let l=0;l<4;l++){const nin=sz[l],nout=sz[l+1];const W=new Float64Array(nout*nin);for(let i=0;i<W.length;i++)W[i]=gauss(R)*std;
        L.push({nin,nout,W,b:new Float64Array(nout),g:new Float64Array(nout).fill(1),be:new Float64Array(nout),rm:new Float64Array(nout),rv:new Float64Array(nout).fill(1)});}
      return {L,bn};}
    function forward(net,X,n,train){const Cs=[];let A=X;
      for(let l=0;l<4;l++){const Ly=net.L[l],nin=Ly.nin,nout=Ly.nout,W=Ly.W,Z=new Float64Array(n*nout);
        for(let s=0;s<n;s++){const ao=s*nin,zo=s*nout;for(let o=0;o<nout;o++){let z=0;const wo=o*nin;for(let i=0;i<nin;i++)z+=W[wo+i]*A[ao+i];Z[zo+o]=z;}}
        const c={A,Z};
        if(l<3){const Yp=new Float64Array(n*nout);
          if(net.bn){const mu=new Float64Array(nout),va=new Float64Array(nout),xh=new Float64Array(n*nout);
            if(train){for(let s=0;s<n;s++)for(let o=0;o<nout;o++)mu[o]+=Z[s*nout+o];for(let o=0;o<nout;o++)mu[o]/=n;
              for(let s=0;s<n;s++)for(let o=0;o<nout;o++){const d=Z[s*nout+o]-mu[o];va[o]+=d*d;}for(let o=0;o<nout;o++)va[o]/=n;   // biased, 1/m
              for(let o=0;o<nout;o++){Ly.rm[o]=0.9*Ly.rm[o]+0.1*mu[o];Ly.rv[o]=0.9*Ly.rv[o]+0.1*va[o]*n/(n-1);}}               // population: m/(m−1)
            else{mu.set(Ly.rm);va.set(Ly.rv);}
            for(let s=0;s<n;s++)for(let o=0;o<nout;o++){const k=s*nout+o;xh[k]=(Z[k]-mu[o])/Math.sqrt(va[o]+EPS);Yp[k]=Ly.g[o]*xh[k]+Ly.be[o];}
            c.mu=mu;c.va=va;c.xh=xh;}
          else{for(let s=0;s<n;s++)for(let o=0;o<nout;o++)Yp[s*nout+o]=Z[s*nout+o]+Ly.b[o];}
          const H=new Float64Array(n*nout);for(let k=0;k<H.length;k++)H[k]=1/(1+Math.exp(-Yp[k]));c.Yp=Yp;c.H=H;A=H;}
        else{for(let s=0;s<n;s++)for(let o=0;o<nout;o++)Z[s*nout+o]+=Ly.b[o];c.out=Z;}
        Cs.push(c);}
      return Cs;}
    function sgd(net,X,Y,n,lr){const Cs=forward(net,X,n,true),out=Cs[3].out;let dA=new Float64Array(n*K);
      for(let s=0;s<n;s++){let mx=-1e9;for(let o=0;o<K;o++)mx=Math.max(mx,out[s*K+o]);let se=0;for(let o=0;o<K;o++)se+=Math.exp(out[s*K+o]-mx);
        for(let o=0;o<K;o++)dA[s*K+o]=(Math.exp(out[s*K+o]-mx)/se-(o===Y[s]?1:0))/n;}
      for(let l=3;l>=0;l--){const Ly=net.L[l],c=Cs[l],nin=Ly.nin,nout=Ly.nout;let dZ;
        if(l===3){dZ=dA;for(let o=0;o<nout;o++){let t=0;for(let s=0;s<n;s++)t+=dZ[s*nout+o];Ly.b[o]-=lr*t;}}
        else{const dY=new Float64Array(n*nout);for(let k=0;k<dY.length;k++)dY[k]=dA[k]*c.H[k]*(1-c.H[k]);
          if(net.bn){dZ=new Float64Array(n*nout);
            for(let o=0;o<nout;o++){const is=1/Math.sqrt(c.va[o]+EPS);let dg=0,db=0,dvar=0,dmu=0,sxm=0;
              // the original study's chain rule, term by term (§04 of the page)
              for(let s=0;s<n;s++){const k=s*nout+o,dxh=dY[k]*Ly.g[o],xm=c.Z[k]-c.mu[o];dg+=dY[k]*c.xh[k];db+=dY[k];dvar+=dxh*xm*(-0.5)*is*is*is;dmu+=-dxh*is;sxm+=-2*xm;}
              dmu+=dvar*sxm/n;
              for(let s=0;s<n;s++){const k=s*nout+o;dZ[k]=dY[k]*Ly.g[o]*is+dvar*2*(c.Z[k]-c.mu[o])/n+dmu/n;}
              Ly.g[o]-=lr*dg;Ly.be[o]-=lr*db;}}
          else{dZ=dY;for(let o=0;o<nout;o++){let t=0;for(let s=0;s<n;s++)t+=dZ[s*nout+o];Ly.b[o]-=lr*t;}}}
        const A=c.A,W=Ly.W,dAp=l>0?new Float64Array(n*nin):null,dW=new Float64Array(nout*nin);
        for(let s=0;s<n;s++){const ao=s*nin,zo=s*nout;for(let o=0;o<nout;o++){const g=dZ[zo+o];if(g===0)continue;const wo=o*nin;
          if(dAp)for(let i=0;i<nin;i++)dAp[ao+i]+=g*W[wo+i];for(let i=0;i<nin;i++)dW[wo+i]+=g*A[ao+i];}}
        for(let i=0;i<W.length;i++)W[i]-=lr*dW[i];dA=dAp;}}
    function measure(net,layer){const Cs=forward(net,probe.X,NPROBE,false),o=Cs[3].out;let ok=0;
      for(let s=0;s<NPROBE;s++){let b=0;for(let k=1;k<K;k++)if(o[s*K+k]>o[s*K+b])b=k;if(b===probe.Y[s])ok++;}
      const v=[];for(let s=0;s<NPROBE;s++)v.push(Cs[layer].Yp[s*HID+UNIT]);v.sort((a,b)=>a-b);
      return {acc:ok/NPROBE,p15:d3.quantileSorted(v,0.15),p50:d3.quantileSorted(v,0.5),p85:d3.quantileSorted(v,0.85)};}

    // ── layout: (a) accuracy, (b) without BN, (c) with BN ──
    const H=360, y0=40, y1=310, pan=[[52,232],[300,500],[560,750]];
    const xs=pan.map(p=>d3.scaleLinear().domain([0,MAXSTEP]).range(p));
    const ya=d3.scaleLinear().domain([0,1]).range([y1,y0]);
    const yp=d3.scaleLinear().domain([-4,4]).range([y1,y0]).clamp(true);
    const titles=["(a) held-out accuracy vs step","(b) without BN: one sigmoid input","(c) with BN: one sigmoid input"];
    pan.forEach((p,i)=>{
      svg.append("g").attr("transform",`translate(0,${y1})`).call(d3.axisBottom(xs[i]).ticks(3).tickFormat(d3.format("~s"))).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${p[0]},0)`).call(d3.axisLeft(i===0?ya:yp).ticks(i===0?5:8)).attr("color",C.muted);
      txt(svg,p[0],y0-16,titles[i],{fill:C.ink});
      txt(svg,(p[0]+p[1])/2,y1+32,"training step",{anchor:"middle",size:10.5});
      if(i>0){svg.append("line").attr("x1",p[0]).attr("x2",p[1]).attr("y1",yp(0)).attr("y2",yp(0)).attr("stroke",C.line);}});
    txt(svg,pan[1][0],H-6,"band = 15th–85th percentile over 200 held-out inputs, line = median; axis clipped to ±4",{size:10.5});
    const accLine=[svg.append("path").attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2),
                   svg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2)];
    txt(svg,pan[0][0]+8,y1-26,"without BN",{fill:C.a2});txt(svg,pan[0][0]+8,y1-12,"with BN",{fill:C.accent});
    const bands=[1,2].map(i=>({area:svg.append("path").attr("fill",i===1?C.a2:C.accent).attr("opacity",0.25),
      med:svg.append("path").attr("fill","none").attr("stroke",i===1?C.a2:C.accent).attr("stroke-width",1.8),
      lo:svg.append("path").attr("fill","none").attr("stroke",i===1?C.a2:C.accent).attr("stroke-width",0.8).attr("opacity",0.7),
      hi:svg.append("path").attr("fill","none").attr("stroke",i===1?C.a2:C.accent).attr("stroke-width",0.8).attr("opacity",0.7)}));

    let nets,hist,step,timer=null,R;
    function cfg(){return {lr:+d3.select("#ics-lr").property("value"),std:+d3.select("#ics-std").property("value"),layer:+d3.select("#ics-layer").property("value")};}
    function reset(){stop();const c=cfg();nets=[makeNet(false,c.std),makeNet(true,c.std)];R=rng(5);step=0;hist=[];record();render();}
    function record(){const c=cfg();hist.push({s:step,m:nets.map(n=>measure(n,c.layer))});}
    function render(){
      accLine.forEach((p,i)=>p.attr("d",d3.line().x(d=>xs[0](d.s)).y(d=>ya(d.m[i].acc))(hist)));
      bands.forEach((b,i)=>{const x=xs[i+1];
        b.area.attr("d",d3.area().x(d=>x(d.s)).y0(d=>yp(d.m[i].p15)).y1(d=>yp(d.m[i].p85))(hist));
        b.med.attr("d",d3.line().x(d=>x(d.s)).y(d=>yp(d.m[i].p50))(hist));
        b.lo.attr("d",d3.line().x(d=>x(d.s)).y(d=>yp(d.m[i].p15))(hist));
        b.hi.attr("d",d3.line().x(d=>x(d.s)).y(d=>yp(d.m[i].p85))(hist));});
      const h=hist[hist.length-1],f=v=>v.toFixed(2);
      const drift=i=>{const a=hist[Math.min(hist.length-1,2)].m[i];return Math.abs(h.m[i].p50-a.p50).toFixed(2);};
      d3.select("#ics-readout").html(`step <b>${step}</b> / ${MAXSTEP} · held-out accuracy: without BN <b>${f(h.m[0].acc)}</b>, with BN <b>${f(h.m[1].acc)}</b>`+
        `<br>sigmoid input of unit 1, hidden layer ${cfg().layer+1} — 15/50/85th percentiles: without BN <b>${f(h.m[0].p15)} / ${f(h.m[0].p50)} / ${f(h.m[0].p85)}</b>, with BN <b>${f(h.m[1].p15)} / ${f(h.m[1].p50)} / ${f(h.m[1].p85)}</b>`+
        (hist.length>3?`<br>median drift since step ${hist[2].s}: without BN <b>${drift(0)}</b>, with BN <b>${drift(1)}</b>`:"")+
        (step===0?" · both nets start from the same weights":""));
    }
    function stop(){if(timer){timer.stop();timer=null;}d3.select("#ics-play").text(step>=MAXSTEP?"done — reset to rerun":"train");}
    function play(){if(timer){stop();return;}if(step>=MAXSTEP)reset();d3.select("#ics-play").text("pause");const lr=cfg().lr;
      timer=d3.timer(()=>{const t0=performance.now();
        while(performance.now()-t0<22&&step<MAXSTEP){const b=sample(R,M);sgd(nets[0],b.X,b.Y,M,lr);sgd(nets[1],b.X,b.Y,M,lr);step++;if(step%PROBE_EVERY===0){record();break;}}
        render();if(step>=MAXSTEP)stop();});}
    d3.select("#ics-play").on("click",play);
    d3.select("#ics-reset").on("click",reset);
    ["#ics-lr","#ics-std","#ics-layer"].forEach(id=>d3.select(id).on("change",reset));
    reset();
  })();

  /* ═══════════════ 2. scale invariance, measured ═══════════════ */
  (function(){
    const svg=d3.select("#sc-svg"); if(svg.empty()) return;
    const m=32,n=8,k=6, R=rng(21);
    const U=d3.range(m).map(()=>d3.range(n).map(()=>gauss(R)));
    const W0=d3.range(k).map(()=>d3.range(n).map(()=>gauss(R)/Math.sqrt(n)));
    const T=d3.range(m).map(()=>d3.range(k).map(()=>gauss(R)));
    // loss ℓ = ½ ∑ (y − t)², y = BN(aW u) with γ = 1, β = 0, or y = aW u without BN
    function run(a,bn,eps){
      const X=U.map(u=>W0.map(w=>a*d3.sum(w,(wi,i)=>wi*u[i])));
      let Y,gX;
      if(bn){const mu=d3.range(k).map(j=>d3.mean(X,x=>x[j])),va=d3.range(k).map(j=>d3.mean(X,x=>(x[j]-mu[j])**2));
        const is=va.map(v=>1/Math.sqrt(v+eps));
        Y=X.map(x=>x.map((v,j)=>(v-mu[j])*is[j]));
        const gY=Y.map((y,s)=>y.map((v,j)=>v-T[s][j]));
        gX=X.map(()=>new Array(k).fill(0));
        for(let j=0;j<k;j++){let dvar=0,dmu=0,sxm=0;
          for(let s=0;s<m;s++){const xm=X[s][j]-mu[j];dvar+=gY[s][j]*xm*(-0.5)*is[j]**3;dmu+=-gY[s][j]*is[j];sxm+=-2*xm;}
          dmu+=dvar*sxm/m;
          for(let s=0;s<m;s++)gX[s][j]=gY[s][j]*is[j]+dvar*2*(X[s][j]-mu[j])/m+dmu/m;}}
      else{Y=X;gX=Y.map((y,s)=>y.map((v,j)=>v-T[s][j]));}
      // ∂ℓ/∂u = (aW)ᵀ gx ; ∂ℓ/∂(aW) = ∑ gx uᵀ
      let gu=0,gw=0,yn=0;
      for(let s=0;s<m;s++)for(let i=0;i<n;i++){let t=0;for(let j=0;j<k;j++)t+=a*W0[j][i]*gX[s][j];gu+=t*t;}
      for(let j=0;j<k;j++)for(let i=0;i<n;i++){let t=0;for(let s=0;s<m;s++)t+=gX[s][j]*U[s][i];gw+=t*t;}
      for(let s=0;s<m;s++)for(let j=0;j<k;j++)yn+=Y[s][j]**2;
      const wn=a*Math.sqrt(d3.sum(W0,w=>d3.sum(w,v=>v*v)));
      return {Y,yn:Math.sqrt(yn),gu:Math.sqrt(gu),gw:Math.sqrt(gw),rel:Math.sqrt(gw)/wn};}
    const sig=(()=>{const X=U.map(u=>W0.map(w=>d3.sum(w,(wi,i)=>wi*u[i])));return d3.mean(d3.range(k),j=>d3.deviation(X,x=>x[j]));})();

    const W=760,H=330,x0=60,x1=470,y0=30,y1=280;
    const xs=d3.scaleLog().domain([0.01,100]).range([x0,x1]);
    const ys=d3.scaleLog().domain([1e-5,1e5]).range([y1,y0]).clamp(true);
    svg.append("g").attr("transform",`translate(0,${y1})`).call(d3.axisBottom(xs).ticks(5,"~g")).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(ys).ticks(6,"~g")).attr("color",C.muted);
    txt(svg,x0,y0-12,"ratio to the value at a = 1, vs weight scale a (log–log)",{fill:C.ink});
    txt(svg,(x0+x1)/2,y1+34,"weight scale a",{anchor:"middle",size:10.5});
    const epsRule=svg.append("line").attr("y1",y0).attr("y2",y1).attr("stroke",C.bad).attr("stroke-dasharray","2 3");
    const epsTxt=txt(svg,0,y1-6,"",{fill:C.bad,size:10});
    const refs=svg.append("g");
    const series=[{key:"yn",name:"output ‖y‖",col:C.teal},{key:"gu",name:"‖∂ℓ/∂u‖",col:C.violet},{key:"gw",name:"‖∂ℓ/∂(aW)‖",col:C.a2},{key:"rel",name:"relative step ‖η∂ℓ/∂(aW)‖ / ‖aW‖",col:C.bad}];
    const paths=series.map(s=>svg.append("path").attr("fill","none").attr("stroke",s.col).attr("stroke-width",2));
    const dots=series.map(s=>svg.append("circle").attr("r",4).attr("fill",s.col));
    const marker=svg.append("line").attr("y1",y0).attr("y2",y1).attr("stroke",C.ink).attr("stroke-dasharray","4 3");
    series.forEach((s,i)=>{svg.append("rect").attr("x",x1+22).attr("y",y0+4+i*18).attr("width",12).attr("height",3).attr("fill",s.col);
      txt(svg,x1+40,y0+9+i*18,s.name,{size:10.5});});
    // right: one unit's outputs over the batch, at a = 1 (outline) and at the current a (filled)
    const bx0=x1+24,bx1=W-14,by0=150,by1=300;
    txt(svg,bx0,by0-12,"unit 1's output for the 32 batch items",{fill:C.ink,size:10.5});
    const bxs=d3.scaleBand().domain(d3.range(m)).range([bx0,bx1]).padding(0.15);
    const bys=d3.scaleLinear().domain([-3,3]).range([by1,by0]).clamp(true);
    svg.append("line").attr("x1",bx0).attr("x2",bx1).attr("y1",bys(0)).attr("y2",bys(0)).attr("stroke",C.line);
    const barsNow=svg.append("g"),barsRef=svg.append("g");
    txt(svg,bx0,by1+16,"outline: a = 1 · filled: current a · axis clipped to ±3",{size:10});

    function draw(){
      const v=+d3.select("#sc-a").property("value"),a=Math.pow(10,v),bn=d3.select("#sc-bn").property("checked"),eps=+d3.select("#sc-eps").property("value");
      d3.select("#sc-av").text(a<1?a.toPrecision(2):a.toFixed(a<10?2:1));
      const grid=d3.range(-2,2.0001,0.05).map(t=>Math.pow(10,t));
      const ref=run(1,bn,eps),rs=grid.map(g=>run(g,bn,eps)),cur=run(a,bn,eps);
      series.forEach((s,i)=>{paths[i].attr("d",d3.line().x((d,j)=>xs(grid[j])).y(d=>ys(Math.max(1e-5,d[s.key]/ref[s.key])))(rs));
        dots[i].attr("cx",xs(a)).attr("cy",ys(Math.max(1e-5,cur[s.key]/ref[s.key])));});
      marker.attr("x1",xs(a)).attr("x2",xs(a));
      refs.selectAll("*").remove();
      if(bn){[[0,"1"],[-1,"1/a"],[-2,"1/a²"]].forEach(([p,l])=>{
        refs.append("line").attr("x1",xs(0.01)).attr("x2",xs(100)).attr("y1",ys(Math.pow(0.01,p))).attr("y2",ys(Math.pow(100,p))).attr("stroke",C.muted).attr("stroke-dasharray","1 4").attr("opacity",0.8);
        txt(refs,xs(60),ys(Math.pow(60,p))-5,l,{size:10,anchor:"end"});});
        const aEps=Math.sqrt(eps)/sig; epsRule.attr("x1",xs(aEps)).attr("x2",xs(aEps)).attr("opacity",aEps>0.01?1:0);
        epsTxt.attr("x",xs(aEps)+4).text(aEps>0.01?"a·σ = √ε":"");}
      else{epsRule.attr("opacity",0);epsTxt.text("");}
      const y1v=ref.Y.map(r=>r[0]),ynv=cur.Y.map(r=>r[0]);
      barsRef.selectAll("rect").data(y1v).join("rect").attr("x",(d,i)=>bxs(i)).attr("width",bxs.bandwidth())
        .attr("y",d=>Math.min(bys(d),bys(0))).attr("height",d=>Math.abs(bys(d)-bys(0))).attr("fill","none").attr("stroke",C.ink).attr("stroke-width",0.8);
      barsNow.selectAll("rect").data(ynv).join("rect").attr("x",(d,i)=>bxs(i)+1).attr("width",Math.max(1,bxs.bandwidth()-2))
        .attr("y",d=>Math.min(bys(d),bys(0))).attr("height",d=>Math.abs(bys(d)-bys(0))).attr("fill",C.teal).attr("opacity",0.75);
      const dmax=d3.max(d3.range(m),s=>d3.max(d3.range(k),j=>Math.abs(cur.Y[s][j]-ref.Y[s][j])));
      const e=x=>x.toExponential(2);
      d3.select("#sc-readout").html(`a = <b>${a.toPrecision(3)}</b> · BN ${bn?"on":"off"}`+(bn?` · ε = ${eps}`:"")+
        ` · max |y(a) − y(1)| over the batch = <b>${e(dmax)}</b>`+
        `<br>ratios to a = 1: ‖y‖ <b>${(cur.yn/ref.yn).toPrecision(4)}</b> · ‖∂ℓ/∂u‖ <b>${(cur.gu/ref.gu).toPrecision(4)}</b> · ‖∂ℓ/∂(aW)‖ <b>${(cur.gw/ref.gw).toPrecision(4)}</b> (1/a = ${(1/a).toPrecision(4)}) · relative step <b>${(cur.rel/ref.rel).toPrecision(4)}</b> (1/a² = ${(1/(a*a)).toPrecision(4)})`);
    }
    d3.select("#sc-a").on("input",draw);d3.select("#sc-bn").on("change",draw);d3.select("#sc-eps").on("change",draw);
    draw();
  })();

  /* ═══════════════ 3. reported single-network ImageNet results ═══════════════ */
  (function(){
    const svg=d3.select("#res-svg"); if(svg.empty()) return;
    // the original study's Figure 3 and §4.2.2; null = never reached 72.2%
    const rows=[{m:"Inception",steps:31.0,acc:72.2,lr:0.0015},{m:"BN-Baseline",steps:13.3,acc:72.7,lr:0.0015},
      {m:"BN-x5",steps:2.1,acc:73.0,lr:0.0075},{m:"BN-x30",steps:2.7,acc:74.8,lr:0.045},{m:"BN-x5-Sigmoid",steps:null,acc:69.8,lr:0.0075}];
    const W=760,H=270,x0=130,x1=650,y0=34,y1=230;
    const ys=d3.scaleBand().domain(rows.map(r=>r.m)).range([y0,y1]).padding(0.25);
    function draw(){
      svg.selectAll("*").remove();
      const v=d3.select("#res-v").property("value");
      let xs,title,val,lab;
      if(v==="steps"){xs=d3.scaleLinear().domain([0,32]).range([x0,x1]);title="training steps to reach 72.2% single-crop top-1 (millions)";val=r=>r.steps;lab=r=>r.steps==null?"never reaches 72.2%":r.steps.toFixed(1)+"M"+(r.m==="Inception"?"":"  ("+(31/r.steps).toFixed(1)+"× fewer)");}
      else if(v==="acc"){xs=d3.scaleLinear().domain([68,75.5]).range([x0,x1]);title="maximum validation accuracy (%) — axis starts at 68, not 0";val=r=>r.acc;lab=r=>r.acc.toFixed(1)+"%";}
      else{xs=d3.scaleLog().domain([0.001,0.06]).range([x0,x1]);title="initial learning rate (log axis)";val=r=>r.lr;lab=r=>r.lr+"  ("+(r.lr/0.0015).toFixed(0)+"× Inception)";}
      txt(svg,x0,y0-14,title,{fill:C.ink});
      svg.append("g").attr("transform",`translate(0,${y1})`).call(d3.axisBottom(xs).ticks(6,v==="lr"?"~g":undefined)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(ys)).attr("color",C.muted);
      rows.forEach(r=>{const y=ys(r.m),h=ys.bandwidth(),vv=val(r),col=r.m==="Inception"?C.muted:(r.m.indexOf("Sigmoid")>0?C.violet:C.accent);
        if(vv!=null){svg.append("rect").attr("x",x0).attr("y",y).attr("width",Math.max(1,xs(vv)-x0)).attr("height",h).attr("fill",col).attr("opacity",0.8);}
        else{svg.append("rect").attr("x",x0).attr("y",y).attr("width",x1-x0).attr("height",h).attr("fill","none").attr("stroke",C.violet).attr("stroke-dasharray","4 3");}
        txt(svg,(vv!=null?xs(vv):x0)+6,y+h/2+4,lab(r),{fill:C.ink,size:10.5});});
      if(v==="acc"){svg.append("line").attr("x1",xs(72.2)).attr("x2",xs(72.2)).attr("y1",y0).attr("y2",y1).attr("stroke",C.a2).attr("stroke-dasharray","4 3");
        txt(svg,xs(72.2)+4,y1-4,"72.2% (Inception's best)",{fill:C.a2,size:10});}
      const ro={steps:"BN-x5 reaches Inception's best in 2.1M steps instead of 31.0M — 31.0 / 2.1 ≈ 14.8 (this page's division; the study rounds to 14×, and to 7% of the steps). BN-x30 is slower to 72.2% (2.7M) but ends higher.",
        acc:"Every ReLU variant beats Inception's 72.2%; BN-x30's 74.8% is +2.6 points. BN-x5-Sigmoid reaches 69.8%, where Inception with sigmoid stayed at chance (no better than 1/1000).",
        lr:"BN-x5 and BN-x30 multiply the initial rate by 5 and 30. The same ×5 increase applied to Inception without BN drove its parameters to machine infinity."};
      d3.select("#res-readout").html(ro[v]);
    }
    d3.select("#res-v").on("change",draw);draw();
  })();
})();
