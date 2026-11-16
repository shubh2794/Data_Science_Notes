/* dropout.viz.js — interactive figures for
   deep-learning/neural-network-training/normalization-regularization/dropout.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-mc  live simulation: thinned networks sampled at retention p on a tiny random net,
              Monte-Carlo average vs the exact 2ⁿ ensemble vs test-time weight scaling
     #viz-rt  the original study's reported tables, with and without dropout */

(function(){
  "use strict";
  const C={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",
           line:"#2a2f3a",panel:"#171a23",teal:"#2dd4bf",violet:"#a78bfa"};

  /* ═══════════════ 1. thinned networks, Monte-Carlo vs weight scaling ═══════════════ */
  (function(){
    const svg=d3.select("#mc-svg"); if(svg.empty()) return;
    const NI=3, H1=6, H2=5, NH=H1+H2, KMAX0=200;
    const X=[1.0,-0.6,0.8];                                     // one fixed input case
    let seed=5, W1,b1,W2,b2,w3,b3, p, act, exact, samples=[], lastMask=null, showTest=false;

    function rng(s0){let s=s0>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    const sig=z=>1/(1+Math.exp(-z));
    const F={relu:z=>Math.max(0,z), tanh:z=>Math.tanh(z), linear:z=>z};

    function build(){
      const R=rng(seed);
      const mat=(o,i,g)=>d3.range(o).map(()=>d3.range(i).map(()=>gauss(R)*g/Math.sqrt(i)));
      W1=mat(H1,NI,1.6); b1=d3.range(H1).map(()=>0.3*gauss(R));
      W2=mat(H2,H1,1.8); b2=d3.range(H2).map(()=>0.3*gauss(R));
      w3=mat(1,H2,2.6)[0]; b3=0.2*gauss(R);
    }
    // m1, m2: per-unit multipliers on the hidden outputs (0/1 for a thinned net, p for the test net)
    function forward(m1,m2){
      const f=F[act];
      const h1=W1.map((row,j)=>f(d3.sum(row,(w,i)=>w*X[i])+b1[j])*m1[j]);
      const h2=W2.map((row,j)=>f(d3.sum(row,(w,i)=>w*h1[i])+b2[j])*m2[j]);
      return d3.sum(w3,(w,j)=>w*h2[j])+b3;                       // logit z
    }
    function maskOf(bits){return [d3.range(H1).map(j=>(bits>>j)&1), d3.range(H2).map(j=>(bits>>(H1+j))&1)];}
    function computeExact(){
      let mA=0,mZ=0,m2=0;                                          // weighted over all 2^11 masks
      for(let bits=0;bits<(1<<NH);bits++){
        const [a,b]=maskOf(bits); let k=0; for(let j=0;j<NH;j++) k+=(bits>>j)&1;
        const w=Math.pow(p,k)*Math.pow(1-p,NH-k);
        const z=forward(a,b), y=sig(z);
        mA+=w*y; mZ+=w*z; m2+=w*y*y;
      }
      const ws=sig(forward(d3.range(H1).map(()=>p), d3.range(H2).map(()=>p)));
      return {arith:mA, ngm:sig(mZ), sd:Math.sqrt(Math.max(0,m2-mA*mA)), ws};
    }
    const R2=rng(999);
    function sampleOne(){
      const a=d3.range(H1).map(()=>R2()<p?1:0), b=d3.range(H2).map(()=>R2()<p?1:0);
      lastMask=[a,b]; samples.push(sig(forward(a,b)));
    }

    // ── layout ──
    const lx=[34,120,206,292], ly=n=>d3.range(n).map(i=>60+ (i-(n-1)/2)*42 + 110);
    const pos=[ly(NI).map(y=>[lx[0],y]), ly(H1).map(y=>[lx[1],y]), ly(H2).map(y=>[lx[2],y]), [[lx[3],170]]];
    const gNet=svg.append("g");
    const netTitle=svg.append("text").attr("x",14).attr("y",22).attr("fill",C.muted).attr("font-size",11.5);
    svg.append("text").attr("x",14).attr("y",352).attr("fill",C.muted).attr("font-size",10.5)
      .text("blue = positive weight, amber = negative; crossed = dropped this pass");
    ["input x","hidden 1","hidden 2","σ(z)"].forEach((t,i)=>svg.append("text").attr("x",lx[i]).attr("y",318)
      .attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text(t));

    const cx0=380,cx1=742,cy0=40,cy1=300;
    const xs=d3.scaleLinear().range([cx0,cx1]), ys=d3.scaleLinear().range([cy1,cy0]).clamp(true);
    const gx=svg.append("g").attr("transform",`translate(0,${cy1})`).attr("color",C.muted);
    const gy=svg.append("g").attr("transform",`translate(${cx0},0)`).attr("color",C.muted);
    svg.append("text").attr("x",cx0).attr("y",cy0-16).attr("fill",C.muted).attr("font-size",11)
      .text("output probability vs number of thinned nets averaged, k");
    svg.append("text").attr("x",cx1).attr("y",cy1+30).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("k");
    const band=svg.append("path").attr("fill",C.good).attr("opacity",0.10);
    const dots=svg.append("g");
    const lineEx=svg.append("line").attr("stroke",C.good).attr("stroke-width",1.6);
    const lineNg=svg.append("line").attr("stroke",C.violet).attr("stroke-dasharray","2 3").attr("stroke-width",1.6);
    const lineWs=svg.append("line").attr("stroke",C.a2).attr("stroke-dasharray","6 4").attr("stroke-width",2);
    const run=svg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    const leg=svg.append("g").attr("transform",`translate(${cx0+8},${cy1+44})`);
    [["Monte-Carlo running mean",C.accent,""],["weight scaling, W_test = pW",C.a2,"6 4"],
     ["exact ensemble mean (all 2¹¹ nets)",C.good,""],["normalised geometric mean",C.violet,"2 3"]].forEach((d,i)=>{
      const g=leg.append("g").attr("transform",`translate(${(i%2)*190},${Math.floor(i/2)*15})`);
      g.append("line").attr("x2",18).attr("y1",-4).attr("y2",-4).attr("stroke",d[1]).attr("stroke-width",2).attr("stroke-dasharray",d[2]);
      g.append("text").attr("x",23).attr("fill",C.muted).attr("font-size",10.5).text(d[0]);
    });

    function drawNet(){
      gNet.selectAll("*").remove();
      const keep=showTest||!lastMask ? [d3.range(NI).map(()=>1), d3.range(H1).map(()=>1), d3.range(H2).map(()=>1), [1]]
                                     : [d3.range(NI).map(()=>1), lastMask[0], lastMask[1], [1]];
      const mats=[W1,W2,[w3]];
      netTitle.text(showTest? `test-time net: every unit present, outgoing hidden weights × p = ${p.toFixed(2)}`
                            : lastMask? `thinned net #${samples.length}: ${d3.sum(lastMask[0])+d3.sum(lastMask[1])} of ${NH} hidden units retained`
                                      : "full net (press sample to draw a thinned one)");
      mats.forEach((M,L)=>M.forEach((row,o)=>row.forEach((w,i)=>{
        if(!keep[L][i]||!keep[L+1][o]) return;
        const scale=(showTest&&L>0)?p:1;
        gNet.append("line").attr("x1",pos[L][i][0]).attr("y1",pos[L][i][1]).attr("x2",pos[L+1][o][0]).attr("y2",pos[L+1][o][1])
          .attr("stroke",w>=0?C.accent:C.a2).attr("stroke-opacity",Math.min(0.9,0.15+0.35*Math.abs(w)))
          .attr("stroke-width",Math.max(0.4,1.8*Math.abs(w)*scale));
      })));
      pos.forEach((layer,L)=>layer.forEach((q,i)=>{
        const on=keep[L][i];
        gNet.append("circle").attr("cx",q[0]).attr("cy",q[1]).attr("r",11)
          .attr("fill",on?C.panel:"#0f1117").attr("stroke",on?(L===3?C.good:C.ink):C.line).attr("stroke-width",1.4);
        if(!on){gNet.append("path").attr("d",`M${q[0]-6},${q[1]-6}L${q[0]+6},${q[1]+6}M${q[0]+6},${q[1]-6}L${q[0]-6},${q[1]+6}`)
          .attr("stroke",C.bad).attr("stroke-width",1.6);}
      }));
    }
    function drawChart(){
      const K=Math.max(KMAX0,samples.length);
      xs.domain([0,K]);
      const ref=[exact.arith,exact.ws,exact.ngm];
      const lo=Math.min(...ref), hi=Math.max(...ref), half=Math.max(0.06,(hi-lo)*1.8, 2.2*exact.sd/Math.sqrt(12));
      const mid=(lo+hi)/2; ys.domain([Math.max(0,mid-half),Math.min(1,mid+half)]);
      gx.call(d3.axisBottom(xs).ticks(6)); gy.call(d3.axisLeft(ys).ticks(6,".3f"));
      const hl=(sel,v)=>sel.attr("x1",cx0).attr("x2",cx1).attr("y1",ys(v)).attr("y2",ys(v));
      hl(lineEx,exact.arith); hl(lineNg,exact.ngm); hl(lineWs,exact.ws);
      const ks=d3.range(1,K+1);
      band.attr("d",d3.area().x(k=>xs(k)).y0(k=>ys(exact.arith-2*exact.sd/Math.sqrt(k))).y1(k=>ys(exact.arith+2*exact.sd/Math.sqrt(k)))(ks));
      let s=0; const pts=samples.map((y,i)=>{s+=y;return [xs(i+1),ys(s/(i+1))];});
      run.attr("d",pts.length?d3.line()(pts):null);
      const d=dots.selectAll("circle").data(samples.slice(-400));
      d.enter().append("circle").attr("r",1.6).attr("fill",C.muted).attr("opacity",0.45).merge(d)
        .attr("cx",(y,i)=>xs(Math.max(0,samples.length-400)+i+1)).attr("cy",y=>ys(y));
      d.exit().remove();
      const k=samples.length, mc=k?s/k:NaN, f=v=>v.toFixed(4);
      d3.select("#mc-readout").html(
        `k = <b>${k}</b> thinned nets · MC mean = <b>${k?f(mc):"—"}</b>`+
        (k?` (|MC − exact| = ${f(Math.abs(mc-exact.arith))}, expected ≈ ${f(exact.sd/Math.sqrt(k))})`:"")+
        `<br>exact ensemble mean over 2¹¹ = 2048 masks = <b>${f(exact.arith)}</b> · weight scaling = <b>${f(exact.ws)}</b>`+
        ` · normalised geometric mean σ(E[z]) = ${f(exact.ngm)}`+
        `<br>weight-scaling gap to the exact mean = <b>${f(Math.abs(exact.ws-exact.arith))}</b>; `+
        `spread of a single thinned net (sd) = ${f(exact.sd)}`+
        (act==="linear"?" — with linear hidden units weight scaling equals σ(E[z]) exactly":""));
    }
    function draw(){drawNet();drawChart();}
    function reset(){
      p=+d3.select("#mc-p").property("value"); act=d3.select("#mc-act").property("value");
      d3.select("#mc-pv").text(p.toFixed(2));
      exact=computeExact(); samples=[]; lastMask=null; draw();
    }
    let timer=null;
    function stop(){if(timer){timer.stop();timer=null;d3.select("#mc-run").text("run to k = 200");}}
    d3.select("#mc-p").on("input",()=>{stop();reset();});
    d3.select("#mc-act").on("change",()=>{stop();reset();});
    d3.select("#mc-one").on("click",()=>{stop();showTest=false;sampleOne();draw();});
    d3.select("#mc-20").on("click",()=>{stop();showTest=false;for(let i=0;i<20;i++)sampleOne();draw();});
    d3.select("#mc-run").on("click",()=>{
      if(timer){stop();return;}
      showTest=false; d3.select("#mc-run").text("pause");
      timer=d3.interval(()=>{ if(samples.length>=KMAX0){stop();return;} sampleOne(); draw(); },40);
    });
    d3.select("#mc-test").on("click",()=>{showTest=!showTest;drawNet();});
    d3.select("#mc-reset").on("click",()=>{stop();reset();});
    d3.select("#mc-new").on("click",()=>{stop();seed+=17;build();reset();});
    build(); reset();
  })();

  /* ═══════════════ 2. the reported tables ═══════════════ */
  (function(){
    const svg=d3.select("#rt-svg"); if(svg.empty()) return;
    // cat: n = no dropout, d = uses dropout, o = reference (human, Bayesian net)
    const T={
      mnist:{t:"Table 2 · MNIST, permutation-invariant · test error %",lo:true,rows:[
        ["Standard NN, logistic, 2×800",1.60,"n"],["SVM, Gaussian kernel",1.40,"n"],
        ["Dropout NN, logistic, 3×1024",1.35,"d"],["Dropout NN, ReLU, 3×1024",1.25,"d"],
        ["Dropout + max-norm, ReLU, 3×1024",1.06,"d"],["Dropout + max-norm, ReLU, 3×2048",1.04,"d"],
        ["Dropout + max-norm, ReLU, 2×4096",1.01,"d"],["Dropout + max-norm, ReLU, 2×8192",0.95,"d"],
        ["Dropout + max-norm, maxout 2×(5×240) †",0.94,"d"],
        ["DBN + finetuning, 500-500-2000",1.18,"n"],["DBM + finetuning, 500-500-2000",0.96,"n"],
        ["DBN + dropout finetuning",0.92,"d"],["DBM + dropout finetuning",0.79,"d"]]},
      svhn:{t:"Table 3 · SVHN · test error %",lo:true,rows:[
        ["Binary features (WDCH)",36.7,"n"],["HOG",15.0,"n"],["Stacked sparse autoencoders",10.3,"n"],["K-means",9.4,"n"],
        ["Multi-stage conv net, average pooling",9.06,"n"],["Multi-stage conv net, L2 pooling",5.36,"n"],
        ["Multi-stage conv net, L4 pooling + padding",4.90,"n"],["Conv net + max-pooling",3.95,"n"],
        ["… + dropout in fully connected layers",3.02,"d"],["Conv net + stochastic pooling",2.80,"n"],
        ["… + dropout in all layers",2.55,"d"],["Conv net + maxout †",2.47,"d"],["Human performance",2.0,"o"]]},
      c10:{t:"Table 4 · CIFAR-10, no augmentation · test error %",lo:true,rows:[
        ["Conv net + max pooling (hand tuned)",15.60,"n"],["Conv net + stochastic pooling",15.13,"n"],
        ["Conv net + max pooling (Bayesian hyperparameter opt.)",14.98,"n"],
        ["… + dropout in fully connected layers",14.32,"d"],["… + dropout in all layers",12.61,"d"],["Conv net + maxout †",11.68,"d"]]},
      c100:{t:"Table 4 · CIFAR-100, no augmentation · test error %",lo:true,rows:[
        ["Conv net + max pooling (hand tuned)",43.48,"n"],["Conv net + stochastic pooling",42.51,"n"],
        ["… + dropout in fully connected layers",41.26,"d"],["… + dropout in all layers",37.20,"d"],["Conv net + maxout †",38.57,"d"]]},
      inet:{t:"Table 5 · ILSVRC-2010 test set · top-5 error %",lo:true,rows:[
        ["Sparse coding",28.2,"n"],["SIFT + Fisher vectors",25.7,"n"],["Conv net + dropout",17.0,"d"]]},
      timit:{t:"Table 7 · TIMIT core test set · phone error rate %",lo:true,rows:[
        ["NN, 6 layers",23.4,"n"],["Dropout NN, 6 layers",21.8,"d"],
        ["DBN-pretrained NN, 4 layers",22.7,"n"],["DBN-pretrained NN, 6 layers",22.4,"n"],
        ["DBN-pretrained NN, 8 layers",20.7,"n"],["mcRBM-DBN-pretrained NN, 5 layers",20.5,"n"],
        ["DBN-pretrained NN, 4 layers + dropout",19.7,"d"],["DBN-pretrained NN, 8 layers + dropout",19.7,"d"]]},
      rcv1:{t:"§6.3 · Reuters-RCV1, 50 topics · test error %",lo:true,rows:[
        ["Best neural net without dropout",31.05,"n"],["… with dropout",29.62,"d"]]},
      splice:{t:"Table 8 · Alternative splicing · code quality, bits (HIGHER is better)",lo:false,rows:[
        ["Neural network, early stopping",440,"n"],["Regression, PCA",463,"n"],["SVM, PCA",487,"n"],
        ["Neural network with dropout",567,"d"],["Bayesian neural network",623,"o"]]},
      reg:{t:"Table 9 · MNIST, 784-1024-1024-2048-10 ReLU, regulariser comparison · test error %",lo:true,rows:[
        ["L2",1.62,"n"],["L2 + L1 towards end of training",1.60,"n"],["L2 + KL-sparsity",1.55,"n"],
        ["Max-norm",1.35,"n"],["Dropout + L2",1.25,"d"],["Dropout + max-norm",1.05,"d"]]}
    };
    const W=760, x0=300, x1=700, top=34, rowH=21;
    const title=svg.append("text").attr("x",14).attr("y",18).attr("fill",C.ink).attr("font-size",12);
    const g=svg.append("g"), gax=svg.append("g").attr("color",C.muted);
    const col=c=>c==="d"?C.accent:c==="o"?C.violet:C.muted;
    function draw(){
      const key=d3.select("#rt-t").property("value"), D=T[key], rows=D.rows, H=top+rows.length*rowH+30;
      svg.attr("viewBox",`0 0 ${W} ${H}`).attr("height",H);
      title.text(D.t);
      const xs=d3.scaleLinear().domain([0,d3.max(rows,r=>r[1])*1.08]).range([x0,x1]).nice();
      gax.attr("transform",`translate(0,${top+rows.length*rowH+4})`).call(d3.axisBottom(xs).ticks(6));
      g.selectAll("*").remove();
      rows.forEach((r,i)=>{
        const y=top+i*rowH;
        g.append("text").attr("x",x0-8).attr("y",y+14).attr("text-anchor","end").attr("font-size",11)
          .attr("fill",r[2]==="d"?C.ink:C.muted).text(r[0]);
        g.append("rect").attr("x",x0).attr("y",y+3).attr("height",rowH-6).attr("width",xs(r[1])-x0)
          .attr("fill",col(r[2])).attr("opacity",r[2]==="n"?0.55:0.9);
        g.append("text").attr("x",xs(r[1])+5).attr("y",y+14).attr("font-size",11).attr("fill",C.ink)
          .attr("font-family","SF Mono,Menlo,monospace").text(r[1]);
      });
      const best=c=>{const v=rows.filter(r=>r[2]===c).map(r=>r[1]);return D.lo?d3.min(v):d3.max(v);};
      const bn=best("n"), bd=best("d"), rel=D.lo?(bn-bd)/bn:(bd-bn)/bn;
      d3.select("#rt-readout").html(
        `best row without dropout: <b>${bn}</b> · best row with dropout: <b>${bd}</b> · `+
        (D.lo?`relative error reduction = <b>${(100*rel).toFixed(1)}%</b>`:`relative improvement = <b>${(100*rel).toFixed(1)}%</b>`)+
        ` <span style="color:var(--muted)">(arithmetic on the reported values, computed here; rows marked † are later work that also trains with dropout)</span>`);
    }
    d3.select("#rt-t").on("change",draw);
    draw();
  })();
})();
