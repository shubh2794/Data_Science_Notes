/* gcn.viz.js — interactive figures for deep-learning/graph-neural-networks/gcn.html.
   Loaded after data.js / notes.js. Shared helpers, then four independent IIFEs:
     #viz-cb  Chebyshev basis T_K(L̃) applied to a delta: exactly K-hop localized
     #viz-sp  eigenvalue spectra of I + D^−½AD^−½ vs D̃^−½ÃD̃^−½ and their k-fold gains
     #viz-kc  a 3-layer tanh GCN on the 34-node karate-club graph, trained live
     #viz-ab  the reported propagation-model ablation (Citeseer / Cora / Pubmed)
   Every number drawn by the first three figures is computed here; the fourth plots
   the reported table. */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const CLS=[C.accent,C.a2,C.violet,C.blue];

  /* ─────────── shared: the standard 78-edge karate-club list (0-indexed) ─────────── */
  const KE=[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],
    [14,32],[14,33],[15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],
    [23,25],[23,27],[23,29],[23,32],[23,33],[24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],
    [28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  const KN=34;
  /* the two factions after the club split, as in the commonly distributed version of the data */
  const CLUB=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1];

  function adjFrom(n,edges){const A=Array.from({length:n},()=>new Float64Array(n));edges.forEach(([a,b])=>{A[a][b]=1;A[b][a]=1;});return A;}
  const rowSum=M=>Array.from(M,r=>{let s=0;for(let j=0;j<r.length;j++)s+=r[j];return s;});
  /* D^−½ A D^−½ (selfLoops=false) or D̃^−½ Ã D̃^−½ (selfLoops=true) */
  function symNorm(A,selfLoops){const n=A.length;const B=A.map((r,i)=>Float64Array.from(r,(v,j)=>v+(selfLoops&&i===j?1:0)));
    const d=rowSum(B);return B.map((r,i)=>Float64Array.from(r,(v,j)=>v?v/Math.sqrt(d[i]*d[j]):0));}
  /* cyclic Jacobi eigenvalues of a small symmetric matrix, ascending */
  function eigSym(M){const n=M.length;const a=M.map(r=>Array.from(r));
    for(let sweep=0;sweep<100;sweep++){let off=0;for(let p=0;p<n;p++)for(let q=p+1;q<n;q++)off+=a[p][q]*a[p][q];if(off<1e-24)break;
      for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){if(Math.abs(a[p][q])<1e-300)continue;
        const th=(a[q][q]-a[p][p])/(2*a[p][q]);const t=(th>=0?1:-1)/(Math.abs(th)+Math.sqrt(th*th+1));const c=1/Math.sqrt(t*t+1),s=t*c;
        for(let k=0;k<n;k++){const x=a[k][p],y=a[k][q];a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
        for(let k=0;k<n;k++){const x=a[p][k],y=a[q][k];a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}}}
    return a.map((r,i)=>r[i]).sort((x,y)=>x-y);}
  function hops(A,src){const n=A.length,d=new Array(n).fill(Infinity);d[src]=0;const q=[src];
    while(q.length){const u=q.shift();for(let v=0;v<n;v++)if(A[u][v]&&d[v]===Infinity){d[v]=d[u]+1;q.push(v);}}return d;}
  /* Louvain modularity clustering (local moving + aggregation), deterministic node order */
  function louvain(A0){let W=A0.map(r=>Array.from(r));let member=W.map((_,i)=>i);let m2=0;W.forEach(r=>r.forEach(x=>m2+=x));
    for(let level=0;level<12;level++){const n=W.length,k=W.map(r=>r.reduce((s,x)=>s+x,0));let c=d3.range(n);const tot=k.slice();let moved=true,any=false;
      while(moved){moved=false;for(let i=0;i<n;i++){const ci=c[i],kin={};for(let j=0;j<n;j++)if(W[i][j]&&j!==i)kin[c[j]]=(kin[c[j]]||0)+W[i][j];
        tot[ci]-=k[i];let best=ci,bg=(kin[ci]||0)-tot[ci]*k[i]/m2;for(const cc in kin){const g=kin[cc]-tot[cc]*k[i]/m2;if(g>bg+1e-12){bg=g;best=+cc;}}
        tot[best]+=k[i];if(best!==ci){c[i]=best;moved=true;any=true;}}}
      if(!any)break;const ids=[...new Set(c)],map={};ids.forEach((v,i)=>map[v]=i);c=c.map(v=>map[v]);member=member.map(v=>c[v]);
      const W2=ids.map(()=>new Array(ids.length).fill(0));for(let i=0;i<n;i++)for(let j=0;j<n;j++)W2[c[i]][c[j]]+=W[i][j];W=W2;}
    return member;}
  function modularity(A,c){const d=rowSum(A);let m2=0;d.forEach(x=>m2+=x);let Q=0;
    for(let i=0;i<A.length;i++)for(let j=0;j<A.length;j++)if(c[i]===c[j])Q+=A[i][j]-d[i]*d[j]/m2;return Q/m2;}

  const KA=adjFrom(KN,KE), KDEG=rowSum(KA);
  /* relabel Louvain communities so that class 0 contains node 0 and so on, for stable colours */
  const COMM=(function(){const raw=louvain(KA),map={};let nx=0;raw.forEach(v=>{if(!(v in map))map[v]=nx++;});return raw.map(v=>map[v]);})();
  const COMM_Q=modularity(KA,COMM), CLUB_Q=modularity(KA,CLUB);

  /* one deterministic force layout, normalised to [0,1]², reused by two figures */
  const KPOS=(function(){const nodes=d3.range(KN).map(i=>({id:i}));const links=KE.map(([s,t])=>({source:s,target:t}));
    const sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).distance(34).strength(0.7))
      .force("charge",d3.forceManyBody().strength(-90)).force("center",d3.forceCenter(0,0)).stop();
    for(let i=0;i<400;i++)sim.tick();
    const xe=d3.extent(nodes,d=>d.x),ye=d3.extent(nodes,d=>d.y);
    return nodes.map(d=>[(d.x-xe[0])/(xe[1]-xe[0]||1),(d.y-ye[0])/(ye[1]-ye[0]||1)]);})();

  function drawGraph(g,x0,y0,w,h){
    const px=i=>x0+KPOS[i][0]*w, py=i=>y0+KPOS[i][1]*h;
    g.append("g").selectAll("line").data(KE).join("line").attr("x1",d=>px(d[0])).attr("y1",d=>py(d[0])).attr("x2",d=>px(d[1])).attr("y2",d=>py(d[1]))
      .attr("stroke",C.line).attr("stroke-width",1.1);
    return {px,py};
  }
  const fmtE=v=>{if(v===0)return "0";const a=Math.abs(v);return (a>=1e4||a<1e-3)?v.toExponential(2):v.toPrecision(4);};

  /* ═══════════════ 1. Chebyshev basis: T_K(L̃) δ is exactly K-hop localized ═══════════════ */
  (function(){
    const svg=d3.select("#cb-svg"); if(svg.empty()) return;
    const W=760,H=400;
    const S=symNorm(KA,false), L=S.map((r,i)=>Float64Array.from(r,(v,j)=>(i===j?1:0)-v));
    const EIG=eigSym(L), LMAX=EIG[KN-1];
    const NNZ=2*KE.length+KN;
    let centre=0;

    const gG=svg.append("g");
    svg.append("text").attr("x",16).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("entries of T_K(L̃) δᵥ on the karate graph (click a node to move δ)");
    const {px,py}=drawGraph(gG,34,40,330,320);
    const nodeG=svg.append("g");
    const circles=nodeG.selectAll("circle").data(d3.range(KN)).join("circle").attr("cx",px).attr("cy",py).attr("r",7.5)
      .style("cursor","pointer").on("click",(e,i)=>{centre=i;render();});
    circles.append("title").text(i=>"node "+i);
    svg.append("text").attr("x",16).attr("y",388).attr("fill",C.muted).attr("font-size",10.5).text("teal = positive, amber = negative, intensity ∝ log |entry| · hollow = exactly 0 · white ring = within K hops");

    // right panel: the polynomials
    const x0=430,x1=740,y0=40,y1=330;
    const xs=d3.scaleLinear().domain([0,2]).range([x0,x1]);
    const ys=d3.scaleLinear().domain([-1.6,1.6]).range([y1,y0]);
    svg.append("g").attr("transform",`translate(0,${y1})`).call(d3.axisBottom(xs).ticks(5)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(ys).ticks(5)).attr("color",C.muted);
    svg.append("text").attr("x",x0).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("Tₖ(2λ/λ_max − 1) for k = 0 … K, over λ ∈ [0, 2]");
    svg.append("text").attr("x",(x0+x1)/2).attr("y",y1+32).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("eigenvalue λ of L = I − D^−½AD^−½ (ticks: this graph's spectrum)");
    const shade=svg.append("rect").attr("y",y0).attr("height",y1-y0).attr("fill",C.bad).attr("opacity",0.07);
    const shadeT=svg.append("text").attr("y",y0+12).attr("fill",C.bad).attr("font-size",10).attr("text-anchor","middle");
    svg.append("g").selectAll("line").data(EIG).join("line").attr("x1",d=>xs(d)).attr("x2",d=>xs(d)).attr("y1",y1-7).attr("y2",y1).attr("stroke",C.ink).attr("opacity",0.7);
    svg.append("line").attr("x1",x0).attr("x2",x1).attr("y1",ys(1)).attr("y2",ys(1)).attr("stroke",C.line).attr("stroke-dasharray","3 3");
    svg.append("line").attr("x1",x0).attr("x2",x1).attr("y1",ys(-1)).attr("y2",ys(-1)).attr("stroke",C.line).attr("stroke-dasharray","3 3");
    const curves=svg.append("g");

    function cheb(k,x){let a=1,b=x;if(k===0)return 1;for(let i=2;i<=k;i++){const c=2*x*b-a;a=b;b=c;}return b;}
    function render(){
      const K=+d3.select("#cb-k").property("value"), scaleMode=d3.select("#cb-s").property("value");
      d3.select("#cb-kv").text(K);
      const lm=scaleMode==="exact"?LMAX:2;
      const Lt=L.map((r,i)=>Float64Array.from(r,(v,j)=>(2/lm)*v-(i===j?1:0)));
      const mv=v=>Lt.map(r=>{let s=0;for(let j=0;j<KN;j++)if(r[j]!==0)s+=r[j]*v[j];return s;});
      // three-term recurrence on vectors: t₀ = δ, t₁ = L̃δ, t_k = 2L̃t_(k−1) − t_(k−2)
      let tPrev=new Array(KN).fill(0);tPrev[centre]=1;let t=tPrev;
      if(K>=1){t=mv(tPrev);for(let k=2;k<=K;k++){const n=mv(t).map((v,i)=>2*v-tPrev[i]);tPrev=t;t=n;}}
      const hd=hops(KA,centre);
      const maxA=d3.max(t,v=>Math.abs(v))||1;
      const inten=v=>{if(v===0)return 0;return Math.max(0.12,1+Math.log10(Math.abs(v)/maxA)/4);};
      circles.attr("fill",i=>t[i]===0?C.panel:d3.interpolateRgb("#1b1f27",t[i]>0?C.accent:C.a2)(inten(t[i])))
        .attr("stroke",i=>i===centre?C.bad:(hd[i]<=K?C.ink:C.muted)).attr("stroke-width",i=>i===centre?3:(hd[i]<=K?1.6:0.8))
        .attr("stroke-dasharray",i=>t[i]===0?"2 2":null);
      circles.select("title").text(i=>`node ${i} · ${hd[i]} hop${hd[i]===1?"":"s"} from ${centre} · entry ${fmtE(t[i])}`);
      // curves
      const pts=d3.range(0,2.0001,0.01);
      curves.selectAll("path").data(d3.range(K+1)).join("path").attr("fill","none")
        .attr("stroke",k=>k===K?C.accent:C.muted).attr("stroke-width",k=>k===K?2.4:1).attr("opacity",k=>k===K?1:0.55)
        .attr("d",k=>d3.line().x(l=>xs(l)).y(l=>ys(Math.max(-1.6,Math.min(1.6,cheb(k,2*l/lm-1)))))(pts));
      if(lm<2){shade.attr("x",xs(lm)).attr("width",x1-xs(lm));shadeT.attr("x",(xs(lm)+x1)/2).text("no eigenvalues");}
      else{shade.attr("width",0);shadeT.text("");}
      const nz=t.filter(v=>v!==0).length, within=hd.filter(v=>v<=K).length;
      let outMax=0;t.forEach((v,i)=>{if(hd[i]>K)outMax=Math.max(outMax,Math.abs(v));});
      const maxIn=d3.max(EIG,l=>Math.abs(cheb(K,2*l/lm-1)));
      d3.select("#cb-readout").html(
        `δ at node <b>${centre}</b> (degree ${KDEG[centre]}) · K = <b>${K}</b> · nonzero entries of T_K(L̃)δ: <b>${nz}</b> · nodes within ${K} hop${K===1?"":"s"}: <b>${within}</b> of ${KN} · largest |entry| beyond ${K} hops: <b>${fmtE(outMax)}</b>`+
        `<br>cost: ${K} sparse mat-vec${K===1?"":"s"} with L̃ (${NNZ} nonzeros = 2·${KE.length} edges + ${KN} diagonal) — O(K·|E|), no eigendecomposition · `+
        `λ_max = ${scaleMode==="exact"?LMAX.toFixed(4)+" (exact, from this graph)":"2 (the first-order approximation; true value "+LMAX.toFixed(4)+")"} · max over this graph's eigenvalues of |T_K| = <b>${maxIn.toFixed(3)}</b>`);
    }
    d3.select("#cb-k").on("input",render);
    d3.select("#cb-s").on("change",render).on("input",render);
    render();
  })();

  /* ═══════════════ 2. spectra of the two propagation operators, and k applications ═══════════════ */
  (function(){
    const svg=d3.select("#sp-svg"); if(svg.empty()) return;
    const W=760;
    function graph(kind){
      if(kind==="karate")return {n:KN,e:KE,name:"karate club (34 nodes)"};
      if(kind==="path"){const n=10;return {n,e:d3.range(n-1).map(i=>[i,i+1]),name:"path, 10 nodes (bipartite)"};}
      if(kind==="star"){const n=9;return {n,e:d3.range(1,n).map(i=>[0,i]),name:"star, 1 hub + 8 leaves (bipartite)"};}
      if(kind==="cycle7"){const n=7;return {n,e:d3.range(n).map(i=>[i,(i+1)%n]),name:"cycle, 7 nodes (odd, not bipartite)"};}
      const n=6,e=[];for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)e.push([i,j]);return {n,e,name:"complete graph K₆"};
    }
    const OPS=[{key:"plain",lab:"I + D^−½ A D^−½   (first order, tied, no renormalization)",col:C.a2},
               {key:"renorm",lab:"D̃^−½ Ã D̃^−½   (renormalization trick)",col:C.accent}];
    // top: number lines
    const nx=d3.scaleLinear().domain([-1.15,2.15]).range([40,W-30]);
    const rowY=[64,140];
    svg.append("g").attr("transform",`translate(0,${rowY[1]+24})`).call(d3.axisBottom(nx).tickValues([-1,-0.5,0,0.5,1,1.5,2])).attr("color",C.muted);
    const bands=[[0,2],[-1,1]];
    OPS.forEach((o,k)=>{
      svg.append("rect").attr("x",nx(bands[k][0])).attr("width",nx(bands[k][1])-nx(bands[k][0])).attr("y",rowY[k]-14).attr("height",28).attr("fill",o.col).attr("opacity",0.08);
      svg.append("line").attr("x1",nx(-1.15)).attr("x2",nx(2.15)).attr("y1",rowY[k]).attr("y2",rowY[k]).attr("stroke",C.line);
      svg.append("text").attr("x",40).attr("y",rowY[k]-24).attr("fill",o.col).attr("font-size",11.5).text("eigenvalues of "+o.lab);
    });
    [-1,1].forEach(v=>svg.append("line").attr("x1",nx(v)).attr("x2",nx(v)).attr("y1",30).attr("y2",rowY[1]+24).attr("stroke",C.muted).attr("stroke-dasharray","3 4").attr("opacity",0.6));
    svg.append("text").attr("x",nx(1)+4).attr("y",rowY[1]+20).attr("fill",C.muted).attr("font-size",10).text("|μ| = 1");
    const dotsG=svg.append("g");
    // bottom: gains
    const by0=210,by1=380;
    const gy=d3.scaleLog().domain([1e-12,1e10]).range([by1,by0]).clamp(true);
    const panels=[[50,380],[420,W-20]];
    panels.forEach((p,k)=>{
      svg.append("g").attr("transform",`translate(${p[0]},0)`).call(d3.axisLeft(gy).tickValues([1e-12,1e-8,1e-4,1,1e4,1e8]).tickFormat(d3.format(".0e"))).attr("color",C.muted);
      svg.append("line").attr("x1",p[0]).attr("x2",p[1]).attr("y1",gy(1)).attr("y2",gy(1)).attr("stroke",C.ink).attr("stroke-dasharray","4 3").attr("opacity",0.6);
    });
    const gainT=OPS.map((o,k)=>svg.append("text").attr("x",panels[k][0]).attr("y",by0-12).attr("fill",o.col).attr("font-size",11));
    const barsG=OPS.map(()=>svg.append("g"));

    function render(){
      const g=graph(d3.select("#sp-g").property("value")), k=+d3.select("#sp-k").property("value");
      d3.select("#sp-kv").text(k);
      const A=adjFrom(g.n,g.e);
      const S=symNorm(A,false);
      const M1=S.map((r,i)=>Float64Array.from(r,(v,j)=>v+(i===j?1:0)));
      const M2=symNorm(A,true);
      const ev=[eigSym(M1),eigSym(M2)].map(e=>e.map(v=>Math.abs(v)<1e-10?0:v).sort((a,b)=>b-a));
      // dots, stacked where eigenvalues repeat
      const dots=[];ev.forEach((e,r)=>{const seen={};e.forEach(v=>{const key=v.toFixed(3);seen[key]=(seen[key]||0)+1;dots.push({r,v,s:seen[key]-1});});});
      dotsG.selectAll("circle").data(dots).join("circle").attr("cx",d=>nx(d.v)).attr("cy",d=>rowY[d.r]+(d.s%2?-1:1)*Math.min(16,Math.ceil(d.s/2)*4.5)).attr("r",3.6)
        .attr("fill",d=>OPS[d.r].col).attr("opacity",0.9);
      // gains |μ|^k
      ev.forEach((e,r)=>{
        const x=d3.scaleBand().domain(d3.range(e.length)).range(panels[r]).padding(0.15);
        const gain=e.map(v=>Math.pow(Math.abs(v),k));
        barsG[r].selectAll("rect").data(gain).join("rect").attr("x",(d,i)=>x(i)).attr("width",x.bandwidth())
          .attr("y",d=>d>=1?gy(d):gy(1)).attr("height",d=>Math.max(0.5,Math.abs(gy(d===0?1e-12:d)-gy(1))))
          .attr("fill",(d,i)=>d===0?C.muted:OPS[r].col).attr("opacity",d=>d===0?0.5:0.85);
        gainT[r].text(`|μᵢ|^${k} per eigenmode, largest μ first (log scale)`);
      });
      const top=ev.map(e=>Math.pow(Math.abs(e[0]),k)), mins=ev.map(e=>d3.min(e,v=>Math.abs(v)));
      const minG=mins.map(m=>Math.pow(m,k));
      const surv=ev.map(e=>{const t=Math.pow(d3.max(e,v=>Math.abs(v)),k);return e.filter(v=>Math.pow(Math.abs(v),k)>=1e-3*t).length;});
      d3.select("#sp-readout").html(
        `<b>${g.name}</b> · k = <b>${k}</b> application${k===1?"":"s"} (linear part only: no weights, no nonlinearity)`+
        `<br><span style="color:${C.a2}">first-order, tied</span>: spectrum [${ev[0][ev[0].length-1].toFixed(3)}, ${ev[0][0].toFixed(3)}] · largest gain <b>${fmtE(top[0])}</b> · smallest <b>${fmtE(minG[0])}</b>`+
        (mins[0]===0?" (an exact 0 eigenvalue: bipartite graph, that mode is wiped out)":"")+` · modes within 10⁻³ of the largest: <b>${surv[0]}</b> of ${g.n}`+
        `<br><span style="color:${C.accent}">renormalized</span>: spectrum [${ev[1][ev[1].length-1].toFixed(3)}, ${ev[1][0].toFixed(3)}] · largest gain <b>${fmtE(top[1])}</b> · smallest <b>${fmtE(minG[1])}</b> · modes within 10⁻³ of the largest: <b>${surv[1]}</b> of ${g.n}`);
    }
    d3.select("#sp-g").on("change",render).on("input",render);
    d3.select("#sp-k").on("input",render);
    render();
  })();

  /* ═══════════════ 3. a 3-layer tanh GCN on the karate club, trained live ═══════════════ */
  (function(){
    const svg=d3.select("#kc-svg"); if(svg.empty()) return;
    const DIMS=[4,4,2], LR=0.01, MAXIT=300, PER_FRAME=3;
    const P_GCN=symNorm(KA,true).map(r=>Array.from(r));
    const P_ID=d3.range(KN).map(i=>d3.range(KN).map(j=>i===j?1:0));
    let seed=1, labels, Ccount, lab, P, params, mA, vA, it, timer=null, cache;

    function rng(s0){let s=s0>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    const mm=(X,Y)=>{const n=X.length,k=Y.length,m=Y[0].length,Z=new Array(n);for(let i=0;i<n;i++){const z=new Array(m).fill(0),xi=X[i];
      for(let t=0;t<k;t++){const x=xi[t];if(!x)continue;const y=Y[t];for(let j=0;j<m;j++)z[j]+=x*y[j];}Z[i]=z;}return Z;};
    const T=X=>X[0].map((_,j)=>X.map(r=>r[j]));

    function init(){
      const which=d3.select("#kc-l").property("value");
      labels=which==="club"?CLUB:COMM; Ccount=d3.max(labels)+1;
      P=d3.select("#kc-p").property("value")==="gcn"?P_GCN:P_ID;
      lab=d3.range(Ccount).map(c=>{let b=-1;for(let i=0;i<KN;i++)if(labels[i]===c&&(b<0||KDEG[i]>KDEG[b]))b=i;return b;});
      const R=rng(seed*7919);
      const glorot=(a,b)=>{const l=Math.sqrt(6/(a+b));return d3.range(a).map(()=>d3.range(b).map(()=>(2*R()-1)*l));};
      const sizes=[KN,...DIMS];params=DIMS.map((d,i)=>glorot(sizes[i],d));
      params.push(glorot(DIMS[DIMS.length-1],Ccount));params.push([new Array(Ccount).fill(0)]);
      mA=params.map(p=>p.map(r=>r.map(()=>0)));vA=params.map(p=>p.map(r=>r.map(()=>0)));
      it=0;cache=forward();
    }
    function forward(){
      const Hs=[null],PH=[];for(let l=0;l<DIMS.length;l++){const ph=l===0?P:mm(P,Hs[l]);PH.push(ph);Hs.push(mm(ph,params[l]).map(r=>r.map(Math.tanh)));}
      const b=params[DIMS.length+1][0];
      const Sc=mm(Hs[DIMS.length],params[DIMS.length]).map(r=>r.map((v,j)=>v+b[j]));
      const Y=Sc.map(r=>{const m=d3.max(r),e=r.map(v=>Math.exp(v-m)),s=d3.sum(e);return e.map(v=>v/s);});
      let loss=0;lab.forEach(i=>{loss-=Math.log(Math.max(1e-12,Y[i][labels[i]]));});
      return {Hs,PH,Y,loss};
    }
    function step(){
      const {Hs,PH,Y}=cache, L=DIMS.length;
      const dS=Y.map(r=>r.map(()=>0));lab.forEach(i=>{for(let j=0;j<Ccount;j++)dS[i][j]=Y[i][j]-(labels[i]===j?1:0);}); // labelled rows only
      const g=new Array(params.length);
      g[L]=mm(T(Hs[L]),dS);g[L+1]=[d3.range(Ccount).map(j=>d3.sum(dS,r=>r[j]))];
      let dH=mm(dS,T(params[L]));
      for(let l=L-1;l>=0;l--){const H=Hs[l+1];const dZ=dH.map((r,i)=>r.map((v,j)=>v*(1-H[i][j]*H[i][j])));
        g[l]=mm(T(PH[l]),dZ); if(l>0)dH=mm(P,mm(dZ,T(params[l])));}  // P symmetric, so Pᵀ = P
      it++;const b1=0.9,b2=0.999,c1=1-Math.pow(b1,it),c2=1-Math.pow(b2,it);
      params.forEach((p,k)=>p.forEach((r,i)=>{for(let j=0;j<r.length;j++){const gr=g[k][i][j];
        mA[k][i][j]=b1*mA[k][i][j]+(1-b1)*gr;vA[k][i][j]=b2*vA[k][i][j]+(1-b2)*gr*gr;
        r[j]-=LR*(mA[k][i][j]/c1)/(Math.sqrt(vA[k][i][j]/c2)+1e-8);}}));
      cache=forward();
    }

    // layout
    svg.append("text").attr("x",16).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("graph: fill = predicted class, outer ring = true class, white halo = the one labelled node per class");
    const gG=svg.append("g");drawGraph(gG,30,42,340,300);
    const px=i=>30+KPOS[i][0]*340, py=i=>42+KPOS[i][1]*300;
    const halo=svg.append("g").selectAll("circle").data(d3.range(KN)).join("circle").attr("cx",px).attr("cy",py).attr("r",12).attr("fill","none");
    const ring=svg.append("g").selectAll("circle").data(d3.range(KN)).join("circle").attr("cx",px).attr("cy",py).attr("r",8.5).attr("fill","none").attr("stroke-width",2.4);
    const core=svg.append("g").selectAll("circle").data(d3.range(KN)).join("circle").attr("cx",px).attr("cy",py).attr("r",5.6);
    core.append("title");
    // embedding panel
    const ex0=430,ex1=740,ey0=42,ey1=342;
    svg.append("rect").attr("x",ex0).attr("y",ey0).attr("width",ex1-ex0).attr("height",ey1-ey0).attr("fill","none").attr("stroke",C.line);
    svg.append("text").attr("x",ex0).attr("y",30).attr("fill",C.muted).attr("font-size",11).text("2-D output of the third GCN layer, coloured by true class (axes rescale)");
    const embG=svg.append("g");
    const axT=svg.append("text").attr("x",ex1).attr("y",ey1+16).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10);
    const curveG=svg.append("g");

    function render(){
      const {Hs,Y,loss}=cache, H3=Hs[DIMS.length];
      const pred=Y.map(r=>r.indexOf(d3.max(r)));
      core.attr("fill",i=>CLS[pred[i]]).select("title").text(i=>`node ${i} · true class ${labels[i]+1} · predicted ${pred[i]+1} (p = ${d3.max(Y[i]).toFixed(2)})`);
      ring.attr("stroke",i=>CLS[labels[i]]);
      halo.attr("stroke",i=>lab.includes(i)?C.ink:"none").attr("stroke-width",2);
      const m=Math.max(1e-6,d3.max(H3,r=>Math.max(Math.abs(r[0]),Math.abs(r[1]))))*1.15;
      const sx=d3.scaleLinear().domain([-m,m]).range([ex0+8,ex1-8]), sy=d3.scaleLinear().domain([-m,m]).range([ey1-8,ey0+8]);
      embG.selectAll("circle").data(d3.range(KN)).join("circle").attr("cx",i=>sx(H3[i][0])).attr("cy",i=>sy(H3[i][1]))
        .attr("r",i=>lab.includes(i)?7:4.8).attr("fill",i=>CLS[labels[i]]).attr("stroke",i=>lab.includes(i)?C.ink:"none").attr("stroke-width",2).attr("opacity",0.9);
      axT.text(`axis half-width ${m.toFixed(3)}`);
      let ok=0,n=0;for(let i=0;i<KN;i++){if(lab.includes(i))continue;n++;if(pred[i]===labels[i])ok++;}
      const which=d3.select("#kc-l").property("value"), prop=d3.select("#kc-p").property("value");
      d3.select("#kc-readout").html(
        `iteration <b>${it}</b> / ${MAXIT} · cross-entropy on the ${lab.length} labelled nodes <b>${loss.toFixed(4)}</b> · accuracy on the other ${n} nodes <b>${ok}/${n} = ${(100*ok/n).toFixed(1)}%</b>`+
        ` (chance with ${Ccount} balanced classes ≈ ${(100/Ccount).toFixed(0)}%)`+
        `<br>labels: ${which==="club"?`the 2 factions after the split (modularity of that partition ${CLUB_Q.toFixed(3)})`:`${Ccount} communities found here by Louvain modularity clustering (Q = ${COMM_Q.toFixed(4)})`}`+
        ` · labelled nodes: ${lab.join(", ")} (highest degree in each class) · propagation: ${prop==="gcn"?"Â = D̃^−½ÃD̃^−½":"Â = I (no graph: an MLP on one-hot node ids)"}`+
        (it===0?` · untrained: random Glorot weights, X = I`:"")+
        (prop!=="gcn"&&it>0?` · without Â, rows of W⁽⁰⁾ for unlabelled nodes never receive a gradient`:""));
    }
    function stop(){if(timer){timer.stop();timer=null;}d3.select("#kc-play").text(it>=MAXIT?"done (click to restart)":"train");}
    function play(){
      if(timer){stop();return;}
      if(it>=MAXIT){init();}
      d3.select("#kc-play").text("pause");
      timer=d3.timer(()=>{for(let i=0;i<PER_FRAME&&it<MAXIT;i++)step();render();if(it>=MAXIT)stop();});
    }
    d3.select("#kc-play").on("click",play);
    d3.select("#kc-step").on("click",()=>{stop();for(let i=0;i<25&&it<MAXIT;i++)step();render();stop();});
    d3.select("#kc-reset").on("click",()=>{stop();seed+=1;init();render();stop();});
    d3.select("#kc-l").on("change",()=>{stop();init();render();stop();});
    d3.select("#kc-p").on("change",()=>{stop();init();render();stop();});
    init();render();
  })();

  /* ═══════════════ 4. reported propagation-model ablation ═══════════════ */
  (function(){
    const svg=d3.select("#ab-svg"); if(svg.empty()) return;
    const W=760,H=340,m={l:290,r:60,t:34,b:40};
    const ROWS=[
      ["Chebyshev, K = 3","∑ₖ Tₖ(L̃) X Θₖ, k = 0…3",[69.8,79.5,74.4],"cheb"],
      ["Chebyshev, K = 2","∑ₖ Tₖ(L̃) X Θₖ, k = 0…2",[69.6,81.2,73.8],"cheb"],
      ["1st-order model","XΘ₀ + D^−½AD^−½XΘ₁",[68.3,80.0,77.5],"first"],
      ["single parameter","(I + D^−½AD^−½) X Θ",[69.3,79.2,77.4],"first"],
      ["renormalization trick","D̃^−½ÃD̃^−½ X Θ",[70.3,81.5,79.0],"renorm"],
      ["1st-order term only","D^−½AD^−½ X Θ",[68.7,80.5,77.8],"first"],
      ["MLP (no graph)","X Θ",[46.5,55.1,71.4],"mlp"]];
    const DS=["Citeseer","Cora","Pubmed"];
    const col=k=>k==="renorm"?C.accent:k==="mlp"?C.bad:k==="cheb"?C.violet:C.a2;
    function val(r,d){return d==="mean"?d3.mean(r[2]):r[2][+d];}
    function draw(){
      svg.selectAll("*").remove();
      const d=d3.select("#ab-d").property("value"), zoom=d3.select("#ab-z").property("checked");
      const name=d==="mean"?"mean of the three":DS[+d];
      const vals=ROWS.map(r=>val(r,d));
      const lo=zoom?Math.floor(d3.min(vals.filter((v,i)=>ROWS[i][3]!=="mlp"))-2):0;
      const x=d3.scaleLinear().domain([lo,Math.ceil(d3.max(vals))+1]).range([m.l,W-m.r]).clamp(true);
      const y=d3.scaleBand().domain(ROWS.map(r=>r[0])).range([m.t,H-m.b]).padding(0.22);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickSize(0).tickPadding(8)
        .tickFormat(k=>{const r=ROWS.find(z=>z[0]===k);return k+"   "+r[1];})).attr("color",C.muted).select(".domain").remove();
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11)
        .text(`${name}: test accuracy (%), mean of 100 runs` + (zoom?` — axis starts at ${lo}, not 0`:""));
      const best=d3.max(vals), mlp=val(ROWS[6],d);
      ROWS.forEach((r,i)=>{
        const v=vals[i], clipped=v<lo;
        svg.append("rect").attr("x",m.l).attr("y",y(r[0])).attr("height",y.bandwidth()).attr("width",Math.max(2,x(v)-m.l))
          .attr("fill",col(r[3])).attr("opacity",clipped?0.35:0.85).style("cursor","pointer")
          .on("mouseenter",()=>hover(r,d,v,best,mlp)).append("title").text(`${r[0]}: ${v.toFixed(1)}`);
        svg.append("text").attr("x",(clipped?m.l:x(v))+6).attr("y",y(r[0])+y.bandwidth()/2+4).attr("fill",C.ink).attr("font-size",11)
          .text(v.toFixed(d==="mean"?2:1)+(clipped?"  (off the zoomed axis)":""));
      });
      summary(d,name,vals);
    }
    function summary(d,name,vals){
      const rn=vals[4], sp=vals[3], fo=vals[2], ch=Math.max(vals[0],vals[1]), mlp=vals[6];
      const f=v=>(v>=0?"+":"")+v.toFixed(d==="mean"?2:1);
      d3.select("#ab-readout").html(`<b>${name}</b> · renormalization trick <b>${rn.toFixed(d==="mean"?2:1)}</b> · vs single parameter without it <b>${f(rn-sp)}</b> · vs two-parameter 1st-order model <b>${f(rn-fo)}</b> · vs best Chebyshev <b>${f(rn-ch)}</b> · vs MLP <b>${f(rn-mlp)}</b><br>hover a bar for its gap to the best and to the MLP`);
    }
    function hover(r,d,v,best,mlp){
      const p=d==="mean"?2:1;
      d3.select("#ab-readout").html(`<b>${r[0]}</b> — ${r[1]} · ${v.toFixed(p)}% · ${v===best?"best on this view":(best-v).toFixed(p)+" below the best"} · ${r[3]==="mlp"?"no propagation at all":(v-mlp).toFixed(p)+" above the MLP"} · Citeseer / Cora / Pubmed = ${r[2].join(" / ")}`);
    }
    d3.select("#ab-d").on("change",draw);
    d3.select("#ab-z").on("change",draw);
    draw();
  })();
})();
