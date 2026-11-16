/* mamba.viz.js — interactive figures for deep-learning/state-space-models/mamba.html.
   Loaded after data.js / notes.js. Four independent IIFEs:
     #viz-sel   selection playground: scalar SSM, LTI (fixed Δ) vs selective Δ (Theorem 1 recurrence)
     #viz-scan  parallel scan over affine steps (a, b): sequential / Hillis–Steele / Blelloch
     #viz-io    memory calculator: expanded (B,L,D,N) state vs fused IO; KV cache vs recurrent state
     #viz-lm    reported zero-shot average accuracy vs parameters (Mamba, Pythia, GPT-J) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>Math.round(n).toLocaleString("en-US");
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}

  /* ═══════════════ 1. selection playground ═══════════════ */
  (function(){
    const svg=d3.select("#sel-svg"); if(svg.empty()) return;
    const T=32, W=760, x0=60, x1=745;
    let seed=3, tok=[];
    function build(){
      const R=rng(seed), noise=+d3.select("#sel-n").property("value");
      const pos=new Set(); while(pos.size<5){pos.add(1+Math.floor(R()*(T-2)));}
      tok=d3.range(T).map(t=>{
        if(pos.has(t)){const s=R()<0.5?-1:1;return {sig:true,v:s*(0.3+0.7*R())};}
        return {sig:false,v:(2*R()-1)*noise};
      });
    }
    function run(deltas){
      let h=0; const H=[],Ab=[],Bb=[];
      for(let t=0;t<T;t++){const a=Math.exp(-deltas[t]), b=1-a; h=a*h+b*tok[t].v; H.push(h);Ab.push(a);Bb.push(b);}
      // weight of x_s in the final state: B̄_s · ∏_{t>s} Ā_t
      const w=new Array(T); let p=1; for(let s=T-1;s>=0;s--){w[s]=Bb[s]*p;p*=Ab[s];}
      return {H,Ab,Bb,w};
    }
    function target(){let cur=null;return tok.map(k=>{if(k.sig)cur=k.v;return cur;});}
    function rmse(H,tg){let s=0,n=0;for(let t=0;t<T;t++){if(tg[t]===null)continue;const d=H[t]-tg[t];s+=d*d;n++;}return n?Math.sqrt(s/n):0;}

    const xs=d3.scaleBand().domain(d3.range(T)).range([x0,x1]).padding(0.12);
    const gTok=svg.append("g"), gA=svg.append("g"), gH=svg.append("g"), gW=svg.append("g");
    const rows={tok:[24,52],A:[82,150],H:[180,320],W:[352,420]};
    const yA=d3.scaleLinear().domain([0,1]).range([rows.A[1],rows.A[0]]);
    const yH=d3.scaleLinear().domain([-1.15,1.15]).range([rows.H[1],rows.H[0]]);
    const yW=d3.scaleLinear().domain([0,1]).range([rows.W[1],rows.W[0]]);
    const lab=(y,t)=>svg.append("text").attr("x",x0).attr("y",y).attr("fill",C.muted).attr("font-size",11).text(t);
    lab(rows.tok[0]-8,"tokens (teal = signal with its value, grey = filler)");
    lab(rows.A[0]-8,"Āₜ = exp(−Δₜ): fraction of hₜ₋₁ kept");
    lab(rows.H[0]-8,"state hₜ (teal) vs the most recent signal value (amber, dashed)");
    lab(rows.W[0]-8,"weight of each token's value in the final state h₃₂:  B̄ₛ × (product of Āₜ over all later t)");
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(yA).ticks(2)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(yH).ticks(5)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(yW).ticks(2)).attr("color",C.muted);
    svg.append("line").attr("x1",x0).attr("x2",x1).attr("y1",yH(0)).attr("y2",yH(0)).attr("stroke",C.line);
    const tgPath=gH.append("path").attr("fill","none").attr("stroke",C.a2).attr("stroke-width",1.6).attr("stroke-dasharray","5 3");
    const hPath=gH.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2);
    const cx=t=>xs(t)+xs.bandwidth()/2;
    svg.append("g").attr("transform",`translate(0,${rows.W[1]})`).call(d3.axisBottom(xs).tickValues([0,7,15,23,31]).tickFormat(t=>"t="+(t+1))).attr("color",C.muted);

    function deltasFor(mode){
      const d=+d3.select("#sel-d").property("value"), ds=+d3.select("#sel-ds").property("value"), df=+d3.select("#sel-df").property("value");
      return tok.map(k=>mode==="lti"?d:(k.sig?ds:df));
    }
    function draw(){
      const mode=d3.select("#sel-m").property("value");
      d3.select("#sel-dv").text((+d3.select("#sel-d").property("value")).toFixed(2));
      d3.select("#sel-dsv").text((+d3.select("#sel-ds").property("value")).toFixed(2));
      d3.select("#sel-dfv").text((+d3.select("#sel-df").property("value")).toFixed(2));
      d3.select("#sel-nv").text((+d3.select("#sel-n").property("value")).toFixed(2));
      d3.select("#sel-d").property("disabled",mode!=="lti");
      d3.select("#sel-ds").property("disabled",mode==="lti");
      d3.select("#sel-df").property("disabled",mode==="lti");
      const r=run(deltasFor(mode)), tg=target();

      gTok.selectAll("rect").data(tok).join("rect").attr("x",(d,t)=>xs(t)).attr("y",rows.tok[0]).attr("width",xs.bandwidth()).attr("height",rows.tok[1]-rows.tok[0])
        .attr("rx",3).attr("fill",d=>d.sig?C.accent:"#2a2f3a").attr("opacity",d=>d.sig?0.9:0.8);
      gTok.selectAll("text").data(tok).join("text").attr("x",(d,t)=>cx(t)).attr("y",rows.tok[0]+18).attr("text-anchor","middle")
        .attr("font-size",9).attr("fill",d=>d.sig?"#0b0d12":C.muted).text(d=>d.sig?d.v.toFixed(1):"·");

      gA.selectAll("rect").data(r.Ab).join("rect").attr("x",(d,t)=>xs(t)).attr("width",xs.bandwidth())
        .attr("y",d=>yA(d)).attr("height",d=>rows.A[1]-yA(d)).attr("fill",(d,t)=>tok[t].sig?C.accent:C.muted).attr("opacity",0.8);

      const tgPts=tg.map((v,t)=>v===null?null:[t,v]);
      tgPath.attr("d",d3.line().defined(d=>d!==null).curve(d3.curveStepAfter).x(d=>cx(d[0])).y(d=>yH(d[1]))(tgPts));
      hPath.attr("d",d3.line().x((d,t)=>cx(t)).y(d=>yH(Math.max(-1.15,Math.min(1.15,d))))(r.H));
      gH.selectAll("circle").data(r.H).join("circle").attr("cx",(d,t)=>cx(t)).attr("cy",d=>yH(Math.max(-1.15,Math.min(1.15,d)))).attr("r",2.4)
        .attr("fill",(d,t)=>tok[t].sig?C.accent:C.ink);

      const wmax=Math.max(1e-9,d3.max(r.w));
      yW.domain([0,Math.max(0.2,wmax)]);
      gW.selectAll("rect").data(r.w).join("rect").attr("x",(d,t)=>xs(t)).attr("width",xs.bandwidth())
        .attr("y",d=>yW(d)).attr("height",d=>rows.W[1]-yW(d)).attr("fill",(d,t)=>tok[t].sig?C.accent:C.a2).attr("opacity",0.85);

      // best any single fixed Δ can do on this sequence
      let best={d:0,e:Infinity};
      for(let i=1;i<=300;i++){const d=i*0.02; const rr=run(tok.map(()=>d)); const e=rmse(rr.H,tg); if(e<best.e)best={d,e};}
      const e=rmse(r.H,tg), hT=r.H[T-1], tT=tg[T-1];
      const wF=d3.sum(r.w.filter((d,t)=>!tok[t].sig)), wAll=d3.sum(r.w);
      let lastSig=-1; tok.forEach((k,t)=>{if(k.sig)lastSig=t;});
      const wLast=lastSig>=0?r.w[lastSig]:0;
      const half=mode==="lti"?Math.LN2/(+d3.select("#sel-d").property("value")):null;
      d3.select("#sel-readout").html(
        `${mode==="lti"?"LTI":"selective"} · final h₃₂ = <b>${hT.toFixed(3)}</b> vs target <b>${tT===null?"–":tT.toFixed(3)}</b> · tracking RMSE (after the first signal) <b>${e.toFixed(3)}</b>`+
        ` · best any single fixed Δ reaches on this sequence: <b>${best.e.toFixed(3)}</b> at Δ = ${best.d.toFixed(2)}`+
        `<br>final-state weight: on the last signal token <b>${(100*wLast).toFixed(1)}%</b>, on filler <b>${(100*wF).toFixed(1)}%</b>, lost to the h₀ = 0 start ${(100*(1-wAll)).toFixed(1)}%`+
        (half!==null?` · with fixed Δ the state's half-life is ln2/Δ = <b>${half.toFixed(1)}</b> tokens, whatever the tokens are`:" · Δ ≈ 0 on filler makes Ā ≈ 1 and B̄ ≈ 0: hold and ignore"));
    }
    ["#sel-m","#sel-d","#sel-ds","#sel-df"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    d3.select("#sel-n").on("input",()=>{build();draw();});
    d3.select("#sel-new").on("click",()=>{seed+=1;build();draw();});
    build();draw();
  })();

  /* ═══════════════ 2. parallel scan over affine steps ═══════════════ */
  (function(){
    const svg=d3.select("#scan-svg"); if(svg.empty()) return;
    const N=8, W=760, H=440, x0=150, x1=748, y0=26, y1=430;
    let seed=21, base=[];
    const ID={a:1,b:0,lo:1,hi:0};                       // identity: empty range
    const empty=e=>e.lo>e.hi;
    // e1 earlier, e2 later:  (a₂,b₂) ∘ (a₁,b₁) = (a₂a₁, a₂b₁ + b₂)
    function comb(e1,e2){return {a:e2.a*e1.a,b:e2.a*e1.b+e2.b,lo:empty(e1)?e2.lo:e1.lo,hi:empty(e2)?e1.hi:e2.hi};}
    function build(){
      const R=rng(seed);
      base=d3.range(N).map(i=>{const a=0.3+0.7*R(), x=2*R()-1, b=(1-a)*x; return {a,b,lo:i+1,hi:i+1};});
    }
    function stages(alg){
      let x=base.map(e=>Object.assign({},e));
      const out=[{name:"inputs (aₜ, bₜ)",cells:x.map(e=>Object.assign({},e)),ops:[],nComb:0}];
      const push=(name,ops,nComb)=>out.push({name,cells:x.map(e=>Object.assign({},e)),ops,nComb});
      if(alg==="seq"){
        for(let t=1;t<N;t++){x[t]=comb(x[t-1],x[t]);push("step "+t,[{i:t,src:[t-1,t]}],1);}
      }else if(alg==="hs"){
        for(let d=1;d<N;d*=2){const old=x.map(e=>Object.assign({},e)),ops=[];
          for(let i=d;i<N;i++){x[i]=comb(old[i-d],old[i]);ops.push({i,src:[i-d,i]});}
          push("offset "+d,ops,ops.length);}
      }else{
        let lev=1;
        for(let s=2;s<=N;s*=2,lev++){const ops=[];
          for(let i=s-1;i<N;i+=s){x[i]=comb(x[i-s/2],x[i]);ops.push({i,src:[i-s/2,i]});}
          push("up-sweep "+lev,ops,ops.length);}
        x[N-1]=Object.assign({},ID); lev=1;
        for(let s=N;s>=2;s/=2,lev++){const ops=[];let nc=0;
          for(let i=s-1;i<N;i+=s){const l=i-s/2, t=x[l]; x[l]=x[i]; x[i]=comb(x[i],t); nc++;
            ops.push({i,src:[i,l]}); ops.push({i:l,src:[i],copy:true});}
          push("down-sweep "+lev+(lev===1?" (root ← identity)":""),ops,nc);}
        const ops=[]; for(let i=0;i<N;i++){x[i]=comb(x[i],base[i]);ops.push({i,src:[i],self:true});}
        push("add own step",ops,N);
      }
      return out;
    }
    function seqH(){let h=0;return base.map(e=>{h=e.a*h+e.b;return h;});}

    function draw(toEnd){
      svg.selectAll("*").remove();
      const alg=d3.select("#scan-a").property("value"), st=stages(alg), S=st.length-1;
      const sl=d3.select("#scan-s"); sl.attr("max",S); if(toEnd===true||+sl.property("value")>S) sl.property("value",S);
      const upto=+sl.property("value"); d3.select("#scan-sv").text(upto);
      const xs=d3.scaleBand().domain(d3.range(N)).range([x0,x1]).padding(0.14);
      const rowH=(y1-y0)/st.length, ch=Math.min(36,rowH-12);
      const yr=k=>y0+k*rowH;
      d3.range(N).forEach(i=>svg.append("text").attr("x",xs(i)+xs.bandwidth()/2).attr("y",y0-10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("t = "+(i+1)));
      st.forEach((s,k)=>{
        const on=k<=upto, upd=new Set(s.ops.map(o=>o.i));
        svg.append("text").attr("x",8).attr("y",yr(k)+ch/2+4).attr("fill",on?C.ink:C.line).attr("font-size",11).text(s.name);
        if(k>0){
          s.ops.forEach(o=>o.src.forEach(j=>{
            if(o.self){
              svg.append("line").attr("x1",xs(o.i)+xs.bandwidth()/2).attr("x2",xs(o.i)+xs.bandwidth()/2).attr("y1",yr(k-1)+ch).attr("y2",yr(k))
                .attr("stroke",on?C.violet:C.line).attr("stroke-width",1.4);
              return;}
            svg.append("line").attr("x1",xs(j)+xs.bandwidth()/2).attr("y1",yr(k-1)+ch).attr("x2",xs(o.i)+xs.bandwidth()/2).attr("y2",yr(k))
              .attr("stroke",on?(o.copy?C.a2:C.accent):C.line).attr("stroke-width",on?1.4:1).attr("stroke-dasharray",o.copy?"4 3":null).attr("opacity",on?0.9:0.6);
          }));
        }
        s.cells.forEach((e,i)=>{
          const active=k===0||upd.has(i);
          svg.append("rect").attr("x",xs(i)).attr("y",yr(k)).attr("width",xs.bandwidth()).attr("height",ch).attr("rx",4)
            .attr("fill",on&&active?(k===S?"#1d2a2a":C.panel):"#11141a").attr("stroke",on&&active?(k===S?C.accent:C.muted):C.line);
          const span=empty(e)?"∅ (1, 0)":(e.lo===e.hi?""+e.lo:e.lo+"–"+e.hi);
          svg.append("text").attr("x",xs(i)+xs.bandwidth()/2).attr("y",yr(k)+(ch>26?14:ch/2+4)).attr("text-anchor","middle").attr("font-size",11)
            .attr("fill",on?(active?C.ink:C.muted):C.line).text(span);
          if(ch>26) svg.append("text").attr("x",xs(i)+xs.bandwidth()/2).attr("y",yr(k)+28).attr("text-anchor","middle").attr("font-size",9.5)
            .attr("fill",on?C.muted:C.line).text("b="+e.b.toFixed(3));
        });
      });
      const ref=seqH(), fin=st[S].cells, err=d3.max(fin.map((e,i)=>Math.abs(e.b-ref[i])));
      const combs=d3.sum(st.slice(1,upto+1),s=>s.nComb), total=d3.sum(st.slice(1),s=>s.nComb);
      const done=st[upto].cells.filter(e=>!empty(e)&&e.lo===1).length;
      const desc={seq:"one combine per stage: depth L − 1, work L − 1 — optimal work, no parallelism",
        hs:"every element combines with the one 2ᵏ back: depth log₂L, work ≈ L·log₂L",
        bl:"up-sweep tree, down-sweep of exclusive prefixes, then each token's own step: depth ≈ 2·log₂L, work O(L)"}[alg];
      d3.select("#scan-readout").html(
        `stage <b>${upto}</b> of ${S} · combines so far <b>${combs}</b> of <b>${total}</b> · cells holding a full prefix 1…t: <b>${done}</b> of ${N}`+
        `<br>${desc}`+
        `<br>final b-components vs the sequential recurrence hₜ = aₜhₜ₋₁ + bₜ: max |difference| = <b>${err.toExponential(1)}</b>`+
        (upto===S?` · h₈ = <b>${ref[N-1].toFixed(4)}</b>`:""));
    }
    d3.select("#scan-a").on("change",()=>draw(true));
    d3.select("#scan-s").on("input",()=>draw(false));
    d3.select("#scan-new").on("click",()=>{seed+=1;build();draw(false);});
    build();draw(true);
  })();

  /* ═══════════════ 3. memory / IO calculator ═══════════════ */
  (function(){
    const svg=d3.select("#io-svg"); if(svg.empty()) return;
    const W=760, H=300, m={l:250,r:110,t:34,b:36}, E=2, K=4;
    const U=["B","kB","MB","GB","TB","PB"];
    function bytes(n){let i=0;while(n>=1000&&i<U.length-1){n/=1000;i++;}return n.toFixed(n<10?2:1)+" "+U[i];}
    function tick(n){let i=0;while(n>=999&&i<U.length-1){n/=1000;i++;}return Math.round(n)+" "+U[i];}
    function draw(){
      svg.selectAll("*").remove();
      const Bt=Math.pow(2,+d3.select("#io-b").property("value")), L=Math.pow(2,+d3.select("#io-l").property("value"));
      const dm=+d3.select("#io-d").property("value"), N=+d3.select("#io-n").property("value"), s=+d3.select("#io-s").property("value");
      d3.select("#io-bv").text(Bt); d3.select("#io-lv").text(fmt(L)); d3.select("#io-nv").text(N);
      const D=E*dm;
      const naive=Bt*L*D*N*s;
      const fused=s*(3*Bt*L*D+2*Bt*L*N+D*N);
      const kv=2*Bt*L*dm*s;                         // one Transformer block, full MHA
      const st=2*Bt*D*(N+(K-1))*s;                  // two Mamba blocks
      const bars=[
        {g:"training · one SSM layer",lab:"expanded state (B, L, D, N) in HBM",v:naive,c:C.bad},
        {g:"",lab:"fused kernel HBM IO (in + out)",v:fused,c:C.accent},
        {g:"inference · same parameter budget",lab:"KV cache, 1 Transformer block",v:kv,c:C.a2},
        {g:"",lab:"recurrent state, 2 Mamba blocks",v:st,c:C.good}];
      const lo=Math.pow(10,Math.floor(Math.log10(d3.min(bars,b=>b.v)))), hi=Math.pow(10,Math.ceil(Math.log10(d3.max(bars,b=>b.v))));
      const x=d3.scaleLog().domain([lo,hi]).range([m.l,W-m.r]);
      const y=d3.scaleBand().domain(d3.range(bars.length)).range([m.t,H-m.b]).padding(0.3);
      const e0=Math.round(Math.log10(lo)), e1=Math.round(Math.log10(hi)), stepE=(e1-e0)>7?2:1;
      const tv=d3.range(e0,e1+1,stepE).map(e=>Math.pow(10,e));
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickValues(tv).tickFormat(tick)).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("bytes (decimal units), logarithmic axis — "+(stepE===1?"one decade":"two decades")+" per tick");
      bars.forEach((b,i)=>{
        if(b.g) svg.append("text").attr("x",8).attr("y",y(i)-6).attr("fill",C.ink).attr("font-size",11).attr("font-weight",600).text(b.g);
        svg.append("text").attr("x",m.l-8).attr("y",y(i)+y.bandwidth()/2+4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",11).text(b.lab);
        svg.append("rect").attr("x",m.l).attr("y",y(i)).attr("height",y.bandwidth()).attr("width",Math.max(1,x(b.v)-m.l)).attr("fill",b.c).attr("opacity",0.85);
        svg.append("text").attr("x",x(b.v)+6).attr("y",y(i)+y.bandwidth()/2+4).attr("fill",C.ink).attr("font-size",11).text(bytes(b.v));
      });
      d3.select("#io-readout").html(
        `D = E·d_model = ${fmt(D)} channels, N = ${N}, ${s} bytes/element · expanded state <b>${bytes(naive)}</b> vs fused IO <b>${bytes(fused)}</b> → <b>${(naive/fused).toFixed(1)}×</b>`+
        ` (≈ N/3 = ${(N/3).toFixed(1)} for D ≫ N; the naive path moves Ā, B̄ and h, each this size, several times, so the real gap is larger)`+
        `<br>inference at context ${fmt(L)}, batch ${Bt}: KV cache <b>${bytes(kv)}</b> vs Mamba state <b>${bytes(st)}</b> → <b>${(kv/st).toFixed(1)}×</b>`+
        ` · the KV cache doubles every time L doubles; the Mamba state, ${fmt(2*D*(N+K-1))} numbers per sequence, never changes`);
    }
    ["#io-b","#io-l","#io-d","#io-n","#io-s"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    draw();
  })();

  /* ═══════════════ 4. reported zero-shot accuracy vs size ═══════════════ */
  (function(){
    const svg=d3.select("#lm-svg"); if(svg.empty()) return;
    const W=760, H=320, m={l:56,r:150,t:24,b:44};
    const S=[
      {name:"Mamba",col:C.accent,pts:[["Mamba-130M",130,44.7],["Mamba-370M",370,50.0],["Mamba-790M",790,57.1],["Mamba-1.4B",1400,59.7],["Mamba-2.8B",2800,63.3]]},
      {name:"Pythia",col:C.a2,pts:[["Pythia-160M",160,40.6],["Pythia-410M",410,48.2],["Pythia-1B",1000,51.9],["Pythia-1.4B",1400,55.2],["Pythia-2.8B",2800,59.1],["Pythia-6.9B",6900,61.7]]},
      {name:"GPT-J",col:C.violet,pts:[["GPT-J-6B",6000,63.0]]}];
    const x=d3.scaleLog().domain([100,8000]).range([m.l,W-m.r]);
    const y=d3.scaleLinear().domain([38,66]).range([H-m.b,m.t]);
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickValues([100,200,500,1000,2000,5000]).tickFormat(d=>d>=1000?(d/1000)+"B":d+"M")).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)).attr("color",C.muted);
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("parameters (log scale)");
    svg.append("text").attr("x",m.l).attr("y",14).attr("fill",C.muted).attr("font-size",11).text("average zero-shot accuracy (%) — axis does not start at 0");
    const all=[].concat(...S.map(s=>s.pts));
    function nearest(p){ // closest model of another family by score, to make "twice its size" checkable
      const fam=p[0].split("-")[0];
      const others=all.filter(q=>q[0].split("-")[0]!==fam);
      return others.reduce((a,q)=>Math.abs(q[2]-p[2])<Math.abs(a[2]-p[2])?q:a,others[0]);
    }
    function show(p){
      const q=nearest(p);
      d3.select("#lm-readout").html(`<b>${p[0]}</b>: ${p[2].toFixed(1)}% · closest-scoring model of another family: <b>${q[0]}</b> at ${q[2].toFixed(1)}% with <b>${(q[1]/p[1]).toFixed(2)}×</b> the parameters`);
    }
    S.forEach((s,si)=>{
      if(s.pts.length>1) svg.append("path").attr("d",d3.line().x(d=>x(d[1])).y(d=>y(d[2]))(s.pts)).attr("fill","none").attr("stroke",s.col).attr("stroke-width",2);
      s.pts.forEach(p=>{
        svg.append("circle").attr("cx",x(p[1])).attr("cy",y(p[2])).attr("r",5).attr("fill",s.col).style("cursor","pointer")
          .on("mouseenter",()=>show(p)).append("title").text(`${p[0]}: ${p[2]}`);
      });
      svg.append("rect").attr("x",W-m.r+20).attr("y",m.t+20+si*20-6).attr("width",12).attr("height",3).attr("fill",s.col);
      svg.append("text").attr("x",W-m.r+38).attr("y",m.t+20+si*20).attr("fill",s.col).attr("font-size",11).text(s.name);
    });
    d3.select("#lm-readout").html(`Mamba is above Pythia at every size · <b>Mamba-2.8B 63.3</b> vs Pythia-6.9B 61.7 (2.5× the parameters) and GPT-J-6B 63.0 (2.1×) · hover any point to compare`);
  })();
})();
