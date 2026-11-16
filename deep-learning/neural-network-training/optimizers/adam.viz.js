/* adam.viz.js — interactive figures for deep-learning/neural-network-training/optimizers/adam.html.
   Loaded after data.js / notes.js. Two independent IIFEs, both honest simulations:
     #viz-st  the effective step |Δt|/α of Adam, Adam without bias correction and AdaMax,
              run on a chosen synthetic gradient stream, against the original study's bounds
     #viz-ce  the later online-convex counterexample: Adam vs AMSGrad vs SGD on
              f_t(x) = C·x once every three steps, −x otherwise, x ∈ [−1, 1] */

(function(){
  "use strict";
  const C={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",teal:"#2dd4bf",violet:"#a78bfa"};
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
  function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  const f3=x=>(Math.abs(x)>=100?x.toFixed(0):Math.abs(x)>=10?x.toFixed(1):x.toFixed(3));

  /* ═══════════════ 1. the effective step, step by step ═══════════════ */
  (function(){
    const svg=d3.select("#st-svg"); if(svg.empty()) return;
    const T=3000, EPS=1e-8;
    let seed=5;
    const W=760, x0=58, x1=736, ty0=34, ty1=250, by0=292, by1=388;
    const xs=d3.scaleLog().domain([1,T]).range([x0,x1]);
    const ys=d3.scaleLinear().range([ty1,ty0]);
    const yg=d3.scaleLinear().range([by1,by0]);
    const gxT=svg.append("g").attr("transform",`translate(0,${ty1})`).attr("color",C.muted);
    const gyT=svg.append("g").attr("transform",`translate(${x0},0)`).attr("color",C.muted);
    const gxB=svg.append("g").attr("transform",`translate(0,${by1})`).attr("color",C.muted);
    const gyB=svg.append("g").attr("transform",`translate(${x0},0)`).attr("color",C.muted);
    svg.append("text").attr("x",x0).attr("y",ty0-14).attr("fill",C.muted).attr("font-size",11)
      .text("effective step |Δt| / α per step (log step axis)");
    svg.append("text").attr("x",x0).attr("y",by0-8).attr("fill",C.muted).attr("font-size",11)
      .text("the gradient stream g_t fed to every method");
    svg.append("text").attr("x",x1).attr("y",by1+30).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("step t");
    const lineA=svg.append("line").attr("x1",x0).attr("x2",x1).attr("stroke",C.ink).attr("stroke-dasharray","4 4").attr("opacity",0.7);
    const txtA=svg.append("text").attr("x",x1-4).attr("text-anchor","end").attr("fill",C.ink).attr("font-size",10.5);
    const lineB=svg.append("line").attr("x1",x0).attr("x2",x1).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
    const txtB=svg.append("text").attr("x",x0+6).attr("fill",C.bad).attr("font-size",10.5);
    const zeroG=svg.append("line").attr("x1",x0).attr("x2",x1).attr("stroke",C.line);
    const stems=svg.append("path").attr("stroke",C.muted).attr("stroke-width",1).attr("fill","none").attr("opacity",0.8);
    const paths={adam:svg.append("path"),raw:svg.append("path"),max:svg.append("path")};
    const col={adam:C.accent,raw:C.a2,max:C.teal};
    const name={adam:"Adam (Algorithm 1)",raw:"Adam without bias correction",max:"AdaMax (Algorithm 2)"};
    for(const k in paths) paths[k].attr("fill","none").attr("stroke",col[k]).attr("stroke-width",1.6);
    const leg=svg.append("g").attr("transform",`translate(${x0+12},${ty0+4})`);
    ["adam","raw","max"].forEach((k,i)=>{
      leg.append("rect").attr("x",0).attr("y",i*15).attr("width",14).attr("height",3).attr("fill",col[k]);
      leg.append("text").attr("x",20).attr("y",i*15+5).attr("fill",C.ink).attr("font-size",10.5).text(name[k]);
    });

    function stream(kind,b,b2s){
      const R=rng(seed), snr=+d3.select("#st-snr").property("value"), p=+d3.select("#st-p").property("value");
      const g=new Float64Array(T+1);
      for(let t=1;t<=T;t++){
        let v;
        if(kind==="const") v=1;
        else if(kind==="noisy") v=snr+gauss(R);
        else if(kind==="sparse") v=R()<p?1+0.3*gauss(R):0;
        else if(kind==="optimum") v=snr*(1-t/T)+gauss(R);
        else if(kind==="spike") v=(t===1500)?1:0;
        else if(kind==="shrink") v=Math.pow(b2s,t);                      // AdaMax's worst case: shrinks ×β₂ per step
        else { v=t<1000?1:Math.pow(1.11,Math.min(t,1100)-999); }   // grows ×1.11 per step for 100 steps
        g[t]=v*b;
      }
      return g;
    }
    function run(g,b1,b2){
      const out={adam:new Float64Array(T+1),raw:new Float64Array(T+1),max:new Float64Array(T+1)};
      let m=0,v=0,u=0;
      for(let t=1;t<=T;t++){
        const gt=g[t];
        m=b1*m+(1-b1)*gt; v=b2*v+(1-b2)*gt*gt; u=Math.max(b2*u,Math.abs(gt));
        const mh=m/(1-Math.pow(b1,t)), vh=v/(1-Math.pow(b2,t));
        out.adam[t]=Math.abs(mh/(Math.sqrt(vh)+EPS));
        out.raw[t]=Math.abs(m/(Math.sqrt(v)+EPS));
        out.max[t]=u>0?Math.abs(mh/u):0;                // Algorithm 2 has no ε; u = 0 only while m = 0
      }
      return out;
    }
    function render(){
      const kind=d3.select("#st-k").property("value");
      const b1=+d3.select("#st-b1").property("value"), b2=+d3.select("#st-b2").property("value");
      const scale=+d3.select("#st-sc").property("value");
      d3.select("#st-b1v").text(b1.toFixed(2));
      d3.select("#st-snrv").text((+d3.select("#st-snr").property("value")).toFixed(2));
      d3.select("#st-pv").text((+d3.select("#st-p").property("value")).toFixed(3));
      d3.select("#st-snr-l").style("opacity",(kind==="noisy"||kind==="optimum")?1:0.4);
      d3.select("#st-p-l").style("opacity",kind==="sparse"?1:0.4);
      const g=stream(kind,scale,b2), out=run(g,b1,b2);
      const show={adam:d3.select("#st-adam").property("checked"),raw:d3.select("#st-raw").property("checked"),max:d3.select("#st-max").property("checked")};
      const bnd=(1-b1)/Math.sqrt(1-b2), case1=(1-b1)>Math.sqrt(1-b2);
      let top=Math.max(1.2,bnd*1.08);
      for(const k in out) if(show[k]) for(let t=1;t<=T;t++) top=Math.max(top,out[k][t]*1.05);
      ys.domain([0,top]).nice();
      gxT.call(d3.axisBottom(xs).ticks(6,"~g"));
      gyT.call(d3.axisLeft(ys).ticks(6));
      let gm=0; for(let t=1;t<=T;t++) gm=Math.max(gm,Math.abs(g[t]));
      gm=gm||1; yg.domain([-gm,gm]);
      gxB.call(d3.axisBottom(xs).ticks(6,"~g"));
      gyB.call(d3.axisLeft(yg).ticks(3,"~g"));
      zeroG.attr("y1",yg(0)).attr("y2",yg(0));
      let sd=""; for(let t=1;t<=T;t++) if(g[t]!==0) sd+=`M${xs(t).toFixed(1)},${yg(0).toFixed(1)}V${yg(g[t]).toFixed(1)}`;
      stems.attr("d",sd);
      lineA.attr("y1",ys(1)).attr("y2",ys(1)); txtA.attr("y",ys(1)-4).text("|Δt| = α");
      lineB.attr("y1",ys(bnd)).attr("y2",ys(bnd)).style("display",case1?null:"none");
      txtB.attr("y",ys(bnd)-4).style("display",case1?null:"none").text("first-case bound (1 − β₁)/√(1 − β₂) = "+f3(bnd));
      const line=d3.line().x((d,i)=>xs(i+1)).y(d=>ys(d));
      for(const k in paths){
        paths[k].style("display",show[k]?null:"none");
        if(show[k]) paths[k].attr("d",line(Array.from(out[k]).slice(1)));
      }
      const stat=k=>{let mx=0,at=1,tail=0;for(let t=1;t<=T;t++){if(out[k][t]>mx){mx=out[k][t];at=t;}if(t>T-500)tail+=out[k][t];}return {first:out[k][1],mx,at,tail:tail/500};};
      const rows=["adam","raw","max"].filter(k=>show[k]).map(k=>{const s=stat(k);
        return `<span style="color:${col[k]}">${name[k]}</span>: first step <b>${f3(s.first)}</b> · largest <b>${f3(s.mx)}</b> at t = ${s.at} · mean of last 500 <b>${f3(s.tail)}</b>`+
          (k==="adam"&&case1&&s.mx>bnd*1.001?` · <span style="color:${C.bad}">exceeds the stated first-case bound</span>`:"")+
          (k==="adam"&&!case1&&s.mx>1.001?` · <span style="color:${C.bad}">exceeds α</span>`:"");}).join("<br>");
      d3.select("#st-readout").html(
        `(1 − β₁) = ${(1-b1).toFixed(3)} vs √(1 − β₂) = ${Math.sqrt(1-b2).toFixed(4)} → `+
        (case1?`<b>first case</b>: the study's bound is |Δt| ≤ α·${f3(bnd)}`:`<b>second case</b>: the study's bound is |Δt| ≤ α`)+
        ` · gradient scale ×${({"1e-9":"10⁻⁹","1e-6":"10⁻⁶","1":"1","1000":"10³"})[d3.select("#st-sc").property("value")]} · ε = 10⁻⁸<br>`+rows);
    }
    d3.selectAll("#st-k,#st-b1,#st-b2,#st-sc,#st-snr,#st-p,#st-adam,#st-raw,#st-max").on("input change",render);
    d3.select("#st-new").on("click",()=>{seed=(seed*7919+13)%100000;render();});
    render();
  })();

  /* ═══════════════ 2. the later counterexample ═══════════════ */
  (function(){
    const svg=d3.select("#ce-svg"); if(svg.empty()) return;
    const T=6000;
    const x0=58,x1=736, ay0=34, ay1=190, ry0=240, ry1=390;
    const xs=d3.scaleLog().domain([1,T]).range([x0,x1]);
    const yx=d3.scaleLinear().domain([-1.05,1.05]).range([ay1,ay0]);
    const yr=d3.scaleLog().range([ry1,ry0]).clamp(true);
    svg.append("g").attr("transform",`translate(0,${ay1})`).call(d3.axisBottom(xs).ticks(6,"~g")).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${x0},0)`).call(d3.axisLeft(yx).tickValues([-1,-0.5,0,0.5,1])).attr("color",C.muted);
    const gxR=svg.append("g").attr("transform",`translate(0,${ry1})`).attr("color",C.muted);
    const gyR=svg.append("g").attr("transform",`translate(${x0},0)`).attr("color",C.muted);
    svg.append("text").attr("x",x0).attr("y",ay0-14).attr("fill",C.muted).attr("font-size",11).text("iterate x_t (feasible set [−1, 1]; the best fixed point is x = −1)");
    svg.append("text").attr("x",x0).attr("y",ry0-12).attr("fill",C.muted).attr("font-size",11).text("average regret R(t)/t (log–log)");
    svg.append("line").attr("x1",x0).attr("x2",x1).attr("y1",yx(-1)).attr("y2",yx(-1)).attr("stroke",C.good).attr("stroke-dasharray","4 4");
    svg.append("text").attr("x",x1-4).attr("y",yx(-1)-4).attr("text-anchor","end").attr("fill",C.good).attr("font-size",10.5).text("optimum x* = −1");
    const floorL=svg.append("line").attr("x1",x0).attr("x2",x1).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
    const floorT=svg.append("text").attr("x",x1-4).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10.5);
    const meth=[{k:"adam",n:"Adam (β₁ = 0)",c:C.a2},{k:"ams",n:"AMSGrad (β₁ = 0)",c:C.accent},{k:"sgd",n:"SGD, α/√t",c:C.muted}];
    const pX={},pR={};
    meth.forEach(m=>{pX[m.k]=svg.append("path").attr("fill","none").attr("stroke",m.c).attr("stroke-width",1.6);
                     pR[m.k]=svg.append("path").attr("fill","none").attr("stroke",m.c).attr("stroke-width",1.6);});
    const leg=svg.append("g").attr("transform",`translate(${x0+300},${ay1-52})`);
    meth.forEach((m,i)=>{leg.append("rect").attr("y",i*15).attr("width",14).attr("height",3).attr("fill",m.c);
      leg.append("text").attr("x",20).attr("y",i*15+5).attr("fill",C.ink).attr("font-size",10.5).text(m.n);});

    function sim(k,Cc,b2,alpha){
      let x=1, v=0, vh=0, R=0; const X=new Float64Array(T+1), A=new Float64Array(T+1); X[0]=x;
      for(let t=1;t<=T;t++){
        const g=(t%3===1)?Cc:-1;
        R+=g*x-g*(-1);                              // f_t(x_t) − f_t(x*) with x* = −1
        A[t]=R/t;
        let step;
        if(k==="sgd") step=g;
        else{
          v=b2*v+(1-b2)*g*g;
          const vc=v/(1-Math.pow(b2,t));            // Algorithm 1's correction (β₁ = 0, so m̂ = g)
          if(k==="ams"){vh=Math.max(vh,vc);step=g/(Math.sqrt(vh)+1e-8);}
          else step=g/(Math.sqrt(vc)+1e-8);
        }
        x=Math.max(-1,Math.min(1,x-alpha/Math.sqrt(t)*step));   // α_t = α/√t, then project onto [−1, 1]
        X[t]=x;
      }
      return {X,A,R};
    }
    function render(){
      const Cc=+d3.select("#ce-c").property("value"), mode=d3.select("#ce-b2").property("value"), alpha=+d3.select("#ce-a").property("value");
      const b2=mode==="ce"?1/(1+Cc*Cc):+mode;
      d3.select("#ce-cv").text(Cc.toFixed(1)); d3.select("#ce-av").text(alpha.toFixed(2));
      const res={}; meth.forEach(m=>res[m.k]=sim(m.k,Cc,b2,alpha));
      let lo=Infinity,hi=0; meth.forEach(m=>{for(let t=1;t<=T;t++){const a=Math.abs(res[m.k].A[t]);if(a>0){lo=Math.min(lo,a);hi=Math.max(hi,a);}}});
      const flo=(2*Cc-4)/3; if(flo>0){lo=Math.min(lo,flo);hi=Math.max(hi,flo);}
      yr.domain([Math.max(lo*0.8,1e-4),hi*1.3]);
      gxR.call(d3.axisBottom(xs).ticks(6,"~g")); gyR.call(d3.axisLeft(yr).ticks(4,"~g"));
      const show=mode==="ce"&&flo>0;
      floorL.style("display",show?null:"none").attr("y1",yr(flo)).attr("y2",yr(flo));
      floorT.style("display",show?null:"none").attr("y",yr(flo)-4).text("lower bound on Adam's average regret, (2C − 4)/3 = "+flo.toFixed(3));
      const idx=d3.range(1,T+1);
      meth.forEach(m=>{
        pX[m.k].attr("d",d3.line().x(t=>xs(t)).y(t=>yx(res[m.k].X[t]))(idx));
        pR[m.k].attr("d",d3.line().x(t=>xs(t)).y(t=>yr(Math.max(Math.abs(res[m.k].A[t]),1e-4)))(idx));
      });
      d3.select("#ce-readout").html(
        `C = ${Cc.toFixed(1)} · β₂ = ${b2.toFixed(4)}${mode==="ce"?" (= 1/(1 + C²), the counterexample's choice)":""} · α_t = ${alpha.toFixed(2)}/√t · ${T} steps<br>`+
        meth.map(m=>`<span style="color:${m.c}">${m.n}</span>: final x = <b>${res[m.k].X[T].toFixed(3)}</b> · R(T)/T = <b>${(res[m.k].R/T).toFixed(4)}</b>`).join(" &nbsp;·&nbsp; ")+
        (Cc<=2?"<br>with C ≤ 2 the construction no longer forces a positive regret floor — the theorem needs C &gt; 2":""));
    }
    d3.selectAll("#ce-c,#ce-b2,#ce-a").on("input change",render);
    render();
  })();
})();
