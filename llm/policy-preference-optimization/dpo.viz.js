/* dpo.viz.js — interactive figures for llm/policy-preference-optimization/dpo.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-dp  simulated: DPO on a one-prompt, six-response toy vs the closed-form optimum
     #viz-jg  reported: judge vs human win rates and agreement (TL;DR human study, Table 2) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. DPO toy: everything computable ═══════════════ */
  (function(){
    const svg=d3.select("#dp-svg"); if(svg.empty()) return;
    const K=6, BETAS=[0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5], SWEEP=[0.3,0.5,1,2,5];
    const ETA=3, ETA_N=3, MAXSTEP=600, PER_FRAME=4, MAXGHOST=6;
    const PRESET={pr:[0.32,0.24,0.18,0.12,0.09,0.05], rs:[0.0,0.6,-0.8,1.2,-0.3,2.0]};
    const SUBS=["₁","₂","₃","₄","₅","₆"];
    const lab=i=>"y"+SUBS[i];

    function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    const sig=z=>1/(1+Math.exp(-z));
    const lse=v=>{const m=d3.max(v);return m+Math.log(d3.sum(v,x=>Math.exp(x-m)));};
    const softmax=v=>{const l=lse(v);return v.map(x=>Math.exp(x-l));};

    let toySeed=0, pairSeed=0, pr, lr, rs, pairs=[], ghosts=[], timer=null;
    let st;           // current run: {beta, mode, data, th, step, hist}

    function buildToy(){
      if(toySeed===0){pr=PRESET.pr.slice();rs=PRESET.rs.slice();}
      else{const R=rng(1000+toySeed*7919);pr=softmax(d3.range(K).map(()=>0.7*gauss(R)));rs=d3.range(K).map(()=>+(0.9*gauss(R)).toFixed(2));}
      lr=pr.map(Math.log);
    }
    const beta=()=>BETAS[+d3.select("#dp-b").property("value")];
    const dataMode=()=>d3.select("#dp-d").property("value");
    const gradMode=()=>d3.select("#dp-g").property("value");
    const nPairs=()=>+d3.select("#dp-n").property("value");

    function samplePairs(){
      const R=rng(500+toySeed*31+pairSeed*977), cdf=d3.cumsum(pr), n=nPairs();
      const draw=()=>{const u=R();let i=0;while(i<K-1&&u>cdf[i])i++;return i;};
      pairs=[];let guard=0;
      while(pairs.length<n&&guard++<100000){const a=draw(),b=draw();if(a===b)continue;
        pairs.push(R()<sig(rs[a]-rs[b])?[a,b]:[b,a]);}
    }
    const piStar=b=>softmax(lr.map((l,i)=>l+rs[i]/b));
    const logZ=b=>lse(lr.map((l,i)=>l+rs[i]/b));
    const KLr=p=>d3.sum(p,(x,i)=>x>1e-300?x*Math.log(x/pr[i]):0);
    const ER=p=>d3.sum(p,(x,i)=>x*rs[i]);

    // weighted list of (w, l, weight): population = all ordered pairs with BT label probability
    function pairList(data){
      if(data==="fin"){const w=1/Math.max(1,pairs.length);return pairs.map(([a,b])=>[a,b,w]);}
      const L=[];let tot=0;
      for(let i=0;i<K;i++)for(let j=0;j<K;j++){if(i===j)continue;const w=pr[i]*pr[j]*sig(rs[i]-rs[j]);L.push([i,j,w]);tot+=w;}
      return L.map(([a,b,w])=>[a,b,w/tot]);
    }
    function dpoLoss(th,b,L){let s=0;for(const [w,l,q] of L){const m=b*((th[w]-th[l])-(lr[w]-lr[l]));s+=q*Math.log1p(Math.exp(-m));}return s;}
    function gradStep(S,L){
      const g=new Float64Array(K);
      if(S.mode==="dpo"){
        for(const [w,l,q] of L){const m=S.beta*((S.th[w]-S.th[l])-(lr[w]-lr[l]));const wt=q*S.beta*sig(-m);g[w]-=wt;g[l]+=wt;}
        const eta=ETA/(S.beta*S.beta);
        for(let i=0;i<K;i++)S.th[i]-=eta*g[i];
      }else{
        for(const [w,l,q] of L){g[w]-=q;g[l]+=q;}
        for(let i=0;i<K;i++)S.th[i]-=ETA_N*g[i];
      }
      S.step++;
    }
    function record(S,L){const p=softmax(S.th);S.hist.push({s:S.step,kl:KLr(p),er:ER(p),loss:dpoLoss(S.th,S.beta,L)});}
    function newRun(b){const S={beta:b,mode:gradMode(),data:dataMode(),th:lr.slice(),step:0,hist:[]};record(S,pairList(S.data));return S;}
    function simulate(b){const S=newRun(b),L=pairList(S.data);while(S.step<MAXSTEP){gradStep(S,L);record(S,L);}return S;}

    // ── layout ──
    const A={x0:50,x1:370,y0:40,y1:178}, B={x0:455,x1:740,y0:40,y1:178}, F={x0:50,x1:400,y0:290,y1:480}, Dr={x0:470,x1:740,y0:290,y1:370}, Dk={x0:470,x1:740,y0:400,y1:480};
    const gA=svg.append("g"), gB=svg.append("g"), gF=svg.append("g"), gD=svg.append("g");
    const txt=(g,x,y,s,o={})=>g.append("text").attr("x",x).attr("y",y).attr("fill",o.fill||C.muted).attr("font-size",o.fs||11).attr("text-anchor",o.anchor||"start").text(s);
    txt(svg,A.x0,18,"policy over the six responses: π_ref (outline) · π* closed form (amber tick) · π_θ (filled)");
    txt(svg,B.x0,18,"implicit reward r̂ = β·log(π_θ/π_ref) vs true r*");
    txt(svg,F.x0,262,"expected true reward vs KL(π_θ‖π_ref): exact optimal frontier and training runs");
    txt(svg,Dr.x0,262,"E[r*] and KL vs training step (dashed: values at π*)");
    txt(svg,760-6,514,"simulated toy, not data from the study",{anchor:"end",fs:10});

    function render(){
      const S=st, b=S.beta, p=softmax(S.th), ps=piStar(b), lz=logZ(b);
      const rhat=p.map((x,i)=>b*(Math.log(x)-lr[i]));
      // A: bars
      gA.selectAll("*").remove();
      const xa=d3.scaleBand().domain(d3.range(K)).range([A.x0,A.x1]).padding(0.25);
      const ymax=Math.min(1,Math.max(d3.max(p),d3.max(ps),d3.max(pr))*1.08);
      const ya=d3.scaleLinear().domain([0,ymax]).range([A.y1,A.y0]).nice();
      gA.append("g").attr("transform",`translate(${A.x0},0)`).call(d3.axisLeft(ya).ticks(4)).attr("color",C.muted);
      gA.append("line").attr("x1",A.x0).attr("x2",A.x1).attr("y1",A.y1).attr("y2",A.y1).attr("stroke",C.muted);
      for(let i=0;i<K;i++){
        const x=xa(i), w=xa.bandwidth();
        gA.append("rect").attr("x",x+w*0.18).attr("width",w*0.64).attr("y",ya(p[i])).attr("height",A.y1-ya(p[i])).attr("fill",S.mode==="dpo"?C.accent:C.bad).attr("opacity",0.85);
        gA.append("rect").attr("x",x).attr("width",w).attr("y",ya(pr[i])).attr("height",A.y1-ya(pr[i])).attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.2);
        gA.append("line").attr("x1",x-3).attr("x2",x+w+3).attr("y1",ya(ps[i])).attr("y2",ya(ps[i])).attr("stroke",C.a2).attr("stroke-width",2.2).attr("stroke-dasharray","5 3");
        txt(gA,x+w/2,A.y1+14,lab(i),{anchor:"middle",fill:C.ink});
        txt(gA,x+w/2,A.y1+27,"r* "+rs[i].toFixed(1),{anchor:"middle",fs:10});
      }
      // B: scatter implicit vs true
      gB.selectAll("*").remove();
      const rx=d3.extent(rs), padx=0.3;
      const xb=d3.scaleLinear().domain([rx[0]-padx,rx[1]+padx]).range([B.x0,B.x1]);
      const tline=rs.map(r=>r-b*lz);
      let lo=Math.min(d3.min(rhat),d3.min(tline)), hi=Math.max(d3.max(rhat),d3.max(tline));
      lo=Math.max(lo,-40);hi=Math.min(hi,40);if(hi-lo<1){const m=(hi+lo)/2;lo=m-0.5;hi=m+0.5;}
      const yb=d3.scaleLinear().domain([lo-0.1*(hi-lo),hi+0.1*(hi-lo)]).range([B.y1,B.y0]).nice();
      gB.append("g").attr("transform",`translate(0,${B.y1})`).call(d3.axisBottom(xb).ticks(5)).attr("color",C.muted);
      gB.append("g").attr("transform",`translate(${B.x0},0)`).call(d3.axisLeft(yb).ticks(4)).attr("color",C.muted);
      const xr=xb.domain();
      gB.append("line").attr("x1",xb(xr[0])).attr("x2",xb(xr[1])).attr("y1",yb(xr[0]-b*lz)).attr("y2",yb(xr[1]-b*lz)).attr("stroke",C.a2).attr("stroke-dasharray","5 4");
      txt(gB,B.x1,B.y0-4,"dashed: r̂ = r* − β·log Z",{anchor:"end",fill:C.a2,fs:10});
      txt(gB,(B.x0+B.x1)/2,B.y1+28,"true reward r*",{anchor:"middle",fs:10});
      for(let i=0;i<K;i++){
        const cy=yb(Math.max(yb.domain()[0],Math.min(yb.domain()[1],rhat[i])));
        gB.append("circle").attr("cx",xb(rs[i])).attr("cy",cy).attr("r",4.5).attr("fill",S.mode==="dpo"?C.accent:C.bad);
        txt(gB,xb(rs[i])+6,cy-5,lab(i),{fill:C.ink,fs:10});
      }
      // F: frontier
      gF.selectAll("*").remove();
      const klmax=-Math.log(d3.min(pr));
      const xf=d3.scaleLinear().domain([0,klmax*1.04]).range([F.x0,F.x1]);
      const yf=d3.scaleLinear().domain([Math.min(d3.min(rs),ER(pr))-0.1,d3.max(rs)+0.15]).range([F.y1,F.y0]);
      gF.append("g").attr("transform",`translate(0,${F.y1})`).call(d3.axisBottom(xf).ticks(6)).attr("color",C.muted);
      gF.append("g").attr("transform",`translate(${F.x0},0)`).call(d3.axisLeft(yf).ticks(5)).attr("color",C.muted);
      txt(gF,(F.x0+F.x1)/2,F.y1+30,"KL(π_θ ‖ π_ref), nats",{anchor:"middle",fs:10});
      const fr=d3.range(-1.7,2.01,0.02).map(e=>{const q=piStar(Math.pow(10,e));return [KLr(q),ER(q)];});
      gF.append("path").attr("d",d3.line().x(d=>xf(d[0])).y(d=>yf(d[1]))(fr)).attr("fill","none").attr("stroke",C.ink).attr("stroke-width",1.4).attr("opacity",0.7);
      txt(gF,F.x1-4,yf(d3.max(rs))-6,"optimal frontier (π* as β varies)",{anchor:"end",fs:10,fill:C.ink});
      gF.append("circle").attr("cx",xf(0)).attr("cy",yf(ER(pr))).attr("r",4).attr("fill",C.muted);
      txt(gF,xf(0)+7,yf(ER(pr))+14,"π_ref",{fs:10});
      const line=d3.line().x(d=>xf(d.kl)).y(d=>yf(d.er));
      ghosts.forEach(G=>{
        gF.append("path").attr("d",line(G.hist)).attr("fill","none").attr("stroke",G.mode==="dpo"?C.blue:C.bad).attr("stroke-width",1.2).attr("opacity",0.45);
        const e=G.hist[G.hist.length-1];
        gF.append("circle").attr("cx",xf(e.kl)).attr("cy",yf(e.er)).attr("r",3).attr("fill",G.mode==="dpo"?C.blue:C.bad).attr("opacity",0.8);
        txt(gF,xf(e.kl)+5,yf(e.er)+12,"β="+G.beta+(G.mode==="dpo"?"":" naive"),{fs:9.5});
      });
      gF.append("circle").attr("cx",xf(KLr(ps))).attr("cy",yf(ER(ps))).attr("r",6.5).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2);
      gF.append("path").attr("d",line(S.hist)).attr("fill","none").attr("stroke",S.mode==="dpo"?C.accent:C.bad).attr("stroke-width",2.2);
      const last=S.hist[S.hist.length-1];
      gF.append("circle").attr("cx",xf(last.kl)).attr("cy",yf(last.er)).attr("r",4).attr("fill",S.mode==="dpo"?C.accent:C.bad);
      // D: E[r] and KL vs step
      gD.selectAll("*").remove();
      const xd=d3.scaleLinear().domain([0,MAXSTEP]).range([Dr.x0,Dr.x1]);
      [[Dr,"er",ER(ps),yf.domain(),"E[r*]"],[Dk,"kl",KLr(ps),[0,klmax*1.04],"KL"]].forEach(([P,key,target,dom,name])=>{
        const y=d3.scaleLinear().domain(dom).range([P.y1,P.y0]);
        gD.append("g").attr("transform",`translate(0,${P.y1})`).call(d3.axisBottom(xd).ticks(5).tickFormat(P===Dk?d3.format("d"):()=>"")).attr("color",C.muted);
        gD.append("g").attr("transform",`translate(${P.x0},0)`).call(d3.axisLeft(y).ticks(3)).attr("color",C.muted);
        gD.append("line").attr("x1",P.x0).attr("x2",P.x1).attr("y1",y(target)).attr("y2",y(target)).attr("stroke",C.a2).attr("stroke-dasharray","5 4");
        gD.append("path").attr("d",d3.line().x(d=>xd(d.s)).y(d=>y(Math.max(dom[0],Math.min(dom[1],d[key]))))(S.hist)).attr("fill","none").attr("stroke",S.mode==="dpo"?C.accent:C.bad).attr("stroke-width",1.8);
        txt(gD,P.x1,P.y0+10,name,{anchor:"end",fill:C.ink,fs:10});
      });
      txt(gD,(Dk.x0+Dk.x1)/2,Dk.y1+30,"training step",{anchor:"middle",fs:10});

      // readout
      const L1=d3.sum(p,(x,i)=>Math.abs(x-ps[i]));
      const diff=rhat.map((v,i)=>v-rs[i]), spread=d3.max(diff)-d3.min(diff), off=d3.mean(diff);
      const top=d3.maxIndex(p);
      const dataTxt=S.data==="pop"?"population pairs (exact)":`${pairs.length} sampled pairs`;
      let html=`<b>simulated</b> · step <b>${S.step}</b> / ${MAXSTEP} · β = <b>${b}</b> · ${dataTxt} · DPO loss <b>${last.loss.toFixed(4)}</b> (log 2 = 0.6931 at the start)`+
        `<br>KL(π_θ‖π_ref) = <b>${last.kl.toFixed(3)}</b> (π*: ${KLr(ps).toFixed(3)}) · E[r*] = <b>${last.er.toFixed(3)}</b> (π*: ${ER(ps).toFixed(3)}; π_ref: ${ER(pr).toFixed(3)}) · ‖π_θ − π*‖₁ = <b>${L1.toExponential(1)}</b>`;
      if(S.mode==="dpo"){
        html+=`<br>implicit vs true reward: spread of r̂ − r* across responses <b>${spread.toExponential(1)}</b> (0 means equal up to a constant) · mean offset ${off.toFixed(3)} vs −β·log Z = ${(-b*lz).toFixed(3)}`;
        if(S.data==="fin"&&S.step>0) html+=` · with finite pairs DPO fits the sampled data, so it lands near π* rather than on it; a response that wins every one of its comparisons keeps overshooting`;
      }else{
        html+=`<br>unweighted update: every pair pushes equally forever, so the logits grow without bound and π_θ is collapsing onto <b>${lab(top)}</b> (π = ${p[top].toFixed(3)}, r* = ${rs[top]}) · β has no effect on this objective · best response is ${lab(d3.maxIndex(rs))}`;
      }
      d3.select("#dp-readout").html(html);
    }

    function pushGhost(){if(st&&st.step>0){ghosts.push(st);if(ghosts.length>MAXGHOST)ghosts.shift();}}
    function stop(){if(timer){timer.stop();timer=null;}d3.select("#dp-play").text(st&&st.step>=MAXSTEP?"done":"train");}
    function reset(keep){stop();if(keep)pushGhost();st=newRun(beta());render();d3.select("#dp-play").text("train");}
    function play(){
      if(timer){stop();return;}
      if(st.step>=MAXSTEP)reset(true);
      d3.select("#dp-play").text("pause");
      const L=pairList(st.data);
      timer=d3.timer(()=>{for(let i=0;i<PER_FRAME&&st.step<MAXSTEP;i++){gradStep(st,L);record(st,L);}render();if(st.step>=MAXSTEP)stop();});
    }
    function syncLabels(){d3.select("#dp-bv").text(beta());d3.select("#dp-n").attr("aria-valuetext",nPairs()+" pairs");d3.select("#dp-nv").text(nPairs());
      d3.select("#dp-b").attr("aria-valuetext","beta "+beta());}
    d3.select("#dp-b").on("input",()=>{syncLabels();reset(true);});
    d3.select("#dp-g").on("change",()=>{reset(true);});
    d3.select("#dp-d").on("change",()=>{ghosts=[];reset(false);});
    d3.select("#dp-n").on("input",()=>{syncLabels();samplePairs();if(dataMode()==="fin"){ghosts=[];reset(false);}});
    d3.select("#dp-play").on("click",play);
    d3.select("#dp-reset").on("click",()=>reset(true));
    d3.select("#dp-resample").on("click",()=>{pairSeed++;samplePairs();if(dataMode()==="fin"){ghosts=[];}reset(false);});
    d3.select("#dp-sweep").on("click",()=>{stop();ghosts=SWEEP.map(simulate);st=newRun(beta());render();});
    d3.select("#dp-new").on("click",()=>{toySeed++;buildToy();samplePairs();ghosts=[];reset(false);});
    syncLabels();buildToy();samplePairs();st=newRun(beta());render();
  })();

  /* ═══════════════ 2. reported: GPT-4 judge vs humans (study Table 2) ═══════════════ */
  (function(){
    const svg=d3.select("#jg-svg"); if(svg.empty()) return;
    const GROUPS=["DPO (temp 0.25)","SFT (temp 0.25)","PPO-1 (temp 1.0)"], NRESP=[272,122,199];
    const WIN={s:[47,27,13],c:[54,32,12],h:[58,43,17]};
    const AGR={s:[70,77,86],c:[67,79,85],h:[65,null,87]};
    const W=760,H=320,m={l:60,r:180,t:30,b:50};
    function draw(){
      svg.selectAll("*").remove();
      const view=d3.select("#jg-v").property("value");
      const keys=view==="win"?[["s","GPT-4 (S) win %",C.a2],["c","GPT-4 (C) win %",C.accent],["h","human win %",C.ink]]
                             :[["s","GPT-4 (S)–human",C.a2],["c","GPT-4 (C)–human",C.accent],["h","human–human",C.ink]];
      const D=view==="win"?WIN:AGR;
      const x0=d3.scaleBand().domain(GROUPS).range([m.l,W-m.r]).padding(0.22);
      const x1=d3.scaleBand().domain(keys.map(k=>k[0])).range([0,x0.bandwidth()]).padding(0.1);
      const y=d3.scaleLinear().domain([0,100]).range([H-m.b,m.t]);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x0)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",16).attr("fill",C.muted).attr("font-size",11)
        .text(view==="win"?"win rate (%) against PPO at temperature 0, TL;DR summaries":"per-judgment agreement (%) between raters");
      if(view==="win"){svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(50)).attr("y2",y(50)).attr("stroke",C.muted).attr("stroke-dasharray","4 4");
        svg.append("text").attr("x",W-m.r+4).attr("y",y(50)+4).attr("fill",C.muted).attr("font-size",10).text("50%: parity");}
      GROUPS.forEach((g,gi)=>{
        keys.forEach(([k,name,col])=>{
          const v=D[k][gi], x=x0(g)+x1(k);
          if(v===null){svg.append("text").attr("x",x+x1.bandwidth()/2).attr("y",y(0)-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text("n/c");return;}
          svg.append("rect").attr("x",x).attr("width",x1.bandwidth()).attr("y",y(v)).attr("height",y(0)-y(v)).attr("fill",col).attr("opacity",0.85).style("cursor","pointer")
            .on("mouseenter",()=>hover(gi,name,v)).append("title").text(`${g} · ${name}: ${v}`);
          svg.append("text").attr("x",x+x1.bandwidth()/2).attr("y",y(v)-4).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(v);
        });
      });
      keys.forEach(([k,name,col],i)=>{
        svg.append("rect").attr("x",W-m.r+20).attr("y",m.t+30+i*20).attr("width",12).attr("height",12).attr("fill",col);
        svg.append("text").attr("x",W-m.r+38).attr("y",m.t+40+i*20).attr("fill",col).attr("font-size",11).text(name);
      });
      if(view==="agree") svg.append("text").attr("x",W-m.r+20).attr("y",m.t+110).attr("fill",C.muted).attr("font-size",10).text("n/c = not collected (one rater)");
      summary(view);
    }
    function summary(view){
      if(view==="win"){
        const gs=WIN.s.map((v,i)=>Math.abs(v-WIN.h[i])), gc=WIN.c.map((v,i)=>Math.abs(v-WIN.h[i]));
        d3.select("#jg-readout").html(`reported (study Table 2) · |judge − human| win-rate gap, DPO / SFT / PPO-1: simple prompt <b>${gs.join(" / ")}</b> (mean <b>${d3.mean(gs).toFixed(1)}</b>), concise prompt <b>${gc.join(" / ")}</b> (mean <b>${d3.mean(gc).toFixed(1)}</b>), our arithmetic · both prompts rank the three methods in the same order as humans · respondents: ${NRESP.join(" / ")}`);
      }else{
        d3.select("#jg-readout").html(`reported (study Table 2) · judge–human agreement 67–86% vs human–human 65% (DPO) and 87% (PPO-1) · the judge agrees with a human about as often as two humans agree with each other · SFT had one rater per comparison, so no human–human figure`);
      }
    }
    function hover(gi,name,v){d3.select("#jg-readout").html(`<b>${GROUPS[gi]}</b> vs greedy PPO · ${name}: <b>${v}%</b> · ${NRESP[gi]} respondents (reported)`);}
    d3.select("#jg-v").on("change",draw);
    draw();
  })();
})();
