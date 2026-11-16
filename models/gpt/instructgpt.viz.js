/* instructgpt.viz.js — interactive figures for models/gpt/instructgpt.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-ds  the prompt dataset: use-case categories, or labeller vs customer counts per stage (as reported)
     #viz-kl  a toy KL-regularised policy: pi*(y) ∝ pi_SFT(y)·exp(r(y)/beta), with two reward-hacking responses */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>n.toLocaleString("en-US");
  function axStyle(a){a.selectAll("text").attr("fill",C.muted);a.selectAll("line,path").attr("stroke",C.line);}

  /* ═══════════════ 1. the prompt dataset ═══════════════ */
  (function(){
    const svg=d3.select("#ds-svg"); if(svg.empty()) return;
    const W=640,H=270,m={l:120,r:70,t:14,b:26};
    const CAT=[["Generation",45.6],["Open QA",12.4],["Brainstorming",11.2],["Chat",8.4],["Rewrite",6.6],["Summarization",4.2],["Classification",3.5],["Other",3.5],["Closed QA",2.6],["Extract",1.9]];
    const SRC=[
      {s:"SFT · train",lab:11295,cus:1430},{s:"SFT · valid",lab:1550,cus:103},
      {s:"RM · train",lab:6623,cus:26584},{s:"RM · valid",lab:3488,cus:14399},
      {s:"PPO · train",lab:0,cus:31144},{s:"PPO · valid",lab:0,cus:16185}];
    const g=svg.append("g"), gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`), gy=svg.append("g").attr("transform",`translate(${m.l},0)`);
    const leg=svg.append("g");
    let mode="cat";
    function draw(){
      g.selectAll("*").remove(); leg.selectAll("*").remove();
      if(mode==="cat"){
        const y=d3.scaleBand().domain(CAT.map(c=>c[0])).range([m.t,H-m.b]).padding(0.18);
        const x=d3.scaleLinear().domain([0,50]).range([m.l,W-m.r]);
        gx.call(d3.axisBottom(x).ticks(5).tickFormat(d=>d+"%")).call(axStyle);
        gy.call(d3.axisLeft(y).tickSize(0)).call(axStyle); gy.selectAll("text").attr("fill",C.ink);
        const open=new Set(["Generation","Brainstorming"]), bench=new Set(["Classification","Closed QA","Extract"]);
        const b=g.selectAll("g").data(CAT).enter().append("g").attr("tabindex",0);
        b.append("rect").attr("x",m.l).attr("y",c=>y(c[0])).attr("height",y.bandwidth()).attr("rx",3)
          .attr("fill",c=>open.has(c[0])?C.accent:bench.has(c[0])?C.violet:C.muted).attr("opacity",.85)
          .attr("width",0).transition().duration(350).attr("width",c=>x(c[1])-m.l);
        b.append("text").attr("x",c=>x(c[1])+6).attr("y",c=>y(c[0])+y.bandwidth()/2+4).attr("font-size",11).attr("fill",C.ink).text(c=>c[1]+"%");
        b.on("mouseenter focus",(ev,c)=>d3.select("#ds-read").text(`${c[0]}: ${c[1]}% of the API prompts in the RM dataset.`));
        [["open-ended (generation, brainstorming)",C.accent],["benchmark-like (classification, closed QA, extract)",C.violet]].forEach((l,i)=>{
          leg.append("rect").attr("x",W-m.r-260).attr("y",H-m.b-58+i*16).attr("width",10).attr("height",10).attr("fill",l[1]);
          leg.append("text").attr("x",W-m.r-245).attr("y",H-m.b-49+i*16).attr("font-size",10.5).attr("fill",C.muted).text(l[0]);});
        d3.select("#ds-read").text("Open-ended generation and brainstorming: 56.8% of API prompts. Classification, closed QA and extraction: 8.0%. (Category shares as reported; sums are our arithmetic.)");
      } else {
        const y=d3.scaleBand().domain(SRC.map(s=>s.s)).range([m.t,H-m.b]).padding(0.18);
        const x=d3.scaleLinear().domain([0,34000]).range([m.l,W-m.r]);
        gx.call(d3.axisBottom(x).ticks(6).tickFormat(d=>d/1000+"k")).call(axStyle);
        gy.call(d3.axisLeft(y).tickSize(0)).call(axStyle); gy.selectAll("text").attr("fill",C.ink);
        const b=g.selectAll("g").data(SRC).enter().append("g").attr("tabindex",0);
        b.append("rect").attr("x",m.l).attr("y",s=>y(s.s)).attr("height",y.bandwidth()).attr("fill",C.a2).attr("width",0).transition().duration(350).attr("width",s=>x(s.lab)-m.l);
        b.append("rect").attr("x",s=>x(s.lab)).attr("y",s=>y(s.s)).attr("height",y.bandwidth()).attr("fill",C.blue).attr("width",0).transition().duration(350).attr("width",s=>x(s.lab+s.cus)-x(s.lab));
        b.append("text").attr("x",s=>x(s.lab+s.cus)+6).attr("y",s=>y(s.s)+y.bandwidth()/2+4).attr("font-size",11).attr("fill",C.ink).text(s=>fmt(s.lab+s.cus));
        b.on("mouseenter focus",(ev,s)=>d3.select("#ds-read").text(`${s.s}: ${fmt(s.lab)} labeller-written + ${fmt(s.cus)} customer prompts = ${fmt(s.lab+s.cus)} (${Math.round(100*s.cus/(s.lab+s.cus))}% from customers).`));
        [["labeller-written",C.a2],["customer (API)",C.blue]].forEach((l,i)=>{
          leg.append("rect").attr("x",W-m.r-120).attr("y",H-m.b-58+i*16).attr("width",10).attr("height",10).attr("fill",l[1]);
          leg.append("text").attr("x",W-m.r-105).attr("y",H-m.b-49+i*16).attr("font-size",10.5).attr("fill",C.muted).text(l[0]);});
        d3.select("#ds-read").text("SFT training prompts are 89% labeller-written; RM training prompts 80% customer; PPO prompts are all customer. (Counts as reported; percentages our arithmetic.)");
      }
    }
    d3.select("#ds-cat").on("click",function(){mode="cat";d3.select("#ds-src").classed("on",false);d3.select(this).classed("on",true);draw();});
    d3.select("#ds-src").on("click",function(){mode="src";d3.select("#ds-cat").classed("on",false);d3.select(this).classed("on",true);draw();});
    draw();
  })();

  /* ═══════════════ 2. KL-regularised policy toy ═══════════════ */
  (function(){
    const svg=d3.select("#kl-svg"); if(svg.empty()) return;
    const W=640,H=300,m={l:44,r:16,t:22,b:96};
    // invented toy values: SFT prior, reward-model score, and "true" human quality per response
    const Y=[
      {n:"off-task",       p:0.16, rm:-1.2, q:-1.2},
      {n:"terse",          p:0.22, rm:-0.2, q:-0.1},
      {n:"adequate",       p:0.24, rm: 0.4, q: 0.5},
      {n:"good",           p:0.18, rm: 1.0, q: 1.1},
      {n:"very good",      p:0.12, rm: 1.5, q: 1.6},
      {n:"excellent",      p:0.06, rm: 1.9, q: 2.0},
      {n:"hack: verbose",  p:0.015,rm: 2.6, q: 0.2},
      {n:"hack: flattery", p:0.005,rm: 3.0, q:-0.6}];
    const x=d3.scaleBand().domain(Y.map(d=>d.n)).range([m.l,W-m.r]).padding(0.2);
    const xi=d3.scaleBand().domain(["sft","pol"]).range([0,x.bandwidth()]).padding(0.08);
    const y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickSize(0)).call(axStyle)
      .selectAll("text").attr("fill",(d,i)=>i>=6?C.bad:C.ink).attr("font-size",10.5).attr("transform","rotate(-24)").attr("text-anchor","end").attr("dy","0.9em");
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).call(axStyle);
    svg.append("text").attr("x",m.l).attr("y",13).attr("font-size",11).attr("fill",C.muted).text("probability of each response:");
    svg.append("rect").attr("x",m.l+170).attr("y",4).attr("width",10).attr("height",10).attr("fill",C.muted);
    svg.append("text").attr("x",m.l+184).attr("y",13).attr("font-size",11).attr("fill",C.muted).text("SFT");
    svg.append("rect").attr("x",m.l+218).attr("y",4).attr("width",10).attr("height",10).attr("fill",C.accent);
    svg.append("text").attr("x",m.l+232).attr("y",13).attr("font-size",11).attr("fill",C.muted).text("optimised policy");
    const bars=svg.append("g");
    Y.forEach(d=>{
      bars.append("rect").attr("class","sft").datum(d).attr("x",x(d.n)+xi("sft")).attr("width",xi.bandwidth()).attr("y",y(d.p)).attr("height",H-m.b-y(d.p)).attr("fill",C.muted).attr("opacity",.6);
      bars.append("rect").attr("class","pol").datum(d).attr("x",x(d.n)+xi("pol")).attr("width",xi.bandwidth()).attr("fill",C.accent);
    });
    const stats=svg.append("g").attr("transform",`translate(${m.l},${H-20})`);
    const sTxt=stats.append("text").attr("font-size",11.5).attr("fill",C.ink);
    function policy(beta){
      const lw=Y.map(d=>Math.log(d.p)+d.rm/beta), mx=d3.max(lw);
      const w=lw.map(v=>Math.exp(v-mx)), Z=d3.sum(w);
      return w.map(v=>v/Z);
    }
    function draw(){
      const beta=Math.pow(10,+d3.select("#kl-b").property("value"));
      d3.select("#kl-bv").text(beta<0.1?beta.toFixed(3):beta.toFixed(2));
      const pi=policy(beta);
      bars.selectAll("rect.pol").each(function(d,i){d3.select(this).transition().duration(120).attr("y",y(pi[i])).attr("height",H-m.b-y(pi[i]));});
      const ERm=d3.sum(Y,(d,i)=>pi[i]*d.rm), EQ=d3.sum(Y,(d,i)=>pi[i]*d.q);
      const EQ0=d3.sum(Y,d=>d.p*d.q);
      const KL=d3.sum(Y,(d,i)=>pi[i]>1e-12?pi[i]*Math.log(pi[i]/d.p):0);
      const hack=pi[6]+pi[7];
      sTxt.text(`β = ${beta<0.1?beta.toFixed(3):beta.toFixed(2)} · reward-model score ${ERm.toFixed(2)} · true quality ${EQ.toFixed(2)} (SFT ${EQ0.toFixed(2)}) · KL ${KL.toFixed(2)} nats · mass on hacks ${(100*hack).toFixed(0)}%`);
      d3.select("#kl-read").text(
        hack>0.5 ? `β = ${beta.toFixed(3)}: the policy has collapsed onto responses the reward model over-scores — reward ${ERm.toFixed(2)} is high but true quality ${EQ.toFixed(2)} has fallen. This is reward over-optimisation.`
        : beta>3 ? `β = ${beta.toFixed(2)}: the KL anchor dominates and the policy stays close to SFT (KL ${KL.toFixed(2)} nats); little is gained.`
        : `β = ${beta.toFixed(2)}: the policy shifts toward better responses; true quality ${EQ.toFixed(2)} vs SFT ${EQ0.toFixed(2)}, at a KL cost of ${KL.toFixed(2)} nats. Lower β further and watch the hacks take over.`);
    }
    d3.select("#kl-b").on("input",draw);
    draw();
  })();
})();
