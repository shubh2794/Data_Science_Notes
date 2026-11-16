/* ci-cd.viz.js — interactive figures for ci-cd.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: gated pipeline simulator ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#1b1f27"};
  const svg=d3.select("#pp-svg"); if(svg.empty()) return;
  const W=680,H=170;
  const SE=0.003, DELTA=0.005;

  const val=id=>+d3.select(id).property("value");
  function inputs(){
    return {unit:d3.select("#pp-unit").property("checked"), nul:val("#pp-null"), psi:val("#pp-psi"),
            dauc:val("#pp-auc"), slice:val("#pp-slice"), lat:val("#pp-lat"), order:d3.select("#pp-order").property("value")};
  }
  const STAGE={
    commit:{name:["commit"],min:1,check:()=>[true,"change pushed"]},
    unit:{name:["unit","tests"],min:3,check:s=>s.unit?[false,"unit test failed: transform returns wrong shape"]:[true,"all unit tests pass"]},
    data:{name:["data","validation"],min:5,check:s=>{
      if(s.nul>2) return [false,`null rate ${s.nul.toFixed(1)}% > 2% fail threshold`];
      if(s.psi>=0.25) return [false,`PSI ${s.psi.toFixed(2)} ≥ 0.25 — distribution shift, quarantine batch`];
      return [true, s.psi>=0.1?`passed with WARN (PSI ${s.psi.toFixed(2)} ≥ 0.1)`:"schema, nulls and distribution OK"];}},
    train:{name:["train"],min:90,check:()=>[true,"model trained"]},
    quality:{name:["quality","gate"],min:10,check:s=>{
      const lb=s.dauc-1.96*SE;
      return lb>=-DELTA?[true,`ΔAUC lower bound ${lb.toFixed(4)} ≥ −${DELTA}`]:[false,`ΔAUC lower bound ${lb.toFixed(4)} < −${DELTA}: not non-inferior to champion`];}},
    slice:{name:["slice","gate"],min:5,check:s=>s.slice>5?[false,`worst slice regresses ${s.slice.toFixed(1)} pts > 5 pt limit`]:[true,"no slice regresses beyond 5 pts"]},
    perf:{name:["perf","gate"],min:15,check:s=>s.lat>100?[false,`p99 ${s.lat} ms > 100 ms budget`]:[true,`p99 ${s.lat} ms within budget`]},
    staging:{name:["staging","integr."],min:20,check:()=>[true,"end-to-end and parity tests pass"]},
    canary:{name:["canary"],min:120,check:()=>[true,"canary guardrails hold"]},
    prod:{name:["prod"],min:0,check:()=>[true,"serving 100%"]}
  };
  const ORDER={good:["commit","unit","data","train","quality","slice","perf","staging","canary","prod"],
               bad:["commit","unit","train","data","quality","slice","perf","staging","canary","prod"]};

  const n=10, gap=8, bw=(W-20-gap*(n-1))/n, bh=58, y0=40;
  const g=svg.append("g").attr("transform","translate(10,0)");
  const conn=g.append("g"), boxes=g.append("g");
  const token=g.append("circle").attr("r",6).attr("fill",C.ink).attr("cy",y0-14).style("display","none");

  function layout(order){
    conn.selectAll("*").remove(); boxes.selectAll("*").remove();
    order.forEach((k,i)=>{
      const x=i*(bw+gap);
      if(i>0) conn.append("line").attr("x1",x-gap).attr("x2",x).attr("y1",y0+bh/2).attr("y2",y0+bh/2).attr("stroke",C.muted);
      const b=boxes.append("g").attr("class","stg").attr("data-k",k).attr("transform",`translate(${x},${y0})`);
      b.append("rect").attr("width",bw).attr("height",bh).attr("rx",7).attr("fill",C.panel).attr("stroke",C.line).attr("stroke-width",1.5);
      STAGE[k].name.forEach((t,j)=>b.append("text").attr("x",bw/2).attr("y",bh/2-4+j*13-(STAGE[k].name.length-1)*6)
        .attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(t));
      b.append("text").attr("x",bw/2).attr("y",bh+16).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5)
        .text(STAGE[k].min?STAGE[k].min+" min":"");
      b.append("text").attr("class","mark").attr("x",bw/2).attr("y",-6).attr("text-anchor","middle").attr("font-size",12).attr("font-weight",700);
    });
    g.selectAll("text.legend-t").remove();
    g.append("text").attr("class","legend-t").attr("x",0).attr("y",H-14).attr("fill",C.muted).attr("font-size",10)
      .text("grey = not reached · amber = running · green = passed · red = blocked");
  }
  function paint(k,state,label){
    const b=boxes.select(`g.stg[data-k="${k}"]`);
    const col={idle:C.line,run:C.a2,pass:C.good,fail:C.bad,skip:"#3a4150"}[state];
    b.select("rect").attr("stroke",col).attr("fill",state==="fail"?"rgba(248,113,113,.12)":state==="pass"?"rgba(74,222,128,.08)":C.panel);
    b.select("text.mark").attr("fill",col).text(label||"");
  }

  let timer=null;
  function preview(){
    const s=inputs();
    d3.select("#pp-null-v").text(s.nul.toFixed(1)+"%");
    d3.select("#pp-psi-v").text(s.psi.toFixed(2));
    d3.select("#pp-auc-v").text((s.dauc>=0?"+":"")+s.dauc.toFixed(4));
    d3.select("#pp-slice-v").text(s.slice.toFixed(1));
    d3.select("#pp-lat-v").text(s.lat);
  }
  function reset(){
    if(timer){timer.stop();timer=null;}
    const s=inputs(); preview(); layout(ORDER[s.order]);
    ORDER[s.order].forEach(k=>paint(k,"idle"));
    token.style("display","none");
    d3.select("#pp-state").html(`<span style="color:${C.muted}">● press run to push this change through the pipeline</span>`);
    d3.select("#pp-read").html("");
  }
  function run(){
    reset();
    const s=inputs(), order=ORDER[s.order];
    let i=0, elapsed=0; const log=[];
    token.style("display",null).raise();
    function tick(){
      const k=order[i]; const x=i*(bw+gap)+bw/2;
      token.transition().duration(220).attr("cx",x);
      paint(k,"run","…");
      timer=d3.timeout(()=>{
        const [ok,msg]=STAGE[k].check(s);
        elapsed+=STAGE[k].min;
        log.push(`${ok?"✓":"✗"} ${STAGE[k].name.join(" ")}: ${msg}`);
        paint(k,ok?"pass":"fail",ok?"✓":"✗");
        if(!ok){
          order.slice(i+1).forEach(r=>paint(r,"skip"));
          const wasted = k==="data" && s.order==="bad" ? ` — <b style="color:${C.bad}">90 min of training were spent before a 5-minute check found the problem</b>`:"";
          d3.select("#pp-state").html(`<span style="color:${C.bad}">● BLOCKED at ${STAGE[k].name.join(" ")} after ${elapsed} min</span>`);
          d3.select("#pp-read").html(log.join("<br>")+wasted);
          token.attr("fill",C.bad); return;
        }
        i++;
        if(i>=order.length){
          d3.select("#pp-state").html(`<span style="color:${C.good}">● PROMOTED to production after ${elapsed} min</span>`);
          d3.select("#pp-read").html(log.join("<br>")); token.attr("fill",C.good); return;
        }
        d3.select("#pp-read").html(log.join("<br>"));
        tick();
      },380);
    }
    token.attr("fill",C.ink).attr("cx",bw/2);
    tick();
  }
  ["#pp-null","#pp-psi","#pp-auc","#pp-slice","#pp-lat"].forEach(id=>d3.select(id).on("input",preview));
  d3.select("#pp-order").on("change",reset);
  d3.select("#pp-unit").on("change",preview);
  d3.select("#pp-run").on("click",run);
  reset();
})();

/* ---- Viz 2: operating characteristic of a non-inferiority gate ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#oc-svg"); if(svg.empty()) return;
  const W=640,H=230,m={t:14,r:18,b:36,l:48};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  // standard normal CDF via an erf approximation (Abramowitz–Stegun 7.1.26)
  function Phi(z){const t=1/(1+0.3275911*Math.abs(z)/Math.SQRT2);
    const e=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-z*z/2);
    return z>=0?0.5*(1+e):0.5*(1-e);}
  const x=d3.scaleLinear().domain([-0.03,0.02]).range([0,iw]);
  const y=d3.scaleLinear().domain([0,1]).range([ih,0]);
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(x).ticks(10).tickFormat(d3.format("+.3f")));
  g.append("g").attr("class","axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",9);
  g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
  g.append("text").attr("x",iw/2).attr("y",ih+30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("true difference Δ = candidate − champion (AUC)");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-36).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("P(gate passes)");
  const badZone=g.append("rect").attr("y",0).attr("height",ih).attr("fill","rgba(248,113,113,.08)");
  const ideal=g.append("path").attr("fill","none").attr("stroke",C.muted).attr("stroke-dasharray","4 3");
  const curve=g.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2);
  const zl=g.append("line").attr("y1",0).attr("y2",ih).attr("stroke",C.muted).attr("stroke-opacity",.4);
  zl.attr("x1",x(0)).attr("x2",x(0));

  function draw(){
    const nEval=Math.round(Math.pow(10,val("#oc-n"))), d=val("#oc-d");
    const se=0.3/Math.sqrt(nEval);
    d3.select("#oc-n-v").text(nEval.toLocaleString());
    d3.select("#oc-d-v").text(d.toFixed(4));
    const xs=d3.range(-0.03,0.02001,0.0005);
    curve.attr("d",d3.line().x(v=>x(v)).y(v=>y(Phi((v+d)/se-1.96)))(xs));
    ideal.attr("d",d3.line().curve(d3.curveStepAfter).x(v=>x(v)).y(v=>y(v>=-d?1:0))(xs));
    badZone.attr("x",0).attr("width",Math.max(0,x(-d)));
    const passBad=Phi((-2*d+d)/se-1.96);          // a model truly 2δ worse
    const passEq=Phi((0+d)/se-1.96);               // a model exactly as good
    d3.select("#oc-read").html(
      `SE(Δ̂) ≈ <b>${se.toFixed(4)}</b> · P(pass) for an equal model: <b style="color:${passEq<0.8?C.a2:C.good}">${(passEq*100).toFixed(0)}%</b>`+
      ` · for a model 2× the margin worse: <b style="color:${passBad>0.05?C.bad:C.good}">${(passBad*100).toFixed(1)}%</b>`+
      `<br><span class="lbl">red zone = candidates worse than the margin; the gate should almost never pass them, and should almost always pass equal or better ones</span>`
    );
  }
  function val(id){return +d3.select(id).property("value");}
  ["#oc-n","#oc-d"].forEach(id=>d3.select(id).on("input",draw));
  draw();
})();
