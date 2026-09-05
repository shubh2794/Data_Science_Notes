/* llmops.viz.js — extracted from llmops.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#lo-svg"),W=640,H=200;
  const stages=["User","Guardrail","Cache","LLM","Guardrail","Response"];
  function draw(){
    svg.selectAll("*").remove();
    const hit=d3.select("#lo-sel").property("value")==="hit";
    const active = hit ? ["User","Guardrail","Cache","Response"] : stages;
    const bw=88, gap=14, y=70, x0=20;
    stages.forEach((s,i)=>{
      const x=x0+i*(bw+gap), on=active.includes(s) && !(hit && (s==="LLM"));
      const skipped = hit && (s==="LLM" || (s==="Guardrail" && i===4));
      svg.append("rect").attr("x",x).attr("y",y).attr("width",bw).attr("height",44).attr("rx",8)
        .attr("fill",skipped?"#15181f":"rgba(163,230,53,.14)").attr("stroke",skipped?"#2a2f3a":C.accent).attr("stroke-opacity",skipped?.5:1);
      svg.append("text").attr("x",x+bw/2).attr("y",y+27).attr("text-anchor","middle").attr("font-size",11).attr("fill",skipped?C.muted:C.ink).text(s);
      if(i<stages.length-1){ const skipArrow = hit && (stages[i]==="Cache");
        svg.append("text").attr("x",x+bw+gap/2).attr("y",y+27).attr("text-anchor","middle").attr("font-size",13).attr("fill",C.muted).text("›"); }
    });
    if(hit){ // arrow from Cache straight to Response
      const cx=x0+2*(bw+gap)+bw/2, rx=x0+5*(bw+gap)+bw/2;
      svg.append("path").attr("d",`M${cx},${y+44} C${cx},${y+90} ${rx},${y+90} ${rx},${y+44}`).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2).attr("stroke-dasharray","4 3");
      svg.append("text").attr("x",(cx+rx)/2).attr("y",y+104).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",10).text("cache hit → skip the model");
    }
    // monitoring tap
    svg.append("text").attr("x",W/2).attr("y",24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("every call is logged → monitoring (quality · cost · latency · drift)");
    const lat=hit?"~20 ms":"~900 ms", cost=hit?"$0.000":"$0.004";
    d3.select("#lo-read").html(hit?`<b style="color:${C.good}">cache hit</b> · latency <b>${lat}</b> · cost <b>${cost}</b> — model skipped entirely`:`<b>cache miss</b> · latency <b>${lat}</b> · cost <b>${cost}</b> — full guardrail + model + guardrail path`);
  }
  d3.select("#lo-sel").on("change",draw);
  draw();
})();

(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  // illustrative price tiers: per-token = per-1M / 1e6. L_miss = full-path latency (ms), small model assumed faster.
  const TIERS={
    frontier:{in:5/1e6, out:15/1e6, lat:1800},
    mid:     {in:0.50/1e6, out:1.50/1e6, lat:900},
    small:   {in:0.10/1e6, out:0.40/1e6, lat:350}
  };
  const SMALL=TIERS.small, L_HIT=20; // cache-hit latency (ms)
  const svg=d3.select("#cc-svg"),W=640,H=240,m={t:18,r:18,b:34,l:64};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const x=d3.scaleLinear().domain([0,0.95]).range([0,iw]);
  const y=d3.scaleLinear().range([ih,0]);
  const area=g.append("path").attr("fill","rgba(163,230,53,.12)");
  const line=g.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
  const dot=g.append("circle").attr("r",4.5).attr("fill",C.a2).attr("stroke","#15181f").attr("stroke-width",1.5);
  const xAxis=g.append("g").attr("transform",`translate(0,${ih})`);
  const yAxis=g.append("g");
  g.append("text").attr("x",iw/2).attr("y",ih+30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("cache hit-rate h");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-50).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("monthly spend ($)");

  const fmt$=v=> v>=1e6 ? "$"+(v/1e6).toFixed(2)+"M" : v>=1e3 ? "$"+(v/1e3).toFixed(1)+"k" : "$"+v.toFixed(2);
  function val(id){return +d3.select(id).property("value");}

  function costPerCall(t){
    // blend full-tier and small-model cost by routing fraction rt
    const rt=val("#cc-rt"), nin=val("#cc-in"), nout=val("#cc-out");
    const cFull = nin*t.in + nout*t.out;
    const cSmall= nin*SMALL.in + nout*SMALL.out;
    return (1-rt)*cFull + rt*cSmall;
  }
  function spendAt(h,t){
    const req=val("#cc-req");
    return (1-h)*costPerCall(t)*req*30;
  }

  function draw(){
    const t=TIERS[d3.select("#cc-tier").property("value")];
    const h=val("#cc-h");
    d3.select("#cc-req-v").text(val("#cc-req").toLocaleString());
    d3.select("#cc-in-v").text(val("#cc-in"));
    d3.select("#cc-out-v").text(val("#cc-out"));
    d3.select("#cc-h-v").text(h.toFixed(2));
    d3.select("#cc-rt-v").text((val("#cc-rt")*100).toFixed(0)+"%");

    const pts=d3.range(0,0.951,0.025).map(hh=>({h:hh,s:spendAt(hh,t)}));
    y.domain([0, d3.max(pts,d=>d.s)*1.08 || 1]);
    const ln=d3.line().x(d=>x(d.h)).y(d=>y(d.s));
    const ar=d3.area().x(d=>x(d.h)).y0(ih).y1(d=>y(d.s));
    line.attr("d",ln(pts));
    area.attr("d",ar(pts));
    dot.attr("cx",x(h)).attr("cy",y(spendAt(h,t)));
    xAxis.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")));
    yAxis.call(d3.axisLeft(y).ticks(5).tickFormat(fmt$));
    svg.selectAll(".domain,.tick line").attr("stroke",C.line);
    svg.selectAll(".tick text").attr("fill",C.muted).attr("font-size",9);

    const cpc=costPerCall(t), ceff=(1-h)*cpc;
    const latMiss=(1-val("#cc-rt"))*t.lat + val("#cc-rt")*SMALL.lat;
    const latEff=h*L_HIT + (1-h)*latMiss;
    d3.select("#cc-read").html(
      `cost/call <b>${"$"+cpc.toFixed(5)}</b> · cost_eff (×(1−h)) <b style="color:${C.a2}">${"$"+ceff.toFixed(5)}</b> · `+
      `monthly spend <b style="color:${C.accent}">${fmt$(spendAt(h,t))}</b> · `+
      `latency_eff <b style="color:${C.good}">${latEff.toFixed(0)} ms</b> &nbsp;|&nbsp; `+
      `<span class="lbl">h=${h.toFixed(2)} ⇒ ≈ ${((1-h)*100).toFixed(0)}% of uncached spend · figures illustrative</span>`
    );
  }
  ["#cc-tier","#cc-req","#cc-in","#cc-out","#cc-h","#cc-rt"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
  draw();
})();
