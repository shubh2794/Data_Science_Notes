/* ai-engineering.viz.js — interactive visualizations for AI Engineering.
   Loaded after data.js / notes.js. Two visualizations:
   1. The 5-layer production architecture (clickable stack diagram)
   2. Cost vs. quality with model routing (interactive scatter/bar) */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

  /* ================================================================
     VIZ 1 — 5-layer architecture stack
     ================================================================ */
  (function archViz(){
    const svg=d3.select("#arch-svg"); if(svg.empty()) return;
    const W=700,H=340;
    const layers=[
      {id:"context",  label:"1 · Enhance Context",  color:"#5b9cff", desc:"RAG retrieval, few-shot examples, conversation memory, tool results — enrich the prompt before the model sees it."},
      {id:"guard",    label:"2 · Guardrails",        color:"#c084fc", desc:"Input filters (injection, scope, PII) and output validators (schema, policy, grounding). Deterministic rules first, classifiers second."},
      {id:"router",   label:"3 · Router / Gateway",  color:"#2dd4bf", desc:"Route by complexity/cost — simple queries to cheap models, hard ones to frontier. Unified API, retries, fallback chains."},
      {id:"cache",    label:"4 · Cache",             color:"#a3e635", desc:"Exact-match and semantic caching for repeat queries. KV cache reuse for shared prefixes. Latency drops from seconds to milliseconds."},
      {id:"agent",    label:"5 · Agent Patterns",    color:"#fb923c", desc:"Multi-step reasoning, tool use, dynamic control flow — add only when simpler layers are insufficient. ReAct, planning, orchestration."},
    ];
    const crosscut=[
      {label:"Monitoring", color:"#f87171"},
      {label:"Orchestration", color:"#818cf8"},
      {label:"Feedback", color:"#ffb454"},
    ];

    let selected=-1;

    function draw(){
      svg.selectAll("*").remove();
      const mx=100, my=16, lw=400, lh=48, gap=8;
      const totalH=layers.length*(lh+gap)-gap;
      const startY=(H-totalH)/2;

      // title
      svg.append("text").attr("x",mx+lw/2).attr("y",startY-8).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("user request  ▼");

      // layers
      layers.forEach((l,i)=>{
        const y=startY+i*(lh+gap);
        const g=svg.append("g").attr("cursor","pointer")
          .on("click",()=>{ selected=selected===i?-1:i; draw(); });

        const isSel=selected===i;
        g.append("rect").attr("x",mx).attr("y",y).attr("width",lw).attr("height",lh)
          .attr("rx",8).attr("fill",l.color).attr("fill-opacity",isSel?0.35:0.18)
          .attr("stroke",l.color).attr("stroke-width",isSel?2.5:1);

        g.append("text").attr("x",mx+lw/2).attr("y",y+lh/2+5).attr("text-anchor","middle")
          .attr("font-size",13).attr("font-weight",700).attr("fill",isSel?C.ink:l.color).text(l.label);

        // arrow between layers
        if(i<layers.length-1){
          const ay=y+lh+gap/2;
          svg.append("text").attr("x",mx+lw/2).attr("y",ay+4).attr("text-anchor","middle")
            .attr("font-size",10).attr("fill","#555e6e").text("▼");
        }
      });

      // cross-cutting concerns — vertical bar on the right
      const ccx=mx+lw+24, ccw=28;
      crosscut.forEach((cc,i)=>{
        const x=ccx+i*(ccw+6);
        svg.append("rect").attr("x",x).attr("y",startY).attr("width",ccw).attr("height",totalH)
          .attr("rx",6).attr("fill",cc.color).attr("fill-opacity",0.15).attr("stroke",cc.color).attr("stroke-width",0.8);
        // rotated label
        svg.append("text").attr("x",x+ccw/2).attr("y",startY+totalH/2)
          .attr("text-anchor","middle").attr("font-size",9).attr("fill",cc.color).attr("font-weight",600)
          .attr("transform",`rotate(-90,${x+ccw/2},${startY+totalH/2})`).text(cc.label);
      });

      // "model call" label
      const modelY=startY+totalH+18;
      svg.append("text").attr("x",mx+lw/2).attr("y",modelY).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("▼  model call  ▼");

      // "response" label
      svg.append("text").attr("x",mx+lw/2).attr("y",modelY+18).attr("text-anchor","middle")
        .attr("font-size",10).attr("fill","#555e6e").text("▲  response back through the stack  ▲");

      // readout
      if(selected>=0){
        const l=layers[selected];
        d3.select("#arch-read").html(`<b style="color:${l.color}">${l.label}:</b> ${l.desc}`);
      } else {
        d3.select("#arch-read").html("Click a layer to see what it does. Each layer wraps the model call and is independently bypassable.");
      }
    }
    draw();
  })();


  /* ================================================================
     VIZ 2 — cost vs. quality with model routing
     ================================================================ */
  (function routeViz(){
    const svg=d3.select("#route-svg"); if(svg.empty()) return;
    const W=640,H=260,pad={t:30,r:24,b:50,l:60};
    const iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;

    // model tiers: cost per 1K tokens (cents), quality score 0-100
    const cheapCost=0.015, cheapQual=72;   // e.g. small model
    const frontierCost=1.5, frontierQual=94; // e.g. frontier model

    function draw(){
      svg.selectAll("*").remove();
      const thresh=+d3.select("#rt-thresh").property("value");
      const vol=+d3.select("#rt-vol").property("value");

      // fraction of queries that are "easy" (below threshold)
      // simulate: queries have a difficulty uniformly 0-100
      const fCheap=thresh/100;
      const fFrontier=1-fCheap;

      const nCheap=Math.round(vol*fCheap);
      const nFrontier=vol-nCheap;

      // costs
      const costCheapTotal=nCheap*cheapCost;
      const costFrontierTotal=nFrontier*frontierCost;
      const costRouted=costCheapTotal+costFrontierTotal;
      const costAllFrontier=vol*frontierCost;

      // blended quality (weighted)
      const qualRouted=fCheap>0||fFrontier>0 ? (nCheap*cheapQual+nFrontier*frontierQual)/vol : 0;
      const qualAllFrontier=frontierQual;

      // bar chart: 3 scenarios
      const scenarios=[
        {label:"All frontier",  cost:costAllFrontier,  qual:qualAllFrontier, color:"#c084fc"},
        {label:"Routed (mixed)",cost:costRouted,        qual:qualRouted,      color:C.accent},
        {label:"All cheap",     cost:vol*cheapCost,     qual:cheapQual,       color:"#5b9cff"},
      ];

      const maxCost=d3.max(scenarios,d=>d.cost)*1.15;
      const xScale=d3.scaleLinear().domain([0,maxCost]).range([0,iW]);
      const yBand=iH/scenarios.length;

      const g=svg.append("g").attr("transform",`translate(${pad.l},${pad.t})`);

      // bars
      scenarios.forEach((s,i)=>{
        const y=i*yBand+yBand*0.15;
        const bh=yBand*0.5;
        g.append("rect").attr("x",0).attr("y",y).attr("width",xScale(s.cost)).attr("height",bh)
          .attr("rx",5).attr("fill",s.color).attr("fill-opacity",0.7);
        // cost label
        g.append("text").attr("x",xScale(s.cost)+6).attr("y",y+bh/2+4)
          .attr("font-size",11).attr("fill",C.ink).attr("font-weight",600)
          .text("$"+s.cost.toFixed(2));
        // quality badge
        g.append("text").attr("x",Math.max(xScale(s.cost)-8,40)).attr("y",y+bh/2+4)
          .attr("text-anchor","end").attr("font-size",10).attr("fill","#06101f").attr("font-weight",700)
          .text("q="+s.qual.toFixed(1));
        // label
        g.append("text").attr("x",-8).attr("y",y+bh/2+4).attr("text-anchor","end")
          .attr("font-size",11).attr("fill",s.color).attr("font-weight",600).text(s.label);
      });

      // x axis
      g.append("line").attr("x1",0).attr("y1",iH).attr("x2",iW).attr("y2",iH).attr("stroke",C.line);
      g.append("text").attr("x",iW/2).attr("y",iH+32).attr("text-anchor","middle")
        .attr("font-size",10).attr("fill",C.muted).text("Cost per "+vol.toLocaleString()+" requests ($ at 1K-token avg)");

      // savings
      const savedPct=((1-costRouted/costAllFrontier)*100).toFixed(0);
      const qualDrop=(qualAllFrontier-qualRouted).toFixed(1);
      d3.select("#route-read").html(
        `<b>${thresh}%</b> of traffic to cheap model → <b style="color:${C.good}">${savedPct}% cost savings</b> · quality drop <b>${qualDrop} pts</b> (${qualRouted.toFixed(1)} vs ${qualAllFrontier}) · ${nCheap.toLocaleString()} cheap + ${nFrontier.toLocaleString()} frontier calls`
      );
    }
    d3.select("#rt-thresh").on("input",draw);
    d3.select("#rt-vol").on("input",draw);
    draw();
  })();

})();
