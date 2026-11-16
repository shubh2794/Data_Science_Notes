/* chain-of-thought.viz.js — interactive figures for llm/reasoning-in-llms/chain-of-thought.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-sc  solve rate vs model size, standard vs CoT (the study's appendix result tables)
     #viz-ab  ablation and robustness bars, mean ± std over exemplar orders (the study's
              arithmetic and commonsense/symbolic ablation tables)
   Every number below is copied from those tables; nothing is interpolated. */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. scaling curves ═══════════════ */
  (function(){
    const svg=d3.select("#sc-svg"); if(svg.empty()) return;

    // parameter counts in billions (LaMDA's smallest is 422M in the text, 420M in the tables)
    const SIZES={"LaMDA":[0.42,2,8,68,137],"GPT-3":[0.35,1.3,6.7,175],"PaLM":[8,62,540]};
    const SLAB={"LaMDA":["420M","2B","8B","68B","137B"],"GPT-3":["350M","1.3B","6.7B","175B"],"PaLM":["8B","62B","540B"]};
    const FCOL={"LaMDA":C.a2,"GPT-3":C.violet,"PaLM":C.accent};
    // each entry: [standard, CoT] per size, in SIZES order
    const D={
      gsm8k:{name:"GSM8K",LaMDA:[[2.6,0.4],[3.6,1.9],[3.2,1.6],[5.7,8.2],[6.5,14.3]],"GPT-3":[[2.2,0.5],[2.4,0.5],[4.0,2.4],[15.6,46.9]],PaLM:[[4.9,4.1],[9.6,29.9],[17.9,56.9]],UL2:[4.1,4.4],Codex:[19.7,63.1],ref:{v:55,lab:"prior supervised best 55"}},
      svamp:{name:"SVAMP",LaMDA:[[2.5,1.6],[3.3,2.4],[4.3,3.4],[13.6,18.8],[29.5,37.5]],"GPT-3":[[1.4,0.8],[1.5,1.7],[6.1,3.1],[65.7,68.9]],PaLM:[[15.1,16.8],[48.2,46.7],[69.4,79.0]],UL2:[10.1,12.5],Codex:[69.9,76.4],ref:{v:57.4,lab:"prior supervised best 57.4"}},
      asdiv:{name:"ASDiv",LaMDA:[[3.2,0.8],[4.1,3.8],[5.9,5.0],[21.8,23.1],[40.1,46.6]],"GPT-3":[[2.1,0.8],[2.6,1.4],[8.6,3.6],[70.3,71.3]],PaLM:[[23.7,25.2],[58.7,61.9],[72.1,73.9]],UL2:[16.0,16.9],Codex:[74.0,80.4],ref:{v:75.3,lab:"prior supervised best 75.3"}},
      aqua:{name:"AQuA",LaMDA:[[23.5,8.3],[22.9,17.7],[22.8,18.6],[22.3,20.2],[25.5,20.6]],"GPT-3":[[18.1,8.7],[12.6,4.3],[15.4,13.4],[24.8,35.8]],PaLM:[[19.3,21.7],[25.6,22.4],[25.2,35.8]],UL2:[20.5,23.6],Codex:[29.5,45.3],ref:{v:37.9,lab:"prior supervised best 37.9"}},
      mawps:{name:"MAWPS",LaMDA:[[3.2,0.9],[3.9,3.1],[5.3,4.8],[21.6,30.6],[43.2,57.9]],"GPT-3":[[2.4,1.1],[3.1,1.7],[8.8,3.5],[72.7,87.1]],PaLM:[[26.2,30.5],[61.8,80.3],[79.2,93.3]],UL2:[16.6,19.1],Codex:[78.7,92.6],ref:{v:88.4,lab:"prior supervised best 88.4"}},
      singleop:{name:"MAWPS SingleOp",LaMDA:[[2.8,1.0],[4.6,4.1],[8.0,7.0],[36.5,40.8],[73.2,76.2]],"GPT-3":[[3.2,1.8],[5.3,3.0],[13.5,3.9],[90.9,88.8]],PaLM:[[41.8,46.6],[87.9,85.6],[94.1,94.1]],UL2:[24.9,27.2],Codex:[93.1,91.8]},
      singleeq:{name:"MAWPS SingleEq",LaMDA:[[2.4,0.4],[2.4,3.3],[4.5,4.4],[23.9,26.0],[48.8,58.7]],"GPT-3":[[2.0,0.2],[2.4,1.6],[8.7,4.9],[82.7,86.6]],PaLM:[[29.5,28.2],[77.2,83.5],[86.5,92.3]],UL2:[18.0,20.2],Codex:[86.8,93.1]},
      addsub:{name:"MAWPS AddSub",LaMDA:[[1.9,0.7],[2.7,3.2],[3.4,5.2],[17.3,23.2],[43.0,51.9]],"GPT-3":[[2.0,1.5],[2.3,1.5],[8.6,2.5],[83.3,81.3]],PaLM:[[29.4,31.4],[74.7,78.2],[93.9,91.9]],UL2:[18.5,18.2],Codex:[90.9,89.1]},
      multiarith:{name:"MAWPS MultiArith",LaMDA:[[5.8,1.5],[5.8,1.8],[5.2,2.4],[8.7,32.4],[7.6,44.9]],"GPT-3":[[2.3,0.8],[2.2,0.5],[4.5,2.8],[33.8,91.7]],PaLM:[[4.2,15.8],[7.3,73.7],[42.2,94.7]],UL2:[5.0,10.7],Codex:[44.0,96.2]},
      csqa:{name:"CSQA",LaMDA:[[20.1,19.2],[20.2,19.6],[19.0,20.3],[37.0,44.1],[53.6,57.9]],"GPT-3":[[14.7,15.2],[12.0,19.2],[19.0,24.0],[79.5,73.5]],PaLM:[[19.8,24.9],[65.4,68.1],[78.1,79.9]],UL2:[34.2,51.4],Codex:[82.3,77.9]},
      strategyqa:{name:"StrategyQA",LaMDA:[[46.4,24.9],[52.6,45.2],[54.1,46.8],[59.6,62.2],[62.4,65.4]],"GPT-3":[[20.6,0.9],[45.8,35.7],[53.6,50.0],[65.9,65.4]],PaLM:[[55.6,53.5],[58.4,63.4],[68.6,77.8]],UL2:[59.0,53.3],Codex:[67.1,73.2],ref:{v:69.4,lab:"prior single-model best 69.4"}},
      date:{name:"Date Understanding",LaMDA:[[1.9,1.6],[8.0,6.8],[9.5,5.4],[15.5,18.6],[21.5,26.8]],"GPT-3":[[4.3,0.9],[4.0,1.4],[8.9,4.9],[43.8,52.1]],PaLM:[[12.9,13.1],[29.8,44.7],[49.0,65.3]],UL2:[13.5,14.0],Codex:[49.0,64.8]},
      sports:{name:"Sports Understanding",LaMDA:[[50.0,49.7],[49.3,57.5],[50.0,52.1],[55.2,77.5],[59.5,85.8]],"GPT-3":[[33.8,41.6],[0.0,26.9],[0.0,4.4],[69.6,82.4]],PaLM:[[55.1,75.2],[72.1,93.6],[80.5,95.4]],UL2:[57.9,65.3],Codex:[71.7,98.5],ref:{v:84,lab:"unaided sports enthusiast 84"}},
      saycan:{name:"SayCan",LaMDA:[[7.5,7.5],[8.3,8.3],[28.3,33.3],[35.0,42.5],[43.3,46.6]],"GPT-3":[[12.5,0.8],[20.8,9.2],[17.5,35.0],[81.7,87.5]],PaLM:[[34.2,40.0],[65.8,70.0],[80.8,91.7]],UL2:[20.0,41.7],Codex:[85.8,88.3]},
      concat2:{name:"Last-letter concat, 2 words (in domain)",LaMDA:[[0.3,1.6],[2.3,6.0],[1.5,11.5],[4.4,52.0],[5.8,77.5]],PaLM:[[2.6,18.8],[6.8,85.0],[7.6,99.4]],UL2:[0.6,18.8]},
      concat3:{name:"Last-letter concat, 3 words (OOD)",LaMDA:[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.8],[0.0,34.4]],PaLM:[[0.0,0.0],[0.0,59.6],[0.2,94.8]],UL2:[0.0,0.2]},
      concat4:{name:"Last-letter concat, 4 words (OOD)",LaMDA:[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,2.5],[0.0,13.5]],PaLM:[[0.0,0.2],[0.0,13.4],[0.0,63.0]],UL2:[0.0,0.0]},
      coin2:{name:"Coin flip, 2 (in domain)",LaMDA:[[52.9,49.6],[54.9,55.3],[52.9,55.5],[56.2,83.2],[49.0,99.6]],PaLM:[[60.0,74.4],[91.4,96.8],[98.1,100.0]],UL2:[70.4,67.1],chance:true},
      coin3:{name:"Coin flip, 3 (OOD)",LaMDA:[[50.0,50.5],[47.4,48.7],[48.2,49.6],[50.4,69.1],[50.7,91.0]],PaLM:[[47.3,57.1],[43.9,91.0],[49.3,98.6]],UL2:[51.6,52.2],chance:true},
      coin4:{name:"Coin flip, 4 (OOD)",LaMDA:[[49.5,49.1],[49.8,50.2],[51.2,50.6],[50.9,59.6],[49.1,74.5]],PaLM:[[50.9,51.8],[38.3,72.4],[54.8,90.2]],UL2:[48.7,50.4],chance:true}
    };

    const W=760,H=380,m={l:56,r:150,t:30,b:46};
    const x=d3.scaleLog().domain([0.25,800]).range([m.l,W-m.r]);
    const fmtP=v=>v>=1?v+"B":Math.round(v*1000)+"M";

    function crossover(fam,rows){
      // smallest size from which CoT stays strictly ahead at every larger size
      let i=rows.length;
      while(i>0&&rows[i-1][1]>rows[i-1][0]) i--;
      if(i===rows.length) return "CoT is not ahead at the largest size";
      if(i===0) return "CoT ahead at every size";
      return "CoT ahead from "+SLAB[fam][i];
    }

    function draw(){
      svg.selectAll("*").remove();
      const ds=d3.select("#sc-d").property("value"), fsel=d3.select("#sc-f").property("value");
      const showUL2=d3.select("#sc-ul2").property("checked"), zoom=d3.select("#sc-zoom").property("checked");
      const d=D[ds];
      const fams=(fsel==="all"?["LaMDA","GPT-3","PaLM"]:[fsel]).filter(f=>d[f]);
      const missing=(fsel==="all"?["LaMDA","GPT-3","PaLM"]:[fsel]).filter(f=>!d[f]);

      let vals=[];
      fams.forEach(f=>d[f].forEach(r=>{vals.push(r[0],r[1]);}));
      if(showUL2&&d.UL2) vals.push(d.UL2[0],d.UL2[1]);
      if(d.ref) vals.push(d.ref.v);
      if(d.chance) vals.push(50);
      let y0=0,y1=100;
      if(zoom&&vals.length){y0=Math.max(0,Math.floor((d3.min(vals)-3)/5)*5);y1=Math.min(100,Math.ceil((d3.max(vals)+3)/5)*5);}
      const y=d3.scaleLinear().domain([y0,y1]).range([H-m.b,m.t]);

      svg.append("g").attr("transform",`translate(0,${H-m.b})`)
        .call(d3.axisBottom(x).tickValues([0.3,1,3,10,30,100,300]).tickFormat(v=>v<1?(v*1000)+"M":v+"B")).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)).attr("color",C.muted);
      svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("model parameters (log scale)");
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11)
        .text(d.name+" — solve rate (%)"+(zoom?" — axis zoomed, does not start at 0":" — axis 0 to 100"));

      if(d.ref&&d.ref.v>=y0&&d.ref.v<=y1){
        svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(d.ref.v)).attr("y2",y(d.ref.v)).attr("stroke",C.ink).attr("stroke-dasharray","1 4").attr("stroke-width",1.5);
        svg.append("text").attr("x",W-m.r-4).attr("y",y(d.ref.v)-5).attr("text-anchor","end").attr("fill",C.ink).attr("font-size",10).text(d.ref.lab);
      }
      if(d.chance&&50>=y0&&50<=y1){
        svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(50)).attr("y2",y(50)).attr("stroke",C.muted).attr("stroke-dasharray","1 4");
        svg.append("text").attr("x",m.l+6).attr("y",y(50)-5).attr("fill",C.muted).attr("font-size",10).text("chance (binary task) 50");
      }

      const parts=[];
      fams.forEach(f=>{
        const rows=d[f], sz=SIZES[f], col=FCOL[f];
        // shaded gain/loss between the two curves, one quad per adjacent pair
        for(let i=0;i<rows.length-1;i++){
          const a=rows[i],b=rows[i+1];
          const mid=((a[1]-a[0])+(b[1]-b[0]))/2;
          svg.append("path").attr("d",`M${x(sz[i])},${y(a[0])}L${x(sz[i+1])},${y(b[0])}L${x(sz[i+1])},${y(b[1])}L${x(sz[i])},${y(a[1])}Z`)
            .attr("fill",mid>=0?C.accent:C.bad).attr("opacity",0.09);
        }
        const line=k=>d3.line().x((r,i)=>x(sz[i])).y(r=>y(r[k]))(rows);
        svg.append("path").attr("d",line(0)).attr("fill","none").attr("stroke",col).attr("stroke-width",1.6).attr("stroke-dasharray","5 4");
        svg.append("path").attr("d",line(1)).attr("fill","none").attr("stroke",col).attr("stroke-width",2.4);
        rows.forEach((r,i)=>{
          [0,1].forEach(k=>{
            const c=svg.append("circle").attr("cx",x(sz[i])).attr("cy",y(r[k])).attr("r",k?4.5:3.5)
              .attr("fill",k?col:C.panel).attr("stroke",col).attr("stroke-width",1.5).style("cursor","pointer")
              .on("mouseenter",()=>hover(f,SLAB[f][i],r,d.name));
            c.append("title").text(`${f} ${SLAB[f][i]} · standard ${r[0]} · CoT ${r[1]}`);
          });
        });
        const last=rows[rows.length-1];
        parts.push(`<b style="color:${col}">${f}</b>: ${crossover(f,rows)}; at ${SLAB[f][rows.length-1]} ${last[0]} → ${last[1]} (<b>${last[1]-last[0]>=0?"+":""}${(last[1]-last[0]).toFixed(1)}</b>)`);
      });
      if(showUL2&&d.UL2){
        const u=d.UL2;
        [0,1].forEach(k=>svg.append("rect").attr("x",x(20)-4).attr("y",y(u[k])-4).attr("width",8).attr("height",8)
          .attr("fill",k?C.blue:C.panel).attr("stroke",C.blue).attr("stroke-width",1.5).attr("transform",`rotate(45,${x(20)},${y(u[k])})`)
          .style("cursor","pointer").on("mouseenter",()=>hover("UL2","20B",u,d.name)).append("title").text(`UL2 20B · standard ${u[0]} · CoT ${u[1]}`));
        parts.push(`<b style="color:${C.blue}">UL2 20B</b>: ${u[0]} → ${u[1]}`);
      }

      // legend
      const lg=svg.append("g").attr("transform",`translate(${W-m.r+8},${m.t+4})`);
      let ly=0;
      fams.forEach(f=>{lg.append("line").attr("x1",0).attr("x2",22).attr("y1",ly).attr("y2",ly).attr("stroke",FCOL[f]).attr("stroke-width",2.4);
        lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",FCOL[f]).attr("font-size",11).text(f);ly+=18;});
      if(showUL2&&d.UL2){lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",C.blue).attr("font-size",11).text("UL2 20B (◆)");ly+=18;}
      ly+=6;
      lg.append("line").attr("x1",0).attr("x2",22).attr("y1",ly).attr("y2",ly).attr("stroke",C.ink).attr("stroke-width",2.4);
      lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",C.ink).attr("font-size",10.5).text("chain of thought");ly+=16;
      lg.append("line").attr("x1",0).attr("x2",22).attr("y1",ly).attr("y2",ly).attr("stroke",C.ink).attr("stroke-width",1.6).attr("stroke-dasharray","5 4");
      lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",C.ink).attr("font-size",10.5).text("standard");ly+=16;
      lg.append("rect").attr("x",0).attr("y",ly-5).attr("width",22).attr("height",10).attr("fill",C.accent).attr("opacity",0.25);
      lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",C.muted).attr("font-size",10.5).text("CoT gain");ly+=16;
      lg.append("rect").attr("x",0).attr("y",ly-5).attr("width",22).attr("height",10).attr("fill",C.bad).attr("opacity",0.25);
      lg.append("text").attr("x",28).attr("y",ly+4).attr("fill",C.muted).attr("font-size",10.5).text("CoT loss");

      let extra="";
      if(missing.length) extra+=`<br>${missing.join(", ")}: not reported for this task in the study (the symbolic tasks were run on LaMDA, PaLM and UL2 only).`;
      if(d.Codex) extra+=`<br>Codex (size not stated, so not plotted): ${d.Codex[0]} → ${d.Codex[1]}.`;
      if(!fams.length) svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H/2).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",13).text("not reported for this family on this task");
      d3.select("#sc-readout").html(`<b>${d.name}</b>, standard → CoT · `+(parts.length?parts.join(" · "):"no data for this selection")+extra+`<br>hover a point for its values`);
    }
    function hover(f,lab,r,name){
      const g=r[1]-r[0];
      d3.select("#sc-readout").html(`<b>${f} ${lab}</b> on ${name}: standard <b>${r[0]}</b> → chain of thought <b>${r[1]}</b> · difference <b style="color:${g>=0?C.accent:C.bad}">${g>=0?"+":""}${g.toFixed(1)}</b> points`+
        (g<0?" — chain of thought hurt at this size":""));
    }
    ["#sc-d","#sc-f","#sc-ul2","#sc-zoom"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();

  /* ═══════════════ 2. ablations & robustness ═══════════════ */
  (function(){
    const svg=d3.select("#ab-svg"); if(svg.empty()) return;
    // [mean, std over five exemplar orders]; null = variant not run for that task
    const T={
      gsm8k:{name:"GSM8K",model:"LaMDA 137B",std:[6.5,0.4],cot:[14.3,0.4],eq:[5.4,0.2],vc:[6.4,0.3],after:[6.1,0.4],B:[15.5,0.6],Cc:[17.6,1.0],concise:[11.1,0.3],ga:[12.6,0.6],gb:[12.7,0.5],gg:[12.6,0.7]},
      svamp:{name:"SVAMP",model:"LaMDA 137B",std:[29.5,0.6],cot:[36.7,0.4],eq:[35.1,0.4],vc:[28.0,0.6],after:[30.7,0.9],B:[35.2,0.4],Cc:[37.5,2.0],concise:[38.7,0.8],ga:[32.8,1.1],gb:[34.8,1.1],gg:[35.6,0.5]},
      asdiv:{name:"ASDiv",model:"LaMDA 137B",std:[40.1,0.6],cot:[46.6,0.7],eq:[45.9,0.6],vc:[39.4,0.4],after:[38.6,0.6],B:[46.5,0.4],Cc:[48.7,0.7],concise:[48.0,0.3],ga:[44.1,0.9],gb:[46.9,0.6],gg:[44.4,2.6]},
      mawps:{name:"MAWPS",model:"LaMDA 137B",std:[43.2,0.9],cot:[57.9,1.5],eq:[50.1,1.0],vc:[41.3,1.1],after:[43.6,1.0],B:[58.2,1.0],Cc:[60.1,2.0],concise:[59.6,0.7],ga:[53.9,1.1],gb:[60.9,0.8],gg:[54.2,4.7]},
      date:{name:"Date Understanding",model:"LaMDA 137B",std:[21.5,0.6],cot:[26.8,2.1],vc:[21.3,0.7],after:[20.9,1.0],B:[27.4,1.7],Cc:[25.5,2.5]},
      sports:{name:"Sports Understanding",model:"LaMDA 137B",std:[59.5,3.0],cot:[85.8,1.8],vc:[61.6,2.2],after:[63.0,2.0],B:[75.4,2.7],Cc:[81.1,3.6]},
      saycan:{name:"SayCan",model:"PaLM 540B",std:[80.8,1.8],cot:[91.7,1.4],vc:[74.2,2.3],after:[83.3,0.6],B:[88.3,1.4],Cc:[85.0,1.8]},
      concat:{name:"Last-letter concatenation",model:"LaMDA 137B",std:[5.8,0.6],cot:[77.5,3.8],vc:[7.2,1.6],after:[0.0,0.0],B:[76.0,1.9],Cc:[68.1,2.2]},
      coin:{name:"Coin flip",model:"LaMDA 137B",std:[49.0,2.1],cot:[99.6,0.3],vc:[50.7,0.7],after:[50.2,0.5],B:[77.5,7.9],Cc:[71.4,11.1]}
    };
    const VIEWS={
      ablation:[["std","standard prompting"],["eq","equation only"],["vc","variable compute only (dots)"],["after","reasoning after answer"],["cot","chain of thought (A)"]],
      robust:[["std","standard prompting"],["cot","CoT · Annotator A"],["B","CoT · Annotator B"],["Cc","CoT · Annotator C"],["concise","CoT · concise style (A)"],["ga","CoT · GSM8K-train exemplars, set α"],["gb","CoT · GSM8K-train exemplars, set β"],["gg","CoT · GSM8K-train exemplars, set γ"]]
    };
    const W=760,H=360,m={l:230,r:40,t:34,b:40};

    function draw(){
      svg.selectAll("*").remove();
      const view=d3.select("#ab-v").property("value"), ds=d3.select("#ab-d").property("value");
      const t=T[ds];
      const rows=VIEWS[view].filter(r=>t[r[0]]);
      const skipped=VIEWS[view].filter(r=>!t[r[0]]).map(r=>r[1]);
      const x=d3.scaleLinear().domain([0,Math.min(100,Math.ceil((d3.max(rows,r=>t[r[0]][0]+t[r[0]][1])+4)/10)*10)]).range([m.l,W-m.r]);
      const y=d3.scaleBand().domain(rows.map(r=>r[0])).range([m.t,H-m.b]).padding(0.28);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickFormat(k=>rows.find(r=>r[0]===k)[1])).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11)
        .text(`${t.name} · ${t.model} · accuracy (%), axis starts at 0 · whiskers = ±1 std over 5 exemplar orders`);

      const s=t.std[0], c=t.cot[0], gap=c-s;
      // gap band
      svg.append("rect").attr("x",x(Math.min(s,c))).attr("y",m.t-4).attr("width",Math.abs(x(c)-x(s))).attr("height",H-m.b-m.t+4).attr("fill",C.accent).attr("opacity",0.06);
      svg.append("line").attr("x1",x(s)).attr("x2",x(s)).attr("y1",m.t-4).attr("y2",H-m.b).attr("stroke",C.muted).attr("stroke-dasharray","5 4");
      svg.append("line").attr("x1",x(c)).attr("x2",x(c)).attr("y1",m.t-4).attr("y2",H-m.b).attr("stroke",C.accent);

      const share=v=>gap>0?100*(v-s)/gap:NaN;
      rows.forEach(r=>{
        const [v,sd]=t[r[0]];
        const isStd=r[0]==="std", isCot=r[0]==="cot";
        const sh=share(v);
        const col=isStd?C.muted:isCot?C.accent:(view==="ablation"?(sh>=50?C.a2:C.bad):(v>s?C.good:C.bad));
        const yy=y(r[0]), bh=y.bandwidth();
        svg.append("rect").attr("x",m.l).attr("y",yy).attr("height",bh).attr("width",Math.max(1,x(v)-m.l)).attr("fill",col).attr("opacity",0.85)
          .style("cursor","pointer").on("mouseenter",()=>hover(r,v,sd,t,view)).append("title").text(`${r[1]}: ${v} ± ${sd}`);
        if(sd>0){
          svg.append("line").attr("x1",x(Math.max(0,v-sd))).attr("x2",x(v+sd)).attr("y1",yy+bh/2).attr("y2",yy+bh/2).attr("stroke",C.ink).attr("stroke-width",1.3);
          [v-sd,v+sd].forEach(e=>svg.append("line").attr("x1",x(Math.max(0,e))).attr("x2",x(Math.max(0,e))).attr("y1",yy+bh/2-4).attr("y2",yy+bh/2+4).attr("stroke",C.ink).attr("stroke-width",1.3));
        }
        svg.append("text").attr("x",x(v+sd)+6).attr("y",yy+bh/2+4).attr("fill",C.ink).attr("font-size",11).text(v.toFixed(1)+(sd>0?" ± "+sd:""));
      });

      const parts=rows.filter(r=>r[0]!=="std"&&r[0]!=="cot").map(r=>{
        const v=t[r[0]][0], sh=share(v);
        return `${r[1].replace("CoT · ","")}: <b>${v}</b> (${Math.round(sh)}% of the gain)`;
      });
      d3.select("#ab-readout").html(
        `<b>${t.name}</b> (${t.model}): standard <b>${s}</b> → chain of thought <b>${c}</b>, a gain of <b>${gap.toFixed(1)}</b> points.`+
        (view==="ablation"?" Share of that gain each ablation recovers, (variant − standard) / (CoT − standard) — our arithmetic: ":" Robustness variants, share of Annotator A's gain (our arithmetic; over 100% means better than A): ")+
        parts.join(" · ")+
        (skipped.length?`<br>Not run for this task in the study: ${skipped.join(", ")}.`:"")+
        (ds==="svamp"?"<br>Note: this table gives LaMDA 137B CoT on SVAMP as 36.7; the main results table gives 37.5.":"")+
        (ds==="saycan"?"<br>SayCan was run with PaLM 540B because its evaluation set is only 120 examples.":""));
    }
    function hover(r,v,sd,t,view){
      const s=t.std[0], c=t.cot[0];
      const sh=(c-s)>0?100*(v-s)/(c-s):NaN;
      d3.select("#ab-readout").html(`<b>${r[1]}</b> on ${t.name}: <b>${v}</b> ± ${sd} · ${v>=s?"+":""}${(v-s).toFixed(1)} vs standard · `+
        (r[0]==="std"?"the baseline":r[0]==="cot"?"the full method":`recovers <b>${Math.round(sh)}%</b> of the chain-of-thought gain (our arithmetic)`));
    }
    ["#ab-v","#ab-d"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();
})();
