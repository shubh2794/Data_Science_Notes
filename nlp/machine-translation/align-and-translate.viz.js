/* align-and-translate.viz.js — interactive figures for nlp/machine-translation/align-and-translate.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-att  soft vs hard alignment heat-map for one sentence pair (illustrative weights, not paper data)
     #viz-tab  Table 1 as printed: BLEU, all sentences vs sentences with no unknown words */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. Alignment heat-map ═══════════════ */
  (function(){
    const svg=d3.select("#att-svg"); if(svg.empty()) return;
    const W=640, L=92, T=78, R=10;
    const src=["The","agreement","on","the","European","Economic","Area","was","signed","in","August","1992",".","<end>"];
    const tgt=["L'","accord","sur","la","zone","économique","européenne","a","été","signé","en","août","1992",".","<end>"];
    // illustrative affinities target→source (0-based source index: strength). The first key is the primary link.
    const AFF=[
      {0:1,1:.35},{1:1},{2:1},{3:1,6:.4},{6:1},{5:1},{4:1,6:.3},{7:1},{7:1,8:.4},{8:1},{9:1},{10:1},{11:1},{12:1},{13:1,12:.3}
    ];
    const nS=src.length, nT=tgt.length;
    const cw=(W-L-R)/nS, ch=16;
    const top=T, hmBot=T+nT*ch;
    const barTop=hmBot+42, barBot=430;
    const color=d3.scaleSequential(d3.interpolateViridis).domain([0,1]);
    const gH=svg.append("g"), gB=svg.append("g");

    src.forEach((s,j)=>svg.append("text").attr("transform",`translate(${L+j*cw+cw/2},${T-6}) rotate(-45)`).attr("font-size",10.5).attr("fill",C.ink).text(s));
    tgt.forEach((t,i)=>svg.append("text").attr("class","tl").attr("x",L-6).attr("y",top+i*ch+ch/2+4).attr("text-anchor","end").attr("font-size",10.5).attr("fill",C.ink).text(t));
    svg.append("text").attr("x",4).attr("y",14).attr("fill",C.muted).attr("font-size",10.5).text("rows: French target words yᵢ · columns: English annotations hⱼ");
    svg.append("text").attr("x",L).attr("y",barTop-8).attr("fill",C.muted).attr("font-size",10.5).attr("class","bt");

    function weights(beta,mode){
      return AFF.map(a=>{
        const keys=Object.keys(a).map(Number), p=keys[0];
        const logits=d3.range(nS).map(j=>beta*(4*(a[j]||0)-0.35*Math.abs(j-p)));
        if(mode==="hard"){ const m=d3.maxIndex(logits); return logits.map((_,j)=>j===m?1:0); }
        const mx=d3.max(logits), ex=logits.map(v=>Math.exp(v-mx)), z=d3.sum(ex);
        return ex.map(v=>v/z);
      });
    }
    let A=null;
    function render(){
      const beta=+d3.select("#att-b").property("value"), mode=d3.select("#att-m").property("value");
      d3.select("#att-bv").text(beta.toFixed(1));
      A=weights(beta,mode);
      gH.selectAll("*").remove();
      A.forEach((row,i)=>{
        const g=gH.append("g").attr("tabindex",0).style("cursor","pointer")
          .on("mouseenter focus",()=>{d3.select("#att-i").property("value",i+1);select(i);});
        row.forEach((v,j)=>g.append("rect").attr("x",L+j*cw+0.5).attr("y",top+i*ch+0.5).attr("width",cw-1).attr("height",ch-1).attr("fill",color(Math.sqrt(v))));
      });
      gH.append("rect").attr("class","sel").attr("x",L).attr("width",nS*cw).attr("height",ch).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2);
      select(+d3.select("#att-i").property("value")-1);
    }
    function select(i){
      const mode=d3.select("#att-m").property("value");
      d3.select("#att-iv").text(`${i+1} · ${tgt[i]}`);
      gH.select("rect.sel").attr("y",top+i*ch);
      svg.selectAll("text.tl").attr("fill",(d,k)=>k===i?C.a2:C.ink);
      const row=A[i];
      const y=d3.scaleLinear().domain([0,1]).range([barBot,barTop]);
      gB.selectAll("*").remove();
      row.forEach((v,j)=>{
        gB.append("rect").attr("x",L+j*cw+3).attr("width",cw-6).attr("y",y(v)).attr("height",barBot-y(v)).attr("fill",C.accent).attr("opacity",.85);
        if(v>=0.05) gB.append("text").attr("x",L+j*cw+cw/2).attr("y",y(v)-3).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.ink).text(v.toFixed(2));
      });
      gB.append("line").attr("x1",L).attr("x2",L+nS*cw).attr("y1",barBot).attr("y2",barBot).attr("stroke",C.line);
      svg.select("text.bt").text(`weights αᵢⱼ for "${tgt[i]}" over the ${nS} annotations (they sum to 1)`);
      const H=-d3.sum(row,v=>v>0?v*Math.log(v):0);
      const top3=row.map((v,j)=>({v,j})).sort((a,b)=>b.v-a.v).slice(0,3).filter(o=>o.v>0.005);
      const desc=top3.map(o=>`${src[o.j]} ${o.v.toFixed(2)}`).join(", ");
      d3.select("#att-read").text(`${mode==="soft"?"Soft":"Hard"} alignment for target word ${i+1} "${tgt[i]}": ${desc}. Entropy ${H.toFixed(2)} nats (max ${Math.log(nS).toFixed(2)}), about ${Math.exp(H).toFixed(1)} annotations read. The context cᵢ is this weighted sum of annotations. Illustrative weights.`);
    }
    d3.select("#att-i").on("input",function(){select(+this.value-1);});
    d3.select("#att-b").on("input",render);
    d3.select("#att-m").on("change",render);
    render();
  })();

  /* ═══════════════ 2. Table 1, as printed ═══════════════ */
  (function(){
    const svg=d3.select("#tb-svg"); if(svg.empty()) return;
    const W=640,H=280,m={l:46,r:20,t:24,b:44};
    // BLEU on news-test-2014, exactly as printed in the paper's Table 1
    const groups=[
      {g:"trained ≤ 30 words",rows:[{s:"RNNencdec-30",all:13.93,unk:24.19,att:false},{s:"RNNsearch-30",all:21.50,unk:31.44,att:true}]},
      {g:"trained ≤ 50 words",rows:[{s:"RNNencdec-50",all:17.82,unk:26.71,att:false},{s:"RNNsearch-50",all:26.75,unk:34.16,att:true}]},
      {g:"trained longer",rows:[{s:"RNNsearch-50*",all:28.45,unk:36.15,att:true}]}
    ];
    const PB={all:33.30,unk:35.63};
    const x0=d3.scaleBand().domain(groups.map(g=>g.g)).range([m.l,W-m.r]).padding(0.2);
    const y=d3.scaleLinear().domain([0,40]).range([H-m.b,m.t]);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(8)).call(a=>{a.selectAll("text").attr("fill",C.muted);a.selectAll("line,path").attr("stroke",C.line);});
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x0)).call(a=>{a.selectAll("text").attr("fill",C.ink).attr("font-size",11);a.selectAll("line,path").attr("stroke",C.line);});
    svg.append("text").attr("transform",`translate(12,${(H-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("test BLEU");
    svg.append("rect").attr("x",W-m.r-190).attr("y",6).attr("width",10).attr("height",10).attr("fill",C.a2);
    svg.append("text").attr("x",W-m.r-176).attr("y",15).attr("fill",C.muted).attr("font-size",10.5).text("fixed vector");
    svg.append("rect").attr("x",W-m.r-104).attr("y",6).attr("width",10).attr("height",10).attr("fill",C.accent);
    svg.append("text").attr("x",W-m.r-90).attr("y",15).attr("fill",C.muted).attr("font-size",10.5).text("attention");
    const pbLine=svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("stroke",C.ink).attr("stroke-dasharray","4 4").attr("opacity",.75);
    const pbText=svg.append("text").attr("x",m.l+4).attr("font-size",10.5).attr("fill",C.ink);
    const g=svg.append("g");
    let key="all";
    function draw(){
      pbLine.transition().duration(300).attr("y1",y(PB[key])).attr("y2",y(PB[key]));
      pbText.transition().duration(300).attr("y",y(PB[key])-4).text(`phrase-based system ${PB[key].toFixed(2)}`);
      g.selectAll("*").remove();
      groups.forEach(gr=>{
        const x1=d3.scaleBand().domain(gr.rows.map(r=>r.s)).range([0,x0.bandwidth()]).padding(0.12);
        gr.rows.forEach(r=>{
          const v=r[key], gg=g.append("g").attr("tabindex",0);
          const bx=x0(gr.g)+x1(r.s), bw=Math.min(x1.bandwidth(),90), off=(x1.bandwidth()-bw)/2;
          gg.append("rect").attr("x",bx+off).attr("width",bw).attr("y",y(0)).attr("height",0).attr("rx",3).attr("fill",r.att?C.accent:C.a2).attr("opacity",.85)
            .transition().duration(300).attr("y",y(v)).attr("height",y(0)-y(v));
          gg.append("text").attr("x",bx+x1.bandwidth()/2).attr("y",y(v)-4).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.ink).text(v.toFixed(2));
          gg.append("text").attr("x",bx+x1.bandwidth()/2).attr("y",y(0)-6).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.panel).text(r.s.replace("RNN",""));
          gg.on("mouseenter focus",()=>d3.select("#tb-read").text(`${r.s}: ${v.toFixed(2)} BLEU (${key==="all"?"all sentences":"no unknown words"}), ${(v-PB[key]>=0?"+":"")+(v-PB[key]).toFixed(2)} against the phrase-based system.`));
        });
      });
      const d30=groups[0].rows[1][key]-groups[0].rows[0][key], d50=groups[1].rows[1][key]-groups[1].rows[0][key], best=groups[2].rows[0][key];
      d3.select("#tb-read").text(`${key==="all"?"All sentences":"Sentences with no unknown words"}: attention adds ${d30.toFixed(2)} BLEU at 30 words and ${d50.toFixed(2)} at 50. RNNsearch-50* ${best.toFixed(2)} vs phrase-based ${PB[key].toFixed(2)} (${(best-PB[key]>=0?"+":"")+(best-PB[key]).toFixed(2)}).`);
    }
    d3.select("#tb-all").on("click",function(){key="all";d3.select("#tb-unk").classed("on",false);d3.select(this).classed("on",true);draw();});
    d3.select("#tb-unk").on("click",function(){key="unk";d3.select("#tb-all").classed("on",false);d3.select(this).classed("on",true);draw();});
    draw();
  })();
})();
