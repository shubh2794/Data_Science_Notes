/* seq2seq-lstm.viz.js — interactive figures for nlp/machine-translation/seq2seq-lstm.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-lag   source reversal: distance from each source word to its translation (our arithmetic)
     #viz-bleu  Tables 1 and 2 as printed: BLEU per system on ntst14 */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. Source reversal and the minimal time lag ═══════════════ */
  (function(){
    const svg=d3.select("#lag-svg"); if(svg.empty()) return;
    const W=640, L=86, R=14;
    const rowY=[52,124];              // baseline of the two timelines
    const chartTop=186, chartBot=296;  // bar chart area
    const gT=svg.append("g"), gC=svg.append("g");
    svg.append("text").attr("x",8).attr("y",rowY[0]+4).attr("fill",C.a2).attr("font-size",11.5).text("in order");
    svg.append("text").attr("x",8).attr("y",rowY[1]+4).attr("fill",C.accent).attr("font-size",11.5).text("reversed");
    svg.append("text").attr("x",8).attr("y",chartTop-6).attr("fill",C.muted).attr("font-size",11).text("distance (time steps) from each target word yᵢ back to its source word, by i");

    // alignment a(i) = index j of the source word that target word i translates (1-based)
    function align(n,mode){
      const a=[];
      for(let i=1;i<=n;i++){
        if(mode==="mono") a.push(i);
        else if(mode==="inv") a.push(n-i+1);
        else { // adjacent pairs swapped: 2,1,4,3,...
          if(i%2===1) a.push(i+1<=n?i+1:i); else a.push(i-1);
        }
      }
      return a;
    }

    function draw(){
      const n=+d3.select("#lag-n").property("value"), mode=d3.select("#lag-a").property("value");
      d3.select("#lag-nv").text(n);
      const a=align(n,mode);
      const x=d3.scaleLinear().domain([0.5,2*n+0.5]).range([L,W-R]);
      const step=(W-R-L)/(2*n);
      gT.selectAll("*").remove();
      const rows=[{rev:false,col:C.a2},{rev:true,col:C.accent}];
      const dist={false:[],true:[]};
      rows.forEach((r,ri)=>{
        const y=rowY[ri];
        // tokens: encoder steps 1..n, decoder steps n+1..2n
        for(let t=1;t<=2*n;t++){
          const enc=t<=n;
          const lab=enc?("x"+(r.rev?(n-t+1):t)):("y"+(t-n));
          gT.append("rect").attr("x",x(t)-step*0.42).attr("y",y-9).attr("width",step*0.84).attr("height",18).attr("rx",3)
            .attr("fill",enc?"rgba(91,156,255,.16)":"rgba(255,180,84,.14)").attr("stroke",C.line);
          if(n<=20) gT.append("text").attr("x",x(t)).attr("y",y+4).attr("text-anchor","middle").attr("font-size",n<=12?10:8).attr("fill",C.ink).text(lab);
        }
        gT.append("line").attr("x1",x(n+0.5)).attr("x2",x(n+0.5)).attr("y1",y-16).attr("y2",y+14).attr("stroke",C.muted).attr("stroke-dasharray","3 3");
        // arcs from source word a(i) to target word i
        for(let i=1;i<=n;i++){
          const j=a[i-1];
          const ts=r.rev?(n-j+1):j, tt=n+i;
          dist[r.rev].push(tt-ts);
          const x1=x(ts), x2=x(tt), h=Math.min(34,6+(x2-x1)*0.09);
          const yy=ri===0?y-10:y+10, dir=ri===0?-1:1;
          gT.append("path").attr("d",`M${x1},${yy} Q${(x1+x2)/2},${yy+dir*h*2} ${x2},${yy}`)
            .attr("fill","none").attr("stroke",r.col).attr("stroke-width",i===1?2.4:1).attr("opacity",i===1?1:0.55);
        }
      });
      // bar chart of distances by target index
      gC.selectAll("*").remove();
      const maxD=d3.max([...dist[false],...dist[true]]);
      const y=d3.scaleLinear().domain([0,maxD]).range([chartBot,chartTop+6]);
      const xb=d3.scaleBand().domain(d3.range(1,n+1)).range([L,W-R]).padding(0.15);
      const bw=xb.bandwidth()/2;
      for(let i=1;i<=n;i++){
        [false,true].forEach((rev,k)=>{
          const v=dist[rev][i-1];
          gC.append("rect").attr("x",xb(i)+k*bw).attr("width",Math.max(1,bw-0.5)).attr("y",y(v)).attr("height",chartBot-y(v))
            .attr("fill",rev?C.accent:C.a2).attr("opacity",0.8);
        });
      }
      const mean=d3.mean(dist[false]); // identical for both orders for any one-to-one alignment
      gC.append("line").attr("x1",L).attr("x2",W-R).attr("y1",y(mean)).attr("y2",y(mean)).attr("stroke",C.ink).attr("stroke-dasharray","4 4").attr("opacity",.8);
      gC.append("text").attr("x",W-R).attr("y",y(mean)-4).attr("text-anchor","end").attr("font-size",10.5).attr("fill",C.ink).text("average = "+mean.toFixed(1)+" (both orders)");
      gC.append("g").attr("transform",`translate(${L},0)`).call(d3.axisLeft(y).ticks(4)).call(g=>{g.selectAll("text").attr("fill",C.muted);g.selectAll("line,path").attr("stroke",C.line);});
      gC.append("text").attr("x",(L+W-R)/2).attr("y",chartBot+16).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.muted).text("target position i = 1 … "+n);

      const s=(arr)=>({mean:d3.mean(arr),min:d3.min(arr),max:d3.max(arr),first:arr[0]});
      const f=s(dist[false]), r=s(dist[true]);
      const lab={mono:"word-for-word",swap:"adjacent pairs swapped",inv:"fully inverted"}[mode];
      d3.select("#lag-read").text(`n = ${n}, ${lab} alignment. In order: lag before the first target word's source = ${f.first}, average ${f.mean.toFixed(1)}, minimum ${f.min}, maximum ${f.max}. Reversed: first ${r.first}, average ${r.mean.toFixed(1)}, minimum ${r.min}, maximum ${r.max}. The average is the same in both orders.`);
    }
    d3.select("#lag-n").on("input",draw);
    d3.select("#lag-a").on("change",draw);
    draw();
  })();

  /* ═══════════════ 2. Tables 1 and 2, as printed ═══════════════ */
  (function(){
    const svg=d3.select("#bl-svg"); if(svg.empty()) return;
    const W=640,H=300,m={l:300,r:46,t:14,b:30};
    const BASE=33.30;
    // values exactly as printed in the paper's Tables 1 and 2 (ntst14); the oracle (≈45) is omitted
    const T={
      dec:[
        {s:"attention-based recurrent model",v:28.45,k:"other"},
        {s:"phrase-based baseline",v:33.30,k:"base"},
        {s:"single forward LSTM, beam 12",v:26.17,k:"lstm"},
        {s:"single reversed LSTM, beam 12",v:30.59,k:"lstm"},
        {s:"5 reversed LSTMs, beam 1",v:33.00,k:"lstm"},
        {s:"2 reversed LSTMs, beam 12",v:33.27,k:"lstm"},
        {s:"5 reversed LSTMs, beam 2",v:34.50,k:"lstm"},
        {s:"5 reversed LSTMs, beam 12",v:34.81,k:"lstm"}
      ],
      res:[
        {s:"phrase-based baseline",v:33.30,k:"base"},
        {s:"encoder–decoder as an SMT feature",v:34.54,k:"other"},
        {s:"rescore 1000-best: forward LSTM",v:35.61,k:"lstm"},
        {s:"rescore 1000-best: reversed LSTM",v:35.85,k:"lstm"},
        {s:"rescore 1000-best: 5 reversed LSTMs",v:36.5,k:"lstm"},
        {s:"best WMT'14 result",v:37.0,k:"other"}
      ]
    };
    const x=d3.scaleLinear().domain([20,38]).range([m.l,W-m.r]);
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`);
    gx.call(d3.axisBottom(x).ticks(9)).call(a=>{a.selectAll("text").attr("fill",C.muted);a.selectAll("line,path").attr("stroke",C.line);});
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-2).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("test BLEU (axis starts at 20)");
    svg.append("line").attr("x1",x(BASE)).attr("x2",x(BASE)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.ink).attr("stroke-dasharray","4 4").attr("opacity",.7);
    const g=svg.append("g");
    let cur="dec";
    const col=k=>k==="lstm"?C.accent:(k==="base"?C.muted:C.violet);
    function draw(){
      const rows=T[cur];
      const y=d3.scaleBand().domain(rows.map(r=>r.s)).range([m.t,H-m.b-6]).padding(0.22);
      g.selectAll("*").remove();
      rows.forEach(r=>{
        const gg=g.append("g").attr("tabindex",0);
        gg.append("text").attr("x",m.l-8).attr("y",y(r.s)+y.bandwidth()/2+4).attr("text-anchor","end").attr("font-size",11.5).attr("fill",C.ink).text(r.s);
        gg.append("rect").attr("x",x(20)).attr("y",y(r.s)).attr("height",y.bandwidth()).attr("rx",3).attr("fill",col(r.k)).attr("opacity",.8)
          .attr("width",0).transition().duration(300).attr("width",x(r.v)-x(20));
        gg.append("text").attr("x",x(r.v)+5).attr("y",y(r.s)+y.bandwidth()/2+4).attr("font-size",11).attr("fill",C.ink).text(r.v.toFixed(2));
        gg.on("mouseenter focus",()=>d3.select("#bl-read").text(`${r.s}: ${r.v.toFixed(2)} BLEU, ${(r.v-BASE>=0?"+":"")+(r.v-BASE).toFixed(2)} against the phrase-based baseline (33.30).`));
      });
      const best=rows.filter(r=>r.k==="lstm").sort((a,b)=>b.v-a.v)[0];
      d3.select("#bl-read").text(cur==="dec"
        ? `LSTM decoding on its own: best ${best.s} at ${best.v.toFixed(2)}, ${(best.v-BASE).toFixed(2)} above the baseline. No single LSTM beats it; reversal adds ${(30.59-26.17).toFixed(2)} on a single model.`
        : `Rescoring the baseline's 1000-best list: best ${best.s} at ${best.v.toFixed(2)}, ${(37.0-best.v).toFixed(1)} below the best WMT'14 result. Oracle rescoring of the same lists would reach about 45.`);
    }
    d3.select("#bl-dec").on("click",function(){cur="dec";d3.select("#bl-res").classed("on",false);d3.select(this).classed("on",true);draw();});
    d3.select("#bl-res").on("click",function(){cur="res";d3.select("#bl-dec").classed("on",false);d3.select(this).classed("on",true);draw();});
    draw();
  })();
})();
