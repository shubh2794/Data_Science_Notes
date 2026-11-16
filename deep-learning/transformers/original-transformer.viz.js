/* original-transformer.viz.js — interactive figures for deep-learning/transformers/original-transformer.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-t1  Table 1 live: per-layer operation count and maximum path length vs n, d, k
     #viz-t2  Table 2: reported BLEU against estimated training FLOPs (log x)
     #viz-t3  Table 3: reported dev BLEU / perplexity for each architecture variation */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const sci=v=>{const e=Math.floor(Math.log10(v));const m=v/Math.pow(10,e);return m.toFixed(m<10?1:0)+"·10^"+e;};

  /* ═══════════════ 1. Table 1, live ═══════════════ */
  (function(){
    const svg=d3.select("#t1-svg"); if(svg.empty()) return;
    const W=640,H=250, L=170, R=20, rowH=26;
    const rows=[
      {k:"self-attention",      cost:(n,d,k)=>n*n*d,   path:(n,d,k)=>1,                           col:C.accent},
      {k:"recurrent",           cost:(n,d,k)=>n*d*d,   path:(n,d,k)=>n,                           col:C.a2},
      {k:"convolution (dilated)",cost:(n,d,k)=>k*n*d*d,path:(n,d,k)=>Math.max(1,Math.ceil(Math.log(n)/Math.log(k))), col:C.violet},
      {k:"convolution (contiguous)",cost:(n,d,k)=>k*n*d*d,path:(n,d,k)=>Math.max(1,Math.ceil(n/k)), col:C.blue}
    ];
    svg.append("text").attr("x",L).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("operations per layer (log scale, relative to the cheapest)");
    svg.append("text").attr("x",L).attr("y",142).attr("fill",C.muted).attr("font-size",11).text("maximum path length between two positions (layers or steps, log scale)");
    const gC=svg.append("g"), gP=svg.append("g");
    function draw(){
      const n=+d3.select("#t1-n").property("value"), d=+d3.select("#t1-d").property("value"), k=+d3.select("#t1-k").property("value");
      d3.select("#t1-nv").text(n); d3.select("#t1-dv").text(d); d3.select("#t1-kv").text(k);
      const costs=rows.slice(0,3).map(r=>({r,v:r.cost(n,d,k)}));
      const minC=d3.min(costs,c=>c.v), maxC=d3.max(costs,c=>c.v);
      const xc=d3.scaleLog().domain([1,Math.max(2,maxC/minC)]).range([0,W-L-R-70]);
      const sc=gC.selectAll("g.b").data(costs,c=>c.r.k);
      const e=sc.enter().append("g").attr("class","b");
      e.append("text").attr("class","l").attr("x",L-8).attr("text-anchor","end").attr("font-size",12).attr("fill",C.ink);
      e.append("rect").attr("height",16).attr("rx",3);
      e.append("text").attr("class","v").attr("font-size",11).attr("fill",C.muted);
      sc.merge(e).each(function(c,i){
        const g=d3.select(this), y=28+i*rowH, w=Math.max(2,xc(c.v/minC));
        g.select("text.l").attr("y",y+12).text(c.r.k);
        g.select("rect").attr("x",L).attr("y",y).attr("fill",c.r.col).transition().duration(250).attr("width",w);
        g.select("text.v").attr("y",y+12).transition().duration(250).attr("x",L+w+6).text(sci(c.v)+(c.v===minC?"  (cheapest)":"  ×"+(c.v/minC).toFixed(c.v/minC<10?1:0)));
      });
      const paths=rows.map(r=>({r,v:r.path(n,d,k)}));
      const xp=d3.scaleLog().domain([1,Math.max(2,d3.max(paths,p=>p.v))]).range([0,W-L-R-70]);
      const sp=gP.selectAll("g.b").data(paths,p=>p.r.k);
      const e2=sp.enter().append("g").attr("class","b");
      e2.append("text").attr("class","l").attr("x",L-8).attr("text-anchor","end").attr("font-size",12).attr("fill",C.ink);
      e2.append("rect").attr("height",14).attr("rx",3);
      e2.append("text").attr("class","v").attr("font-size",11).attr("fill",C.muted);
      sp.merge(e2).each(function(p,i){
        const g=d3.select(this), y=152+i*22, w=Math.max(3,xp(p.v)+3);
        g.select("text.l").attr("y",y+11).text(p.r.k);
        g.select("rect").attr("x",L).attr("y",y).attr("fill",p.r.col).transition().duration(250).attr("width",w);
        g.select("text.v").attr("y",y+11).transition().duration(250).attr("x",L+w+6).text(p.v);
      });
      const cheaper = n<d ? "self-attention is cheaper per layer than recurrence (n < d)" : "recurrence is cheaper per layer than self-attention (n ≥ d)";
      d3.select("#t1-read").text(`n = ${n}, d = ${d}: ${cheaper}. Self-attention needs 1 step to connect any two positions; recurrence needs ${n}; dilated convolutions ${paths[2].v}; contiguous convolutions ${paths[3].v}.`);
    }
    d3.selectAll("#t1-n,#t1-d,#t1-k").on("input",draw);
    draw();
  })();

  /* ═══════════════ 2. Table 2: BLEU vs training FLOPs ═══════════════ */
  (function(){
    const svg=d3.select("#t2-svg"); if(svg.empty()) return;
    const W=640,H=290,m={l:56,r:24,t:18,b:42};
    // values as printed in the paper's Table 2; null = not reported
    const data=[
      {s:"deep attention + PosUnk",           de:null, fr:39.2,  fde:null,   ffr:1.0e20, ens:false},
      {s:"recurrent + RL",                    de:24.6, fr:39.92, fde:2.3e19, ffr:1.4e20, ens:false},
      {s:"convolutional seq2seq",             de:25.16,fr:40.46, fde:9.6e18, ffr:1.5e20, ens:false},
      {s:"mixture-of-experts LSTM",           de:26.03,fr:40.56, fde:2.0e19, ffr:1.2e20, ens:false},
      {s:"deep attention + PosUnk, ensemble", de:null, fr:40.4,  fde:null,   ffr:8.0e20, ens:true},
      {s:"recurrent + RL, ensemble",          de:26.30,fr:41.16, fde:1.8e20, ffr:1.1e21, ens:true},
      {s:"convolutional seq2seq, ensemble",   de:26.36,fr:41.29, fde:7.7e19, ffr:1.2e21, ens:true},
      {s:"Transformer base",                  de:27.3, fr:38.1,  fde:3.3e18, ffr:3.3e18, ens:false, tf:true},
      {s:"Transformer big",                   de:28.4, fr:41.8,  fde:2.3e19, ffr:2.3e19, ens:false, tf:true}
    ];
    const x=d3.scaleLog().domain([1e18,3e21]).range([m.l,W-m.r]);
    const y=d3.scaleLinear().range([H-m.b,m.t]);
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`), gy=svg.append("g").attr("transform",`translate(${m.l},0)`);
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("estimated training FLOPs (log)");
    svg.append("text").attr("transform",`translate(14,${(H-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("BLEU, newstest2014");
    const g=svg.append("g");
    let pair="de";
    function style(a){a.selectAll("text").attr("fill",C.muted);a.selectAll("line,path").attr("stroke",C.line);}
    function draw(){
      const pts=data.filter(d=>d[pair]!=null && d["f"+pair]!=null);
      const ys=pts.map(d=>d[pair]);
      y.domain([Math.floor(d3.min(ys)-0.6),Math.ceil(d3.max(ys)+0.6)]);
      gx.call(d3.axisBottom(x).ticks(4,"~e")).call(style);
      gy.transition().duration(300).call(d3.axisLeft(y).ticks(6)).on("end",function(){style(gy);}); style(gy);
      const s=g.selectAll("g.p").data(pts,d=>d.s);
      s.exit().remove();
      const e=s.enter().append("g").attr("class","p").attr("tabindex",0);
      e.append("circle").attr("r",6);
      e.append("text").attr("font-size",10.5).attr("dx",9).attr("dy",4);
      s.merge(e).each(function(d){
        const gg=d3.select(this);
        gg.transition().duration(300).attr("transform",`translate(${x(d["f"+pair])},${y(d[pair])})`);
        gg.select("circle").attr("fill",d.tf?C.accent:(d.ens?"none":C.a2)).attr("stroke",d.tf?C.accent:C.a2).attr("stroke-width",2);
        gg.select("text").attr("fill",d.tf?C.ink:C.muted).text(d.tf?d.s:"");
      }).on("mouseenter focus",(ev,d)=>d3.select("#t2-read").text(`${d.s}: ${d[pair]} BLEU at ≈ ${sci(d["f"+pair])} training FLOPs${d.ens?" (ensemble)":""}.`));
      const best=pts.filter(d=>!d.tf).sort((a,b)=>b[pair]-a[pair])[0], big=pts.find(d=>d.s==="Transformer big");
      d3.select("#t2-read").text(`${pair==="de"?"English→German":"English→French"}: Transformer big ${big[pair]} BLEU at ${sci(big["f"+pair])} FLOPs; best earlier system ${best.s} ${best[pair]} at ${sci(best["f"+pair])}. Filled = single model, hollow = ensemble.`);
    }
    d3.select("#t2-de").on("click",function(){pair="de";d3.select("#t2-fr").classed("on",false);d3.select(this).classed("on",true);draw();});
    d3.select("#t2-fr").on("click",function(){pair="fr";d3.select("#t2-de").classed("on",false);d3.select(this).classed("on",true);draw();});
    draw();
  })();

  /* ═══════════════ 3. Table 3: ablations ═══════════════ */
  (function(){
    const svg=d3.select("#t3-svg"); if(svg.empty()) return;
    const W=640,H=280,m={l:46,r:46,t:20,b:58};
    const BASE={bleu:25.8,ppl:4.92};
    // values as printed in the paper's Table 3 (newstest2013, dev)
    const G={
      "A · heads":[{v:"h=1",bleu:24.9,ppl:5.29},{v:"h=4",bleu:25.5,ppl:5.00},{v:"h=8 (base)",bleu:25.8,ppl:4.92},{v:"h=16",bleu:25.8,ppl:4.91},{v:"h=32",bleu:25.4,ppl:5.01}],
      "B · key size":[{v:"d_k=16",bleu:25.1,ppl:5.16},{v:"d_k=32",bleu:25.4,ppl:5.01},{v:"d_k=64 (base)",bleu:25.8,ppl:4.92}],
      "C · size":[{v:"N=2",bleu:23.7,ppl:6.11},{v:"N=4",bleu:25.3,ppl:5.19},{v:"N=8",bleu:25.5,ppl:4.88},{v:"d=256",bleu:24.5,ppl:5.75},{v:"d=1024",bleu:26.0,ppl:4.66},{v:"d_ff=1024",bleu:25.4,ppl:5.12},{v:"d_ff=4096",bleu:26.2,ppl:4.75},{v:"big",bleu:26.4,ppl:4.33}],
      "D · regularisation":[{v:"P_drop=0",bleu:24.6,ppl:5.77},{v:"P_drop=0.2",bleu:25.5,ppl:4.95},{v:"ε_ls=0",bleu:25.3,ppl:4.67},{v:"ε_ls=0.2",bleu:25.7,ppl:5.47}],
      "E · positions":[{v:"sinusoidal (base)",bleu:25.8,ppl:4.92},{v:"learned",bleu:25.7,ppl:4.92}]
    };
    let cur="A · heads";
    const btns=d3.select("#t3-btns");
    Object.keys(G).forEach(k=>btns.append("button").attr("type","button").text(k).classed("on",k===cur).on("click",function(){cur=k;btns.selectAll("button").classed("on",false);d3.select(this).classed("on",true);draw();}));
    const yB=d3.scaleLinear().domain([23,26.8]).range([H-m.b,m.t]);
    const yP=d3.scaleLinear().domain([4,6.4]).range([H-m.b,m.t]);
    const x0=d3.scaleBand().range([m.l,W-m.r]).padding(0.22);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(yB).ticks(6)).call(a=>{a.selectAll("text").attr("fill",C.accent);a.selectAll("line,path").attr("stroke",C.line);});
    svg.append("g").attr("transform",`translate(${W-m.r},0)`).call(d3.axisRight(yP).ticks(6)).call(a=>{a.selectAll("text").attr("fill",C.a2);a.selectAll("line,path").attr("stroke",C.line);});
    svg.append("text").attr("x",m.l).attr("y",12).attr("fill",C.accent).attr("font-size",11).text("dev BLEU (bars, higher better)");
    svg.append("text").attr("x",W-m.r).attr("y",12).attr("text-anchor","end").attr("fill",C.a2).attr("font-size",11).text("dev perplexity (dots, lower better)");
    svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",yB(BASE.bleu)).attr("y2",yB(BASE.bleu)).attr("stroke",C.accent).attr("stroke-dasharray","4 4").attr("opacity",.7);
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`);
    const g=svg.append("g");
    function draw(){
      const rows=G[cur];
      x0.domain(rows.map(r=>r.v));
      gx.call(d3.axisBottom(x0)).call(a=>{a.selectAll("text").attr("fill",C.ink).attr("font-size",10.5).attr("transform","rotate(-18)").attr("text-anchor","end");a.selectAll("line,path").attr("stroke",C.line);});
      const s=g.selectAll("g.r").data(rows,r=>cur+r.v);
      s.exit().remove();
      const e=s.enter().append("g").attr("class","r").attr("tabindex",0);
      e.append("rect").attr("rx",3);
      e.append("circle").attr("r",5);
      e.append("text").attr("class","bv").attr("font-size",10.5).attr("text-anchor","middle").attr("fill",C.ink);
      s.merge(e).each(function(r){
        const gg=d3.select(this), bx=x0(r.v), bw=x0.bandwidth();
        gg.select("rect").attr("x",bx).attr("width",bw).attr("y",yB(r.bleu)).attr("height",H-m.b-yB(r.bleu)).attr("fill",r.bleu>=BASE.bleu?C.accent:C.bad).attr("opacity",.75);
        gg.select("circle").attr("cx",bx+bw/2).attr("cy",yP(r.ppl)).attr("fill",C.a2).attr("stroke",C.panel).attr("stroke-width",1.5);
        gg.select("text.bv").attr("x",bx+bw/2).attr("y",yB(r.bleu)-5).text(r.bleu.toFixed(1));
      }).on("mouseenter focus",(ev,r)=>d3.select("#t3-read").text(`${cur}, ${r.v}: ${r.bleu} BLEU (${(r.bleu-BASE.bleu>=0?"+":"")+(r.bleu-BASE.bleu).toFixed(1)} vs base), perplexity ${r.ppl} (${(r.ppl-BASE.ppl>=0?"+":"")+(r.ppl-BASE.ppl).toFixed(2)}).`));
      const best=rows.slice().sort((a,b)=>b.bleu-a.bleu)[0], worst=rows.slice().sort((a,b)=>a.bleu-b.bleu)[0];
      d3.select("#t3-read").text(`${cur}: best ${best.v} at ${best.bleu} BLEU, worst ${worst.v} at ${worst.bleu}. Base model: ${BASE.bleu} BLEU, ${BASE.ppl} PPL. Single runs on one dev set.`);
    }
    draw();
  })();
})();
