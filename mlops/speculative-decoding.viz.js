/* speculative-decoding.viz.js — extracted from speculative-decoding.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",blue:"#60a5fa",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#15181f"};
  const svg=d3.select("#sd-svg"),W=640,H=360;

  function eTokens(a,g){
    if(a>=0.999) return g+1;            // limit of (1-a^(g+1))/(1-a) as a->1
    return (1-Math.pow(a,g+1))/(1-a);
  }
  function speedup(a,g,c){
    return eTokens(a,g)/(g*c+1);
  }

  // simulate one accept/reject pattern for the stream given alpha & gamma
  function simCycle(a,g){
    // walk gamma drafts; each accepted w.p. a until first reject
    let accepted=0, rejectedAt=-1;
    for(let i=0;i<g;i++){
      if(Math.random()<a){ accepted++; }
      else { rejectedAt=i; break; }
    }
    return {accepted, rejectedAt}; // if rejectedAt<0 all g accepted -> bonus token
  }

  function draw(){
    const a=+d3.select("#sd-a").property("value");
    const g=+d3.select("#sd-g").property("value");
    const c=+d3.select("#sd-c").property("value");
    d3.select("#sd-a-v").text(a.toFixed(2));
    d3.select("#sd-g-v").text(g);
    d3.select("#sd-c-v").text(c.toFixed(2));

    svg.selectAll("*").remove();

    // ---- token stream (top) ----
    const {accepted,rejectedAt}=simCycle(a,g);
    const allAccepted = rejectedAt<0;
    const boxN=g+1;                 // gamma drafts + 1 correction/bonus slot
    const bx0=20, bw=Math.min(54,(W-40)/boxN-8), gap=8, by=44, bh=40;
    svg.append("text").attr("x",bx0).attr("y",26).attr("font-size",12).attr("fill",C.muted)
       .text("one cycle:  γ drafted tokens + 1 correction / bonus");

    for(let i=0;i<boxN;i++){
      const x=bx0+i*(bw+gap);
      let fill, op=1, label, lblColor;
      if(i<g){ // a drafted position
        if(allAccepted){ fill=C.good; label="✓"; lblColor="#0a0d12"; }       // all accepted
        else if(i<accepted){ fill=C.good; label="✓"; lblColor="#0a0d12"; }   // accepted prefix
        else if(i===accepted){ fill=C.bad; label="✗"; lblColor="#0a0d12"; }  // first reject
        else { fill="#2a2f3a"; op=0.35; label=""; lblColor=C.muted; }        // trailing drafts fade
      } else { // the final slot
        if(allAccepted){ fill=C.accent; label="+"; lblColor="#0a0d12"; }     // bonus token
        else { fill=C.blue; label="~"; lblColor="#0a0d12"; }                 // resampled correction
      }
      const grp=svg.append("g");
      grp.append("rect").attr("x",x).attr("y",by).attr("width",bw).attr("height",bh).attr("rx",6)
         .attr("fill",fill).attr("fill-opacity", (i<g && i>accepted && !allAccepted)?0.35:0.9)
         .attr("opacity",op).attr("stroke",C.line);
      if(label) grp.append("text").attr("x",x+bw/2).attr("y",by+bh/2+5).attr("text-anchor","middle")
         .attr("font-size",16).attr("font-weight",700).attr("fill",lblColor).text(label);
      // ghost outline for drafted positions
      grp.append("text").attr("x",x+bw/2).attr("y",by-5).attr("text-anchor","middle")
         .attr("font-size",9).attr("fill",C.muted).text(i<g?("d"+(i+1)):"out");
    }
    const emitted = allAccepted ? g+1 : accepted+1;
    svg.append("text").attr("x",bx0).attr("y",by+bh+22).attr("font-size",11).attr("fill",C.ink)
       .html(null).text("this cycle emitted "+emitted+" token"+(emitted!==1?"s":"")+" from 1 target pass");

    // legend
    const lg=[[C.good,"accepted"],[C.bad,"first reject"],[C.blue,"resampled fix"],[C.accent,"bonus"]];
    lg.forEach((d,i)=>{
      const lx=bx0+i*120, ly=by+bh+38;
      svg.append("rect").attr("x",lx).attr("y",ly-9).attr("width",11).attr("height",11).attr("rx",2).attr("fill",d[0]);
      svg.append("text").attr("x",lx+16).attr("y",ly).attr("font-size",10).attr("fill",C.muted).text(d[1]);
    });

    // ---- speedup vs gamma curve (bottom) ----
    const m={l:46,r:18,t:175,b:42};
    const pw=W-m.l-m.r, ph=H-m.t-m.b;
    const gs=d3.range(1,11);
    const ys=gs.map(gg=>speedup(a,gg,c));
    const ymax=Math.max(1.05, d3.max(ys)*1.12);
    const x=d3.scaleLinear().domain([1,10]).range([m.l,m.l+pw]);
    const y=d3.scaleLinear().domain([0,ymax]).range([m.t+ph,m.t]);

    // axes
    svg.append("line").attr("x1",m.l).attr("y1",m.t+ph).attr("x2",m.l+pw).attr("y2",m.t+ph).attr("stroke",C.line);
    svg.append("line").attr("x1",m.l).attr("y1",m.t).attr("x2",m.l).attr("y2",m.t+ph).attr("stroke",C.line);
    // 1x baseline (no speedup)
    if(1<=ymax){
      svg.append("line").attr("x1",m.l).attr("y1",y(1)).attr("x2",m.l+pw).attr("y2",y(1))
         .attr("stroke",C.muted).attr("stroke-dasharray","4 4").attr("opacity",0.6);
      svg.append("text").attr("x",m.l+pw).attr("y",y(1)-4).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text("1× (no gain)");
    }
    gs.forEach(gg=>{
      svg.append("text").attr("x",x(gg)).attr("y",m.t+ph+14).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(gg);
    });
    svg.append("text").attr("x",m.l+pw/2).attr("y",m.t+ph+34).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("draft length γ");
    svg.append("text").attr("x",m.l-34).attr("y",m.t-6).attr("font-size",10).attr("fill",C.muted).text("speedup");
    [0.5,1,1.5,2,2.5,3].filter(t=>t<=ymax).forEach(t=>{
      svg.append("text").attr("x",m.l-8).attr("y",y(t)+3).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text(t+"×");
    });

    // curve
    const line=d3.line().x((d,i)=>x(gs[i])).y(d=>y(d)).curve(d3.curveMonotoneX);
    const winning = d3.max(ys)>1;
    svg.append("path").datum(ys).attr("fill","none")
       .attr("stroke", winning?C.accent:C.bad).attr("stroke-width",2.2).attr("d",line);
    gs.forEach((gg,i)=>{
      svg.append("circle").attr("cx",x(gg)).attr("cy",y(ys[i])).attr("r",2.6).attr("fill",winning?C.accent:C.bad);
    });

    // optimal gamma marker
    const bestI=ys.indexOf(d3.max(ys)), bestG=gs[bestI], bestS=ys[bestI];
    svg.append("line").attr("x1",x(bestG)).attr("y1",m.t+ph).attr("x2",x(bestG)).attr("y2",y(bestS))
       .attr("stroke",C.a2).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",x(bestG)).attr("cy",y(bestS)).attr("r",5).attr("fill",C.a2);
    svg.append("text").attr("x",x(bestG)).attr("y",y(bestS)-9).attr("text-anchor","middle")
       .attr("font-size",10).attr("font-weight",700).attr("fill",C.a2).text("γ*="+bestG);

    // current gamma marker on curve
    svg.append("circle").attr("cx",x(g)).attr("cy",y(speedup(a,g,c))).attr("r",4).attr("fill",C.ink).attr("stroke",C.line);

    // ---- readout ----
    const Et=eTokens(a,g), Sp=speedup(a,g,c);
    const verdict = (a>c) ? `<span style="color:${C.good}">α &gt; c → net win possible</span>`
                          : `<span style="color:${C.bad}">α ≤ c → draft does not pay for itself</span>`;
    d3.select("#sd-read").html(
      `E[tokens] = <b>${Et.toFixed(2)}</b> &nbsp;·&nbsp; Speedup at γ=${g}: <b style="color:${Sp>1?C.good:C.bad}">${Sp.toFixed(2)}×</b>`+
      ` &nbsp;·&nbsp; optimal γ* = <b style="color:${C.a2}">${bestG}</b> (<b>${bestS.toFixed(2)}×</b>) &nbsp;·&nbsp; ${verdict}`
    );
  }

  ["#sd-a","#sd-g","#sd-c"].forEach(s=>d3.select(s).on("input",draw));
  d3.select("#sd-step").on("click",draw);   // re-roll the random accept/reject pattern
  draw();
})();
