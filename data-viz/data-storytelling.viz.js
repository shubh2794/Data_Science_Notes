/* data-storytelling.viz.js — interactive figures for data-storytelling.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: declutter → highlight → annotate, four stages ---- */
(function(){
  const C={accent:"#fcd34d",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",grey:"#5b6272"};
  const svg=d3.select("#dc-svg"); if(svg.empty()) return;
  const W=660,H=290,m={t:58,r:96,b:30,l:62};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const MONTHS=["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar"];
  const EV=5; // March price change
  const rng=d3.randomLcg(0.42), nz=()=>rng()-0.5;
  const base={Basic:62000,Plus:88000,Pro:120000,Team:45000,Enterprise:30000};
  const growth={Basic:0.006,Plus:0.01,Pro:0.004,Team:0.02,Enterprise:0.012};
  const series=Object.keys(base).map(k=>({k,v:MONTHS.map((_,i)=>{
    let v=base[k]*(1+growth[k]*i)*(1+0.025*nz());
    if(k==="Pro" && i>EV) v*= 1-0.30*Math.min(1,(i-EV)/4);
    return v;})}));
  const x=d3.scalePoint().domain(d3.range(MONTHS.length)).range([0,iw]);
  const y=d3.scaleLinear().domain([0,140000]).range([ih,0]);
  const cat=d3.scaleOrdinal().domain(Object.keys(base)).range(["#ef4444","#3b82f6","#22c55e","#a855f7","#f97316"]);
  const NOTES=[
    "Default output: border, heavy gridlines, markers, a legend the eye must shuttle to, five competing colours, and a topic title. The point is in there somewhere.",
    "Decluttered: border, gridlines and markers removed; legend replaced by direct labels; thousands formatted as k. Same information, less to decode.",
    "Highlighted: context series pushed to grey, one series in a strong colour. The eye goes straight to Pro before reading anything.",
    "Annotated: an assertion title states the message, an event marker explains the timing, and a callout quantifies the drop. The chart now tells the story on its own."
  ];

  function draw(){
    const s=+d3.select("#dc-s").property("value");
    d3.select("#dc-s-v").text(["default","declutter","highlight","annotate"][s]);
    svg.selectAll("*").remove();
    const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
    const col=k=> s<=1?cat(k): (k==="Pro"?C.accent:C.grey);
    if(s===0){
      svg.append("rect").attr("x",4).attr("y",4).attr("width",W-8).attr("height",H-8).attr("fill","none").attr("stroke",C.ink).attr("stroke-width",1.5);
      g.append("g").call(d3.axisLeft(y).ticks(7).tickSize(-iw).tickFormat("")).selectAll("line").attr("stroke","#4b5263");
      g.append("g").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(x).tickSize(-ih).tickFormat("")).selectAll("line").attr("stroke","#4b5263");
    }
    const ya=g.append("g").call(d3.axisLeft(y).ticks(s===0?7:4).tickFormat(s===0?d3.format("d"):d=>d/1000+"k"));
    const xa=g.append("g").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(x).tickFormat(i=>MONTHS[i]).tickValues(s===0?d3.range(18):d3.range(0,18,3)));
    [ya,xa].forEach(a=>{a.selectAll("text").attr("fill",s===0?C.ink:C.muted).attr("font-size",9);
      a.selectAll("path,line").attr("stroke",s===0?C.ink:"#3a4150"); if(s>0) a.select(".domain").attr("stroke-opacity",s>=1?0.5:1);});
    if(s===0) g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-50).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text("MAU");
    if(s===3){
      g.append("rect").attr("x",x(EV)).attr("y",0).attr("width",iw-x(EV)).attr("height",ih).attr("fill","rgba(252,211,77,.05)");
      g.append("line").attr("x1",x(EV)).attr("x2",x(EV)).attr("y1",-6).attr("y2",ih).attr("stroke",C.muted).attr("stroke-dasharray","4 3");
      g.append("text").attr("x",x(EV)+4).attr("y",-8+12).attr("fill",C.muted).attr("font-size",10).text("price change");
    }
    const ln=d3.line().x((v,i)=>x(i)).y(v=>y(v));
    const order=s>=2?series.filter(d=>d.k!=="Pro").concat(series.filter(d=>d.k==="Pro")):series;
    order.forEach(d=>{
      g.append("path").attr("d",ln(d.v)).attr("fill","none").attr("stroke",col(d.k)).attr("stroke-width",s>=2&&d.k==="Pro"?3:s===0?2:1.6);
      if(s===0) g.selectAll(null).data(d.v).join("circle").attr("cx",(v,i)=>x(i)).attr("cy",v=>y(v)).attr("r",3).attr("fill",col(d.k));
      if(s>=1) g.append("text").attr("x",iw+6).attr("y",y(d.v[d.v.length-1])+4).attr("fill",col(d.k)).attr("font-size",10.5)
        .attr("font-weight",s>=2&&d.k==="Pro"?700:400).text(d.k);
    });
    if(s===0){
      const lg=svg.append("g").attr("transform",`translate(${W-118},${m.t+4})`);
      lg.append("rect").attr("width",104).attr("height",88).attr("fill","#1b1f27").attr("stroke",C.ink);
      series.forEach((d,i)=>{lg.append("rect").attr("x",8).attr("y",8+i*16).attr("width",10).attr("height",10).attr("fill",cat(d.k));
        lg.append("text").attr("x",24).attr("y",17+i*16).attr("fill",C.ink).attr("font-size",10).text(d.k);});
    }
    if(s===3){
      const pro=series.find(d=>d.k==="Pro").v;
      const i2=EV+4;
      g.append("text").attr("x",x(i2)+8).attr("y",y(pro[i2])+26).attr("fill",C.accent).attr("font-size",11).attr("font-weight",600)
        .text(`−${Math.round((1-pro[i2]/pro[EV])*100)}% in 4 months`);
      g.append("text").attr("x",x(i2)+8).attr("y",y(pro[i2])+40).attr("fill",C.muted).attr("font-size",10).text("other plans unaffected");
    }
    const title = s===3 ? "Pro lost 30% of its users in the four months after the March price change"
                        : "Monthly active users by product";
    svg.append("text").attr("x",s===0?W/2:m.l).attr("y",26).attr("text-anchor",s===0?"middle":"start")
      .attr("fill",C.ink).attr("font-size",s===3?14:s===0?15:13).attr("font-weight",s===0?700:600).text(title);
    if(s===3) svg.append("text").attr("x",m.l).attr("y",43).attr("fill",C.muted).attr("font-size",10.5)
      .text("Monthly active users by plan, 18 months · Pro highlighted");
    d3.select("#dc-read").html(NOTES[s]);
  }
  d3.select("#dc-s").on("input",draw);
  d3.select("#dc-next").on("click",()=>{const el=d3.select("#dc-s"); el.property("value",(+el.property("value")+1)%4); draw();});
  draw();
})();

/* ---- Viz 2: distortion workbench (truncated axis, dual axis, icon area) ---- */
(function(){
  const C={accent:"#fcd34d",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",blue:"#60a5fa"};
  const svg=d3.select("#ms-svg"); if(svg.empty()) return;
  const W=660,H=260;
  const CFG={
    trunc:{lab:"bar baseline",min:0,max:80,step:1,val:0},
    dual:{lab:"right-axis minimum",min:0,max:19,step:0.5,val:0},
    area:{lab:"true ratio r (×10)",min:10,max:30,step:1,val:20}
  };
  const lfCol=lf=>lf>2?C.bad:lf>1.25?C.a2:C.good;

  function setMode(){
    const mode=d3.select("#ms-mode").property("value"), c=CFG[mode];
    d3.select("#ms-lab").text(c.lab);
    d3.select("#ms-p").attr("min",c.min).attr("max",c.max).attr("step",c.step).property("value",c.val);
    draw();
  }
  function axisStyle(a){a.selectAll("text").attr("fill",C.muted).attr("font-size",9); a.selectAll("path,line").attr("stroke","#3a4150");}

  function draw(){
    const mode=d3.select("#ms-mode").property("value"), p=+d3.select("#ms-p").property("value");
    CFG[mode].val=p;
    svg.selectAll("*").remove();
    if(mode==="trunc"){
      d3.select("#ms-p-v").text(p);
      const v1=82, v2=88, m={t:20,r:20,b:30,l:50}, iw=300, ih=H-m.t-m.b;
      const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
      const y=d3.scaleLinear().domain([p,100]).range([ih,0]);
      const x=d3.scaleBand().domain(["last year","this year"]).range([0,iw]).padding(0.35);
      axisStyle(g.append("g").call(d3.axisLeft(y).ticks(5)));
      axisStyle(g.append("g").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(x)));
      [[ "last year",v1],["this year",v2]].forEach(([k,v])=>{
        g.append("rect").attr("x",x(k)).attr("width",x.bandwidth()).attr("y",y(v)).attr("height",ih-y(v)).attr("fill",k==="this year"?C.accent:C.muted);
        g.append("text").attr("x",x(k)+x.bandwidth()/2).attr("y",y(v)-5).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",11).text(v);
      });
      const lf=v1/(v1-p), dataEff=(v2-v1)/v1, gfx=(v2-p)/(v1-p)-1;
      const t=svg.append("g").attr("transform","translate(400,50)");
      t.append("text").attr("fill",C.muted).attr("font-size",11).text("satisfaction score");
      t.append("text").attr("y",30).attr("fill",C.ink).attr("font-size",12).text(`data: +${(dataEff*100).toFixed(1)}%`);
      t.append("text").attr("y",52).attr("fill",C.ink).attr("font-size",12).text(`bar lengths: +${(gfx*100).toFixed(0)}%`);
      t.append("text").attr("y",86).attr("fill",lfCol(lf)).attr("font-size",22).attr("font-weight",700).text(`LF = ${lf.toFixed(1)}`);
      d3.select("#ms-read").html(`baseline <b>${p}</b>: the "this year" bar looks <b>${(1+gfx).toFixed(2)}×</b> as long, though the value is only <b>${(v2/v1).toFixed(3)}×</b>. `+
        `LF = v₁/(v₁ − b) = 82/(82 − ${p}) = <b style="color:${lfCol(lf)}">${lf.toFixed(2)}</b>`);
    } else if(mode==="dual"){
      d3.select("#ms-p-v").text(p.toFixed(1));
      const n=12, A=d3.range(n).map(i=>50+i*1.0+Math.sin(i*1.3)*0.8), B=d3.range(n).map(i=>20+i*0.35+Math.cos(i*1.1)*0.5);
      const m={t:20,r:60,b:30,l:50}, iw=W-m.l-m.r-150, ih=H-m.t-m.b;
      const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
      const x=d3.scaleLinear().domain([0,n-1]).range([0,iw]);
      const yL=d3.scaleLinear().domain([0,70]).range([ih,0]);
      const yR=d3.scaleLinear().domain([p,26]).range([ih,0]);
      axisStyle(g.append("g").call(d3.axisLeft(yL).ticks(5)));
      const ra=g.append("g").attr("transform",`translate(${iw},0)`).call(d3.axisRight(yR).ticks(5)); axisStyle(ra); ra.selectAll("text").attr("fill",C.blue);
      axisStyle(g.append("g").attr("transform",`translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickFormat(i=>"Q"+(i+1))));
      g.append("path").attr("d",d3.line().x((v,i)=>x(i)).y(v=>yL(v))(A)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2.2);
      g.append("path").attr("d",d3.line().x((v,i)=>x(i)).y(v=>yR(v))(B)).attr("fill","none").attr("stroke",C.blue).attr("stroke-width",2.2);
      g.append("text").attr("x",4).attr("y",yL(A[0])-8).attr("fill",C.accent).attr("font-size",10).text("revenue (left)");
      g.append("text").attr("x",4).attr("y",yR(B[0])+14).attr("fill",C.blue).attr("font-size",10).text("ad spend (right)");
      let cross=0; for(let i=1;i<n;i++){ if(Math.sign(yL(A[i])-yR(B[i]))!==Math.sign(yL(A[i-1])-yR(B[i-1]))) cross++; }
      const gA=(A[n-1]-A[0])/A[0], gB=(B[n-1]-B[0])/B[0];
      const lfB=B[0]/(B[0]-p);
      d3.select("#ms-read").html(`true growth: revenue <b>+${(gA*100).toFixed(0)}%</b>, ad spend <b>+${(gB*100).toFixed(0)}%</b> · `+
        `visual growth of the blue line is exaggerated by LF ≈ <b style="color:${lfCol(lfB)}">${lfB.toFixed(1)}</b> · visual crossings: <b>${cross}</b>`+
        `<br><span class="lbl">the right-axis range is a free choice, so where the lines sit and whether they cross carries no information; index both to 100 instead</span>`);
    } else {
      const r=p/10; d3.select("#ms-p-v").text(r.toFixed(1)+"×");
      const bag=(gg,s,cx,cy,col)=>{
        const w=60*s,h=60*s;
        gg.append("rect").attr("x",cx-w/2).attr("y",cy-h).attr("width",w).attr("height",h).attr("rx",8*s).attr("fill",col);
        gg.append("rect").attr("x",cx-w/5).attr("y",cy-h-12*s).attr("width",w*0.4).attr("height",14*s).attr("rx",4*s).attr("fill","none").attr("stroke",col).attr("stroke-width",3*s);
      };
      const g=svg.append("g");
      g.append("text").attr("x",150).attr("y",20).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("icon scaled in both dimensions");
      bag(g,0.6,80,225,C.muted); bag(g,0.6*r,200,225,C.accent);
      g.append("text").attr("x",470).attr("y",20).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("bars (length encodes value)");
      const by=d3.scaleLinear().domain([0,3]).range([225,40]);
      [[0,1,C.muted],[1,r,C.accent]].forEach(([i,v,c])=>g.append("rect").attr("x",420+i*70).attr("width",44).attr("y",by(v)).attr("height",225-by(v)).attr("fill",c));
      g.append("line").attr("x1",400).attr("x2",560).attr("y1",225).attr("y2",225).attr("stroke","#3a4150");
      const lf=r>1?(r*r-1)/(r-1):1;
      d3.select("#ms-read").html(`true ratio <b>${r.toFixed(1)}×</b> · icon area ratio <b>${(r*r).toFixed(1)}×</b> · `+
        `LF = (r² − 1)/(r − 1) = r + 1 = <b style="color:${lfCol(lf)}">${r>1?lf.toFixed(1):"1.0"}</b> (bars: 1.0)`);
    }
  }
  d3.select("#ms-mode").on("change",setMode);
  d3.select("#ms-p").on("input",draw);
  setMode();
})();

/* ---- Viz 3: Simpson's paradox with adjustable case mix ---- */
(function(){
  const C={accent:"#fcd34d",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",blue:"#60a5fa"};
  const svg=d3.select("#sp-svg"); if(svg.empty()) return;
  const W=640,H=250,m={t:20,r:20,b:34,l:46};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const RATE={A:{mild:0.93,severe:0.73},B:{mild:0.87,severe:0.69}};
  const y=d3.scaleLinear().domain([0,1]).range([ih,0]);
  const ya=g.append("g").attr("class","axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`);
  const bars=g.append("g");
  const lg=g.append("g").attr("transform",`translate(${iw-130},0)`);
  [["treatment A",C.accent],["treatment B",C.blue]].forEach((d,i)=>{lg.append("rect").attr("x",i*70).attr("y",-12).attr("width",10).attr("height",10).attr("fill",d[1]);
    lg.append("text").attr("x",i*70+14).attr("y",-3).attr("fill",C.muted).attr("font-size",10).text(d[0]);});

  function draw(){
    const wa=+d3.select("#sp-a").property("value")/100, wb=+d3.select("#sp-b").property("value")/100;
    const agg=d3.select("#sp-agg").property("checked");
    d3.select("#sp-a-v").text(Math.round(wa*100)+"%"); d3.select("#sp-b-v").text(Math.round(wb*100)+"%");
    const oA=(1-wa)*RATE.A.mild+wa*RATE.A.severe, oB=(1-wb)*RATE.B.mild+wb*RATE.B.severe;
    const groups=agg?["overall"]:["mild cases","severe cases","overall"];
    const x0=d3.scaleBand().domain(groups).range([0,iw]).padding(0.3);
    const x1=d3.scaleBand().domain(["A","B"]).range([0,x0.bandwidth()]).padding(0.08);
    xa.call(d3.axisBottom(x0));
    g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",10);
    g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
    const val={"mild cases":{A:RATE.A.mild,B:RATE.B.mild},"severe cases":{A:RATE.A.severe,B:RATE.B.severe},overall:{A:oA,B:oB}};
    const data=groups.flatMap(gr=>["A","B"].map(t=>({gr,t,v:val[gr][t]})));
    const sel=bars.selectAll("g.b").data(data,d=>d.gr+d.t).join(enter=>{const e=enter.append("g").attr("class","b"); e.append("rect"); e.append("text"); return e;});
    sel.select("rect").attr("x",d=>x0(d.gr)+x1(d.t)).attr("width",x1.bandwidth()).attr("y",d=>y(d.v)).attr("height",d=>ih-y(d.v))
      .attr("fill",d=>d.t==="A"?C.accent:C.blue).attr("stroke",d=>d.gr==="overall"?C.ink:"none").attr("stroke-width",1);
    sel.select("text").attr("x",d=>x0(d.gr)+x1(d.t)+x1.bandwidth()/2).attr("y",d=>y(d.v)-5).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10.5)
      .text(d=>(d.v*100).toFixed(1)+"%");
    const paradox=oA<oB;
    d3.select("#sp-read").html(
      `overall: A <b>${(oA*100).toFixed(1)}%</b> vs B <b>${(oB*100).toFixed(1)}%</b> → `+
      (paradox?`<b style="color:${C.bad}">aggregate says B is better — reversed</b>, because A treats ${Math.round(wa*100)}% severe cases vs B's ${Math.round(wb*100)}%`
              :`<b style="color:${C.good}">aggregate agrees with the strata (A better)</b>`)+
      `<br><span class="lbl">within each stratum A wins by 4–6 points; only the weights differ. set both sliders equal and the paradox can never appear</span>`
    );
  }
  ["#sp-a","#sp-b"].forEach(id=>d3.select(id).on("input",draw));
  d3.select("#sp-agg").on("change",draw);
  draw();
})();
