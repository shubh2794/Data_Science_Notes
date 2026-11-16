/* nosql.viz.js — visualizations for nosql.html.
   1. partition-key chooser (balance vs query fan-out)  2. denormalization break-even. */

const NQ_C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",warn:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016"};

/* ---- Viz 1: partition-key chooser ---- */
(function(){
  const C=NQ_C, svg=d3.select("#pk-svg"), W=640, S=8, NORD=2400, NCUST=300, DAYS=30;
  function rng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed);
    t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  function fnv(str){ let h=0x811c9dc5; for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,0x01000193); } return h>>>0; }
  const COUNTRIES=[["US",.45],["DE",.20],["FR",.15],["IN",.12],["BR",.08]];
  function gen(whale){
    const r=rng(42), w=d3.range(1,NCUST).map(i=>1/Math.pow(i,0.8)), tot=d3.sum(w);
    const cum=[]; w.reduce((a,b,i)=>(cum[i]=a+b/tot),0);
    const custCountry=d3.range(NCUST).map(()=>{ let u=r(),a=0; for(const [c,p] of COUNTRIES){a+=p; if(u<a) return c;} return "US"; });
    return d3.range(NORD).map(id=>{
      let cust; if(r()<whale) cust=0; else { const u=r(); cust=1+cum.findIndex(c=>c>=u); if(cust===0) cust=NCUST-1; }
      return {id, cust, day:Math.floor(r()*DAYS), country:custCountry[cust]};
    });
  }
  const KEYF={customer:o=>"c"+o.cust, customer_sort:o=>"c"+o.cust, date:o=>"d"+o.day, country:o=>o.country, order:o=>"o"+o.id};
  const LABEL={customer:"customer_id",customer_sort:"customer_id + sort(order_date)",date:"order_date",country:"country",order:"order_id"};
  function draw(){
    const key=d3.select("#pk-key").property("value"), whale=+d3.select("#pk-whale").property("value");
    const orders=gen(whale), part=o=>fnv(KEYF[key](o))%S;
    const store=new Array(S).fill(0), today=new Array(S).fill(0);
    orders.forEach(o=>{ const p=part(o); store[p]++; if(o.day===DAYS-1) today[p]++; });
    const ctryParts=new Set(COUNTRIES.map(([c])=>fnv(c)%S)).size;
    const Q=[
      {q:"customer's latest 20 orders", v:{customer:["1","whole partition, sort in app",C.warn],customer_sort:["1","sorted range read",C.good],date:[String(S),"scatter-gather",C.bad],country:["1","scans the entire country",C.warn],order:[String(S),"scatter-gather",C.bad]}[key]},
      {q:"today's orders (dashboard)", v:{customer:[String(S),"scatter-gather",C.bad],customer_sort:[String(S),"scatter-gather",C.bad],date:["1","single partition",C.good],country:[String(ctryParts),"every country partition",C.bad],order:[String(S),"scatter-gather",C.bad]}[key]},
      {q:"order by order_id", v:{customer:[String(S),"unless a global index",C.bad],customer_sort:[String(S),"unless a global index",C.bad],date:[String(S),"unless a global index",C.bad],country:[String(S),"unless a global index",C.bad],order:["1","direct key lookup",C.good]}[key]}
    ];
    svg.selectAll("*").remove();
    const px=46, py=14, pw=380, ph=130, bw=pw/S;
    const maxS=d3.max(store), maxT=Math.max(1,d3.max(today));
    const meanS=NORD/S, meanT=d3.sum(today)/S;
    const Y=(v,m)=>py+ph-v/m*ph;
    for(let i=0;i<S;i++){
      const x=px+i*bw;
      svg.append("rect").attr("x",x+4).attr("y",Y(store[i],maxS)).attr("width",bw/2-5).attr("height",py+ph-Y(store[i],maxS)).attr("fill",C.node).attr("opacity",.85);
      svg.append("rect").attr("x",x+bw/2+1).attr("y",Y(today[i],maxT)).attr("width",bw/2-5).attr("height",py+ph-Y(today[i],maxT)).attr("fill",C.warn).attr("opacity",.85);
      svg.append("text").attr("x",x+bw/2).attr("y",py+ph+12).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("P"+i);
    }
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",py+ph).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",Y(meanS,maxS)).attr("y2",Y(meanS,maxS)).attr("stroke",C.node).attr("stroke-dasharray","3 3").attr("stroke-opacity",.7);
    svg.append("text").attr("x",px+2).attr("y",Y(meanS,maxS)-3).attr("fill",C.node).attr("font-size",9).text("storage mean");
    const lx=px+pw+6;
    svg.append("rect").attr("x",lx).attr("y",py+4).attr("width",9).attr("height",9).attr("fill",C.node);
    svg.append("text").attr("x",lx+14).attr("y",py+12).attr("fill",C.ink).attr("font-size",10).text("stored orders");
    svg.append("rect").attr("x",lx).attr("y",py+20).attr("width",9).attr("height",9).attr("fill",C.warn);
    svg.append("text").attr("x",lx+14).attr("y",py+28).attr("fill",C.ink).attr("font-size",10).text("today's writes");
    svg.append("text").attr("x",lx).attr("y",py+52).attr("fill",C.muted).attr("font-size",9.5).text("(each colour scaled");
    svg.append("text").attr("x",lx).attr("y",py+62).attr("fill",C.muted).attr("font-size",9.5).text(" to its own max)");
    // query fan-out table
    const ty=py+ph+34;
    svg.append("text").attr("x",px).attr("y",ty).attr("fill",C.ink).attr("font-size",10.5).attr("font-weight",600).text("query");
    svg.append("text").attr("x",px+240).attr("y",ty).attr("fill",C.ink).attr("font-size",10.5).attr("font-weight",600).text("partitions touched");
    Q.forEach((row,i)=>{
      const y=ty+22+i*22;
      svg.append("rect").attr("x",px-6).attr("y",y-14).attr("width",560).attr("height",19).attr("rx",4).attr("fill",i%2?"transparent":"rgba(255,255,255,.025)");
      svg.append("text").attr("x",px).attr("y",y).attr("fill",C.muted).attr("font-size",10.5).text(row.q);
      svg.append("text").attr("x",px+240).attr("y",y).attr("fill",row.v[2]).attr("font-size",11).attr("font-weight",700).text(row.v[0]+" / "+S);
      svg.append("text").attr("x",px+300).attr("y",y).attr("fill",row.v[2]).attr("font-size",10).text(row.v[1]);
    });
    const used=store.filter(v=>v>0).length;
    d3.select("#pk-read").html(`key <b>${LABEL[key]}</b> · storage max/mean <b>${(maxS/meanS).toFixed(2)}×</b> · today's write max/mean <b style="color:${maxT/Math.max(meanT,1e-9)>3?C.bad:C.ink}">${(maxT/Math.max(meanT,1e-9)).toFixed(2)}×</b> · partitions holding data <b>${used}/${S}</b>`+
      (key==="date"?` — <span style="color:${C.bad}">every new write lands on one partition (monotonic key)</span>`:"")+
      (key==="country"?` — <span style="color:${C.bad}">low cardinality caps parallelism</span>`:"")+
      ((key==="customer"||key==="customer_sort")&&whale>0.15?` — <span style="color:${C.warn}">the whale customer is a hot partition; consider write sharding (key salting)</span>`:""));
  }
  d3.selectAll("#pk-key,#pk-whale").on("input change",draw);
  draw();
})();

/* ---- Viz 2: denormalization break-even ---- */
(function(){
  const C=NQ_C, svg=d3.select("#dn-svg"), W=640, H=210, px=56, py=14, pw=460, ph=160;
  function draw(){
    const J=+d3.select("#dn-j").property("value"), F=+d3.select("#dn-f").property("value"), lr=+d3.select("#dn-r").property("value");
    const R=Math.pow(10,lr), xs=d3.range(-1,3.001,0.05);
    const norm=r=>r*J+1, den=r=>r+F;
    const ymax=Math.log10(norm(1000)+F), ymin=0;
    const X=x=>px+(x+1)/4*pw, Y=v=>py+ph-(Math.log10(Math.max(v,1))-ymin)/(ymax-ymin)*ph;
    svg.selectAll("*").remove();
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",py+ph).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("x2",px).attr("y1",py).attr("y2",py+ph).attr("stroke","#3a4150");
    [-1,0,1,2,3].forEach(t=>svg.append("text").attr("x",X(t)).attr("y",py+ph+13).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text(["0.1","1","10","100","1000"][t+1]));
    svg.append("text").attr("x",px+pw/2).attr("y",H-2).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("reads per write R (log)");
    svg.append("text").attr("x",14).attr("y",py+ph/2).attr("fill",C.muted).attr("font-size",10).attr("text-anchor","middle").attr("transform",`rotate(-90,14,${py+ph/2})`).text("ops per write (log)");
    svg.append("path").attr("d",d3.line().x(X).y(x=>Y(norm(Math.pow(10,x))))(xs)).attr("fill","none").attr("stroke",C.node).attr("stroke-width",2.3);
    svg.append("path").attr("d",d3.line().x(X).y(x=>Y(den(Math.pow(10,x))))(xs)).attr("fill","none").attr("stroke",C.warn).attr("stroke-width",2.3).attr("stroke-dasharray","5 3");
    svg.append("text").attr("x",px+pw+6).attr("y",Y(norm(1000))+4).attr("fill",C.node).attr("font-size",10).text("normalized");
    svg.append("text").attr("x",px+pw+6).attr("y",Y(den(1000))+14).attr("fill",C.warn).attr("font-size",10).text("denormalized");
    const be=(F-1)/(J-1);
    if(be>0.1 && be<1000){
      const bx=X(Math.log10(be));
      svg.append("line").attr("x1",bx).attr("x2",bx).attr("y1",py).attr("y2",py+ph).attr("stroke",C.muted).attr("stroke-dasharray","2 4");
      svg.append("text").attr("x",bx+4).attr("y",py+10).attr("fill",C.muted).attr("font-size",9.5).text("break-even R* = "+be.toFixed(1));
    }
    svg.append("line").attr("x1",X(lr)).attr("x2",X(lr)).attr("y1",py).attr("y2",py+ph).attr("stroke",C.accent).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(lr)).attr("cy",Y(norm(R))).attr("r",4.5).attr("fill",C.node);
    svg.append("circle").attr("cx",X(lr)).attr("cy",Y(den(R))).attr("r",4.5).attr("fill",C.warn);
    const win=den(R)<norm(R);
    d3.select("#dn-read").html(`R = <b>${R<1?R.toFixed(2):R.toFixed(R<10?1:0)}</b> reads/write · normalized <b>${norm(R).toFixed(0)}</b> vs denormalized <b>${den(R).toFixed(0)}</b> ops → `+
      (win?`<span style="color:${C.good}">denormalize</span> (saves ${(100*(1-den(R)/norm(R))).toFixed(0)}%)`:`<span style="color:${C.bad}">stay normalized</span>: ${F} copies per write cost more than the joins saved`)+
      ` · break-even at R* = (F−1)/(J−1) = <b>${be.toFixed(1)}</b>`);
  }
  d3.selectAll("#dn-j,#dn-f,#dn-r").on("input",draw);
  draw();
})();
