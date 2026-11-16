/* dashboards.viz.js — interactive figures for dashboards.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: judge-the-ratio experiment across visual encodings ---- */
(function(){
  const C={accent:"#fcd34d",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#1b1f27"};
  const svg=d3.select("#en-svg"); if(svg.empty()) return;
  const W=660,H=250;
  const ENC=[["pos","position"],["len","length"],["ang","angle"],["area","area"],["col","lightness"]];
  const scores={}; ENC.forEach(e=>scores[e[0]]=[]);
  const stim=svg.append("g");
  const board=svg.append("g").attr("transform","translate(400,30)");
  svg.append("line").attr("x1",380).attr("x2",380).attr("y1",14).attr("y2",H-14).attr("stroke",C.line);
  board.append("text").attr("x",0).attr("y",-10).attr("fill",C.ink).attr("font-size",11).attr("font-weight",600).text("your mean absolute error (percentage points)");
  const bx=d3.scaleLinear().domain([0,30]).range([0,170]).clamp(true);

  let trial;
  function newTrial(){ trial={r:0.1+0.8*Math.random(), s:0.75+0.25*Math.random(), c:0.3+0.5*Math.random(), flip:Math.random()<0.5}; }

  function drawStim(){
    stim.selectAll("*").remove();
    const enc=d3.select("#en-type").property("value");
    const {r,s,c,flip}=trial;
    const lab=(x,y,t)=>stim.append("text").attr("x",x).attr("y",y).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",12).attr("font-weight",600).text(t);
    const xa=flip?250:120, xb=flip?120:250;   // randomize left/right placement
    if(enc==="pos"){
      stim.append("line").attr("x1",60).attr("x2",60).attr("y1",30).attr("y2",215).attr("stroke",C.muted);
      stim.append("line").attr("x1",60).attr("x2",320).attr("y1",215).attr("y2",215).attr("stroke",C.muted);
      const hA=180*s, hB=hA*r;
      [[xa,hA,"A"],[xb,hB,"B"]].forEach(([x,h,t])=>{
        stim.append("line").attr("x1",x).attr("x2",x).attr("y1",215).attr("y2",215-h).attr("stroke",C.line);
        stim.append("circle").attr("cx",x).attr("cy",215-h).attr("r",7).attr("fill",C.accent);
        lab(x,234,t);});
    } else if(enc==="len"){
      const LA=250*s, LB=LA*r;
      stim.append("rect").attr("x",flip?70:30).attr("y",flip?140:60).attr("width",LA).attr("height",34).attr("fill",C.accent);
      stim.append("rect").attr("x",flip?20:95).attr("y",flip?60:140).attr("width",LB).attr("height",34).attr("fill",C.accent);
      stim.append("text").attr("x",8).attr("y",flip?162:82).attr("fill",C.muted).attr("font-size",12).attr("font-weight",600).text("A");
      stim.append("text").attr("x",8).attr("y",flip?82:162).attr("fill",C.muted).attr("font-size",12).attr("font-weight",600).text("B");
    } else if(enc==="ang"){
      const vals=[{k:"A",v:1},{k:"B",v:r},{k:"",v:c*1.5}];
      const pie=d3.pie().sort(null).value(d=>d.v).startAngle(Math.random()*Math.PI*2);
      const arcs=pie(vals); const arc=d3.arc().innerRadius(0).outerRadius(95);
      const gp=stim.append("g").attr("transform","translate(185,125)");
      gp.selectAll("path").data(arcs).join("path").attr("d",arc)
        .attr("fill",d=>d.data.k?C.accent:"#3a4150").attr("stroke",C.panel).attr("stroke-width",2)
        .attr("fill-opacity",d=>d.data.k==="B"?0.7:1);
      gp.selectAll("text").data(arcs.filter(a=>a.data.k)).join("text")
        .attr("transform",d=>`translate(${d3.arc().innerRadius(112).outerRadius(112).centroid(d)})`)
        .attr("text-anchor","middle").attr("dy","0.35em").attr("fill",C.muted).attr("font-size",12).attr("font-weight",600).text(d=>d.data.k);
    } else if(enc==="area"){
      const rA=80*s, rB=rA*Math.sqrt(r);
      stim.append("circle").attr("cx",flip?250:110).attr("cy",120).attr("r",rA).attr("fill",C.accent).attr("fill-opacity",.85);
      stim.append("circle").attr("cx",flip?95:270).attr("cy",120).attr("r",rB).attr("fill",C.accent).attr("fill-opacity",.85);
      lab(flip?250:110,234,"A"); lab(flip?95:270,234,"B");
    } else {
      const ramp=d3.interpolateLab("#1b1f27","#fcd34d");
      stim.append("rect").attr("x",xa-60).attr("y",50).attr("width",120).attr("height",140).attr("fill",ramp(s)).attr("stroke",C.line);
      stim.append("rect").attr("x",xb-60).attr("y",50).attr("width",120).attr("height",140).attr("fill",ramp(s*r)).attr("stroke",C.line);
      lab(xa,214,"A"); lab(xb,214,"B");
      lab(185,238,"darker = smaller value");
    }
  }
  function drawBoard(){
    const rows=board.selectAll("g.row").data(ENC).join(enter=>{
      const gg=enter.append("g").attr("class","row");
      gg.append("text").attr("class","nm").attr("x",0).attr("fill",C.muted).attr("font-size",11);
      gg.append("rect").attr("class","bg").attr("x",72).attr("height",16).attr("width",170).attr("fill",C.panel).attr("stroke",C.line);
      gg.append("rect").attr("class","fg").attr("x",72).attr("height",16);
      gg.append("text").attr("class","vv").attr("fill",C.ink).attr("font-size",10.5);
      return gg;});
    rows.attr("transform",(d,i)=>`translate(0,${12+i*36})`);
    rows.select("text.nm").attr("y",12).text(d=>d[1]);
    rows.select("rect.fg").attr("width",d=>scores[d[0]].length?bx(d3.mean(scores[d[0]])):0)
      .attr("fill",d=>{const m=d3.mean(scores[d[0]])||0; return m>15?C.bad:m>8?C.a2:C.good;});
    rows.select("text.vv").attr("x",248).attr("y",12)
      .text(d=>scores[d[0]].length?`${d3.mean(scores[d[0]]).toFixed(1)}  (n=${scores[d[0]].length})`:"—");
  }
  function guessLbl(){d3.select("#en-g-v").text(d3.select("#en-g").property("value")+"%");}
  d3.select("#en-g").on("input",guessLbl);
  d3.select("#en-type").on("change",()=>{newTrial();drawStim();});
  d3.select("#en-sub").on("click",()=>{
    const enc=d3.select("#en-type").property("value");
    const gss=+d3.select("#en-g").property("value"), truth=trial.r*100, err=Math.abs(gss-truth);
    scores[enc].push(err);
    d3.select("#en-read").html(`true ratio <b>${truth.toFixed(0)}%</b> · your guess <b>${gss}%</b> · error <b style="color:${err>15?C.bad:err>8?C.a2:C.good}">${err.toFixed(0)} pts</b> — a new pair has been drawn.`+
      `<br><span class="lbl">try five or more guesses per encoding; position and length usually land well below angle, area and lightness</span>`);
    newTrial(); drawStim(); drawBoard();
  });
  d3.select("#en-clr").on("click",()=>{ENC.forEach(e=>scores[e[0]]=[]);drawBoard();d3.select("#en-read").html("");});
  newTrial(); guessLbl(); drawStim(); drawBoard();
  d3.select("#en-read").html(`<span class="lbl">estimate B as a percentage of A, submit, repeat; switch encodings and compare your error</span>`);
})();

/* ---- Viz 2: Shewhart individuals chart with optional run rules ---- */
(function(){
  const C={accent:"#fcd34d",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#sp-svg"); if(svg.empty()) return;
  const W=660,H=250,m={t:14,r:60,b:30,l:44};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const N=120, T0=70, NB=40, MU=100, SIG=5;
  function Phi(z){const t=1/(1+0.3275911*Math.abs(z)/Math.SQRT2);
    const e=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-z*z/2);
    return z>=0?0.5*(1+e):0.5*(1-e);}

  let seed=7, noise=[];
  function gen(){const rn=d3.randomNormal.source(d3.randomLcg(seed/1000))(0,1); noise=d3.range(N).map(()=>rn());}

  const x=d3.scaleLinear().domain([0,N-1]).range([0,iw]);
  const y=d3.scaleLinear().range([ih,0]);
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`);
  const ya=g.append("g").attr("class","axis");
  const base=g.append("rect").attr("fill","rgba(154,163,178,.07)");
  const shiftZone=g.append("rect").attr("fill","rgba(248,113,113,.06)");
  const zones=g.append("g"), path=g.append("path").attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.1);
  const pts=g.append("g"), lbls=g.append("g");
  g.append("text").attr("x",x(NB/2)).attr("y",12).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("baseline (sets limits)");
  g.append("text").attr("x",x(T0)+4).attr("y",12).attr("fill",C.bad).attr("font-size",9.5).text("shift begins");

  function draw(){
    const k=+d3.select("#sp-k").property("value"), d=+d3.select("#sp-d").property("value");
    const rule=d3.select("#sp-rule").property("value"), season=d3.select("#sp-season").property("checked");
    d3.select("#sp-k-v").text(k.toFixed(1)+"σ"); d3.select("#sp-d-v").text(d.toFixed(1)+"σ");
    const v=noise.map((z,t)=>MU+SIG*z+(season?1.4*SIG*Math.sin(2*Math.PI*t/7):0)+(t>=T0?d*SIG:0));
    const b=v.slice(0,NB), mu=d3.mean(b);
    const mr=d3.mean(d3.range(1,NB).map(i=>Math.abs(b[i]-b[i-1])));
    const sg=mr/1.128;    // individuals chart: sigma from the average moving range
    const z=v.map(u=>(u-mu)/sg);
    const flag=v.map(()=>null);
    z.forEach((zi,i)=>{
      if(Math.abs(zi)>k){flag[i]="1";return;}
      if(rule!=="we") return;
      const side=Math.sign(zi);
      const win=(n)=>d3.range(Math.max(0,i-n+1),i+1).map(j=>z[j]);
      if(i>=2 && Math.abs(zi)>2 && win(3).filter(q=>Math.sign(q)===side&&Math.abs(q)>2).length>=2){flag[i]="2";return;}
      if(i>=4 && Math.abs(zi)>1 && win(5).filter(q=>Math.sign(q)===side&&Math.abs(q)>1).length>=4){flag[i]="3";return;}
      if(i>=7 && win(8).every(q=>Math.sign(q)===side&&q!==0)){flag[i]="4";}
    });
    const lo=Math.min(d3.min(v),mu-Math.max(k,3)*sg)-2, hi=Math.max(d3.max(v),mu+Math.max(k,3)*sg)+2;
    y.domain([lo,hi]);
    xa.call(d3.axisBottom(x).ticks(12)); ya.call(d3.axisLeft(y).ticks(6));
    g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",9);
    g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
    base.attr("x",0).attr("width",x(NB-1)).attr("y",0).attr("height",ih);
    shiftZone.attr("x",x(T0)).attr("width",iw-x(T0)).attr("y",0).attr("height",ih);
    const lines=[{v:mu,c:C.ink,l:"centre",dash:"none"},{v:mu+k*sg,c:C.bad,l:"UCL",dash:"5 3"},{v:mu-k*sg,c:C.bad,l:"LCL",dash:"5 3"}];
    if(rule==="we"){[1,2,-1,-2].forEach(s=>lines.push({v:mu+s*sg,c:C.muted,l:(s>0?"+":"−")+Math.abs(s)+"σ",dash:"1 4"}));}
    zones.selectAll("line").data(lines).join("line").attr("x1",0).attr("x2",iw)
      .attr("y1",l=>y(l.v)).attr("y2",l=>y(l.v)).attr("stroke",l=>l.c).attr("stroke-dasharray",l=>l.dash).attr("stroke-opacity",.85);
    lbls.selectAll("text").data(lines).join("text").attr("x",iw+4).attr("y",l=>y(l.v)+3).attr("fill",l=>l.c).attr("font-size",9).text(l=>l.l);
    path.attr("d",d3.line().x((u,i)=>x(i)).y(u=>y(u))(v));
    pts.selectAll("circle").data(v).join("circle").attr("cx",(u,i)=>x(i)).attr("cy",u=>y(u))
      .attr("r",(u,i)=>flag[i]?4.2:2.4).attr("fill",(u,i)=>flag[i]?C.bad:C.accent);

    const fa=flag.slice(0,T0).filter(Boolean).length;
    const first=flag.findIndex((f,i)=>i>=T0&&f);
    const alpha=2*(1-Phi(k)), arl0=1/alpha;
    const p1=Phi(-k+d)+Phi(-k-d), arl1=1/p1;
    d3.select("#sp-read").html(
      `per-point false-alarm rate α = <b>${(alpha*100).toFixed(2)}%</b> · ARL₀ ≈ <b>${arl0>=1e4?"&gt;10,000":arl0.toFixed(0)}</b> · `+
      `ARL₁ for a ${d.toFixed(1)}σ shift (limit only) ≈ <b>${d===0?"—":arl1.toFixed(1)}</b> points`+
      `<br>this run: false alarms before the shift <b style="color:${fa?C.bad:C.good}">${fa}</b> · `+
      (d===0?`no shift injected`:`first alarm after shift: <b style="color:${first<0?C.bad:C.good}">${first<0?"never (missed)":"t = "+first+" (delay "+(first-T0)+")"}</b>`)+
      ` · estimated σ̂ = <b>${sg.toFixed(2)}</b> (true 5.00)`+
      (season?`<br><span class="lbl">seasonality distorts the moving-range estimate and therefore the limits; chart the deseasonalised residual instead</span>`:"")
    );
  }
  ["#sp-k","#sp-d"].forEach(id=>d3.select(id).on("input",draw));
  ["#sp-rule","#sp-season"].forEach(id=>d3.select(id).on("change",draw));
  d3.select("#sp-re").on("click",()=>{seed=(seed*7919+13)%1000; gen(); draw();});
  gen(); draw();
})();
