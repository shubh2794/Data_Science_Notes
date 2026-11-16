/* dqn.viz.js — interactive figures for machine-learning/reinforcement-learning/dqn.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-rp  toy corridor: online vs recent-window vs uniform-replay semi-gradient Q-learning
     #viz-rs  the seven-game results table, human-normalised (own normalisation) or raw */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. replay toy ═══════════════ */
  (function(){
    const svg=d3.select("#rp-svg"); if(svg.empty()) return;
    // corridor: states 0..19, 19 terminal (+1 on entry), left wall at 0 reflects
    const NS=20, TERM=19, G=0.9, EPS_MIN=0.1, T=30000, ANNEAL=6000, B=32, PER_FRAME=100, REC=25;
    const CEN=[0,3,6,9,12,15,18], SIG=2.2, D=CEN.length+1;
    const NAMES=["online (newest sample)","recent window (last 32, consecutive)","uniform replay (32 from memory)"];
    const SHORT=["online","recent window","replay"];
    const COL=[C.bad,C.a2,C.accent];
    let seed=5, timer=null, L=null, step=0;

    function rng(sd){let s=sd>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    const PH=d3.range(NS).map(s=>{const f=CEN.map(c=>Math.exp(-((s-c)*(s-c))/(2*SIG*SIG)));f.push(1);return f;});
    function envStep(s,a){const n=a===1?s+1:Math.max(0,s-1);return {n:n,r:n===TERM?1:0,d:n===TERM};}
    // Q* by value iteration
    const QS=d3.range(NS).map(()=>[0,0]);
    for(let it=0;it<400;it++)for(let s=0;s<TERM;s++)for(let a=0;a<2;a++){const e=envStep(s,a);QS[s][a]=e.r+(e.d?0:G*Math.max(QS[e.n][0],QS[e.n][1]));}
    const VS=d3.range(TERM).map(s=>Math.max(QS[s][0],QS[s][1]));

    function q(w,s,a){const f=PH[s],wa=w[a];let v=0;for(let i=0;i<D;i++)v+=wa[i]*f[i];return v;}
    function rmse(w){let e=0;for(let s=0;s<TERM;s++)for(let a=0;a<2;a++){const d=q(w,s,a)-QS[s][a];e+=d*d;}return Math.sqrt(e/(2*TERM));}
    function update(w,batch,al){
      const g=[new Float64Array(D),new Float64Array(D)];
      for(const t of batch){
        const y=t.r+(t.d?0:G*Math.max(q(w,t.n,0),q(w,t.n,1)));   // target from CURRENT weights (as in the 2013 algorithm)
        const dl=y-q(w,t.s,t.a), f=PH[t.s];
        for(let i=0;i<D;i++)g[t.a][i]+=dl*f[i]/batch.length;
      }
      for(let a=0;a<2;a++)for(let i=0;i<D;i++)w[a][i]+=al*g[a][i];
    }
    const epsAt=t=>Math.max(EPS_MIN,1-(1-EPS_MIN)*t/ANNEAL);

    function init(){
      const cap=+d3.select("#rp-n").property("value");
      L=d3.range(3).map(k=>{const R=rng(seed);return {k:k,R:R,w:[new Float64Array(D),new Float64Array(D)],s:Math.floor(R()*TERM),
        buf:new Array(Math.max(cap,B)),cap:Math.max(cap,B),len:0,head:0,last:[],hist:[{t:0,e:0}],worst:0};});
      L.forEach(l=>{l.hist[0].e=rmse(l.w);});
      step=0;
    }
    function push(l,tr){l.buf[l.head]=tr;l.head=(l.head+1)%l.cap;if(l.len<l.cap)l.len++;}
    function recent(l,n){const out=[];for(let i=1;i<=Math.min(n,l.len);i++)out.push(l.buf[(l.head-i+l.cap)%l.cap]);return out;}
    function simStep(){   // one environment step for each learner; replay draws index the ring directly
      step++;
      const al=+d3.select("#rp-a").property("value"), eps=epsAt(step), cap=+d3.select("#rp-n").property("value");
      for(const l of L){
        let a; if(l.R()<eps)a=l.R()<0.5?1:0; else a=q(l.w,l.s,1)>=q(l.w,l.s,0)?1:0;
        const e=envStep(l.s,a), tr={s:l.s,a:a,r:e.r,n:e.n,d:e.d};
        push(l,tr);
        l.s=e.d?Math.floor(l.R()*TERM):e.n;
        let batch;
        if(l.k===0)batch=[tr];
        else if(l.k===1)batch=recent(l,B);
        else{const n=Math.min(l.len,cap);batch=new Array(B);for(let i=0;i<B;i++){const j=1+Math.floor(l.R()*n);batch[i]=l.buf[(l.head-j+l.cap)%l.cap];}}
        update(l.w,batch,al);
        l.last=batch.map(b=>b.s);
        if(step%REC===0){const er=rmse(l.w);l.hist.push({t:step,e:er});if(step>T/2)l.worst=Math.max(l.worst,er);}
      }
    }

    // ── layout ──
    const cx0=16, cw=18, rowY=[44,96,148], rh=20;
    svg.append("text").attr("x",cx0).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("states used by each learner's latest update (agent ▲)");
    const rowG=d3.range(3).map(k=>{
      const g=svg.append("g").attr("transform",`translate(${cx0},${rowY[k]})`);
      g.append("text").attr("x",0).attr("y",-5).attr("fill",COL[k]).attr("font-size",10.5).text(NAMES[k]);
      const cells=d3.range(NS).map(s=>g.append("rect").attr("x",s*cw).attr("y",0).attr("width",cw-2).attr("height",rh).attr("rx",2)
        .attr("fill",s===TERM?"#1d2a22":C.panel).attr("stroke",s===TERM?C.good:C.line));
      g.append("text").attr("x",TERM*cw+(cw-2)/2).attr("y",14).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",9.5).text("+1");
      const mark=g.append("path").attr("fill",C.ink);
      return {cells:cells,mark:mark};
    });

    // value panel
    const vx0=420,vx1=740,vy0=30,vy1=168;
    const vxs=d3.scaleLinear().domain([0,TERM-1]).range([vx0,vx1]);
    const vys=d3.scaleLinear().domain([0,1.15]).range([vy1,vy0]);
    svg.append("g").attr("transform",`translate(0,${vy1})`).call(d3.axisBottom(vxs).ticks(6)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${vx0},0)`).call(d3.axisLeft(vys).ticks(4)).attr("color",C.muted);
    svg.append("text").attr("x",vx0).attr("y",vy0-12).attr("fill",C.muted).attr("font-size",11).text("max_a Q(s, a) by state: true (dashed) vs learned");
    svg.append("path").attr("d",d3.line().x((d,i)=>vxs(i)).y(d=>vys(d))(VS)).attr("fill","none").attr("stroke",C.ink).attr("stroke-dasharray","5 4").attr("stroke-width",1.5);
    const vLines=d3.range(3).map(k=>svg.append("path").attr("fill","none").attr("stroke",COL[k]).attr("stroke-width",2));

    // error panel
    const ex0=60,ex1=740,ey0=232,ey1=410;
    const exs=d3.scaleLinear().domain([0,T]).range([ex0,ex1]);
    const eys=d3.scaleLog().domain([0.005,1]).range([ey1,ey0]).clamp(true);
    const epsY=d3.scaleLinear().domain([0,1]).range([ey1,ey0]);
    svg.append("path").attr("d",d3.area().x(d=>exs(d)).y0(ey1).y1(d=>epsY(epsAt(d)))(d3.range(0,T+1,250))).attr("fill",C.violet).attr("opacity",0.10);
    svg.append("text").attr("x",exs(ANNEAL)+6).attr("y",epsY(0.1)-4).attr("fill",C.violet).attr("font-size",10).attr("opacity",0.8).text("ε (shaded, linear scale): 1.0 → 0.1, then fixed");
    svg.append("g").attr("transform",`translate(0,${ey1})`).call(d3.axisBottom(exs).ticks(6).tickFormat(d3.format("~s"))).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${ex0},0)`).call(d3.axisLeft(eys).tickValues([0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]).tickFormat(d3.format("~g"))).attr("color",C.muted);
    svg.append("text").attr("x",ex0).attr("y",ey0-10).attr("fill",C.muted).attr("font-size",11).text("RMS error of Q against Q* (log) vs environment step");
    svg.append("line").attr("x1",exs(T/2)).attr("x2",exs(T/2)).attr("y1",ey0).attr("y2",ey1).attr("stroke",C.line).attr("stroke-dasharray","3 3");
    svg.append("text").attr("x",exs(T/2)+4).attr("y",ey0+10).attr("fill",C.muted).attr("font-size",10).text("second half: worst error tracked from here");
    const eLines=d3.range(3).map(k=>svg.append("path").attr("fill","none").attr("stroke",COL[k]).attr("stroke-width",1.6).attr("opacity",0.9));
    d3.range(3).forEach(k=>{svg.append("rect").attr("x",ex1-150).attr("y",ey0+22+k*15).attr("width",12).attr("height",3).attr("fill",COL[k]);
      svg.append("text").attr("x",ex1-134).attr("y",ey0+26+k*15).attr("fill",COL[k]).attr("font-size",10.5).text(SHORT[k]);});
    svg.append("text").attr("x",ex0).attr("y",440).attr("fill",C.muted).attr("font-size",10).text("toy: 20-state corridor, linear Q over 7 Gaussian bumps + bias, γ = 0.9, one update per step per learner, targets from current weights");

    function render(){
      const show=d3.select("#rp-show").property("value");
      L.forEach((l,k)=>{
        const cnt=new Array(NS).fill(0); l.last.forEach(s=>cnt[s]++);
        const mx=Math.max(1,d3.max(cnt));
        rowG[k].cells.forEach((c,s)=>{if(s===TERM)return;c.attr("fill",cnt[s]?COL[k]:C.panel).attr("fill-opacity",cnt[s]?0.25+0.75*cnt[s]/mx:1);});
        const ax=l.s*cw+(cw-2)/2; rowG[k].mark.attr("d",`M${ax},${rh+2} l-5,8 l10,0 z`);
        const vis=show==="all"||+show===k;
        vLines[k].attr("opacity",vis?1:0).attr("d",d3.line().x((d,i)=>vxs(i)).y(d=>vys(Math.max(-0.05,Math.min(1.15,d))))(d3.range(TERM).map(s=>Math.max(q(l.w,s,0),q(l.w,s,1)))));
        eLines[k].attr("d",d3.line().x(d=>exs(d.t)).y(d=>eys(Math.max(0.005,d.e)))(l.hist));
      });
      const cur=L.map(l=>l.hist[l.hist.length-1].e);
      const half=step>T/2;
      d3.select("#rp-readout").html(
        `step <b>${step.toLocaleString("en-US")}</b> / ${T.toLocaleString("en-US")} · ε = <b>${epsAt(Math.max(step,0)).toFixed(2)}</b> · α = ${(+d3.select("#rp-a").property("value")).toFixed(1)} · memory N = ${(+d3.select("#rp-n").property("value")).toLocaleString("en-US")}<br>`+
        L.map((l,k)=>`<span style="color:${COL[k]}">${SHORT[k]}</span>: now <b>${cur[k].toFixed(3)}</b>`+(half?`, worst in 2nd half <b>${l.worst.toFixed(3)}</b>`:"")).join(" · ")+
        (half?"":" · worst-case errors are reported once the run passes the halfway mark")+
        (step===0?" · press run: all three start from zero weights and the same seed":""));
    }
    function stop(){if(timer){timer.stop();timer=null;}d3.select("#rp-play").text(step>=T?"done":"run");}
    function play(){
      if(timer){stop();return;}
      if(step>=T){init();}
      d3.select("#rp-play").text("pause");
      timer=d3.timer(()=>{for(let i=0;i<PER_FRAME&&step<T;i++)simStep();render();if(step>=T)stop();});
    }
    d3.select("#rp-a").on("input",()=>{d3.select("#rp-av").text((+d3.select("#rp-a").property("value")).toFixed(1));stop();init();render();});
    d3.select("#rp-n").on("change",()=>{stop();init();render();});
    d3.select("#rp-show").on("change",render);
    d3.select("#rp-play").on("click",play);
    d3.select("#rp-reset").on("click",()=>{stop();init();render();});
    d3.select("#rp-seed").on("click",()=>{stop();seed+=1;init();render();});
    init();render();
  })();

  /* ═══════════════ 2. results chart ═══════════════ */
  (function(){
    const svg=d3.select("#rs-svg"); if(svg.empty()) return;
    const GAMES=["Beam Rider","Breakout","Enduro","Pong","Q*bert","Seaquest","Space Invaders"];
    const ROWS={
      "Random":[354,1.2,0,-20.4,157,110,179],
      "Sarsa":[996,5.2,129,-19,614,665,271],
      "Contingency":[1743,6,159,-17,960,723,268],
      "DQN":[4092,168,470,20,1952,1705,581],
      "Human":[7456,31,368,-3,18900,28010,3690],
      "HNeat Best":[3616,52,106,19,1800,920,1720],
      "HNeat Pixel":[1332,4,91,-16,1325,800,1145],
      "DQN Best":[5184,225,661,21,4500,1740,1075]};
    const COL={"Random":"#5b6170","Sarsa":C.blue,"Contingency":C.violet,"DQN":C.accent,"Human":C.ink,"HNeat Best":C.a2,"HNeat Pixel":"#c9925a","DQN Best":C.accent};
    const BEST=new Set(["HNeat Best","HNeat Pixel","DQN Best"]);
    const norm=(m,g)=>(ROWS[m][g]-ROWS.Random[g])/(ROWS.Human[g]-ROWS.Random[g]);
    const pct=v=>(100*v).toFixed(v<0.1?1:0)+"%";
    const W=760,H=380,m={l:62,r:16,t:30,b:58};
    // hatch pattern for best-episode rows
    const defs=svg.append("defs");
    function hatch(id,col){defs.append("pattern").attr("id",id).attr("patternUnits","userSpaceOnUse").attr("width",5).attr("height",5).attr("patternTransform","rotate(45)")
      .append("rect").attr("width",2).attr("height",5).attr("fill",col);}
    hatch("rs-h-hb",COL["HNeat Best"]);hatch("rs-h-hp",COL["HNeat Pixel"]);hatch("rs-h-db",COL["DQN Best"]);
    const fillOf=mm=>mm==="HNeat Best"?"url(#rs-h-hb)":mm==="HNeat Pixel"?"url(#rs-h-hp)":mm==="DQN Best"?"url(#rs-h-db)":COL[mm];
    const plot=svg.append("g");

    function say(mm,g){
      const raw=ROWS[mm][g], n=norm(mm,g), kind=BEST.has(mm)?"single best episode":"average over episodes (ε = 0.05)";
      const vsH=raw>ROWS.Human[g]?"above":"below";
      d3.select("#rs-readout").html(`<b>${mm}</b> on <b>${GAMES[g]}</b>: raw score <b>${raw}</b> (${kind}) · human-normalised <b>${pct(n)}</b> · ${vsH} the human expert's ${ROWS.Human[g]}`+
        (mm==="DQN"?` · DQN / best prior learning method = <b>${(Math.max(ROWS.Sarsa[g],ROWS.Contingency[g])>0?(raw/Math.max(ROWS.Sarsa[g],ROWS.Contingency[g])).toFixed(1)+"×":"n/a")}</b>`:"")+
        (BEST.has(mm)?" · not comparable with the average rows":""));
    }
    function summary(order){
      const above=d3.range(7).filter(g=>ROWS.DQN[g]>ROWS.Human[g]).map(g=>GAMES[g]);
      d3.select("#rs-readout").html(`DQN average above the human expert on <b>${above.length}</b> of 7 (${above.join(", ")}) · median human-normalised DQN score <b>${pct(d3.median(d3.range(7),g=>norm("DQN",g)))}</b> · normalisation (score − random)/(human − random) is this page's own, not the study's · hover a bar for its numbers`);
    }

    function draw(){
      plot.selectAll("*").remove();
      const view=d3.select("#rs-v").property("value"), sc=d3.select("#rs-s").property("value"), best=d3.select("#rs-best").property("checked"), ord=d3.select("#rs-o").property("value");
      let order=d3.range(7); if(ord==="dqn")order=order.slice().sort((a,b)=>norm("DQN",b)-norm("DQN",a));
      if(view==="norm"){
        const methods=["Sarsa","Contingency","DQN"].concat(best?["HNeat Best","HNeat Pixel","DQN Best"]:[]);
        const vmax=d3.max(methods,mm=>d3.max(order,g=>norm(mm,g)));
        const y=(sc==="sqrt"?d3.scaleSqrt():d3.scaleLinear()).domain([0,vmax*1.05]).range([H-m.b,m.t]);
        const x0=d3.scaleBand().domain(order).range([m.l,W-m.r]).paddingInner(0.18);
        const x1=d3.scaleBand().domain(methods).range([0,x0.bandwidth()]).padding(0.08);
        const ticks=sc==="sqrt"?[0,0.05,0.25,0.5,1,2,3,4,5,6,8].filter(t=>t<=vmax*1.05):y.ticks(6);
        plot.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(t=>Math.round(t*100)+"%")).attr("color",C.muted);
        plot.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x0).tickFormat(g=>GAMES[g])).attr("color",C.muted);
        plot.append("text").attr("x",m.l).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("human-normalised score: random = 0%, human expert = 100% (own normalisation, "+(sc==="sqrt"?"square-root":"linear")+" axis)");
        plot.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(1)).attr("y2",y(1)).attr("stroke",C.ink).attr("stroke-dasharray","6 4").attr("opacity",0.7);
        plot.append("text").attr("x",W-m.r-4).attr("y",y(1)-4).attr("text-anchor","end").attr("fill",C.ink).attr("font-size",10.5).text("human expert = 100%");
        order.forEach(g=>{
          methods.forEach(mm=>{
            const v=Math.max(0,norm(mm,g));
            plot.append("rect").attr("x",x0(g)+x1(mm)).attr("width",x1.bandwidth()).attr("y",y(v)).attr("height",Math.max(0.5,H-m.b-y(v)))
              .attr("fill",fillOf(mm)).attr("stroke",BEST.has(mm)?COL[mm]:"none").attr("stroke-width",1).attr("opacity",mm==="DQN"?1:0.85).style("cursor","pointer")
              .on("mouseenter",()=>say(mm,g)).append("title").text(`${mm}, ${GAMES[g]}: ${ROWS[mm][g]} (${pct(norm(mm,g))})`);
          });
          const v=norm("DQN",g);
          plot.append("text").attr("x",x0(g)+x1("DQN")+x1.bandwidth()/2).attr("y",y(v)-4).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",9.5).text(pct(v));
        });
        legend(methods);
      }else{
        const methods=["Random","Sarsa","Contingency","DQN","Human"].concat(best?["HNeat Best","HNeat Pixel","DQN Best"]:[]);
        const pw=(W-m.l+30-m.r)/7, top=m.t+6, bot=H-m.b;
        plot.append("text").attr("x",m.l-30).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("raw game scores, each panel on its own linear axis (scale option applies to the normalised view)");
        order.forEach((g,pi)=>{
          const px=m.l-30+pi*pw, vals=methods.map(mm=>ROWS[mm][g]);
          const lo=Math.min(0,d3.min(vals)), hi=Math.max(0,d3.max(vals));
          const y=d3.scaleLinear().domain([lo,hi]).range([bot,top]).nice();
          const x=d3.scaleBand().domain(methods).range([px+30,px+pw-6]).padding(0.12);
          plot.append("g").attr("transform",`translate(${px+30},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format("~s"))).attr("color",C.muted).attr("font-size",9);
          plot.append("line").attr("x1",px+30).attr("x2",px+pw-6).attr("y1",y(0)).attr("y2",y(0)).attr("stroke",C.muted);
          plot.append("text").attr("x",px+30+(pw-36)/2).attr("y",bot+16).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10.5).text(GAMES[g]);
          methods.forEach(mm=>{
            const v=ROWS[mm][g];
            plot.append("rect").attr("x",x(mm)).attr("width",x.bandwidth()).attr("y",Math.min(y(v),y(0))).attr("height",Math.max(0.5,Math.abs(y(v)-y(0))))
              .attr("fill",fillOf(mm)).attr("stroke",BEST.has(mm)?COL[mm]:"none").style("cursor","pointer")
              .on("mouseenter",()=>say(mm,g)).append("title").text(`${mm}, ${GAMES[g]}: ${v}`);
          });
        });
        legend(methods);
      }
      summary(order);
    }
    function legend(methods){
      const lg=plot.append("g").attr("transform",`translate(${m.l},${H-22})`);
      let xo=0;
      methods.forEach(mm=>{
        lg.append("rect").attr("x",xo).attr("y",-9).attr("width",11).attr("height",11).attr("fill",fillOf(mm)).attr("stroke",BEST.has(mm)?COL[mm]:"none");
        const t=lg.append("text").attr("x",xo+15).attr("y",0).attr("fill",C.muted).attr("font-size",10.5).text(mm+(BEST.has(mm)?" (best ep.)":""));
        xo+=15+(mm.length+(BEST.has(mm)?10:0))*6.1+12;
      });
    }
    ["#rs-v","#rs-s","#rs-best","#rs-o"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();
})();
