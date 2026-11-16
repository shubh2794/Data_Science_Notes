/* react.viz.js — interactive figures for agentic/agents/react.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-tr  four prompting styles on one toy multi-hop question (illustrative, written for the page)
     #viz-rs  reported results explorer: HotpotQA, FEVER, ALFWorld, WebShop
     #viz-er  reported HotpotQA error analysis, ReAct vs CoT */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",panel2:"#1b1f27",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. four-lane trajectory comparator ═══════════════ */
  (function(){
    const svg=d3.select("#tr-svg"); if(svg.empty()) return;
    const Q="How long is the river that flows through the birthplace of the composer of the opera Glass Harbour?";
    const GOLD="212 km";
    // k: Thought | Act | Obs | Answer ; m: mark (halluc | lost | ground | right | wrong)
    const LANES=[
      {name:"Standard", sub:"no steps", steps:[
        {k:"Answer",t:"340 km",m:"wrong",n:"Standard prompting maps the question straight to an answer. Nothing is shown, so there is nothing to check or correct."}]},
      {name:"CoT", sub:"reason only", steps:[
        {k:"Thought",t:"Glass Harbour was composed by Ilse Varn.",n:"Recalled correctly from memory — this hop happens to be known."},
        {k:"Thought",t:"Ilse Varn was born in Tessel.",m:"halluc",n:"Hallucinated premise: Tessel is the valley; her birthplace is the town of Merrow. A reason-only trace has no observation that could contradict it."},
        {k:"Thought",t:"The Tessel river runs through Tessel and is about 90 km long.",m:"halluc",n:"Error propagation: valid reasoning from a false premise, plus a second invented fact."},
        {k:"Answer",t:"90 km",m:"wrong",n:"Confident, fluent and wrong."}]},
      {name:"Act", sub:"act only", steps:[
        {k:"Act",t:"search[Glass Harbour]"},
        {k:"Obs",t:"Could not find [Glass Harbour]. Similar: ['Glass Harbour (opera)', 'Glass Harbour (film)']."},
        {k:"Act",t:"search[Glass Harbour (opera)]",n:"A sensible retry — the similar-titles list makes it easy."},
        {k:"Obs",t:"Glass Harbour is a 1931 opera in three acts with music by Ilse Varn."},
        {k:"Act",t:"search[Ilse Varn]"},
        {k:"Obs",t:"Ilse Varn (1894–1962) was a composer born in Merrow, a market town in the Tessel valley."},
        {k:"Act",t:"search[Tessel]",m:"lost",n:"Loses track: the birthplace (Merrow) is now in the context, but with no thought to extract it the next query follows the more salient name, the valley."},
        {k:"Obs",t:"The Tessel is a valley whose towns include Merrow and Hale."},
        {k:"Act",t:"lookup[river]",m:"lost",n:"Fishing for the word 'river' on the wrong page — there is no stated plan to say what is being sought."},
        {k:"Obs",t:"No more results."},
        {k:"Act",t:"finish[Tessel]",m:"wrong",n:"Answers with a place, not a length: nothing in a bare action history records that the question asked for a length."}]},
      {name:"ReAct", sub:"reason + act", steps:[
        {k:"Thought",t:"I need the composer of Glass Harbour, their birthplace, the river there, then its length.",m:"ground",n:"Decomposes the goal into an explicit plan, written into the context."},
        {k:"Act",t:"search[Glass Harbour]"},
        {k:"Obs",t:"Could not find [Glass Harbour]. Similar: ['Glass Harbour (opera)', 'Glass Harbour (film)']."},
        {k:"Thought",t:"No exact page. The opera is the one I want.",m:"ground",n:"Handles an exception: reads the failed observation and reformulates the search."},
        {k:"Act",t:"search[Glass Harbour (opera)]"},
        {k:"Obs",t:"Glass Harbour is a 1931 opera in three acts with music by Ilse Varn."},
        {k:"Thought",t:"The composer is Ilse Varn. Next, her birthplace.",n:"Tracks progress: one hop done, names the next sub-goal."},
        {k:"Act",t:"search[Ilse Varn]"},
        {k:"Obs",t:"Ilse Varn (1894–1962) was a composer born in Merrow, a market town in the Tessel valley."},
        {k:"Thought",t:"She was born in Merrow; Tessel is only the valley. Find the river through Merrow.",m:"ground",n:"Extracts the right fact from the observation — exactly the step the Act lane skipped."},
        {k:"Act",t:"search[Merrow]"},
        {k:"Obs",t:"Merrow is a market town on the river Asker."},
        {k:"Thought",t:"Merrow is on the Asker. Now I need the Asker's length."},
        {k:"Act",t:"search[Asker]"},
        {k:"Obs",t:"The Asker is a river 212 km long."},
        {k:"Thought",t:"The Asker is 212 km long, so the answer is 212 km.",n:"Decides it has enough to stop."},
        {k:"Act",t:"finish[212 km]",m:"right",n:"Correct — every fact in the chain came from an observation."}]}
    ];
    const MAX=d3.max(LANES,l=>l.steps.length);
    const W=760, PAD=6, LWS=[118,168,212,232], FS=10.5, LH=13, TOP=78;   // long lanes get more width
    const LX=LWS.map((w,i)=>PAD+d3.sum(LWS.slice(0,i))+i*PAD), CHS=LWS.map(w=>Math.floor((w-26)/5.7));
    const KC={Thought:C.violet,Act:C.accent,Obs:C.muted,Answer:C.ink};
    const MC={halluc:C.bad,lost:C.a2,ground:C.good,right:C.good,wrong:C.bad};
    function wrap(s,n){const w=s.split(" "),out=[];let cur="";for(const x of w){if((cur+" "+x).trim().length>n&&cur){out.push(cur);cur=x;}else cur=(cur+" "+x).trim();}if(cur)out.push(cur);return out;}
    // pre-layout every entry so the figure height is fixed
    let Hmax=0;
    LANES.forEach((l,li)=>{let y=TOP;l.steps.forEach(s=>{s.lines=wrap(s.k+": "+s.t,CHS[li]);s.y=y;s.h=s.lines.length*LH+8;y+=s.h+4;});l.h=y;Hmax=Math.max(Hmax,y);});
    const H=Math.ceil(Hmax+12);
    svg.attr("height",H).attr("viewBox",`0 0 ${W} ${H}`);
    svg.append("text").attr("x",PAD).attr("y",16).attr("fill",C.ink).attr("font-size",11.5).text("Q: "+Q);
    svg.append("text").attr("x",PAD).attr("y",32).attr("fill",C.muted).attr("font-size",10.5).text("toy encyclopedia, written for this page · gold answer: "+GOLD);
    const lanes=LANES.map((l,i)=>{
      const LW=LWS[i], x=LX[i], g=svg.append("g").attr("transform",`translate(${x},0)`);
      g.append("rect").attr("y",44).attr("width",LW).attr("height",H-50).attr("rx",7).attr("fill",C.panel).attr("stroke",C.line);
      g.append("text").attr("x",10).attr("y",60).attr("fill",C.ink).attr("font-size",12.5).attr("font-weight",600).text(l.name);
      g.append("text").attr("x",LW-10).attr("y",60).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text(l.sub);
      const items=l.steps.map(s=>{
        const e=g.append("g").attr("transform",`translate(6,${s.y})`);
        const box=e.append("rect").attr("width",LW-12).attr("height",s.h).attr("rx",4).attr("fill",C.panel2).attr("stroke",C.line);
        const tx=e.append("text").attr("x",7).attr("y",13).attr("font-size",FS).attr("fill",C.ink);
        s.lines.forEach((ln,j)=>{
          const ts=tx.append("tspan").attr("x",7).attr("dy",j?LH:0);
          if(j===0){const c=ln.indexOf(":");ts.append("tspan").attr("fill",KC[s.k]).attr("font-weight",600).text(ln.slice(0,c+1));ts.append("tspan").text(ln.slice(c+1));}
          else ts.text(ln);
        });
        return {e,box};
      });
      return {items};
    });
    let step=0, timer=null;
    function lastIdx(l){return Math.min(step,l.steps.length)-1;}
    function render(){
      const marks=d3.select("#tr-marks").property("checked");
      d3.select("#tr-step").property("value",step); d3.select("#tr-stepv").text(step+" / "+MAX);
      LANES.forEach((l,i)=>l.steps.forEach((s,j)=>{
        const it=lanes[i].items[j], shown=j<step, newest=j===step-1;
        it.e.attr("opacity",shown?1:0).attr("visibility",shown?"visible":"hidden");
        const mc=marks&&s.m?MC[s.m]:null;
        it.box.attr("stroke",mc||(newest?C.ink:C.line)).attr("stroke-width",mc?2:(newest?1.4:1))
          .attr("fill",mc?d3.color(mc).copy({opacity:0.10}).formatRgb():C.panel2);
      }));
      if(step===0){d3.select("#tr-readout").html("step 0 — press <b>next step</b>. Each step reveals one more line in every lane; lanes are aligned by step count, not by meaning. The question and every fact are invented for this page.");return;}
      const rows=LANES.map(l=>{
        const j=lastIdx(l), s=l.steps[j], calls=l.steps.slice(0,j+1).filter(z=>z.k==="Act"&&!z.t.startsWith("finish")).length;
        const fin=l.steps[l.steps.length-1], done=j===l.steps.length-1;
        let status=done?(fin.m==="right"?"finished — <b style='color:"+C.good+"'>correct</b>":"finished — <b style='color:"+C.bad+"'>wrong</b>"):"running";
        const fresh=j===step-1;
        return `<b>${l.name}</b> (${calls} tool call${calls===1?"":"s"}, ${status})`+(fresh&&s.n?`: ${s.n}`:fresh?`: ${s.k.toLowerCase()} line`:"");
      });
      d3.select("#tr-readout").html(`step <b>${step}</b> of ${MAX}<br>`+rows.join("<br>"));
    }
    function stop(){if(timer){clearInterval(timer);timer=null;}d3.select("#tr-play").text("play");}
    d3.select("#tr-step").on("input",function(){stop();step=+this.value;render();});
    d3.select("#tr-next").on("click",()=>{stop();step=Math.min(MAX,step+1);render();});
    d3.select("#tr-prev").on("click",()=>{stop();step=Math.max(0,step-1);render();});
    d3.select("#tr-reset").on("click",()=>{stop();step=0;render();});
    d3.select("#tr-marks").on("change",render);
    d3.select("#tr-play").on("click",()=>{
      if(timer){stop();return;}
      if(step>=MAX)step=0;
      d3.select("#tr-play").text("pause");
      timer=setInterval(()=>{step++;render();if(step>=MAX)stop();},900);
      render();
    });
    render();
  })();

  /* ═══════════════ 2. reported results explorer ═══════════════ */
  (function(){
    const svg=d3.select("#rs-svg"); if(svg.empty()) return;
    // kind: prompt | react | hybrid | learned | ref
    const T1=[["Standard","prompt",28.7,57.1],["CoT","prompt",29.4,56.3],["CoT-SC (21 samples)","prompt",33.4,60.4],["Act","prompt",25.7,58.9],
              ["ReAct","react",27.4,60.9],["CoT-SC → ReAct","hybrid",34.2,64.6],["ReAct → CoT-SC","hybrid",35.1,62.0]];
    const SOTA={hotpot:67.5,fever:89.5};
    const TASKS=["All","Pick","Clean","Heat","Cool","Look","Pick 2"];
    const T3=[["Act (best of 6)","prompt",[45,88,42,74,67,72,41]],["ReAct (avg of 6)","react",[57,65,39,83,76,55,24]],["ReAct (best of 6)","react",[71,92,58,96,86,78,41]],
              ["ReAct-IM (avg of 6)","hybrid",[48,55,59,60,55,23,24]],["ReAct-IM (best of 6)","hybrid",[53,62,68,87,57,39,33]],
              ["BUTLERg (best of 8)","learned",[22,33,26,70,76,17,12]],["BUTLER (best of 8)","learned",[37,46,39,74,100,22,24]]];
    const T4=[["Act (one-shot)","prompt",62.3,30.1],["ReAct (one-shot)","react",66.6,40.0],["IL","learned",59.9,29.1],["IL+RL","learned",62.4,28.7]];
    const HUMAN={score:82.1,sr:59.6};
    const KC={prompt:C.muted,react:C.accent,hybrid:C.violet,learned:C.a2};
    const W=760,H=340,m={l:170,r:70,t:34,b:40};
    const SLICES={hotpot:[["all","all methods (Table 1)"]],fever:[["all","all methods (Table 1)"]],
                  alf:TASKS.map((t,i)=>[String(i),t==="All"?"all task types":t]),web:[["score","score"],["sr","success rate (%)"]]};
    function fillSlices(){
      const b=d3.select("#rs-b").property("value");
      const sel=d3.select("#rs-s"); sel.selectAll("option").remove();
      SLICES[b].forEach(([v,lab])=>sel.append("option").attr("value",v).text(lab));
      sel.property("value",SLICES[b][0][0]);
      sel.property("disabled",SLICES[b].length===1);
    }
    function draw(){
      svg.selectAll("*").remove();
      const b=d3.select("#rs-b").property("value"), s=d3.select("#rs-s").property("value"), ref=d3.select("#rs-ref").property("checked");
      let rows,title,refV=null,refLab="",unit="";
      if(b==="hotpot"||b==="fever"){
        const c=b==="hotpot"?2:3; rows=T1.map(r=>({lab:r[0],k:r[1],v:r[c]}));
        title=(b==="hotpot"?"HotpotQA exact match":"FEVER accuracy")+" — PaLM-540B prompting";
        refV=SOTA[b]; refLab="supervised SoTA "+refV;
      }else if(b==="alf"){
        const i=+s; rows=T3.map(r=>({lab:r[0],k:r[1],v:r[2][i]}));
        title="ALFWorld success rate (%) — "+(TASKS[i]==="All"?"all 134 unseen games":TASKS[i]+" tasks");
      }else{
        const c=s==="score"?2:3; rows=T4.map(r=>({lab:r[0],k:r[1],v:r[c]}));
        title="WebShop "+(s==="score"?"score":"success rate (%)")+" — 500 test instructions";
        refV=HUMAN[s]; refLab="human expert "+refV; unit=s==="sr"?"%":"";
      }
      const x=d3.scaleLinear().domain([0,100]).range([m.l,W-m.r]);
      const y=d3.scaleBand().domain(rows.map(r=>r.lab)).range([m.t,H-m.b]).padding(0.22);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(10)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y)).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11).text(title+" · axis starts at 0");
      const best=d3.max(rows,r=>r.v);
      rows.forEach(r=>{
        svg.append("rect").attr("x",m.l).attr("y",y(r.lab)).attr("height",y.bandwidth()).attr("width",Math.max(1,x(r.v)-m.l))
          .attr("fill",KC[r.k]).attr("opacity",r.v===best?0.95:0.7).append("title").text(`${r.lab}: ${r.v}${unit}`);
        svg.append("text").attr("x",x(r.v)+5).attr("y",y(r.lab)+y.bandwidth()/2+4).attr("fill",C.ink).attr("font-size",11).text(r.v.toFixed(b==="alf"?0:1));
      });
      if(ref&&refV!==null){
        svg.append("line").attr("x1",x(refV)).attr("x2",x(refV)).attr("y1",m.t-6).attr("y2",H-m.b).attr("stroke",C.bad).attr("stroke-dasharray","6 4");
        svg.append("text").attr("x",Math.min(x(refV)+4,W-m.r-60)).attr("y",m.t-10).attr("fill",C.bad).attr("font-size",10.5).text(refLab);
      }
      const lg=[["ReAct","react"],["ReAct hybrid / variant","hybrid"],["prompting baseline","prompt"],["learned baseline","learned"]];
      lg.forEach(([t,k],i)=>{svg.append("rect").attr("x",W-m.r-430+i*112).attr("y",H-12).attr("width",10).attr("height",10).attr("fill",KC[k]);
        svg.append("text").attr("x",W-m.r-416+i*112).attr("y",H-3).attr("fill",C.muted).attr("font-size",10).text(t);});
      d3.select("#rs-readout").html(summary(b,s,rows));
    }
    const f=v=>(Math.round(v*10)/10).toFixed(1);
    const get=(rows,lab)=>rows.find(r=>r.lab===lab).v;
    function summary(b,s,rows){
      const lead=rows.reduce((a,r)=>r.v>a.v?r:a,rows[0]);
      if(b==="hotpot"||b==="fever"){
        const re=get(rows,"ReAct"),cot=get(rows,"CoT"),act=get(rows,"Act"),sc=get(rows,"CoT-SC (21 samples)");
        return `leader: <b>${lead.lab}</b> at <b>${lead.v}</b> · ReAct − CoT = <b>${f(re-cot)}</b> · ReAct − Act = <b>${f(re-act)}</b> · best hybrid − CoT-SC = <b>${f(lead.v-sc)}</b> · gap to supervised SoTA = <b>${f(SOTA[b]-lead.v)}</b>`+
          `<br>differences are our arithmetic on the study's Table 1`+(b==="hotpot"?" · note that ReAct alone is <i>below</i> CoT here (§07 explains why)":"");
      }
      if(b==="alf"){
        const rb=get(rows,"ReAct (best of 6)"),ra=get(rows,"ReAct (avg of 6)"),ab=get(rows,"Act (best of 6)"),bu=get(rows,"BUTLER (best of 8)"),ib=get(rows,"ReAct-IM (best of 6)");
        return `leader: <b>${lead.lab}</b> at <b>${lead.v}%</b> · ReAct best − Act best = <b>${rb-ab}</b> · ReAct best − BUTLER = <b>${rb-bu}</b> · ReAct best − ReAct-IM best = <b>${rb-ib}</b> · ReAct best − ReAct avg = <b>${rb-ra}</b> (prompt sensitivity)`+
          `<br>differences are our arithmetic on the study's Table 3; Act has only a best-of-6 row`;
      }
      const re=get(rows,"ReAct (one-shot)"),ac=get(rows,"Act (one-shot)"),il=get(rows,"IL"),ilrl=get(rows,"IL+RL"), hv=s==="score"?HUMAN.score:HUMAN.sr;
      return `leader among automatic methods: <b>${lead.lab}</b> at <b>${lead.v}</b> · ReAct − Act = <b>${f(re-ac)}</b> · ReAct − IL = <b>${f(re-il)}</b> · ReAct − IL+RL = <b>${f(re-ilrl)}</b> · human expert − ReAct = <b>${f(hv-re)}</b>`+
        `<br>differences are our arithmetic on the study's Table 4`;
    }
    d3.select("#rs-b").on("change",()=>{fillSlices();draw();});
    d3.select("#rs-s").on("change",draw);
    d3.select("#rs-ref").on("change",draw);
    fillSlices(); draw();
  })();

  /* ═══════════════ 3. error analysis, ReAct vs CoT ═══════════════ */
  (function(){
    const svg=d3.select("#er-svg"); if(svg.empty()) return;
    const CAT={tp:{lab:"true positive",col:C.good,def:"correct reasoning trace and facts"},
               fp:{lab:"false positive",col:C.a2,def:"right answer, but hallucinated reasoning or facts"},
               re:{lab:"reasoning error",col:C.violet,def:"wrong reasoning trace, including failing to recover from repetitive steps"},
               se:{lab:"search result error",col:C.blue,def:"search returns empty or nothing useful (ReAct only — CoT does not search)"},
               ha:{lab:"hallucination",col:C.bad,def:"hallucinated reasoning or facts"},
               la:{lab:"label ambiguity",col:C.muted,def:"right prediction that does not match the gold label precisely"}};
    const BARS=[{g:"succ",lab:"ReAct · correct",parts:[["tp",94],["fp",6]]},{g:"succ",lab:"CoT · correct",parts:[["tp",86],["fp",14]]},
                {g:"fail",lab:"ReAct · wrong",parts:[["re",47],["se",23],["ha",0],["la",29]]},{g:"fail",lab:"CoT · wrong",parts:[["re",16],["ha",56],["la",28]]}];
    const W=760,H=300,m={l:120,r:40,t:40,b:56};
    function val(bar,c){const p=bar.parts.find(z=>z[0]===c);return p?p[1]:null;}
    let bars=BARS;
    function highlight(){
      const hc=d3.select("#er-c").property("value");
      svg.selectAll("[data-cat]").attr("opacity",function(){const on=hc==="all"||hc===this.getAttribute("data-cat");return on?+this.getAttribute("data-on"):+this.getAttribute("data-off");});
      readout(hc,bars);
    }
    function draw(){
      svg.selectAll("*").remove();
      const hc="all", gsel=d3.select("#er-g").property("value");
      bars=BARS.filter(b=>gsel==="both"||b.g===gsel);
      const x=d3.scaleLinear().domain([0,100]).range([m.l,W-m.r]);
      const y=d3.scaleBand().domain(bars.map(b=>b.lab)).range([m.t,H-m.b]).padding(0.28);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(10).tickFormat(d=>d+"%")).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y)).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11).text("share of hand-labelled HotpotQA trajectories (correct and wrong answers labelled separately)");
      bars.forEach(b=>{
        let acc=0;
        b.parts.forEach(([c,v])=>{
          if(v>0){
            svg.append("rect").attr("x",x(acc)).attr("y",y(b.lab)).attr("width",x(acc+v)-x(acc)).attr("height",y.bandwidth())
              .attr("fill",CAT[c].col).attr("data-cat",c).attr("data-on",0.9).attr("data-off",0.15).attr("stroke",C.panel).style("cursor","pointer")
              .on("mouseenter",()=>{if(d3.select("#er-c").property("value")!==c){d3.select("#er-c").property("value",c);highlight();}})
              .append("title").text(`${b.lab}: ${CAT[c].lab} ${v}%`);
            if(v>=6) svg.append("text").attr("x",x(acc+v/2)).attr("y",y(b.lab)+y.bandwidth()/2+4).attr("text-anchor","middle")
              .attr("fill","#0b0d12").attr("font-size",11).attr("data-cat",c).attr("data-on",1).attr("data-off",0.3).style("pointer-events","none").text(v+"%");
          }
          acc+=v;
        });
        if(acc<100) svg.append("text").attr("x",x(acc)+4).attr("y",y(b.lab)+y.bandwidth()/2+4).attr("fill",C.muted).attr("font-size",10).text("sums to "+acc+"%");
      });
      const used=Object.keys(CAT).filter(c=>bars.some(b=>b.parts.some(p=>p[0]===c)));
      used.forEach((c,i)=>{svg.append("rect").attr("x",m.l+i*104).attr("y",H-20).attr("width",10).attr("height",10).attr("fill",CAT[c].col).attr("data-cat",c).attr("data-on",0.9).attr("data-off",0.25);
        svg.append("text").attr("x",m.l+14+i*104).attr("y",H-11).attr("fill",C.muted).attr("font-size",10).text(CAT[c].lab);});
      highlight();
    }
    function readout(hc,bars){
      if(hc==="all"){
        d3.select("#er-readout").html("ReAct's failures are mostly <b>reasoning errors (47%)</b> — including repetitive thought–action loops — and <b>uninformative searches (23%)</b>, with <b>0% hallucination</b>. CoT's failures are dominated by <b>hallucination (56%)</b>, and 14% of its <i>correct</i> answers rest on hallucinated facts (ReAct: 6%). Hover a segment or pick a category.");
        return;
      }
      const cat=CAT[hc];
      const vals=BARS.filter(b=>b.parts.some(p=>p[0]===hc)).map(b=>`${b.lab}: <b>${val(b,hc)}%</b>`);
      const hidden=BARS.filter(b=>b.parts.some(p=>p[0]===hc)).some(b=>!bars.includes(b));
      d3.select("#er-readout").html(`<b>${cat.lab}</b> — ${cat.def}<br>`+vals.join(" · ")+(hc==="se"?" · CoT: no such category":"")+(hidden?" · (some bars are hidden by the show control)":""));
    }
    d3.select("#er-c").on("change",highlight);
    d3.select("#er-g").on("change",draw);
    draw();
  })();
})();
