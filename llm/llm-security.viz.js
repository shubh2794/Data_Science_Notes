/* llm-security.viz.js — figures for llm-security.html.
   Loaded after data.js / notes.js. One IIFE per figure; each bails out if its
   elements are missing. All simulations are illustrative, not measurements. */

const SEC_C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#11141b",blue:"#5b9cff"};

/* ---------- (a) injection flow through an agent ---------- */
(function(){
  const svgEl=document.getElementById("inj-svg"), read=document.getElementById("inj-read");
  if(!svgEl||!read) return;
  const C=SEC_C, svg=d3.select(svgEl);
  const val=id=>{const e=document.getElementById(id);return e?e.value:"";};
  const on=id=>{const e=document.getElementById(id);return !!(e&&e.checked);};
  const ENTRY={direct:"User prompt",rag:"RAG chunk",web:"Web page",email:"Email",tool:"Tool output"};
  const N={
    src:{x:62,y:135,label:()=>ENTRY[val("inj-entry")]||"Source",sub:"payload enters"},
    cls:{x:178,y:135,label:()=>"Input filter",sub:"classifier",def:"inj-cls"},
    llm:{x:296,y:135,label:()=>"LLM",sub:"delimiting",def:"inj-delim"},
    lp:{x:418,y:62,label:()=>"Tool gateway",sub:"least privilege",def:"inj-lp"},
    hitl:{x:532,y:62,label:()=>"Confirm",sub:"human in loop",def:"inj-hitl"},
    eff:{x:630,y:62,label:()=>"Side effect",sub:"data sent"},
    out:{x:418,y:208,label:()=>"Output filter",sub:"URL allow-list",def:"inj-out"},
    ren:{x:532,y:208,label:()=>"Client",sub:"renders reply"},
    usr:{x:630,y:208,label:()=>"Outcome",sub:""}
  };
  const EDGES=[["src","cls"],["cls","llm"],["llm","lp"],["lp","hitl"],["hitl","eff"],["llm","out"],["out","ren"],["ren","usr"]];

  function simulate(){
    const entry=val("inj-entry"), goal=val("inj-goal"), naive=val("inj-str")==="naive";
    const direct=entry==="direct";
    const path=["src","cls"];
    const stop=(node,why)=>({path,stopAt:node,why});
    if(on("inj-cls")&&naive) return stop("cls","the classifier recognises a known-looking payload");
    path.push("llm");
    if(on("inj-delim")&&naive&&!direct) return stop("llm","the payload sits inside marked data and reads as content, not a command");
    if(goal==="tool"){
      path.push("lp");
      if(on("inj-lp")) return stop("lp","no tool in scope can both read the private data and send it out");
      path.push("hitl");
      if(on("inj-hitl")&&!direct) return stop("hitl","the user sees an outbound action they never asked for and declines");
      path.push("eff");
      return {path,stopAt:null,why: on("inj-hitl")&&direct
        ? "breach — confirmation is useless when the user is the attacker"
        : "breach — data leaves through a tool call"};
    }
    path.push("out");
    if(goal==="md"){
      if(on("inj-out")) return stop("out","the image URL points to a non-allow-listed host and is stripped");
      path.push("ren","usr");
      return {path,stopAt:null,why:"breach — the client fetches the image and the URL carries the data"+(on("inj-lp")||on("inj-hitl")?" (no tool call, so least privilege and confirmation never fire)":"")};
    }
    if(on("inj-out")&&naive) return stop("out","the policy classifier flags the generated text");
    path.push("ren","usr");
    return {path,stopAt:null,why:"breach — policy-violating text reaches the user"+(on("inj-out")?" (adaptive phrasing slips past the learned output filter)":"")};
  }

  function draw(){
    svg.selectAll("*").remove();
    const r=simulate(), inPath=new Set(r.path);
    const edgeOn=(a,b)=>{const i=r.path.indexOf(a);return i>=0&&r.path[i+1]===b;};
    svg.append("defs").append("marker").attr("id","inj-arrow").attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5)
      .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto")
      .append("path").attr("d","M0,0L10,5L0,10z").attr("fill",C.muted);
    EDGES.forEach(([a,b])=>{
      const A=N[a],B=N[b],hot=edgeOn(a,b);
      const dx=B.x-A.x, dy=B.y-A.y, L=Math.hypot(dx,dy), ux=dx/L, uy=dy/L, pad=44;
      svg.append("line").attr("x1",A.x+ux*pad).attr("y1",A.y+uy*pad*0.6).attr("x2",B.x-ux*pad).attr("y2",B.y-uy*pad*0.6)
        .attr("stroke",hot?(r.stopAt?C.a2:C.bad):C.line).attr("stroke-width",hot?2.6:1.4)
        .attr("marker-end","url(#inj-arrow)");
    });
    // private data store feeding the LLM
    svg.append("rect").attr("x",250).attr("y",10).attr("width",92).attr("height",26).attr("rx",5).attr("fill",C.panel).attr("stroke",C.blue).attr("stroke-dasharray","3 2");
    svg.append("text").attr("x",296).attr("y",27).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.blue).text("private data");
    svg.append("line").attr("x1",296).attr("y1",36).attr("x2",296).attr("y2",112).attr("stroke",C.blue).attr("stroke-dasharray","3 2").attr("stroke-width",1);
    Object.entries(N).forEach(([k,n])=>{
      const isDef=!!n.def, enabled=isDef&&on(n.def), stopped=r.stopAt===k;
      const breachEnd=!r.stopAt&&k===r.path[r.path.length-1];
      const fill=stopped?"rgba(74,222,128,.18)":breachEnd?"rgba(248,113,113,.2)":C.panel;
      const stroke=stopped?C.good:breachEnd?C.bad:enabled?C.accent:(inPath.has(k)?C.ink:C.line);
      const g=svg.append("g").attr("opacity",inPath.has(k)||enabled?1:0.45);
      g.append("rect").attr("x",n.x-46).attr("y",n.y-22).attr("width",92).attr("height",44).attr("rx",8)
        .attr("fill",fill).attr("stroke",stroke).attr("stroke-width",stopped||breachEnd?2.4:1.3)
        .attr("stroke-dasharray",isDef&&!enabled?"4 3":null);
      g.append("text").attr("x",n.x).attr("y",n.y-3).attr("text-anchor","middle").attr("font-size",11).attr("font-weight",700).attr("fill",C.ink).text(n.label());
      const sub=isDef?(n.sub+(enabled?" · on":" · off")):(k==="usr"?(r.stopAt?"":"user harmed"):n.sub);
      g.append("text").attr("x",n.x).attr("y",n.y+13).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",enabled?C.accent:C.muted).text(sub);
      if(stopped) g.append("text").attr("x",n.x).attr("y",n.y+38).attr("text-anchor","middle").attr("font-size",10).attr("font-weight",700).attr("fill",C.good).text("✓ stopped here");
      if(breachEnd) g.append("text").attr("x",n.x).attr("y",n.y+38).attr("text-anchor","middle").attr("font-size",10).attr("font-weight",700).attr("fill",C.bad).text("✗ breach");
    });
    svg.append("text").attr("x",8).attr("y",262).attr("font-size",9.5).attr("fill",C.muted).text("Illustrative rule model: learned filters stop naive payloads only; capability limits, confirmation and allow-lists are deterministic.");
    const hop=r.stopAt?N[r.stopAt].label():null;
    read.innerHTML=r.stopAt
      ? `<span style="color:${C.good}">Stopped at <b>${hop}</b></span> after ${r.path.length-1} hop(s): ${r.why}.`
      : `<span style="color:${C.bad}"><b>${r.why.split(" — ")[0]}</b></span> — ${r.why.split(" — ").slice(1).join(" — ")}.`;
  }
  ["inj-entry","inj-goal","inj-str","inj-delim","inj-cls","inj-lp","inj-hitl","inj-out"].forEach(id=>{
    const e=document.getElementById(id); if(e){e.addEventListener("input",draw);e.addEventListener("change",draw);}
  });
  draw();
})();

/* ---------- (b) guardrail threshold trade-off ---------- */
(function(){
  const svgEl=document.getElementById("thr-svg"), read=document.getElementById("thr-read");
  if(!svgEl||!read) return;
  const C=SEC_C, svg=d3.select(svgEl), W=640, H=240, m={l:36,r:16,t:20,b:34};
  const num=(id,d)=>{const e=document.getElementById(id);const v=e?+e.value:d;return isFinite(v)?v:d;};
  // erf approximation (Abramowitz–Stegun 7.1.26)
  const erf=x=>{const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);
    return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x));};
  const Phi=(x,mu,sd)=>0.5*(1+erf((x-mu)/(sd*Math.SQRT2)));
  const pdf=(x,mu,sd)=>Math.exp(-0.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
  function draw(){
    svg.selectAll("*").remove();
    const t=num("thr-t",0.55), sep=num("thr-sep",0.35), pi=num("thr-pi",0.01), sd=0.12;
    const muB=0.5-sep/2, muM=0.5+sep/2;
    const x=d3.scaleLinear().domain([0,1]).range([m.l,W-m.r]);
    const peak=pdf(0,0,sd);
    const y=d3.scaleLinear().domain([0,peak*1.08]).range([H-m.b,m.t]);
    const xs=d3.range(0,1.0001,0.005);
    const area=(mu,lo,hi)=>d3.area().x(d=>x(d)).y0(H-m.b).y1(d=>y(pdf(d,mu,sd)))(xs.filter(d=>d>=lo&&d<=hi));
    const line=mu=>d3.line().x(d=>x(d)).y(d=>y(pdf(d,mu,sd)))(xs);
    // shaded regions: blocked benign (over-refusal) and missed malicious
    svg.append("path").attr("d",area(muB,t,1)).attr("fill",C.a2).attr("fill-opacity",.35);
    svg.append("path").attr("d",area(muM,0,t)).attr("fill",C.bad).attr("fill-opacity",.35);
    svg.append("path").attr("d",line(muB)).attr("fill","none").attr("stroke",C.blue).attr("stroke-width",2);
    svg.append("path").attr("d",line(muM)).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2);
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(10))
      .call(g=>g.selectAll("text").attr("fill",C.muted)).call(g=>g.selectAll("line,path").attr("stroke",C.line));
    svg.append("text").attr("x",W/2).attr("y",H-4).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("filter score s(x) — higher = more likely malicious");
    svg.append("line").attr("x1",x(t)).attr("x2",x(t)).attr("y1",m.t-6).attr("y2",H-m.b).attr("stroke",C.accent).attr("stroke-width",2);
    svg.append("text").attr("x",x(t)+4).attr("y",m.t+2).attr("font-size",10).attr("fill",C.accent).text("block ≥ t");
    svg.append("text").attr("x",x(muB)).attr("y",y(peak)-4).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.blue).text("benign");
    svg.append("text").attr("x",x(muM)).attr("y",y(peak)-4).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.bad).text("malicious");
    // legend
    [[C.a2,"benign blocked (over-refusal)"],[C.bad,"malicious missed"]].forEach(([c,l],i)=>{
      svg.append("rect").attr("x",W-m.r-170).attr("y",m.t+4+i*15).attr("width",10).attr("height",10).attr("fill",c).attr("fill-opacity",.6);
      svg.append("text").attr("x",W-m.r-155).attr("y",m.t+13+i*15).attr("font-size",9.5).attr("fill",C.muted).text(l);
    });
    const tpr=1-Phi(t,muM,sd), fpr=1-Phi(t,muB,sd);
    const denom=pi*tpr+(1-pi)*fpr, prec=denom>0?pi*tpr/denom:0;
    const perAttack=pi*tpr>1e-12?((1-pi)*fpr)/(pi*tpr):Infinity;
    const pct=v=>(v*100).toFixed(v<0.01?2:1)+"%";
    read.innerHTML=`catch rate (TPR) <b>${pct(tpr)}</b> · over-refusal (FPR) <b>${pct(fpr)}</b> · precision at π=${pi} <b>${pct(prec)}</b> · `+
      (isFinite(perAttack)?`≈ <b>${perAttack<10?perAttack.toFixed(1):Math.round(perAttack).toLocaleString()}</b> benign requests blocked per attack caught`:`no attacks caught at this threshold`)+
      ` <span style="color:${C.muted}">(illustrative distributions)</span>`;
  }
  ["thr-t","thr-sep","thr-pi"].forEach(id=>{const e=document.getElementById(id);if(e){e.addEventListener("input",draw);e.addEventListener("change",draw);}});
  draw();
})();

/* ---------- (c) defence-in-depth residual risk ---------- */
(function(){
  const svgEl=document.getElementById("did-svg"), read=document.getElementById("did-read");
  if(!svgEl||!read) return;
  const C=SEC_C, svg=d3.select(svgEl), W=640, H=250, m={l:52,r:16,t:22,b:40};
  const num=(id,d)=>{const e=document.getElementById(id);const v=e?+e.value:d;return isFinite(v)?v:d;};
  const names=["prompt only","+ input filter","+ output filter","+ gate"];
  function draw(){
    svg.selectAll("*").remove();
    const c=[1,2,3,4].map(i=>Math.min(0.99,Math.max(0,num("did-c"+i,0))));
    const rho=Math.min(1,Math.max(0,num("did-rho",0)));
    const p=c.map(v=>1-v);
    const rows=[];
    for(let k=1;k<=4;k++){
      const ps=p.slice(0,k), prod=ps.reduce((a,b)=>a*b,1), mn=Math.min(...ps);
      rows.push({k,name:names[k-1],ind:prod,cor:(1-rho)*prod+rho*mn});
    }
    const floor=1e-5;
    const y=d3.scaleLog().domain([floor,1]).range([H-m.b,m.t]).clamp(true);
    const x0=d3.scaleBand().domain(rows.map(r=>r.k)).range([m.l,W-m.r]).padding(0.28);
    const x1=d3.scaleBand().domain(["ind","cor"]).range([0,x0.bandwidth()]).padding(0.08);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5,"~e").tickValues([1e-5,1e-4,1e-3,1e-2,1e-1,1]).tickFormat(d=>d>=0.01?(d*100)+"%":"1e"+Math.round(Math.log10(d))))
      .call(g=>g.selectAll("text").attr("fill",C.muted)).call(g=>g.selectAll("line,path").attr("stroke",C.line));
    svg.append("text").attr("transform",`translate(12,${(H-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("P(breach), log");
    rows.forEach(r=>{
      [["ind",r.ind,C.accent],["cor",r.cor,C.a2]].forEach(([key,v,col])=>{
        const vv=Math.max(v,floor), bx=x0(r.k)+x1(key);
        svg.append("rect").attr("x",bx).attr("y",y(vv)).attr("width",x1.bandwidth()).attr("height",Math.max(0,(H-m.b)-y(vv))).attr("fill",col).attr("fill-opacity",.8).attr("rx",2);
        svg.append("text").attr("x",bx+x1.bandwidth()/2).attr("y",y(vv)-4).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.ink)
          .text(v<1e-4?v.toExponential(0):v<0.01?(v*100).toFixed(2)+"%":(v*100).toFixed(1)+"%");
      });
      svg.append("text").attr("x",x0(r.k)+x0.bandwidth()/2).attr("y",H-m.b+16).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(r.name);
    });
    [[C.accent,"independent  ∏ pᵢ"],[C.a2,`with correlation ρ = ${rho.toFixed(2)}`]].forEach(([col,l],i)=>{
      svg.append("rect").attr("x",W-m.r-190).attr("y",m.t-14+i*15).attr("width",10).attr("height",10).attr("fill",col);
      svg.append("text").attr("x",W-m.r-175).attr("y",m.t-5+i*15).attr("font-size",10).attr("fill",C.muted).text(l);
    });
    const last=rows[3], oneIn=v=>v>0?"1 in "+Math.round(1/v).toLocaleString():"never";
    const ratio=last.ind>0?last.cor/last.ind:Infinity;
    read.innerHTML=`All four layers: independent <b>${(last.ind*100).toPrecision(2)}%</b> (${oneIn(last.ind)}) · correlated <b>${(last.cor*100).toPrecision(2)}%</b> (${oneIn(last.cor)})`+
      (isFinite(ratio)&&ratio>1.05?` — correlation makes the stack <b>${ratio<100?ratio.toFixed(1):Math.round(ratio).toLocaleString()}×</b> leakier`:"")+
      ` <span style="color:${C.muted}">(illustrative; random attacks — an adaptive attacker does worse for you)</span>`;
  }
  ["did-c1","did-c2","did-c3","did-c4","did-rho"].forEach(id=>{const e=document.getElementById(id);if(e)e.addEventListener("input",draw);});
  draw();
})();
