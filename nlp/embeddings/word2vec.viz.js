/* word2vec.viz.js — interactive figures for nlp/embeddings/word2vec.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-hs  Huffman-tree hierarchical softmax: path length vs word frequency (toy Zipf vocabulary)
     #viz-q   per-example cost Q of NNLM / RNNLM / CBOW / skip-gram, term by term, log scale
     #viz-an  toy 2-D analogy space: vector offset, cosine ray, exclusion and offset averaging */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>Math.round(n).toLocaleString("en-US");
  const sup=e=>String(e).split("").map(ch=>"⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(ch)]||(ch==="-"?"⁻":ch)).join("");
  const sci=n=>{if(!isFinite(n)||n<=0)return "0";const e=Math.floor(Math.log10(n)),m=n/Math.pow(10,e);return m.toFixed(2)+"×10"+sup(e);};

  /* Entropy in bits of a Zipf(s) law over ranks 1..V: exact sum for the head, integral for the tail. */
  function zipfEntropy(V,s){
    const K=Math.min(V,2000); let Z=0,A=0;
    for(let k=1;k<=K;k++){const p=Math.pow(k,-s);Z+=p;A+=p*Math.log(k);}
    if(V>K){
      const a=K+0.5,b=V+0.5;
      if(Math.abs(s-1)<1e-9){Z+=Math.log(b)-Math.log(a);A+=(Math.log(b)**2-Math.log(a)**2)/2;}
      else{const g=1-s,F=x=>Math.pow(x,g)/g,G=x=>Math.pow(x,g)*(Math.log(x)/g-1/(g*g));Z+=F(b)-F(a);A+=G(b)-G(a);}
    }
    return (Math.log(Z)+s*A/Z)/Math.LN2;
  }

  /* ═══════════════ 1. Huffman hierarchical softmax ═══════════════ */
  (function(){
    const svg=d3.select("#hs-svg"); if(svg.empty()) return;
    const WORDS=["the","of","and","to","in","is","for","on","with","city","king","river","queen","france","paris","bigger","smallest","walked","kwanza","tallahassee"];
    const W=760,H=400,m={l:30,r:190,t:28,b:20};
    let chosen="the";

    function build(V,s){
      const raw=d3.range(V).map(k=>Math.pow(k+1,-s)), Z=d3.sum(raw);
      const leaves=raw.map((p,k)=>({name:WORDS[k],p:p/Z,rank:k+1,id:k}));
      let pool=leaves.map(x=>x), nid=V;
      while(pool.length>1){
        pool.sort((a,b)=>a.p-b.p||a.id-b.id);
        const a=pool.shift(), b=pool.shift();          // two least frequent
        pool.push({children:[b,a],p:a.p+b.p,id:nid++}); // heavier on the left: edge 0
      }
      return {root:pool[0],leaves};
    }
    function draw(){
      const V=+d3.select("#hs-v").property("value"), s=+d3.select("#hs-s").property("value");
      d3.select("#hs-vv").text(V); d3.select("#hs-sv").text(s.toFixed(1));
      const {root,leaves}=build(V,s);
      // word picker
      const names=leaves.map(l=>l.name); if(names.indexOf(chosen)<0) chosen=names[0];
      const sel=d3.select("#hs-w");
      sel.selectAll("option").data(names,d=>d).join("option").attr("value",d=>d).text(d=>d);
      sel.property("value",chosen);

      const h=d3.hierarchy(root,d=>d.children);
      const maxD=d3.max(h.leaves(),l=>l.depth)||1;
      const tree=d3.tree().size([H-m.t-m.b,1]).separation(()=>1);
      tree(h);
      const dx=Math.min(48,(W-m.l-m.r)/maxD);
      const X=d=>m.l+d.depth*dx, Y=d=>m.t+d.x;
      svg.selectAll("*").remove();
      svg.append("text").attr("x",m.l).attr("y",16).attr("fill",C.muted).attr("font-size",11)
        .text("root → leaf: each inner node is one sigmoid decision with its own learned vector; 0 = upper branch, 1 = lower");
      // path to chosen
      const leafNode=h.leaves().find(l=>l.data.name===chosen);
      const onPath=new Set(leafNode.ancestors());
      const codeBits=[];
      leafNode.ancestors().reverse().forEach((n,i,arr)=>{if(i<arr.length-1){const kid=arr[i+1];codeBits.push(n.children.indexOf(kid));}});
      // edges
      h.links().forEach(lk=>{
        const hot=onPath.has(lk.source)&&onPath.has(lk.target), bit=lk.source.children.indexOf(lk.target);
        svg.append("path").attr("d",`M${X(lk.source)},${Y(lk.source)} C${(X(lk.source)+X(lk.target))/2},${Y(lk.source)} ${(X(lk.source)+X(lk.target))/2},${Y(lk.target)} ${X(lk.target)},${Y(lk.target)}`)
          .attr("fill","none").attr("stroke",hot?C.a2:C.line).attr("stroke-width",hot?2.4:1.2);
        if(hot) svg.append("text").attr("x",(X(lk.source)+X(lk.target))/2).attr("y",(Y(lk.source)+Y(lk.target))/2-4)
          .attr("text-anchor","middle").attr("fill",C.a2).attr("font-size",11).attr("font-weight",600).text(bit);
      });
      // nodes
      h.descendants().forEach(n=>{
        const leaf=!n.children, hot=onPath.has(n);
        if(!leaf){svg.append("circle").attr("cx",X(n)).attr("cy",Y(n)).attr("r",hot?4.5:3.5).attr("fill",hot?C.a2:C.panel).attr("stroke",hot?C.a2:C.muted);}
        else{
          const g=svg.append("g").style("cursor","pointer").on("click",()=>{chosen=n.data.name;draw();});
          g.append("rect").attr("x",X(n)-4).attr("y",Y(n)-4).attr("width",8).attr("height",8).attr("fill",n.data.name===chosen?C.a2:C.accent);
          g.append("text").attr("x",X(n)+8).attr("y",Y(n)+4).attr("font-size",11)
            .attr("fill",n.data.name===chosen?C.a2:C.ink).text(n.data.name+"  ·  p = "+n.data.p.toFixed(3)+"  ·  L = "+n.depth);
          g.append("title").text(n.data.name+": rank "+n.data.rank+", probability "+n.data.p.toFixed(4)+", path length "+n.depth);
        }
      });
      // balanced-depth guide
      const bal=Math.ceil(Math.log2(V));
      if(bal<=maxD+1){
        const gx=m.l+bal*dx;
        svg.append("line").attr("x1",gx).attr("x2",gx).attr("y1",m.t-6).attr("y2",H-m.b+6).attr("stroke",C.violet).attr("stroke-dasharray","4 4").attr("opacity",0.8);
        svg.append("text").attr("x",gx+4).attr("y",H-4).attr("fill",C.violet).attr("font-size",10.5).text("balanced depth ⌈log₂ V⌉ = "+bal);
      }
      const Lbar=d3.sum(h.leaves(),l=>l.data.p*l.depth);
      const Hbits=-d3.sum(leaves,l=>l.p>0?l.p*Math.log2(l.p):0);
      const leaf=leafNode.data;
      d3.select("#hs-readout").html(
        `<b>${leaf.name}</b> (rank ${leaf.rank}, p = ${leaf.p.toFixed(4)}): code <b>${codeBits.join("")}</b> · <b>${leafNode.depth}</b> sigmoid decision${leafNode.depth===1?"":"s"}`+
        ` vs ${bal} in a balanced tree${leafNode.depth>bal?" — longer than a balanced tree's path: rare words pay":leafNode.depth<bal?" — shorter than a balanced tree's path":""}`+
        `<br>average over training tokens: Huffman L̄ = ∑ p·L = <b>${Lbar.toFixed(3)}</b> · entropy H = log₂(unigram perplexity) = <b>${Hbits.toFixed(3)}</b> bits · log₂ V = <b>${Math.log2(V).toFixed(3)}</b>`+
        ` · H ≤ L̄ &lt; H + 1 · saving vs log₂ V: <b>${(Math.log2(V)/Lbar).toFixed(2)}×</b>`);
    }
    d3.select("#hs-v").on("input",draw);
    d3.select("#hs-s").on("input",draw);
    d3.select("#hs-w").on("change",function(){chosen=this.value;draw();});
    draw();
  })();

  /* ═══════════════ 2. per-example cost Q ═══════════════ */
  (function(){
    const svg=d3.select("#q-svg"); if(svg.empty()) return;
    const W=760,H=380,xl=250,xr=740,top=26,axisY=350;
    const x=d3.scaleLog().domain([1e2,1e11]).range([xl,xr]).clamp(true);
    const COL={proj:C.blue,hid:C.bad,out:C.a2,tot:C.accent};
    function draw(){
      const V=Math.round(Math.pow(10,+d3.select("#q-v").property("value")));
      const D=+d3.select("#q-d").property("value"), Hh=+d3.select("#q-h").property("value");
      const N=+d3.select("#q-n").property("value"), Cc=+d3.select("#q-c").property("value");
      const mode=d3.select("#q-o").property("value");
      d3.select("#q-vv").text(fmt(V)); d3.select("#q-dv").text(D); d3.select("#q-hv").text(Hh);
      d3.select("#q-nv").text(N); d3.select("#q-cv").text(Cc);
      const out=mode==="full"?V:mode==="tree"?Math.log2(V):zipfEntropy(V,1);
      const outLab=mode==="full"?"V":mode==="tree"?"log₂V":"H(Zipf)";
      const archs=[
        {name:"feedforward NNLM",terms:[["N·D",N*D,"proj"],["N·D·H",N*D*Hh,"hid"],["H·"+outLab,Hh*out,"out"]]},
        {name:"recurrent LM (D = H)",terms:[["H·H",Hh*Hh,"hid"],["H·"+outLab,Hh*out,"out"]]},
        {name:"CBOW",terms:[["N·D",N*D,"proj"],["D·"+outLab,D*out,"out"]]},
        {name:"skip-gram",terms:[["C·D",Cc*D,"proj"],["C·D·"+outLab,Cc*D*out,"out"]]}
      ];
      archs.forEach(a=>{a.Q=d3.sum(a.terms,t=>t[1]);a.dom=a.terms.reduce((p,t)=>t[1]>p[1]?t:p);});
      svg.selectAll("*").remove();
      svg.append("text").attr("x",xl).attr("y",14).attr("fill",C.muted).attr("font-size",11)
        .text("parameters accessed per training example (log scale) · output layer: "+(mode==="full"?"full softmax":mode==="tree"?"balanced tree":"Huffman, Zipf(1) entropy = "+out.toFixed(1)+" bits"));
      x.ticks(10).filter(t=>Math.log10(t)%1===0).forEach(t=>{
        svg.append("line").attr("x1",x(t)).attr("x2",x(t)).attr("y1",top).attr("y2",axisY).attr("stroke",C.line).attr("stroke-width",0.7);
      });
      svg.append("g").attr("transform",`translate(0,${axisY})`)
        .call(d3.axisBottom(x).tickValues(d3.range(2,12).map(e=>Math.pow(10,e))).tickFormat(t=>"10"+sup(Math.round(Math.log10(t)))))
        .attr("color",C.muted);
      let y=top+6;
      archs.forEach((a,ai)=>{
        svg.append("text").attr("x",10).attr("y",y+11).attr("fill",C.ink).attr("font-size",12.5).attr("font-weight",600).text(a.name);
        svg.append("rect").attr("x",xl).attr("y",y).attr("height",14).attr("width",Math.max(1,x(a.Q)-xl)).attr("fill",COL.tot).attr("opacity",0.85);
        svg.append("text").attr("x",Math.min(x(a.Q)+6,xr-90)).attr("y",y+11).attr("fill",C.ink).attr("font-size",11).text("Q ≈ "+sci(a.Q));
        y+=18;
        a.terms.forEach(t=>{
          const isDom=t===a.dom;
          svg.append("text").attr("x",xl-8).attr("y",y+7).attr("text-anchor","end").attr("fill",isDom?C.ink:C.muted).attr("font-size",10.5).text(t[0]+(isDom?"  ◀ dominant":""));
          svg.append("rect").attr("x",xl).attr("y",y).attr("height",8).attr("width",Math.max(1,x(Math.max(t[1],1e2))-xl)).attr("fill",COL[t[2]]).attr("opacity",isDom?0.95:0.55);
          svg.append("text").attr("x",Math.min(x(Math.max(t[1],1e2))+5,xr-60)).attr("y",y+8).attr("fill",C.muted).attr("font-size",10).text(fmt(t[1]));
          y+=12;
        });
        y+=14;
      });
      // legend
      [["projection / lookup",COL.proj],["hidden layer",COL.hid],["output layer",COL.out]].forEach((l,i)=>{
        svg.append("rect").attr("x",10).attr("y",axisY-44+i*14).attr("width",10).attr("height",8).attr("fill",l[1]);
        svg.append("text").attr("x",26).attr("y",axisY-36+i*14).attr("fill",C.muted).attr("font-size",10.5).text(l[0]);
      });
      const [nn,rn,cb,sg]=archs, T=1e9;
      d3.select("#q-readout").html(
        `Q: NNLM <b>${sci(nn.Q)}</b> · RNNLM <b>${sci(rn.Q)}</b> · CBOW <b>${sci(cb.Q)}</b> · skip-gram <b>${sci(sg.Q)}</b>`+
        ` · NNLM / CBOW = <b>${(nn.Q/cb.Q).toFixed(1)}×</b> · skip-gram / CBOW = <b>${(sg.Q/cb.Q).toFixed(1)}×</b>`+
        `<br>one epoch over T = 10⁹ words (O = E·T·Q, E = 1): NNLM ${sci(nn.Q*T)} · CBOW ${sci(cb.Q*T)} · skip-gram ${sci(sg.Q*T)} parameter accesses`+
        (mode==="full"?" · with a full softmax every model is dominated by its × V term":
         mode==="huff"?` · Huffman (Zipf(1) model) saves ${(Math.log2(V)/out).toFixed(2)}× on output terms vs a balanced tree; the paper reports ≈ 2× at V = 10⁶`:""));
    }
    ["#q-v","#q-d","#q-h","#q-n","#q-c"].forEach(id=>d3.select(id).on("input",draw));
    d3.select("#q-o").on("change",draw);
    draw();
  })();

  /* ═══════════════ 3. toy analogy space ═══════════════ */
  (function(){
    const svg=d3.select("#an-svg"); if(svg.empty()) return;
    const REL=[
      {name:"man–woman",th:45,pairs:[["man","woman"],["king","queen"],["brother","sister"],["uncle","aunt"],["grandson","granddaughter"]]},
      {name:"capital",th:145,pairs:[["France","Paris"],["Italy","Rome"],["Germany","Berlin"],["Greece","Athens"],["Norway","Oslo"],["Japan","Tokyo"]]},
      {name:"comparative",th:235,pairs:[["big","bigger"],["small","smaller"],["cold","colder"],["quick","quicker"],["tough","tougher"]]},
      {name:"present participle",th:320,pairs:[["think","thinking"],["read","reading"],["walk","walking"],["swim","swimming"],["work","working"]]}
    ];
    const RC=[C.accent,C.a2,C.violet,C.blue];
    const rad=a=>a*Math.PI/180;
    // clean positions
    const clean={}, relOf={};
    REL.forEach((r,ri)=>{
      const u=[Math.cos(rad(r.th)),Math.sin(rad(r.th))], t=[-u[1],u[0]];
      const off=[1.0*u[0]+0.45*t[0],1.0*u[1]+0.45*t[1]];
      r.pairs.forEach((p,i)=>{
        const ang=rad(r.th+(i-(r.pairs.length-1)/2)*13), R=4+0.3*(((i*7)%3)-1);
        const a=[R*Math.cos(ang),R*Math.sin(ang)];
        clean[p[0]]=a; clean[p[1]]=[a[0]+off[0],a[1]+off[1]];
        relOf[p[0]]=ri; relOf[p[1]]=ri;
      });
    });
    const WORDS=Object.keys(clean);
    // every question in the toy: pair i gives the offset, pair j is asked
    const QS=[];
    REL.forEach((r,ri)=>r.pairs.forEach((pi,i)=>r.pairs.forEach((pj,j)=>{if(i!==j)QS.push({ri,i,j,a:pi[0],b:pi[1],c:pj[0],d:pj[1]});})));
    const SHOW=[["man","woman","king"],["France","Paris","Italy"],["big","bigger","small"],["think","thinking","read"],["brother","sister","grandson"],["Greece","Athens","Norway"],["cold","colder","quick"],["walk","walking","swim"]]
      .map(t=>QS.findIndex(q=>q.a===t[0]&&q.b===t[1]&&q.c===t[2]));
    const qsel=d3.select("#an-q");
    qsel.selectAll("option").data(SHOW).join("option").attr("value",d=>d).text(d=>{const q=QS[d];return `${q.a} : ${q.b} :: ${q.c} : ?`;});
    qsel.property("value",SHOW[0]);
    let seed=3;
    function rng(sd){let s=sd>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    const cos=(p,q)=>(p[0]*q[0]+p[1]*q[1])/(Math.hypot(p[0],p[1])*Math.hypot(q[0],q[1])||1);
    function positions(noise){const r=rng(seed),P={};WORDS.forEach(w=>{const g=[gauss(r),gauss(r)];P[w]=[clean[w][0]+noise*g[0],clean[w][1]+noise*g[1]];});return P;}
    function solve(P,q,excl,avg){
      let off;
      if(avg){const prs=REL[q.ri].pairs.filter((p,k)=>k!==q.j);off=[d3.mean(prs,p=>P[p[1]][0]-P[p[0]][0]),d3.mean(prs,p=>P[p[1]][1]-P[p[0]][1])];}
      else off=[P[q.b][0]-P[q.a][0],P[q.b][1]-P[q.a][1]];
      const X=[P[q.c][0]+off[0],P[q.c][1]+off[1]];
      const skip=excl?new Set([q.a,q.b,q.c]):new Set();
      const ranked=WORDS.filter(w=>!skip.has(w)).map(w=>({w,s:cos(X,P[w])})).sort((u,v)=>v.s-u.s);
      return {X,off,ranked,ok:ranked[0].w===q.d};
    }
    const px=d3.scaleLinear().domain([-6.6,6.6]).range([30,450]);
    const py=d3.scaleLinear().domain([-6.6,6.6]).range([430,10]);
    function arrow(x1,y1,x2,y2,col,dash){
      const g=svg.append("g");
      g.append("line").attr("x1",px(x1)).attr("y1",py(y1)).attr("x2",px(x2)).attr("y2",py(y2)).attr("stroke",col).attr("stroke-width",2).attr("stroke-dasharray",dash||null);
      const ang=Math.atan2(py(y2)-py(y1),px(x2)-px(x1)), L=8;
      g.append("path").attr("d",`M${px(x2)},${py(y2)} L${px(x2)-L*Math.cos(ang-0.4)},${py(y2)-L*Math.sin(ang-0.4)} L${px(x2)-L*Math.cos(ang+0.4)},${py(y2)-L*Math.sin(ang+0.4)} Z`).attr("fill",col);
    }
    function draw(){
      const noise=+d3.select("#an-n").property("value"), excl=d3.select("#an-x").property("checked"), avg=d3.select("#an-avg").property("checked");
      d3.select("#an-nv").text(noise.toFixed(2));
      const q=QS[+qsel.property("value")];
      const P=positions(noise), res=solve(P,q,excl,avg);
      svg.selectAll("*").remove();
      // axes through origin
      svg.append("line").attr("x1",px(-6.6)).attr("x2",px(6.6)).attr("y1",py(0)).attr("y2",py(0)).attr("stroke",C.line);
      svg.append("line").attr("y1",py(-6.6)).attr("y2",py(6.6)).attr("x1",px(0)).attr("x2",px(0)).attr("stroke",C.line);
      svg.append("circle").attr("cx",px(0)).attr("cy",py(0)).attr("r",2.5).attr("fill",C.muted);
      // cosine ray through X
      const nX=Math.hypot(res.X[0],res.X[1])||1, far=6.4/nX;
      svg.append("line").attr("x1",px(0)).attr("y1",py(0)).attr("x2",px(res.X[0]*far)).attr("y2",py(res.X[1]*far))
        .attr("stroke",C.ink).attr("stroke-opacity",0.25).attr("stroke-width",6).attr("stroke-linecap","round");
      // words
      WORDS.forEach(w=>{
        const p=P[w], inQ=[q.a,q.b,q.c].indexOf(w)>=0, isAns=w===res.ranked[0].w, isExp=w===q.d;
        const col=isAns?(res.ok?C.good:C.bad):isExp?C.good:inQ?C.ink:RC[relOf[w]];
        svg.append("circle").attr("cx",px(p[0])).attr("cy",py(p[1])).attr("r",isAns||inQ||isExp?5:3).attr("fill",col).attr("opacity",isAns||inQ||isExp?1:0.6);
        svg.append("text").attr("x",px(p[0])+6).attr("y",py(p[1])-5).attr("font-size",isAns||inQ||isExp?11.5:9.5)
          .attr("fill",col).attr("opacity",isAns||inQ||isExp?1:0.75).attr("font-weight",isAns||isExp?600:400).text(w);
      });
      arrow(P[q.a][0],P[q.a][1],P[q.b][0],P[q.b][1],C.ink);
      arrow(P[q.c][0],P[q.c][1],res.X[0],res.X[1],C.a2,"5 3");
      svg.append("path").attr("d",d3.symbol(d3.symbolCross,90)()).attr("transform",`translate(${px(res.X[0])},${py(res.X[1])}) rotate(45)`).attr("fill",C.a2);
      svg.append("text").attr("x",px(res.X[0])+7).attr("y",py(res.X[1])+13).attr("fill",C.a2).attr("font-size",11).text("X");
      // right panel: top candidates
      const rx=480;
      svg.append("text").attr("x",rx).attr("y",30).attr("fill",C.muted).attr("font-size",11).text("nearest to X by cosine"+(excl?" (question words excluded)":" (nothing excluded)"));
      res.ranked.slice(0,5).forEach((r,i)=>{
        const y=48+i*30, col=i===0?(res.ok?C.good:C.bad):r.w===q.d?C.good:C.muted;
        svg.append("text").attr("x",rx).attr("y",y+12).attr("fill",col).attr("font-size",12).text((i+1)+". "+r.w);
        const bw=Math.max(1,(r.s-0.9)/0.1*150);
        svg.append("rect").attr("x",rx+110).attr("y",y+2).attr("width",Math.min(150,Math.max(1,bw))).attr("height",12).attr("fill",col).attr("opacity",0.75);
        svg.append("text").attr("x",rx+266).attr("y",y+12).attr("text-anchor","end").attr("fill",C.ink).attr("font-size",10.5).text(r.s.toFixed(4));
      });
      svg.append("text").attr("x",rx).attr("y",208).attr("fill",C.muted).attr("font-size",10).text("bars show cosine from 0.9 to 1.0 — in 2-D, cosine is angle only");
      REL.forEach((r,ri)=>{
        svg.append("circle").attr("cx",rx+5).attr("cy",246+ri*18).attr("r",4).attr("fill",RC[ri]);
        svg.append("text").attr("x",rx+16).attr("y",250+ri*18).attr("fill",C.muted).attr("font-size",11).text(r.name);
      });
      svg.append("text").attr("x",rx).attr("y",332).attr("fill",C.muted).attr("font-size",10.5).text("solid arrow: offset b − a"+(avg?" (shown for this pair; averaged offset used)":""));
      svg.append("text").attr("x",rx).attr("y",348).attr("fill",C.muted).attr("font-size",10.5).text("dashed arrow: c + offset = X");
      svg.append("text").attr("x",rx).attr("y",364).attr("fill",C.muted).attr("font-size",10.5).text("grey band: the ray cosine ranks along");
      // accuracy over the whole toy set
      let ok=0; QS.forEach(qq=>{if(solve(P,qq,excl,avg).ok)ok++;});
      const pct=100*ok/QS.length;
      svg.append("text").attr("x",rx).attr("y",404).attr("fill",C.ink).attr("font-size",12.5).text(`toy-set exact-match accuracy: ${ok}/${QS.length} = ${pct.toFixed(1)}%`);
      const top=res.ranked.slice(0,3).map(r=>`${r.w} ${r.s.toFixed(4)}`).join(" · ");
      d3.select("#an-readout").html(
        `${q.a} : ${q.b} :: ${q.c} : ? → <b>${res.ranked[0].w}</b> ${res.ok?"— correct (exact match)":"— wrong; expected <b>"+q.d+"</b>"}`+
        ` · top 3: ${top}`+
        `<br>offset from ${avg?"the mean of "+(REL[q.ri].pairs.length-1)+" other pairs":"one pair"} · noise ${noise.toFixed(2)} · ${excl?"question words excluded":"question words allowed"} · toy-set accuracy <b>${pct.toFixed(1)}%</b> (${ok}/${QS.length})`);
    }
    qsel.on("change",draw);
    d3.select("#an-n").on("input",draw);
    d3.select("#an-x").on("change",draw);
    d3.select("#an-avg").on("change",draw);
    d3.select("#an-seed").on("click",()=>{seed+=1;draw();});
    draw();
  })();
})();
