/* rag-sequence-token.viz.js — interactive figures for nlp/retrieval-augmented-generation/rag-sequence-token.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-mx  marginalisation explorer: RAG-Sequence vs RAG-Token likelihood on a toy (illustrative values)
     #viz-rs  reported results explorer: open-domain QA (test), generation & classification (test), ablations (dev) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",panel2:"#1b1f27",violet:"#a78bfa",blue:"#5b9cff"};

  /* ═══════════════ 1. marginalisation explorer ═══════════════ */
  (function(){
    const svg=d3.select("#mx-svg"); if(svg.empty()) return;
    const TOK=["Farewell","Arms","Sun","Rises"], N=TOK.length, POOL=10;
    // Illustrative generator probabilities p_θ(y_i | x, z, y_<i) before noise.
    // Tokens 2 and 4 complete a title, so every document predicts them (parametric memory).
    const DISTRACT=[0.03,0.60,0.03,0.85];
    const PRESETS={
      split:{scores:[2.0,1.7,0.9,0.6,0.4,0.2,0.1,0.0,-0.2,-0.4],
             base:z=>z===0?[0.70,0.90,0.04,0.85]:z===1?[0.04,0.60,0.70,0.90]:DISTRACT},
      single:{scores:[2.2,1.0,0.8,0.6,0.4,0.2,0.1,0.0,-0.2,-0.4],
             base:z=>z===0?[0.70,0.90,0.70,0.90]:DISTRACT},
      flat:{scores:[0,0,0,0,0,0,0,0,0,0],
             base:()=>[0.05,0.60,0.05,0.85]}
    };
    let seed=3, scores=PRESETS.split.scores.slice(), eps=null;
    function rng(s){s=s>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    function drawNoise(){const r=rng(seed*7919+13);eps=d3.range(POOL).map(()=>d3.range(N).map(()=>gauss(r)));}
    const val=id=>+d3.select(id).property("value");
    const fmtE=v=>v>=0.01?v.toFixed(4):v.toExponential(2);

    // doc selector options
    d3.select("#mx-doc").selectAll("option").data(d3.range(POOL)).join("option").attr("value",d=>d).text(d=>"d"+(d+1));

    function model(){
      const pre=PRESETS[d3.select("#mx-preset").property("value")];
      const k=val("#mx-k"), sharp=val("#mx-t"), noise=val("#mx-n");
      const order=d3.range(POOL).sort((a,b)=>scores[b]-scores[a]||a-b);
      const top=order.slice(0,k);
      const s=top.map(z=>sharp*scores[z]), mx=d3.max(s);
      const ex=s.map(v=>Math.exp(v-mx)), Z=d3.sum(ex), prior=ex.map(v=>v/Z);
      const P=top.map(z=>pre.base(z).map((p,i)=>Math.max(0.005,Math.min(0.97,p*Math.exp(noise*eps[z][i])))));
      const seqProd=P.map(row=>row.reduce((a,b)=>a*b,1));
      const pSeq=d3.sum(seqProd,(v,j)=>prior[j]*v);
      const mix=d3.range(N).map(i=>d3.sum(P,(row,j)=>prior[j]*row[i]));
      const pTok=mix.reduce((a,b)=>a*b,1);
      const postTok=P.map((row,j)=>row.map((p,i)=>prior[j]*p/mix[i]));
      const postSeq=d3.range(N).map(i=>{const w=P.map((row,j)=>prior[j]*row.slice(0,i+1).reduce((a,b)=>a*b,1));const t=d3.sum(w);return w.map(v=>v/t);});
      const postSeqT=P.map((row,j)=>d3.range(N).map(i=>postSeq[i][j]));
      const gSeq=P.map((row,j)=>postSeq[N-1][j]-prior[j]);
      const gTok=P.map((row,j)=>d3.sum(d3.range(N),i=>postTok[j][i]-prior[j]));
      return {k,top,prior,P,seqProd,pSeq,mix,pTok,postTok,postSeqT,gSeq,gTok};
    }

    // ── layout ──
    const W=760, y0=58, RH=21, colW=58;
    const xP0=50, xP1=200, xG=232, xH=500;
    const gCell=d3.interpolateRgb(C.panel2,C.accent), hCell=d3.interpolateRgb(C.panel2,C.a2);
    svg.append("text").attr("x",xP0).attr("y",22).attr("fill",C.muted).attr("font-size",11).text("retriever prior p(z | x)");
    svg.append("text").attr("x",xP0).attr("y",36).attr("fill",C.muted).attr("font-size",10).text("softmax over the k retrieved scores");
    svg.append("text").attr("x",xG).attr("y",22).attr("fill",C.muted).attr("font-size",11).text("generator p(yᵢ | x, z, y<ᵢ)");
    const postTitle=svg.append("text").attr("x",xH).attr("y",22).attr("fill",C.muted).attr("font-size",11);
    TOK.forEach((t,i)=>{
      svg.append("text").attr("x",xG+i*colW+colW/2).attr("y",y0-8).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",11).text(t);
      svg.append("text").attr("x",xH+i*colW+colW/2).attr("y",y0-8).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",11).text(t);
    });
    svg.append("text").attr("x",xG).attr("y",36).attr("fill",C.muted).attr("font-size",10).text("toy answer y = ⟨Farewell, Arms, Sun, Rises⟩");
    const xp=d3.scaleLinear().domain([0,1]).range([xP0,xP1]);
    const gRows=svg.append("g");

    // likelihood comparison
    const by0=318, xl0=170, xl1=640;
    const xl=d3.scaleLog().domain([1e-6,1]).range([xl0,xl1]).clamp(true);
    svg.append("text").attr("x",xl0).attr("y",by0-12).attr("fill",C.muted).attr("font-size",11).text("likelihood of the toy answer, p(y | x) — log scale");
    svg.append("g").attr("transform",`translate(0,${by0+78})`).call(d3.axisBottom(xl).ticks(6,"~e")).attr("color",C.muted);
    const lbars=[{key:"pSeq",lab:"RAG-Sequence",sub:"∑_z p(z|x) ∏ᵢ p(yᵢ|…)",col:C.a2},{key:"pTok",lab:"RAG-Token",sub:"∏ᵢ ∑_z p(z|x) p(yᵢ|…)",col:C.accent}];
    const lb=lbars.map((b,i)=>{
      const y=by0+i*36;
      svg.append("text").attr("x",xl0-8).attr("y",y+12).attr("text-anchor","end").attr("fill",b.col).attr("font-size",12).text(b.lab);
      svg.append("text").attr("x",xl0-8).attr("y",y+25).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9.5).text(b.sub);
      const r=svg.append("rect").attr("x",xl0).attr("y",y).attr("height",26).attr("fill",b.col).attr("opacity",0.85);
      const t=svg.append("text").attr("y",y+17).attr("fill",C.ink).attr("font-size",11);
      return {r,t};
    });
    svg.append("text").attr("x",W-14).attr("y",452).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("illustrative toy values — not numbers from the study");

    function render(){
      d3.select("#mx-kv").text(val("#mx-k"));
      d3.select("#mx-tv").text(val("#mx-t").toFixed(1));
      d3.select("#mx-nv").text(val("#mx-n").toFixed(2));
      const doc=val("#mx-doc");
      d3.select("#mx-sv").text(scores[doc].toFixed(1));
      const M=model(), mode=d3.select("#mx-post").property("value");
      const post=mode==="token"?M.postTok:M.postSeqT;
      postTitle.text(mode==="token"?"RAG-Token posterior p(z | x, y₁:ᵢ), per token":"RAG-Sequence posterior p(z | x, y₁:ᵢ), after prefix");
      gRows.selectAll("*").remove();
      M.top.forEach((z,j)=>{
        const y=y0+j*RH, hl=z===doc;
        gRows.append("text").attr("x",xP0-8).attr("y",y+14).attr("text-anchor","end").attr("fill",hl?C.ink:C.muted).attr("font-size",11).attr("font-weight",hl?600:400).text("d"+(z+1));
        gRows.append("rect").attr("x",xP0).attr("y",y+3).attr("width",xp(1)-xP0).attr("height",RH-6).attr("fill",C.panel2);
        gRows.append("rect").attr("x",xP0).attr("y",y+3).attr("width",Math.max(0.5,xp(M.prior[j])-xP0)).attr("height",RH-6).attr("fill",C.blue).attr("opacity",0.85);
        gRows.append("text").attr("x",xP1+4).attr("y",y+14).attr("fill",C.ink).attr("font-size",9.5).text(M.prior[j].toFixed(2));
        for(let i=0;i<N;i++){
          const p=M.P[j][i], q=post[j][i];
          gRows.append("rect").attr("x",xG+i*colW+1).attr("y",y+1).attr("width",colW-2).attr("height",RH-2).attr("fill",gCell(p));
          gRows.append("text").attr("x",xG+i*colW+colW/2).attr("y",y+14).attr("text-anchor","middle").attr("fill",p>0.55?"#0b0d12":C.ink).attr("font-size",10).text(p.toFixed(2));
          gRows.append("rect").attr("x",xH+i*colW+1).attr("y",y+1).attr("width",colW-2).attr("height",RH-2).attr("fill",hCell(q));
          gRows.append("text").attr("x",xH+i*colW+colW/2).attr("y",y+14).attr("text-anchor","middle").attr("fill",q>0.55?"#0b0d12":C.ink).attr("font-size",10).text(q.toFixed(2));
        }
      });
      lbars.forEach((b,i)=>{
        const v=M[b.key];
        lb[i].r.attr("width",Math.max(1,xl(v)-xl0));
        lb[i].t.attr("x",Math.min(xl(v),xl1)+6).text(fmtE(v)+"   (log p = "+Math.log(v).toFixed(2)+")");
      });
      const ratio=M.pTok/M.pSeq, lr=Math.log(ratio);
      let verdict;
      if(M.k===1) verdict="k = 1: one term in every sum, so the two formulations are <b>identical by definition</b>";
      else if(Math.abs(lr)<0.01) verdict="the two agree: every retrieved document gives (nearly) the same token probabilities, so mixing per token or per sequence makes no difference";
      else if(lr>0) verdict="<b>RAG-Token is higher</b>: different tokens are supported by different documents, and a per-token mixture can take each from where it lives";
      else verdict="<b>RAG-Sequence is higher</b>: one document supports the whole answer; its sequence product is counted once, while the per-token mixture pays the prior's dilution at every token";
      const j0=0, z0=M.top[0];
      d3.select("#mx-readout").html(
        `k = <b>${M.k}</b> · RAG-Sequence p(y|x) = <b>${fmtE(M.pSeq)}</b> · RAG-Token p(y|x) = <b>${fmtE(M.pTok)}</b> · Token / Sequence = <b>${ratio.toFixed(2)}×</b><br>`+verdict+
        `<br>gradient on the top document d${z0+1}'s retrieval score (posterior − prior): RAG-Sequence <b>${M.gSeq[j0].toFixed(3)}</b>, RAG-Token <b>${M.gTok[j0].toFixed(3)}</b> (summed over tokens)`+
        ` · illustrative toy, not reported values`);
    }
    function loadPreset(){scores=PRESETS[d3.select("#mx-preset").property("value")].scores.slice();syncSlider();}
    function syncSlider(){d3.select("#mx-s").property("value",scores[val("#mx-doc")]);}
    d3.select("#mx-preset").on("change",()=>{loadPreset();render();});
    ["#mx-k","#mx-t","#mx-n"].forEach(id=>d3.select(id).on("input",render));
    d3.select("#mx-post").on("change",render);
    d3.select("#mx-doc").on("change",()=>{syncSlider();render();});
    d3.select("#mx-s").on("input",()=>{scores[val("#mx-doc")]=val("#mx-s");render();});
    d3.select("#mx-seed").on("click",()=>{seed+=1;drawNoise();render();});
    drawNoise();loadPreset();render();
  })();

  /* ═══════════════ 2. reported results explorer ═══════════════ */
  (function(){
    const svg=d3.select("#rs-svg"); if(svg.empty()) return;
    const CL={closed:C.muted,open:C.blue,bart:C.muted,sota:C.violet,tok:C.accent,seq:C.a2};
    const EQ="same model as RAG-Token (one-token target)";
    const T={
      t1:{name:"open-domain QA, test exact match",
        cols:[["NQ","Natural Questions"],["TQAo","TriviaQA (open-domain test)"],["TQAw","TriviaQA (Wiki test)"],["WQ","WebQuestions"],["CT","CuratedTrec"]],
        rows:[
          {m:"T5-large (770M, closed book)",k:"closed",v:{NQ:28.9}},
          {m:"T5-11B (closed book)",k:"closed",v:{NQ:34.5,TQAw:50.1,WQ:37.4}},
          {m:"T5-11B + SSM (closed book)",k:"closed",v:{NQ:36.6,TQAw:60.5,WQ:44.7}},
          {m:"REALM (open book)",k:"open",v:{NQ:40.4,WQ:40.7,CT:46.8}},
          {m:"DPR (open book)",k:"open",v:{NQ:41.5,TQAo:57.9,WQ:41.1,CT:50.6}},
          {m:"RAG-Token",k:"tok",v:{NQ:44.1,TQAo:55.2,TQAw:66.1,WQ:45.5,CT:50.0}},
          {m:"RAG-Sequence",k:"seq",v:{NQ:44.5,TQAo:56.8,TQAw:68.0,WQ:45.2,CT:52.2}}]},
      t2:{name:"generation & classification, test",
        cols:[["JB1","Jeopardy BLEU-1"],["JQB1","Jeopardy Q-BLEU-1"],["MRL","MS-MARCO ROUGE-L"],["MB1","MS-MARCO BLEU-1"],["F3","FEVER 3-way label accuracy"],["F2","FEVER 2-way label accuracy"]],
        rows:[
          {m:"previous state of the art",k:"sota",v:{MRL:49.8,MB1:49.9,F3:76.8,F2:92.2},gold:{MRL:1,MB1:1,F2:1}},
          {m:"BART (no retrieval)",k:"bart",v:{JB1:15.1,JQB1:19.7,MRL:38.2,MB1:41.6,F3:64.0,F2:81.1}},
          {m:"RAG-Token",k:"tok",v:{JB1:17.3,JQB1:22.2,MRL:40.1,MB1:41.5,F3:72.5,F2:89.5}},
          {m:"RAG-Sequence",k:"seq",v:{JB1:14.7,JQB1:21.4,MRL:40.8,MB1:44.2},same:{F3:1,F2:1}}]},
      t6:{name:"ablations, dev set",
        cols:[["NQ","Natural Questions EM"],["TQA","TriviaQA EM"],["WQ","WebQuestions EM"],["CT","CuratedTrec EM"],["JB1","Jeopardy BLEU-1"],["JQB1","Jeopardy Q-BLEU-1"],["MRL","MS-MARCO ROUGE-L"],["MB1","MS-MARCO BLEU-1"],["F3","FEVER-3 label accuracy"],["F2","FEVER-2 label accuracy"]],
        rows:[
          {m:"RAG-Token · BM25",k:"tok",o:0.4,v:{NQ:29.7,TQA:41.5,WQ:32.1,CT:33.1,JB1:17.5,JQB1:22.3,MRL:55.5,MB1:48.4,F3:75.1,F2:91.6}},
          {m:"RAG-Sequence · BM25",k:"seq",o:0.4,v:{NQ:31.8,TQA:44.1,WQ:36.6,CT:33.8,JB1:11.1,JQB1:19.5,MRL:56.5,MB1:46.9},same:{F3:1,F2:1}},
          {m:"RAG-Token · frozen",k:"tok",o:0.65,v:{NQ:37.8,TQA:50.1,WQ:37.1,CT:51.1,JB1:16.7,JQB1:21.7,MRL:55.9,MB1:49.4,F3:72.9,F2:89.4}},
          {m:"RAG-Sequence · frozen",k:"seq",o:0.65,v:{NQ:41.2,TQA:52.1,WQ:41.8,CT:52.6,JB1:11.8,JQB1:19.6,MRL:56.7,MB1:47.3},same:{F3:1,F2:1}},
          {m:"RAG-Token · learned",k:"tok",o:1,v:{NQ:43.5,TQA:54.8,WQ:46.5,CT:51.9,JB1:17.9,JQB1:22.6,MRL:56.2,MB1:49.4,F3:74.5,F2:90.6}},
          {m:"RAG-Sequence · learned",k:"seq",o:1,v:{NQ:44.0,TQA:55.8,WQ:44.9,CT:53.4,JB1:15.3,JQB1:21.5,MRL:57.2,MB1:47.5},same:{F3:1,F2:1}}]}
    };
    const NOTES={
      "t1:TQAo":"DPR's 57.9 is above both RAG models on the open-domain TriviaQA test; the state-of-the-art claim on TriviaQA is for the Wiki test split only.",
      "t1:TQAw":"the Wiki test set is the one the closed-book T5 work used; the study finds it easier to answer from Wikipedia than the open-domain test set.",
      "t1:NQ":"T5-large (770M) comes from the study's parameter appendix, not its main table.",
      "t2:MRL":"the previous state of the art used the gold passages; RAG retrieves from Wikipedia instead.",
      "t2:MB1":"the previous state of the art used the gold passages; RAG retrieves from Wikipedia instead.",
      "t2:F2":"the 2-way comparison model was given the gold evidence sentence.",
      "t2:F3":"the previous best was a pipeline trained with retrieval supervision; RAG uses none.",
      "t6:F3":"BM25 is the best retriever on FEVER — the one task where it is.",
      "t6:F2":"BM25 is the best retriever on FEVER — the one task where it is."
    };
    const H=340, m={l:200,r:70,t:34,b:36}, W=760;
    function fillCols(){
      const t=d3.select("#rs-t").property("value");
      const sel=d3.select("#rs-c");
      sel.selectAll("option").data(T[t].cols,d=>t+d[0]).join("option").attr("value",d=>d[0]).text(d=>d[1]);
      sel.property("value",T[t].cols[0][0]);
    }
    function draw(){
      svg.selectAll("*").remove();
      const t=d3.select("#rs-t").property("value"), c=d3.select("#rs-c").property("value"), zoom=d3.select("#rs-z").property("checked");
      const tab=T[t], colName=(tab.cols.find(d=>d[0]===c)||tab.cols[0])[1];
      const rows=tab.rows;
      const vals=rows.map(r=>r.v[c]).filter(v=>v!=null);
      const lo=zoom?Math.max(0,Math.floor(d3.min(vals)-3)):0, hi=Math.ceil(d3.max(vals)+2);
      const x=d3.scaleLinear().domain([lo,hi]).range([m.l,W-m.r]).nice();
      const y=d3.scaleBand().domain(rows.map(r=>r.m)).range([m.t,H-m.b]).padding(0.22);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).tickSize(0)).attr("color",C.muted).selectAll("text").attr("fill",C.ink).attr("font-size",11);
      svg.append("text").attr("x",m.l).attr("y",18).attr("fill",C.muted).attr("font-size",11)
        .text(colName+" · "+tab.name+(zoom?" — axis zoomed, does not start at 0":" — axis starts at 0"));
      const defs=svg.append("defs");
      defs.append("pattern").attr("id","rs-hatch").attr("width",6).attr("height",6).attr("patternUnits","userSpaceOnUse").attr("patternTransform","rotate(45)")
        .append("rect").attr("width",3).attr("height",6).attr("fill",C.violet);
      const best=d3.max(vals);
      rows.forEach(r=>{
        const v=r.v[c], yy=y(r.m), bh=y.bandwidth();
        if(v==null){
          const txt=r.same&&r.same[c]?EQ:"not reported";
          svg.append("text").attr("x",m.l+6).attr("y",yy+bh/2+4).attr("fill",C.muted).attr("font-size",10.5).attr("font-style","italic").text(txt);
          return;
        }
        const gold=r.gold&&r.gold[c];
        svg.append("rect").attr("x",m.l).attr("y",yy).attr("height",bh).attr("width",Math.max(1,x(v)-m.l))
          .attr("fill",gold?"url(#rs-hatch)":CL[r.k]).attr("opacity",r.o!=null?r.o:0.9)
          .attr("stroke",v===best?C.ink:"none").attr("stroke-width",1)
          .append("title").text(`${r.m}: ${v}${gold?" (uses gold passages / evidence)":""}`);
        svg.append("text").attr("x",x(v)+6).attr("y",yy+bh/2+4).attr("fill",C.ink).attr("font-size",11).text(v.toFixed(1)+(gold?" *":""));
      });
      // readout
      const bestRows=rows.filter(r=>r.v[c]===best).map(r=>r.m).join(" and ");
      const tokR=rows.filter(r=>r.k==="tok"), seqR=rows.filter(r=>r.k==="seq");
      const pairs=tokR.map((tr,i)=>[tr,seqR[i]]).filter(p=>p[1]&&p[0].v[c]!=null&&p[1].v[c]!=null);
      const diff=pairs.map(p=>{const d=p[1].v[c]-p[0].v[c];const lab=p[0].m.replace("RAG-Token","").replace(" · ","")||"";
        return (lab?lab+": ":"")+(d>0?"RAG-Sequence +":d<0?"RAG-Token +":"tie ")+Math.abs(d).toFixed(1);}).join(" · ");
      const note=NOTES[t+":"+c];
      d3.select("#rs-readout").html(`<b>${colName}</b> (${tab.name}) · best: <b>${bestRows}</b> at <b>${best.toFixed(1)}</b>`+
        (diff?` · RAG-Sequence vs RAG-Token: ${diff}`:` · the two RAG variants are one model for this one-token target`)+
        (note?`<br>${note}`:"")+(t==="t2"&&(c==="MRL"||c==="MB1"||c==="F2")?" Hatched bars (*) used gold context.":"")+
        `<br>differences computed from the reported table values (our arithmetic)`);
    }
    d3.select("#rs-t").on("change",()=>{fillCols();draw();});
    d3.select("#rs-c").on("change",draw);
    d3.select("#rs-z").on("change",draw);
    fillCols();draw();
  })();
})();
