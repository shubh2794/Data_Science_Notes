/* qlora.viz.js — interactive figures for llm/parameter-efficient-fine-tuning/qlora.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-qz   Int4 / FP4 / NF4 on blockwise-absmax-normalized Gaussian weights (simulated)
     #viz-mem  training-memory arithmetic: 16-bit full FT vs 16-bit LoRA vs QLoRA (our assumptions)
     #viz-elo  reported Elo ratings, human raters vs GPT-4 as the judge */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>n.toLocaleString("en-US");

  /* inverse CDF of N(0,1), rational approximation (relative error ~1e-9) */
  function normInv(p){
    const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00];
    const b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155833011e+01];
    const c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00];
    const d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
    const pl=0.02425;
    if(p<pl){const q=Math.sqrt(-2*Math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
    if(p>1-pl){const q=Math.sqrt(-2*Math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
    const q=p-0.5,r=q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }

  /* ═══════════════ 1. quantizer explorer ═══════════════ */
  (function(){
    const svg=d3.select("#qz-svg"); if(svg.empty()) return;
    const N=16384, BS=[16,32,64,128,256,512,1024,2048,4096], NOUT=8, OUTMAG=12;
    // the paper's published NF4 table (appendix), used only to check the computed levels
    const NF4_PUB=[-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,
      0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0];
    // NF4 from normal quantiles: 8 positive, 7 negative, one exact zero (see the page's callout)
    function buildNF4(){
      const off=0.5*((1-1/32)+(1-1/30)), top=normInv(off);
      const pos=d3.range(8).map(i=>normInv(off+(0.5-off)*i/8));
      const neg=d3.range(7).map(i=>-normInv(off+(0.5-off)*i/7));
      return pos.concat(neg,[0]).map(v=>v/top).sort((x,y)=>x-y);
    }
    const NF4=buildNF4();
    const nfDev=d3.max(NF4,(v,i)=>Math.abs(v-NF4_PUB[i]));
    const INT4=d3.range(-7,8).map(j=>j/7);                                         // 15 levels, absmax analogue of 127/absmax
    const FP4=(()=>{const m=[0,0.5,1,1.5,2,3,4,6];const s=new Set();m.forEach(v=>{s.add(v/6);s.add(-v/6);});return [...s].sort((x,y)=>x-y);})();
    const FMT=[{key:"nf4",lab:"NF4",lv:NF4,col:C.accent},{key:"fp4",lab:"FP4 (E2M1)",lv:FP4,col:C.a2},{key:"int4",lab:"Int4",lv:INT4,col:C.violet}];

    let seed=3, X=null, sigma2=1;
    function rng(s0){let s=s0>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
    function sample(){
      const R=rng(seed), out=d3.select("#qz-o").property("checked");
      X=new Float64Array(N);
      for(let i=0;i<N;i++){const u=R(),v=R();X[i]=Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
      let s=0;for(let i=0;i<N;i++)s+=X[i]*X[i];sigma2=s/N;                             // variance of the Gaussian part
      if(out){for(let k=0;k<NOUT;k++){const j=Math.floor(R()*N);X[j]=(R()<0.5?-1:1)*OUTMAG;}}
    }
    function nearest(lv,v){let bi=0,bd=Infinity;for(let i=0;i<lv.length;i++){const d=Math.abs(lv[i]-v);if(d<bd){bd=d;bi=i;}}return bi;}
    // quantize the whole tensor blockwise; returns errors, code counts and the normalized values
    function run(lv,B,keepNorm){
      let se=0,ae=0;const cnt=new Float64Array(lv.length);const norm=keepNorm?new Float64Array(N):null;
      for(let b0=0;b0<N;b0+=B){
        let m=0;for(let i=b0;i<b0+B;i++)m=Math.max(m,Math.abs(X[i]));
        for(let i=b0;i<b0+B;i++){const z=X[i]/m, k=nearest(lv,z), xr=lv[k]*m, e=X[i]-xr;se+=e*e;ae+=Math.abs(e);cnt[k]++;if(norm)norm[i]=z;}
      }
      let H=0;for(const c of cnt){if(c>0){const p=c/N;H-=p*Math.log2(p);}}
      return {mse:se/N/sigma2, mae:ae/N/Math.sqrt(sigma2), H, used:cnt.filter(c=>c>0).length, norm};
    }

    const W=760, hx0=50, hx1=740, hy0=34, hy1=200;
    const ex0=70, ex1=600, ey0=262, ey1=400;
    const hxs=d3.scaleLinear().domain([-1.02,1.02]).range([hx0,hx1]);
    const gHistAx=svg.append("g").attr("transform",`translate(0,${hy1})`).attr("color",C.muted);
    gHistAx.call(d3.axisBottom(hxs).ticks(10));
    const histTitle=svg.append("text").attr("x",hx0).attr("y",18).attr("fill",C.muted).attr("font-size",11);
    const gBars=svg.append("g"), gTicks=svg.append("g");
    const exs=d3.scaleLog().base(2).domain([16,4096]).range([ex0,ex1]);
    const gEx=svg.append("g").attr("transform",`translate(0,${ey1})`).attr("color",C.muted);
    gEx.call(d3.axisBottom(exs).tickValues(BS).tickFormat(d3.format("d")));
    const gEy=svg.append("g").attr("transform",`translate(${ex0},0)`).attr("color",C.muted);
    svg.append("text").attr("x",ex0).attr("y",ey0-12).attr("fill",C.muted).attr("font-size",11).text("simulated error  MSE / σ²  (log) vs block size B (log) — not reported numbers");
    svg.append("text").attr("x",(ex0+ex1)/2).attr("y",ey1+32).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("block size B (values sharing one absmax constant)");
    const gLines=svg.append("g"), bRule=svg.append("line").attr("stroke",C.ink).attr("stroke-dasharray","3 3");
    const gLeg=svg.append("g").attr("transform",`translate(${ex1+24},${ey0+4})`);
    FMT.forEach((f,i)=>{gLeg.append("rect").attr("x",0).attr("y",i*20).attr("width",14).attr("height",3).attr("fill",f.col);
      gLeg.append("text").attr("x",20).attr("y",i*20+5).attr("fill",f.col).attr("font-size",11).text(f.lab);});

    let curve=null;
    function computeCurves(){curve=FMT.map(f=>BS.map(B=>({B,...run(f.lv,B,false)})));}
    function render(){
      const bi=+d3.select("#qz-b").property("value"), B=BS[bi], fk=d3.select("#qz-f").property("value");
      const f=FMT.find(z=>z.key===fk);
      d3.select("#qz-bv").text(B);
      const cur=run(f.lv,B,true);
      // histogram of normalized values
      const bins=d3.bin().domain([-1.02,1.02]).thresholds(d3.range(-1.02,1.0201,2.04/96))(Array.from(cur.norm));
      const hys=d3.scaleLinear().domain([0,d3.max(bins,b=>b.length)||1]).range([hy1,hy0+8]);
      gBars.selectAll("rect").data(bins).join("rect").attr("x",b=>hxs(b.x0)+0.5).attr("width",b=>Math.max(0.5,hxs(b.x1)-hxs(b.x0)-1))
        .attr("y",b=>hys(b.length)).attr("height",b=>hy1-hys(b.length)).attr("fill","#3a4150");
      gTicks.selectAll("line").data(f.lv).join("line").attr("x1",v=>hxs(v)).attr("x2",v=>hxs(v)).attr("y1",hy0).attr("y2",hy1)
        .attr("stroke",f.col).attr("stroke-width",1.4).attr("opacity",0.85);
      histTitle.text(`all ${fmt(N)} weights after per-block absmax normalization, with the 16 ${f.lab} levels (B = ${B}${d3.select("#qz-o").property("checked")?", with outliers":""})`);
      // error curves
      const all=[].concat(...curve.map(c=>c.map(p=>p.mse))).filter(v=>v>0);
      const eys=d3.scaleLog().domain([d3.min(all)*0.8,d3.max(all)*1.25]).range([ey1,ey0]);
      gEy.call(d3.axisLeft(eys).ticks(4,"~g"));
      gLines.selectAll("path").data(curve).join("path").attr("fill","none").attr("stroke",(c,i)=>FMT[i].col).attr("stroke-width",(c,i)=>FMT[i].key===fk?2.6:1.4)
        .attr("d",c=>d3.line().x(p=>exs(p.B)).y(p=>eys(p.mse))(c));
      gLines.selectAll("circle").data([].concat(...curve.map((c,i)=>c.map(p=>({...p,i}))))).join("circle")
        .attr("cx",p=>exs(p.B)).attr("cy",p=>eys(p.mse)).attr("r",p=>p.B===B?4.5:2.5).attr("fill",p=>FMT[p.i].col);
      bRule.attr("x1",exs(B)).attr("x2",exs(B)).attr("y1",ey0).attr("y2",ey1);
      const at=FMT.map((z,i)=>({z,p:curve[i][bi]}));
      const parts=at.map(({z,p})=>`${z.lab}: MSE/σ² <b>${p.mse.toFixed(4)}</b>, mean |err|/σ ${p.mae.toFixed(3)}, code entropy ${p.H.toFixed(2)} bits`).join(" · ");
      const c1=32/B, c2=8/B+32/(256*B);
      d3.select("#qz-readout").html(`B = <b>${B}</b> · ${parts}`+
        `<br>constants: <b>${c1.toFixed(3)}</b> bits/param single, <b>${c2.toFixed(3)}</b> with double quantization (8-bit constants, blocks of 256) · total ≈ ${(4+c2).toFixed(3)} bits/param`+
        `<br>simulated on a seeded N(0,1) sample, not reported figures · computed NF4 levels match the paper's table to within ${nfDev.toExponential(1)}`);
    }
    function resample(){sample();computeCurves();render();}
    d3.select("#qz-b").on("input",render);
    d3.select("#qz-f").on("change",render).on("input",render);
    d3.select("#qz-o").on("change",resample);
    d3.select("#qz-new").on("click",()=>{seed+=1;resample();});
    resample();
  })();

  /* ═══════════════ 2. memory budget ═══════════════ */
  (function(){
    const svg=d3.select("#mem-svg"); if(svg.empty()) return;
    const MODELS=[{lab:"7B",d:4096,f:11008,L:32},{lab:"13B",d:5120,f:13824,L:40},{lab:"33B",d:6656,f:17920,L:60},{lab:"65B",d:8192,f:22016,L:80}];
    const V=32000, GB=1e9;
    const PARTS=[{key:"w",lab:"base weights",col:"#5b6577"},{key:"c",lab:"quant. constants",col:C.violet},{key:"g",lab:"base gradients",col:C.a2},
                 {key:"o",lab:"base optimizer state",col:C.bad},{key:"a",lab:"adapters + their grads + Adam",col:C.accent}];
    const METH=[{key:"full",lab:"full 16-bit"},{key:"lora",lab:"LoRA 16-bit"},{key:"qlora",lab:"QLoRA"}];
    function est(m,meth,r,dq){
      const lin=m.L*(4*m.d*m.d+3*m.d*m.f), emb=2*V*m.d, P=lin+emb;
      const ad=m.L*r*(4*2*m.d+3*(m.d+m.f));
      const z={w:0,c:0,g:0,o:0,a:0,P,ad};
      if(meth==="full"){z.w=2*P;z.g=2*P;z.o=8*P;}
      else if(meth==="lora"){z.w=2*P;z.a=12*ad;}
      else{z.w=lin*0.5+emb*2;z.c=lin*(dq?(8/64+32/(64*256)):32/64)/8;z.a=12*ad;}
      for(const k of ["w","c","g","o","a"])z[k]/=GB;
      z.tot=z.w+z.c+z.g+z.o+z.a;return z;
    }
    const W=760,H=380,m={l:56,r:170,t:28,b:52};
    function draw(){
      svg.selectAll("*").remove();
      const r=+d3.select("#mem-r").property("value"), dq=d3.select("#mem-dq").property("checked"), zoom=d3.select("#mem-z").property("value")==="zoom";
      const data=MODELS.map(mm=>({m:mm,v:METH.map(me=>({me,z:est(mm,me.key,r,dq)}))}));
      const ymax=zoom?160:d3.max(data,d=>d3.max(d.v,v=>v.z.tot))*1.06;
      const y=d3.scaleLinear().domain([0,ymax]).range([H-m.b,m.t]).nice();
      const x0=d3.scaleBand().domain(MODELS.map(d=>d.lab)).range([m.l,W-m.r]).paddingInner(0.22).paddingOuter(0.1);
      const x1=d3.scaleBand().domain(METH.map(d=>d.key)).range([0,x0.bandwidth()]).padding(0.12);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(8)).attr("color",C.muted);
      svg.append("text").attr("x",m.l).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("estimated GB (10⁹ bytes), activations excluded — our arithmetic"+(zoom?" · axis cut at 160 GB":""));
      const top=y.domain()[1];
      data.forEach(dd=>{
        const gx=x0(dd.m.lab);
        dd.v.forEach(({me,z})=>{
          const bx=gx+x1(me.key), bw=x1.bandwidth();let acc=0;
          PARTS.forEach(p=>{
            const v=z[p.key];if(v<=0)return;
            const a=Math.min(acc,top), b=Math.min(acc+v,top);acc+=v;if(b<=a)return;
            svg.append("rect").attr("x",bx).attr("width",bw).attr("y",y(b)).attr("height",Math.max(0.5,y(a)-y(b))).attr("fill",p.col).attr("opacity",0.9)
              .append("title").text(`${dd.m.lab} ${me.lab}: ${p.lab} ${v.toFixed(2)} GB`);
          });
          const cut=z.tot>top;
          if(cut){svg.append("path").attr("d",`M${bx-2},${y(top)+6} l${bw/2+2},-5 l${bw/2+2},5`).attr("fill","none").attr("stroke",C.ink).attr("stroke-width",1.4);}
          svg.append("text").attr("x",bx+bw/2).attr("y",y(Math.min(z.tot,top))-(cut?10:4)).attr("text-anchor","middle").attr("fill",cut?C.ink:(z.tot<=48?C.good:C.muted))
            .attr("font-size",9.5).text(z.tot>=100?Math.round(z.tot):z.tot.toFixed(1));
          svg.append("text").attr("transform",`translate(${bx+bw/2},${H-m.b+12}) rotate(0)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",8.5)
            .text(me.key==="full"?"full":me.key==="lora"?"LoRA":"QLoRA");
        });
        svg.append("text").attr("x",gx+x0.bandwidth()/2).attr("y",H-m.b+30).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12).text(dd.m.lab+" ("+(dd.v[0].z.P/1e9).toFixed(1)+"B params)");
      });
      svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(48)).attr("y2",y(48)).attr("stroke",C.good).attr("stroke-dasharray","6 4");
      svg.append("text").attr("x",W-m.r+4).attr("y",y(48)+4).attr("fill",C.good).attr("font-size",10.5).text("48 GB GPU");
      const lg=svg.append("g").attr("transform",`translate(${W-m.r+4},${m.t+4})`);
      PARTS.forEach((p,i)=>{lg.append("rect").attr("y",i*18).attr("width",11).attr("height",11).attr("fill",p.col);
        lg.append("text").attr("x",16).attr("y",i*18+10).attr("fill",C.muted).attr("font-size",10.5).text(p.lab);});
      const rows=data.map(dd=>`${dd.m.lab}: full <b>${dd.v[0].z.tot.toFixed(0)}</b> · LoRA <b>${dd.v[1].z.tot.toFixed(1)}</b> · QLoRA <b>${dd.v[2].z.tot.toFixed(1)}</b> GB`).join(" &nbsp;|&nbsp; ");
      const m65=MODELS[3], on=est(m65,"qlora",r,true).tot, off=est(m65,"qlora",r,false).tot;
      d3.select("#mem-readout").html(rows+
        `<br>adapters at r = ${r} on all 7 linear layers: ${MODELS.map(mm=>mm.lab+" "+(est(mm,"lora",r,dq).ad/1e6).toFixed(0)+"M").join(", ")} params · `+
        `65B QLoRA with DQ ${on.toFixed(1)} GB vs without ${off.toFixed(1)} GB (saves ${(off-on).toFixed(2)} GB)`+
        `<br>estimates from the stated assumptions, not measurements; the paper reports only the totals (&gt;780 GB full FT at 65B, &lt;48 GB QLoRA)`);
    }
    ["#mem-r","#mem-dq","#mem-z"].forEach(id=>d3.select(id).on("change",draw).on("input",draw));
    draw();
  })();

  /* ═══════════════ 3. reported Elo: human vs GPT-4 judge ═══════════════ */
  (function(){
    const svg=d3.select("#elo-svg"); if(svg.empty()) return;
    // the paper's Elo table: human raters on Vicuna prompts, GPT-4 on Vicuna prompts, GPT-4 on OA prompts
    const M=[["GPT-4",1176,1348,1294],["Guanaco 65B",1023,1022,1008],["Guanaco 33B",1009,992,1002],["ChatGPT",916,966,1015],
             ["Vicuna 13B",984,974,936],["Guanaco 13B",975,913,885],["Guanaco 7B",1010,879,860],["Bard",909,902,null]];
    const COL=n=>n.startsWith("Guanaco")?C.accent:n==="GPT-4"?C.violet:n==="ChatGPT"?C.a2:n==="Bard"?C.blue:C.muted;
    const W=760,H=360, xl=230, xr=530, y0=40, y1=330;
    const y=d3.scaleLinear().domain([850,1360]).range([y1,y0]);
    function ranks(vals){const idx=vals.map((v,i)=>i).filter(i=>vals[i]!=null).sort((a,b)=>vals[b]-vals[a]);const r=vals.map(()=>null);idx.forEach((i,k)=>r[i]=k+1);return r;}
    function kendall(a,b){let c=0,d=0;for(let i=0;i<a.length;i++)for(let j=i+1;j<a.length;j++){if(a[i]==null||b[i]==null||a[j]==null||b[j]==null)continue;const s=(a[i]-a[j])*(b[i]-b[j]);if(s>0)c++;else if(s<0)d++;}return (c-d)/(c+d);}
    function dodge(items,gap){items.sort((p,q)=>p.y-q.y);for(let k=1;k<items.length;k++){if(items[k].y-items[k-1].y<gap)items[k].y=items[k-1].y+gap;}return items;}
    function draw(){
      svg.selectAll("*").remove();
      const j=d3.select("#elo-j").property("value"), col=j==="g4v"?2:3;
      const jName=j==="g4v"?"GPT-4 judge, Vicuna prompts":"GPT-4 judge, OA prompts";
      svg.append("g").attr("transform",`translate(${xl},0)`).call(d3.axisLeft(y).ticks(6)).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${xr},0)`).call(d3.axisRight(y).ticks(6)).attr("color",C.muted);
      svg.append("text").attr("x",xl).attr("y",22).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12).text("human raters, Vicuna prompts");
      svg.append("text").attr("x",xr).attr("y",22).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12).text(jName);
      const hr=ranks(M.map(m=>m[1])), gr=ranks(M.map(m=>m[col]));
      const L=dodge(M.map((m,i)=>({i,y:y(m[1])})),13), R=dodge(M.filter(m=>m[col]!=null).map(m=>({i:M.indexOf(m),y:y(m[col])})),13);
      M.forEach((m,i)=>{
        const c=COL(m[0]), g=svg.append("g").style("cursor","pointer").on("mouseenter",()=>hover(i));
        if(m[col]!=null){const moved=Math.abs(hr[i]-gr[i])>=2;
          g.append("line").attr("x1",xl).attr("x2",xr).attr("y1",y(m[1])).attr("y2",y(m[col])).attr("stroke",c).attr("stroke-width",moved?2.6:1.5).attr("opacity",moved?1:0.75);
          g.append("circle").attr("cx",xr).attr("cy",y(m[col])).attr("r",4).attr("fill",c);}
        g.append("circle").attr("cx",xl).attr("cy",y(m[1])).attr("r",4).attr("fill",c);
        const ly=L.find(p=>p.i===i).y;
        g.append("text").attr("x",xl-40).attr("y",ly+4).attr("text-anchor","end").attr("fill",c).attr("font-size",11).text(`${m[0]}  ${m[1]} (#${hr[i]})`);
        const rp=R.find(p=>p.i===i);
        if(rp)g.append("text").attr("x",xr+40).attr("y",rp.y+4).attr("fill",c).attr("font-size",11).text(`#${gr[i]} ${m[col]}  ${m[0]}`);
        g.append("title").text(`${m[0]}: human ${m[1]} (rank ${hr[i]})`+(m[col]!=null?`, ${jName} ${m[col]} (rank ${gr[i]})`:", no rating for this judge"));
      });
      summary();
    }
    function summary(){
      const j=d3.select("#elo-j").property("value"), col=j==="g4v"?2:3;
      const tau=kendall(M.map(m=>m[1]),M.map(m=>m[col]));
      const n=M.filter(m=>m[col]!=null).length;
      d3.select("#elo-readout").html(`reported, Vicuna prompts, GPT-4 vs human: Kendall τ <b>0.43</b>, Spearman r <b>0.55</b> (system level) · Fleiss κ <b>0.25</b> per answer vs <b>0.42</b> among humans`+
        `<br>our arithmetic: Kendall τ over just these ${n} systems = ${tau.toFixed(2)}`+(j==="g4o"?" (note the two axes now use different prompt sets)":"")+
        ` — the reported figure need not be computed on this set · hover a model for its ranks`);
    }
    function hover(i){
      const j=d3.select("#elo-j").property("value"), col=j==="g4v"?2:3, m=M[i];
      const hr=ranks(M.map(z=>z[1])), gr=ranks(M.map(z=>z[col]));
      d3.select("#elo-readout").html(`<b>${m[0]}</b>: human raters ${m[1]} (rank <b>${hr[i]}</b> of 8)`+
        (m[col]!=null?` · GPT-4 judge ${m[col]} (rank <b>${gr[i]}</b> of ${gr.filter(v=>v!=null).length}) · Elo difference ${m[col]-m[1]>0?"+":""}${m[col]-m[1]}`:" · not rated on the OA prompts"));
    }
    d3.select("#elo-j").on("change",draw);
    draw();
  })();
})();
