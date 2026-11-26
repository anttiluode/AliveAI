/* AliveAI: explicit adaptive control on a frozen associative bank. */
const Alive=(function(P){
 const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
 const defaults={k:12,feedback:.48,anchor:.18,temperature:.09,coverage:.16,fatigue:.44,burn:.12,recovery:.015,spread:.55};
 const mean=(b,ids)=>P.mix(b.d,ids.map(i=>[b.x[i],1]));
 // Hash normalized coordinates and exact word order, not just a model name.
 function signature(b){let h=2166136261;const add=s=>{for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619)};for(let i=0;i<b.words.length;i++){add(JSON.stringify(b.words[i]));for(const x of b.x[i])add(x.toPrecision(12)+",")}return b.words.length+":"+b.d+":"+(h>>>0).toString(16)}
 class Memory{
  constructor(b){this.b=b;this.signature=signature(b);this.records=[]}
  export(){return {version:1,bank:this.signature,records:this.records.map(r=>({...r,cues:r.cues.slice(),tag:r.tag.slice(),state:r.state.slice(),package:r.package.slice()}))}}
  load(raw){if(raw?.version!==1||raw.bank!==this.signature||!Array.isArray(raw.records)||raw.records.length>512)throw Error("Memory does not match this bank or format");
   const valid=i=>Number.isInteger(i)&&i>=0&&i<this.b.words.length;
   for(const r of raw.records){if(![r.cues,r.tag,r.package].every(a=>Array.isArray(a)&&a.every(valid))||!r.cues.length||!r.package.length||!Array.isArray(r.state)||r.state.length!==this.b.d||!r.state.every(Number.isFinite)||!valid(r.action)||!Number.isFinite(r.reward)||Math.abs(r.reward)>1||!Number.isFinite(r.expected)||Math.abs(r.expected)>1)throw Error("Invalid memory record")}
   this.records=raw.records.map(r=>({cues:r.cues.slice(),tag:r.tag.slice(),package:r.package.slice(),state:P.unit(r.state),action:r.action,reward:r.reward,expected:r.expected}));}
  add(r){this.records.push(r);if(this.records.length>512)this.records.shift()}
  matches(cues,tag,state){
   const q=mean(this.b,cues),t=tag.length?mean(this.b,tag):null;
   return this.records.map(r=>{
    const c=P.dot(q,mean(this.b,r.cues)),a=P.dot(state,r.state),v=t&&r.tag.length?P.dot(t,mean(this.b,r.tag)):(!t&&!r.tag.length?1:0);
    // Explicit context gate. No universal good/bad word score.
    const fit=c>.35&&v>.35?clamp(c)*(.5+.5*clamp(a))*clamp(v):0;
    return {r,fit};
   }).filter(z=>z.fit>0).sort((a,b)=>b.fit-a.fit).slice(0,16);
  }
 }
 class Agent{
  constructor(b,cues,{memory=null,tag=[],learn=true,useMemory=true,regulate=true,strength=.35,reader={}}={}){
   this.b=b;this.memory=memory||new Memory(b);this.learn=learn;this.useMemory=useMemory;this.regulate=regulate;this.strength=strength;
   this.p=new P.Player(b,cues,{...defaults,...reader,tag});this.base={...this.p.opts};
   this.value=0;this.patience=.5;this.novelty=0;this.lastError=0;this.pending=null;this.last=null;this.events=[];this.used=new Set();this.seen=new Set();this.total=0;
  }
  step(){
   // A new observation expires any unscored action; no imaginary reward.
   this.pending=null;
   const p=this.p,b=this.b,matched=this.useMemory?this.memory.matches(p.cues,p.tagIds,p.x):[];
   const bias=new Float64Array(b.words.length);let mass=0;
   for(const {r,fit} of matched){mass+=fit;bias[r.action]+=fit*r.reward;for(const i of r.package)bias[i]+=.25*fit*r.reward/r.package.length}
   if(mass)for(let i=0;i<bias.length;i++)bias[i]=this.strength*clamp(bias[i]/Math.max(1,mass),-1,1);
   p.selectionBias=bias;
   p.opts.burn=this.base.burn*(this.regulate?clamp(1.5-this.patience,.35,1.5):1);
   const r=p.step();this.total++;
   this.novelty=.9*this.novelty+.1*r.novel/Math.max(1,r.workspace.length);
   for(const w of r.workspace)this.seen.add(w.i);
   this.last={...r,value:this.value,patience:this.patience,novelty:this.novelty,predictionError:this.lastError,recalled:matched.length,burn:p.opts.burn};
   // Bound live history; keep cumulative counts separately.
   if(p.history.length>1000)p.history.shift();
   return this.last;
  }
  nominate(i){
   if(!this.last||!this.last.workspace.some(w=>w.i===i))throw Error("Choose a concept from the current workspace");
   if(this.used.has(i))throw Error("Already evaluated this concept in this episode");
   this.pending={action:i,cues:this.p.cues.slice(),tag:this.p.tagIds.slice(),state:this.last.query.slice(),package:this.last.workspace.map(w=>w.i),tick:this.total};return i;
  }
  feedback(reward){
   if(!this.pending||!Number.isFinite(reward)||Math.abs(reward)>1)throw Error("Nominate first; reward must be in [-1,1]");
   const r=this.pending;this.pending=null;this.used.add(r.action);
   const expected=this.value,delta=reward-expected;this.lastError=delta;this.value=clamp(expected+.2*delta,-1,1);
   if(this.regulate){
    this.patience=clamp(this.patience+.15*delta,.1,.95);
    // Replenish only on positive external feedback, not novelty itself.
    if(reward>0)for(const i of r.package)this.p.energy[i]=clamp(this.p.energy[i]+.2*reward);
   }
   if(this.learn)this.memory.add({...r,reward,expected});
   const event={tick:r.tick,action:r.action,reward,expected,delta,patience:this.patience};this.events.push(event);if(this.events.length>1000)this.events.shift();return event;
  }
  setTag(tag){this.pending=null;this.p.setTag(tag)}
  // New sensory context without discarding the state, resource field or memories.
  cue(cues){if(!cues.length||cues.length>=this.b.words.length)throw Error("Invalid cues");const p=this.p;p.cues=cues.slice();p.seed=mean(this.b,cues);p.exclude=new Set(cues);p.baseline=new Set(cues.flatMap(i=>P.rank(this.b,this.b.x[i],p.opts.k,this.b.d,new Set([i])).map(z=>z.i)));p.x=P.mix(this.b.d,[[p.x,.6],[p.seed,.4]]);this.pending=null;this.used.clear();this.last=null}
 }
 // Independent answer lists; evaluator is the only object that knows answers.
 function task(b,raw){
  const cues=P.ids(b,raw.cues),answers=P.ids(b,raw.answers),tag=P.ids(b,raw.tag||"");
  if(!cues.length||!answers.length||answers.some(i=>cues.includes(i)))throw Error("Invalid task or answer leaked as cue");
  const set=new Set(answers);
  return {cues,tag,evaluate:i=>set.has(i)?1:-.15};
 }
 function trial(b,t,{memory,learn=false,useMemory=false,regulate=true,mode="recurrent",budget=40,beats=80}={}){
  const a=new Agent(b,t.cues,{memory,learn,useMemory,regulate,tag:t.tag,reader:{mode}});
  let calls=0,found=false,answer=null;
  for(let j=0;j<beats&&calls<budget&&!found;j++){
   const r=a.step();
   // At most one external evaluation per beat, never repeat an answer.
   const z=r.workspace.find(w=>!a.used.has(w.i));
   if(!z)continue;
   a.nominate(z.i);const v=t.evaluate(z.i);a.feedback(v);calls++;
   if(v===1){found=true;answer=b.words[z.i]}
  }
  return {found,answer,calls,beats:a.total,distinct:a.seen.size};
 }
 function staticTrial(b,t,budget){
  const rows=P.rank(b,mean(b,t.cues),budget,b.d,new Set(t.cues));let calls=0;
  for(const z of rows){calls++;if(t.evaluate(z.i)===1)return {found:true,answer:b.words[z.i],calls,beats:0}}
  return {found:false,answer:null,calls,beats:0};
 }
 function benchmark(b,raw,{budget=40,beats=80,epochs=2}={}){
  if(!Array.isArray(raw.train)||!Array.isArray(raw.test)||!raw.train.length||!raw.test.length)throw Error("Need train and test tasks");
  const train=raw.train.map(r=>task(b,r)),test=raw.test.map(r=>task(b,r));
  const key=t=>t.cues.slice().sort((a,b)=>a-b).join(",");
  if(test.some(t=>train.some(s=>key(s)===key(t))))throw Error("Train/test cue overlap");
  const mem=new Memory(b),training=[];
  for(let e=0;e<epochs;e++)for(const t of train)training.push(trial(b,t,{memory:mem,learn:true,useMemory:true,budget,beats}));
  const before=JSON.stringify(mem.export()),arms={};
  for(const [name,ts] of [["heldout",test],["seen",train]]){
   arms[name]={};
   for(const arm of ["static","reader","regulated","learned"]){
    const rows=ts.map(t=>arm==="static"?staticTrial(b,t,budget):trial(b,t,{memory:mem,learn:false,useMemory:arm==="learned",regulate:arm!=="reader",budget,beats}));
    arms[name][arm]={success:rows.filter(r=>r.found).length,total:rows.length,meanCalls:rows.reduce((s,r)=>s+r.calls,0)/rows.length,meanBeats:rows.reduce((s,r)=>s+r.beats,0)/rows.length,rows};
   }
  }
  if(JSON.stringify(mem.export())!==before)throw Error("Evaluation changed memory");
  return {budget,beats,epochs,training,memoryRecords:mem.records.length,arms};
 }
 function selfTest(){
  const b=P.prepare(P.demo()),c=P.ids(b,"bird,shadow,winter"),m=new Memory(b),before=JSON.stringify(b.x),a=new Agent(b,c,{memory:m}),r=a.step(),i=r.workspace[0].i;
  a.nominate(i);const e=a.feedback(1);if(e.delta!==1||m.records.length!==1||a.patience<=.5)throw Error("Positive feedback");
  let ok=false;try{a.feedback(1)}catch{ok=true}if(!ok)throw Error("Double feedback");
  const clone=new Memory(b);clone.load(JSON.parse(JSON.stringify(m.export())));if(clone.records.length!==1||clone.records[0].action!==i||clone.records[0].reward!==1||P.dot(clone.records[0].state,m.records[0].state)<1-1e-9)throw Error("Persistence");
  const learned=new Agent(b,c,{memory:m}),blank=new Agent(b,c,{memory:m,useMemory:false});learned.step();blank.step();
  if(learned.p.selectionBias[i]<=0||blank.p.selectionBias[i]!==0)throw Error("Memory is not in selection");
  const tag=new Agent(b,c,{memory:m,tag:P.ids(b,"music")});tag.step();if(tag.p.selectionBias.some(x=>x!==0))throw Error("Context gate");
  const n=new Agent(b,c,{memory:m});let z=n.step();n.nominate(z.workspace[0].i);n.feedback(-1);if(n.patience>=.5||m.records.at(-1).reward!==-1)throw Error("Negative feedback");
  const count=m.records.length;n.cue(P.ids(b,"music,song"));n.step();if(m.records.length!==count||JSON.stringify(b.x)!==before)throw Error("Cue or bank mutation");
  const t={train:[{cues:"bird,winter",answers:"snow"}],test:[{cues:"oak,tree",answers:"wood"}]};const bench=benchmark(b,t,{budget:5,beats:10,epochs:1});if(Object.keys(bench.arms.heldout).length!==4)throw Error("Benchmark");
  return [...P.selfTest(),"signed external feedback and one-shot credit","memory round-trip and selection bias","context gate","cue injection with frozen bank","held-out frozen-memory comparison"];
 }
 return {defaults,Memory,Agent,benchmark,selfTest,signature};
})(typeof module!=="undefined"?require("./play.js"):Play);
if(typeof module!=="undefined")module.exports=Alive;
