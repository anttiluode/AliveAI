const fs=require("node:fs"),P=require("./play.js"),A=require("./alive.js");
const args=process.argv.slice(2);
if(args.includes("--self-test")){console.log(A.selfTest().join("\n"));process.exit(0)}
const opt={};for(let i=0;i<args.length;i+=2){if(!args[i].startsWith("--")||args[i+1]===undefined)throw Error("Use --key value");opt[args[i].slice(2)]=args[i+1]}
for(const k of Object.keys(opt))if(!["bank","tasks","budget","beats","epochs","out","cues","tag","ticks","memory"].includes(k))throw Error("Unknown option "+k);
const n=(k,d)=>{const x=Number(opt[k]??d);if(!Number.isInteger(x)||x<1||x>10000)throw Error("Invalid "+k);return x};
const b=P.prepare(opt.bank==="demo"?P.demo():JSON.parse(fs.readFileSync(opt.bank||"bank.json","utf8")));
let report;
if(opt.cues){const m=new A.Memory(b);if(opt.memory)m.load(JSON.parse(fs.readFileSync(opt.memory,"utf8")));const a=new A.Agent(b,P.ids(b,opt.cues),{tag:P.ids(b,opt.tag||""),memory:m});for(let i=0;i<n("ticks",100);i++){const r=a.step();console.log(r.tick,r.workspace.map(w=>b.words[w.i]).join(", "))}report={mode:"free",meta:b.meta,words:b.words,distinct:a.seen.size,total:a.total,trace:a.p.history,memory:m.export()};}
else{const tasks=opt.tasks?JSON.parse(fs.readFileSync(opt.tasks,"utf8")):opt.bank==="demo"?{train:[{cues:"bird,winter",answers:"snow"},{cues:"guitar,piano",answers:"music"}],test:[{cues:"bird,shadow",answers:"sky"},{cues:"song,voice",answers:"music"},{cues:"oak,tree",answers:"wood"}]}:JSON.parse(fs.readFileSync("tasks.json","utf8"));report=A.benchmark(b,tasks,{budget:n("budget",40),beats:n("beats",80),epochs:n("epochs",2)});report.meta=b.meta;report.tasks=tasks;for(const [split,arms] of Object.entries(report.arms)){console.log(split);console.table(Object.entries(arms).map(([arm,r])=>({arm,success:r.success,total:r.total,meanCalls:r.meanCalls,meanBeats:r.meanBeats})));}}
const out=opt.out||"alive-results.json";fs.writeFileSync(out,JSON.stringify(report,null,2));console.log("Wrote",out);
