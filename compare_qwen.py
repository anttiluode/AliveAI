"""Optional native frozen-Qwen reference, one free generated answer per task."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="tasks.json")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out", default="alive-results-qwen.json")
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if a.device.startswith("cuda") else torch.float32,
        local_files_only=a.local_files_only).to(a.device).eval()
    model.requires_grad_(False)
    tasks = json.loads(Path(a.tasks).read_text(encoding="utf-8"))
    results = {}
    for split, rows in tasks.items():
        out = []
        for row in rows:
            cues = row["cues"]
            context = ("Context: " + row["tag"] + "\n") if row.get("tag") else ""
            question = context + "Find one word associated with all these clues. Answer with just the word: " + cues
            prompt = tok.apply_chat_template([{"role":"user","content":question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(a.device)
            with torch.inference_mode():
                seq = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            text = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            answer = text.strip(" \n\t.!?\"'").casefold()
            accepted = {x.strip().casefold() for x in row["answers"].split(",")}
            out.append({"cues":cues,"generated":text,"correct":answer in accepted})
            print(cues, "=>", text, out[-1]["correct"], flush=True)
        results[split] = {"success":sum(r["correct"] for r in out),"total":len(out),"rows":out}
    Path(a.out).write_text(json.dumps({"model":a.model,"results":results},indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()
