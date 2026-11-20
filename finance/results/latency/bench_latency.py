# Controlled timing benchmark for Laya-Finance (J checkpoint).
# Usage: python bench_latency.py <cuda|cpu> <items_per_task> <batch_sizes e.g. 1,32> <out.json>
# Reads test rows from the private multitask data folder (paths below); only timings are written out.
import json, time, statistics, subprocess, platform, sys, importlib.metadata as md
from pathlib import Path
import torch, laya

MT = Path("/home/ravichandran/Documents/GitHub/ollama-trading-optimizer/data/laya_multitask")
CKPT = "/home/ravichandran/Documents/GitHub/ollama-trading-optimizer/scripts/laya_finetune/runs/mt_j/final"
TASKS = ["sentiment/test_phrasebank", "topic/test", "fomc/test", "headline/test", "term_domain/test", "trading_style/test_rare"]
device, n_items, batches, repeats = sys.argv[1], int(sys.argv[2]), [int(b) for b in sys.argv[3].split(",")], 3
out_path = sys.argv[4]

def rows_for(task):
    rows = [json.loads(l) for l in open(MT / (task + ".jsonl"))]
    return rows[:n_items]

def gpu_busy():
    try:
        return subprocess.run(["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=20).stdout.strip()[-300:]
    except Exception as e:
        return f"unavailable: {e}"

agent = laya.load(CKPT, device=device)
data = {t: rows_for(t) for t in TASKS}
def run(rows, bs):
    groups = {}
    for r in rows:
        groups.setdefault(json.dumps(r["q"], sort_keys=True), []).append(r)
    t0 = time.perf_counter()
    for g in groups.values():
        for s in range(0, len(g), bs):
            chunk = g[s:s+bs]
            agent.predict_batch([r["text"] for r in chunk], {"a": chunk[0]["q"]}, batch_size=bs)
    if device == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t0

# warm-up on the first task
run(data[TASKS[0]][:64], 32)
res = {"meta": {
    "date": time.strftime("%F %T"), "device": device,
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine(),
    "torch": torch.__version__, "hip": torch.version.hip, "laya": md.version("laya"),
    "python": platform.python_version(), "checkpoint": "runs/mt_j/final (Laya-Finance J)",
    "items_per_task": n_items, "repeats": repeats, "threads": torch.get_num_threads(),
    "gpu_busy_before": gpu_busy() if device == "cuda" else None,
}, "results": {}}
for bs in batches:
    for t in TASKS:
        times = [run(data[t], bs) for _ in range(repeats)]
        ms = [1000 * x / len(data[t]) for x in times]
        res["results"][f"bs{bs}/{t}"] = {"n": len(data[t]), "ms_per_item_median": round(statistics.median(ms), 2), "ms_per_item_all": [round(m, 2) for m in ms]}
        print(f"bs{bs} {t:28s} median {statistics.median(ms):8.2d}" if False else f"bs{bs} {t:28s} median {statistics.median(ms):8.2f} ms/item", flush=True)
res["meta"]["gpu_busy_after"] = gpu_busy() if device == "cuda" else None
Path(out_path).write_text(json.dumps(res, indent=2))
print("wrote", out_path)
