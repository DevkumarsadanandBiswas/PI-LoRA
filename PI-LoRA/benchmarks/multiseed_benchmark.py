"""Multi-dataset, multi-seed benchmark (low-rank regime, r=4).
Usage: python benchmarks/multiseed_benchmark.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from transformers import AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from pilora import PILoRATrainer, lora_config, training_args, run_single
from pilora.data import get_splits, tokenize_dataset

MODEL, MAX_LEN = "distilgpt2", 256
SEEDS = [42, 123, 2024]
SPECS = {
    "wikitext-2": dict(path="wikitext", name="wikitext-2-raw-v1", field="text"),
    "ag_news":    dict(path="ag_news", name=None, field="text"),
    "imdb":       dict(path="imdb", name=None, field="text"),
}
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
cfg = lora_config(r=4, alpha=8)

all_results = {}
for key, spec in SPECS.items():
    print(f"\n=== {key} ===")
    try:
        tr, va, te, field = get_splits(spec, 4000, 500, 500)
    except Exception as e:
        print(f"skip {key}: {e}"); continue
    tt, tv, tte = (tokenize_dataset(d, tok, field, MAX_LEN, key) for d in (tr, va, te))
    res = {"Standard LoRA": [], "PI-LoRA": []}
    for seed in SEEDS:
        args = training_args(f"./out/{key}_{seed}", seed, epochs=2, warmup=30,
                             logging_steps=200, disable_tqdm=True)
        for name, cls in (("Standard LoRA", Trainer), ("PI-LoRA", PILoRATrainer)):
            r = run_single(name, cls, MODEL, cfg, args, tt, tv, tte, collator, seed)
            print(f"  seed {seed} {name:<14} test_ppl={r['test_ppl']:.3f}")
            res[name].append(r)
    all_results[key] = res

print("\n=== SUMMARY (mean ± std test PPL) ===")
summary = {}
for key, res in all_results.items():
    m = {n: (float(np.mean([r["test_ppl"] for r in rs])), float(np.std([r["test_ppl"] for r in rs])))
         for n, rs in res.items()}
    d = 100 * (m["Standard LoRA"][0] - m["PI-LoRA"][0]) / m["Standard LoRA"][0]
    print(f"{key:<12} LoRA={m['Standard LoRA'][0]:.3f}±{m['Standard LoRA'][1]:.3f}  "
          f"PI-LoRA={m['PI-LoRA'][0]:.3f}±{m['PI-LoRA'][1]:.3f}  Δ={d:+.2f}%")
    summary[key] = {**{n: {"mean": v[0], "std": v[1]} for n, v in m.items()}, "delta_pct": d}
os.makedirs("results", exist_ok=True)
with open("results/multiseed_results.json", "w") as f:
    json.dump({"raw": all_results, "summary": summary}, f, indent=2)
