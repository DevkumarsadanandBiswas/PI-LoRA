"""PI-LoRA vs Standard LoRA on WikiText-2 (distilgpt2, r=16). Single seed.
Usage: python benchmarks/efficiency_benchmark.py
"""
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from pilora import PILoRATrainer, compute_ppl, lora_config, training_args, run_single
from pilora.data import get_splits, tokenize_dataset
from pilora.runner import DEVICE, load_base, set_seed

MODEL, MAX_LEN, SEED = "distilgpt2", 256, 42
SPEC = dict(path="wikitext", name="wikitext-2-raw-v1", field="text")

set_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

train, valid, test, field = get_splits(SPEC)
tt = tokenize_dataset(train, tok, field, MAX_LEN, "train")
tv = tokenize_dataset(valid, tok, field, MAX_LEN, "valid")
te = tokenize_dataset(test, tok, field, MAX_LEN, "test")

m = load_base(MODEL)
zs = compute_ppl(m, te, collator, DEVICE)
print(f"Zero-shot test PPL: {zs:.3f}  (expected ~20-80; >500 means a PPL bug)")
del m; gc.collect()

cfg = lora_config(r=16, alpha=32)
args = training_args("./out/eff", SEED)
results = [
    run_single("Standard LoRA", Trainer, MODEL, cfg, args, tt, tv, te, collator, SEED),
    run_single("PI-LoRA", PILoRATrainer, MODEL, cfg, args, tt, tv, te, collator, SEED),
]
a, b = results
gain = 100 * (a["test_ppl"] - b["test_ppl"]) / a["test_ppl"]
print(f"\n{'Method':<16}{'TestPPL':>10}{'ValidPPL':>10}{'Time(s)':>10}{'Mem(MB)':>10}")
for r in results:
    print(f"{r['method']:<16}{r['test_ppl']:>10.3f}{r['valid_ppl']:>10.3f}"
          f"{r['train_time_sec']:>10.1f}{r['peak_mem_mb']:>10.0f}")
print(f"\nPI-LoRA test-PPL improvement vs LoRA: {gain:+.2f}%")
os.makedirs("results", exist_ok=True)
with open("results/efficiency_results.json", "w") as f:
    json.dump({"zero_shot_test_ppl": zs, "results": results, "ppl_gain_pct": gain}, f, indent=2)
