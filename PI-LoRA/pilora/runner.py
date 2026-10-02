import gc, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from .evaluation import compute_ppl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def lora_config(r=16, alpha=32, dropout=0.05, target_modules=("c_attn", "c_proj")):
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=list(target_modules),
                      lora_dropout=dropout, bias="none", task_type=TaskType.CAUSAL_LM)


def training_args(out_dir, seed, epochs=3, lr=3e-4, warmup=50, batch_size=8,
                  grad_accum=2, logging_steps=50, disable_tqdm=False):
    return TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        warmup_steps=warmup, weight_decay=0.01, lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, logging_steps=logging_steps,
        save_strategy="no", eval_strategy="no", fp16=(DEVICE == "cuda"),
        optim="adamw_torch", report_to="none", seed=seed, disable_tqdm=disable_tqdm)


def load_base(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(DEVICE)


def run_single(name, trainer_cls, model_name, lora_cfg, args, tok_train, tok_valid,
               tok_test, collator, seed, trainer_kwargs=None):
    """Train one method once; return a metrics dict."""
    set_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    model = get_peft_model(load_base(model_name), lora_cfg)
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())

    trainer = trainer_cls(model=model, args=args, train_dataset=tok_train,
                          eval_dataset=tok_valid, data_collator=collator,
                          **(trainer_kwargs or {}))
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    steps = int(trainer.state.global_step)
    peak = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0

    result = {
        "method": name, "seed": seed,
        "trainable_params": train_p, "total_params": total_p,
        "trainable_pct": round(100 * train_p / total_p, 4),
        "train_time_sec": round(elapsed, 1), "steps": steps,
        "steps_per_sec": round(steps / elapsed, 4) if elapsed > 0 else 0,
        "peak_mem_mb": round(peak, 1),
        "valid_ppl": round(compute_ppl(model, tok_valid, collator, DEVICE), 4),
        "test_ppl": round(compute_ppl(model, tok_test, collator, DEVICE), 4),
    }
    del model, trainer
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result
