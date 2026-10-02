# Path Integral LoRA (PI-LoRA)

![architecture](docs/pilora_architecture.png)

PI-LoRA keeps the **standard LoRA forward pass** (`h' = W h + B(A h)`, W frozen,
nothing between A and B) and changes only the **training loop**:

1. Forward pass, task loss `L_LM` (cross-entropy)
2. Add L2 penalty: `L = L_LM + λ · mean(A² + B²)`
3. Backward + optimizer step (A, B only)
4. Anneal temperature `T`, then inject noise into A, B after the step:
   `noise_std = sqrt(2·lr·T)`, scaled by `1 / (1 + log1p(‖grad‖))`

Trainable parameters, inference cost and adapter size are identical to LoRA.

## Install & run

```bash
pip install -r requirements.txt
python benchmarks/efficiency_benchmark.py   # WikiText-2, r=16, ~50 min on a T4
python benchmarks/multiseed_benchmark.py    # 3 datasets x 3 seeds, r=4
```

Use it in your own code:

```python
from pilora import PILoRATrainer
trainer = PILoRATrainer(model=peft_model, args=args, train_dataset=ds,
                        data_collator=collator,
                        temperature=1e-5, anneal_rate=0.9999, lambda_l2=1e-6)
trainer.train()
```

## Results (Tesla T4, distilgpt2, corrected evaluation)

Single seed, WikiText-2, r=16 (811,008 trainable params, 0.98%):

| Method | Test PPL | Valid PPL | Time (s) | Peak mem (MB) |
|---|---|---|---|---|
| Zero-shot | 74.84 | – | – | – |
| Standard LoRA | 36.227 | 34.755 | 1428.8 | 3294 |
| PI-LoRA | 36.269 | 34.785 | 1445.5 | 3176 |

Multi-seed, r=4, test PPL mean ± std over seeds 42/123/2024:

| Dataset | Standard LoRA | PI-LoRA | Δ |
|---|---|---|---|
| WikiText-2 | 45.341 ± 0.091 | 45.375 ± 0.114 | -0.08% (tie) |
| AG News | 57.870 ± 0.191 | 58.442 ± 0.161 | -0.99% (LoRA better) |
| IMDB | 48.504 ± 0.032 | 48.497 ± 0.043 | +0.02% (tie) |

**Honest summary:** with the current hyperparameters PI-LoRA is statistically
indistinguishable from standard LoRA (slightly worse on AG News). It does not
yet show a benefit.

## Known caveats / things to investigate

- **Evaluation bug fixed (v2).** Early runs overwrote collator labels with
  `input_ids`, counting padding as targets (PPL in the thousands). Those numbers
  are invalid and were removed.
- **The perturbation is very small.** With lr=3e-4, T=1e-5: `noise_std ≈ 8e-5`,
  and λ=1e-6 makes the L2 term negligible, so near-identical results to LoRA are
  expected. Try larger `temperature` / `lambda_l2` sweeps.
- **Noise frequency.** By default noise is injected every micro-batch, not only
  every optimizer step (gradient accumulation = 2). Set `noise_on_sync_only=True`
  to inject only on real optimizer steps.
- **Logged loss ~2x higher for PI-LoRA** (≈7.4 vs ≈3.7), while eval PPL matches.
  Likely a gradient-accumulation scaling artifact in Trainer logging with the
  overridden `compute_loss`/`training_step`; verify before reporting training curves.
- Small model (distilgpt2), short training, subsampled data. Do not generalize.

## Layout

```
pilora/        trainer, evaluation, data helpers, runner
benchmarks/    efficiency + multi-seed scripts
results/       reported numbers (JSON)
docs/          architecture diagram
```

## License
MIT
