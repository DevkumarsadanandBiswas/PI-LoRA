"""PI-LoRA trainer.

Forward pass is identical to standard LoRA. Per training step:
  1. Forward pass, task loss (cross-entropy)
  2. Add L2 penalty:  loss += lambda * mean(A^2 + B^2)
  3. Backward + optimizer step (A, B only; W frozen)
  4. Anneal temperature T, then inject noise into A, B (post-step):
         noise_std = sqrt(2 * lr * T)
         scale     = 1 / (1 + log1p(grad_norm))
         A, B     += N(0, 1) * noise_std * scale
"""
import math
import torch
from transformers import Trainer


class PILoRATrainer(Trainer):
    def __init__(self, *args,
                 temperature: float = 1e-5,
                 anneal_rate: float = 0.9999,
                 lambda_l2: float = 1e-6,
                 min_temp: float = 1e-8,
                 noise_on_sync_only: bool = False,
                 **kwargs):
        """
        noise_on_sync_only: if True, inject noise only on real optimizer steps
            (every `gradient_accumulation_steps` micro-batches). The default
            (False) reproduces the benchmarked behaviour: noise every micro-batch.
        """
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.anneal_rate = anneal_rate
        self.lambda_l2 = lambda_l2
        self.min_temp = min_temp
        self.noise_on_sync_only = noise_on_sync_only

    # Step 1 + 2
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        l2_reg = sum(p.float().pow(2).mean()
                     for p in model.parameters() if p.requires_grad)
        total = outputs.loss + self.lambda_l2 * l2_reg
        return (total, outputs) if return_outputs else total

    # Step 3 + 4
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.temperature = max(self.temperature * self.anneal_rate, self.min_temp)

        if self.noise_on_sync_only and not self.accelerator.sync_gradients:
            return loss

        with torch.no_grad():
            lr = self.optimizer.param_groups[0]["lr"]
            noise_std = math.sqrt(2.0 * lr * self.temperature)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    gn = torch.norm(p.grad).item()
                    scale = 1.0 / (1.0 + math.log1p(max(gn, 1e-8)))
                    p.add_(torch.randn_like(p) * noise_std * scale)
        return loss
