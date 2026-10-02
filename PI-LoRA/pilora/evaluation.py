import math
import torch


@torch.no_grad()
def compute_ppl(model, dataset, collator, device, batch_size=16):
    """Perplexity over non-masked label positions only.

    IMPORTANT: do not overwrite batch["labels"] with input_ids. The collator
    masks padding with -100; overwriting counts padding as targets and inflates
    PPL far beyond the random-guess level (this was the v1 bug).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    total_loss, total_tok = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        if out.loss is not None and not torch.isnan(out.loss):
            n = (batch["labels"] != -100).sum().item()
            if n > 0:
                total_loss += out.loss.item() * n
                total_tok += n
    if total_tok == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tok, 100))
