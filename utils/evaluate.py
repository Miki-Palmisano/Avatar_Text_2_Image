import torch

@torch.no_grad()
def evaluate(unet, text_encoder, diffusion, val_loader, device, tokenizer, amp):
    """Media della training_loss DDPM sul validation set (analogo a 'evaluate' della segmentazione,
    ma qui non c'è un Dice score: la metrica naturale durante il training è la stessa loss."""
    unet.eval()
    text_encoder.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch['image'].to(device=device, dtype=torch.float32)
        input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
        B = images.shape[0]
        cond_mask = torch.ones(B, device=device)  # in validation usiamo sempre il testo vero

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            text_hidden, _ = text_encoder(input_ids, cond_mask)
            text_pad_mask = input_ids.eq(tokenizer.pad_id)
            t = torch.randint(0, diffusion.T, (B,), device=device).long()
            loss = diffusion.training_loss(unet, images, t, text_hidden, text_pad_mask)

        total_loss += loss.item()
        n_batches += 1

    unet.train()
    text_encoder.train()
    return total_loss / max(n_batches, 1)