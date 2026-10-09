import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(unet, text_encoder, diffusion, val_loader, device, tokenizer, amp, seed=21):
    """Media della training_loss DDPM sul validation set (analogo a 'evaluate' della segmentazione,
    ma qui non c'è un Dice score: la metrica naturale durante il training è la stessa loss."""
    unet_was_training = unet.training
    unet.eval()
    if text_encoder is not None:
        te_was_training = text_encoder.training
        text_encoder.eval()

    g = torch.Generator(device="cpu")  # CPU: funziona uguale su cuda / mps / cpu
    g.manual_seed(seed)

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch['image'].to(device=device, dtype=torch.float32)
        B = images.shape[0]

        t = torch.randint(0, diffusion.T, (B,), generator=g).to(device)
        noise = torch.randn(images.shape, generator=g).to(device)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            if text_encoder is not None:
                input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
                cond_mask = torch.ones(B, device=device)  # in validation si usa sempre il testo vero
                text_hidden, _ = text_encoder(input_ids, cond_mask)
                text_pad_mask = input_ids.eq(tokenizer.pad_id)
            else:
                text_hidden, text_pad_mask = None, None

            x_t, _ = diffusion.q_sample(images, t, noise)
            pred_noise = unet(x_t, t, text_hidden, text_pad_mask)
            loss = F.mse_loss(pred_noise.float(), noise)

            loss = diffusion.training_loss(unet, images, t, text_hidden, text_pad_mask)

        total_loss += loss.item()
        n_batches += 1

    unet.train(unet_was_training)
    if text_encoder is not None:
        text_encoder.train(te_was_training)
    return total_loss / max(n_batches, 1)