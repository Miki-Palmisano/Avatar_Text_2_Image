"""
Train script for the tiny text-conditioned avatar diffusion model.

This module prepares datasets, initializes logging with WANDB, sets up training parameters,
and handles model training, evaluation, and checkpoint saving.
"""

import argparse
import json
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stdout
)
import torch
import torch.nn as nn
from pathlib import Path

import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

from utils.data_loading import AvatarDataset
from utils.evaluate import evaluate

from tokenizer import Tokenizer
from text_encoder import TextEncoder
from unet import UNet
from diffusion import GaussianDiffusion

import os
os.environ["WANDB_MODE"] = "offline"

dir_img = Path('./dataset/cartoonset100k')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_split_ids(split_file: str, split_name: str):
    """split_file is the JSON saved during preprocessing, e.g.:
       {"train": [...ids...], "val": [...], "test": [...], "ood_compositional": [...]}"""
    with open(split_file) as f:
        splits = json.load(f)
    return splits[split_name]


def build_tokenizer(args, train_ids):
    """Build (or load) the tokenizer, guaranteeing the vocab only ever sees
    training-split captions, per the assignment's requirement."""
    tok_path = Path(args.output_dir) / "tokenizer.json"
    if tok_path.exists() and not args.rebuild_tokenizer:
        return Tokenizer.load(str(tok_path))

    # Build a throwaway dataset restricted to the TRAIN ids only, purely to
    # harvest captions for vocab construction (no val/test text is touched).
    from utils.data_loading import parse_attribute_legend, parse_image_attributes, build_deterministic_caption

    legend = parse_attribute_legend(args.attribute_legend_path)
    metadata, _id_to_path = parse_image_attributes(args.image_attribute_path, legend)  # FIX: parse_image_attributes ritorna una TUPLA (metadata, id_to_path) — va spacchettata
    train_captions = [
        build_deterministic_caption(metadata[i]) for i in train_ids if i in metadata
    ]

    tok = Tokenizer(max_len=args.max_caption_len)
    tok.build_vocab(train_captions)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_path))
    logging.info(f"Built tokenizer: vocab_size={tok.vocab_size} from {len(train_captions)} train captions")
    return tok


def timed_step(fn, device, *args, **kwargs):
    """Misura il tempo di una singola operazione: torch.cuda.Event su CUDA
    (accurato), fallback su time.time() per MPS/CPU (meno preciso ma non crasha)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end)
    else:
        import time
        t0 = time.time()
        result = fn(*args, **kwargs)
        return result, (time.time() - t0) * 1000  # ms


def train_model(
        train_loader,
        val_loader,
        unet,
        text_encoder,
        diffusion,           # GaussianDiffusion, già istanziata fuori
        tokenizer,            # per pad_id e per salvare il vocab nel checkpoint
        device,
        run_name: str = "run",
        dir_checkpoint: Path = Path("checkpoints"),
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 2e-4,
        save_checkpoint: bool = True,
        save_every: int = 2,
        amp: bool = False,
        weight_decay: float = 1e-4,
        gradient_clipping: float = 1.0,
        uncond_prob: float = 0.1,     # 1.0 = baseline unconditional, es. 0.1 = modello conditioned
        sample_prompt_ids: torch.Tensor = None,  # (1, T) token ids per il sample "spia" ad ogni eval
        load_path: str = None,        # checkpoint da cui caricare i pesi (None = training da zero)
        resume: bool = False,         # se True, riprende anche optimizer/scheduler/epoca (richiede load_path)
        profile: bool = False,        # se True, misura data/forward/backward/optimizer una tantum prima del training
):
    """
    Train U-Net + text encoder con l'obiettivo DDPM (MSE sul rumore predetto).

    Fix consolidati rispetto alle versioni precedenti:
      - grad_scaler ora condizionale su (device=='cuda' AND amp) invece di
        essere hardcoded enabled=False — con AMP attivo ma senza grad
        scaling, il forward gira in FP16 senza la protezione da under/overflow
        che il GradScaler fornisce: causa plausibile degli spike di validation
        loss (fino a 66+) osservati nei log precedenti.
      - `model`/`unet` allineati: la funzione riceveva `model` ma il corpo
        usava `unet` (variabile globale) — funzionava per coincidenza di
        scope, ora è coerente sul parametro `unet`.
      - `args.load`/`args.resume` non più letti da una globale: parametri
        espliciti `load_path`/`resume`.
      - Blocco di profiling isolato e opzionale (flag `profile`), NON dentro
        il loop di training (dove causava un secondo backward sullo stesso
        grafo e rompeva persistent_workers).
      - `save_every` è un parametro, non più hardcoded.
    """
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    use_cuda_amp = (device.type == "cuda") and amp
    autocast_device = device.type if device.type != "mps" else "cpu"

    experiment = wandb.init(
        entity="shadow",
        project="U-Net",
        name=run_name,
        resume="allow",
    )
    experiment.config.update(dict(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        save_checkpoint=save_checkpoint, amp=amp, uncond_prob=uncond_prob,
        timesteps=diffusion.T,
    ))

    logging.info(f'''Starting training:
            Run name:         {run_name}
            Epochs:            {epochs}
            Batch size:        {batch_size}
            Learning rate:     {learning_rate}
            Training size:     {n_train}
            Validation size:   {n_val}
            Checkpoints:       {save_checkpoint}
            Device:            {device.type}
            Mixed Precision:   {amp} (grad scaler attivo: {use_cuda_amp})
            Uncond prob:       {uncond_prob}
        ''')

    params = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=use_cuda_amp)
    global_step = 0

    # --- Load model checkpoint if specified ---
    start_epoch = 1
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        unet.load_state_dict(checkpoint['unet_state'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state'])
        logging.info(f"Checkpoint caricato, chiavi disponibili: {list(checkpoint.keys())}")

        if resume:
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                logging.warning("Nessun optimizer_state nel checkpoint: optimizer riparte da zero nonostante --resume")
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            else:
                logging.warning("Nessun scheduler_state nel checkpoint: scheduler riparte senza memoria del plateau")
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f'Resuming training from epoch {start_epoch}')
        else:
            logging.info(f'Loaded weights only from {load_path} (fresh optimizer/epoch)')

    # --- Profiling isolato, opzionale ---
    if profile:
        data_iter = iter(train_loader)
        n_warmup, n_measure = 2, 5

        for _ in range(n_warmup):
            batch = next(data_iter)
            images_dbg = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            input_ids_dbg = batch["input_ids"].to(device=device, dtype=torch.long)
            cond_mask_dbg = torch.ones(images_dbg.shape[0], device=device)
            text_hidden_dbg, _ = text_encoder(input_ids_dbg, cond_mask_dbg)
            pad_mask_dbg = input_ids_dbg.eq(tokenizer.pad_id)
            t_dbg = torch.randint(0, diffusion.T, (images_dbg.shape[0],), device=device).long()
            loss_dbg = diffusion.training_loss(unet, images_dbg, t_dbg, text_hidden_dbg, pad_mask_dbg)
            loss_dbg.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        times = {"data": [], "forward": [], "backward": [], "optim": []}
        for _ in range(n_measure):
            _, t_data = timed_step(lambda: next(data_iter), device)
            batch = next(data_iter)

            images_dbg = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            input_ids_dbg = batch["input_ids"].to(device=device, dtype=torch.long)
            cond_mask_dbg = torch.ones(images_dbg.shape[0], device=device)
            text_hidden_dbg, _ = text_encoder(input_ids_dbg, cond_mask_dbg)
            pad_mask_dbg = input_ids_dbg.eq(tokenizer.pad_id)
            t_dbg = torch.randint(0, diffusion.T, (images_dbg.shape[0],), device=device).long()

            loss_dbg, t_forward = timed_step(diffusion.training_loss, device, unet, images_dbg, t_dbg, text_hidden_dbg, pad_mask_dbg)
            _, t_backward = timed_step(loss_dbg.backward, device)
            _, t_optim = timed_step(optimizer.step, device)
            optimizer.zero_grad(set_to_none=True)

            times["data"].append(t_data); times["forward"].append(t_forward)
            times["backward"].append(t_backward); times["optim"].append(t_optim)

        avg = {k: sum(v) / len(v) for k, v in times.items()}
        logging.info(
            f"[PROFILING] (media su {n_measure} step, dopo warmup) "
            f"data: {avg['data']:.1f}ms | forward+loss: {avg['forward']:.1f}ms | "
            f"backward: {avg['backward']:.1f}ms | optimizer: {avg['optim']:.1f}ms"
        )

    # --- Training loop ---
    for epoch in range(start_epoch, epochs + 1):
        unet.train()
        text_encoder.train()
        epoch_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
                B = images.shape[0]

                cond_mask = (torch.rand(B, device=device) >= uncond_prob).float()

                with torch.autocast(autocast_device, enabled=amp):
                    text_hidden, _ = text_encoder(input_ids, cond_mask)
                    text_pad_mask = input_ids.eq(tokenizer.pad_id)
                    t = torch.randint(0, diffusion.T, (B,), device=device).long()
                    loss = diffusion.training_loss(unet, images, t, text_hidden, text_pad_mask)

                optimizer.zero_grad(set_to_none=True)
                if use_cuda_amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, gradient_clipping)
                    optimizer.step()

                pbar.update(B)
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if global_step % 10 == 0:
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch,
                    })

            # --- fine epoca ---
            histograms = {}
            for tag, value in list(unet.named_parameters()) + list(text_encoder.named_parameters()):
                tag = tag.replace('/', '.')
                if value.grad is None:
                    continue
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_loss = evaluate(unet, text_encoder, diffusion, val_loader, device, tokenizer, amp)
            scheduler.step(val_loss)
            logging.info(f'Validation loss: {val_loss}')

            try:
                log_dict = {
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation loss': val_loss,
                    'step': global_step,
                    'epoch': epoch,
                    **histograms,
                }

                if sample_prompt_ids is not None:
                    unet.eval()
                    text_encoder.eval()
                    with torch.no_grad():
                        prompt_ids = sample_prompt_ids.to(device)
                        # coerente con la distribuzione vista in training: se uncond_prob=1,
                        # il modello ha sempre e solo visto cond_mask=0 (null) — valutarlo
                        # con cond_mask=1 lo metterebbe fuori distribuzione
                        cond_mask_eval = torch.zeros(1, device=device) if uncond_prob >= 1.0 else torch.ones(1, device=device)
                        text_hidden_eval, _ = text_encoder(prompt_ids, cond_mask_eval)
                        pad_mask_eval = prompt_ids.eq(tokenizer.pad_id)
                        shape = (1, 3, images.shape[-2], images.shape[-1])

                        sample = diffusion.sample(
                            unet, shape, text_hidden_eval, pad_mask_eval,
                            device=device, seed=global_step,
                        )

                        from predict import tensor_to_image
                        pil_img = tensor_to_image(sample[0].cpu())
                        log_dict['sample'] = wandb.Image(pil_img)

                    unet.train()
                    text_encoder.train()

                experiment.log(log_dict)
            except Exception as e:
                logging.warning(f"Logging fallito: {e}")

        # --- checkpoint ---
        if epoch % 2 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'unet_state': unet.state_dict(),
                'text_encoder_state': text_encoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'tokenizer_vocab': tokenizer.token2id,
                'uncond_prob': uncond_prob,
                'args': {
                    'image_size': images.shape[-1],
                    'text_dim': text_encoder.dim,
                    'base_ch': unet.base_ch,
                    'timesteps': diffusion.T,
                    'schedule': getattr(diffusion, 'schedule', 'cosine'),
                },
            }, str(dir_checkpoint / f'checkpoint_{run_name}_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(description='Train the UNet on images and caption (optional)')
    p.add_argument("--images_dir", required=True, help="Images Directory")
    p.add_argument("--attribute_legend_path", required=True,
                    help="csv 'cartoon_image_attributes_labels.csv': attribute,value,text_label")
    p.add_argument("--image_attribute_path", required=True,
                    help="csv 'cartoon_image_attributes.csv': per-image numeric attribute codes")
    p.add_argument("--split_file", required=True, help="JSON with train/val/test/ood_compositional id lists")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--run_name", required=True, help="A name for the run")
    p.add_argument('--load', '-f', type=str, default=False, help='Checkpoint Path')
    p.add_argument('--resume', action='store_true', help='Resume optimizer/scheduler/epoch (--load required)')

    p.add_argument("--image_size", type=int, default=32, help="Size of image (32x32 or 64x64)")
    p.add_argument("--max_caption_len", type=int, default=24)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--text_dim", type=int, default=96)
    p.add_argument("--text_layers", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--uncond_prob", type=float, default=0.1,
                    help="Probability of dropping text conditioning per sample. "
                         "Set to 1.0 for the unconditional baseline run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--rebuild_tokenizer", action="store_true")
    p.add_argument("--amp", action='store_true', help="Only for Cuda")
    p.add_argument("--cache_img_dir", type=str, help="Path to Cache Images Directory")
    p.add_argument("--profile", action='store_true', help="Misura data/forward/backward/optimizer una tantum prima del training")

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ids = load_split_ids(args.split_file, "train")
    val_ids = load_split_ids(args.split_file, "val")

    tokenizer = build_tokenizer(args, train_ids)

    train_ds = AvatarDataset(
        args.images_dir, args.attribute_legend_path, args.image_attribute_path, tokenizer,
        split_ids=train_ids, image_size=args.image_size, cache_dir=args.cache_img_dir,
    )
    val_ds = AvatarDataset(
        args.images_dir, args.attribute_legend_path, args.image_attribute_path, tokenizer,
        split_ids=val_ids, image_size=args.image_size, cache_dir=args.cache_img_dir,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=True, drop_last=True, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True, drop_last=True, pin_memory=False,
    )

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size, max_len=args.max_caption_len,
        dim=args.text_dim, n_layers=args.text_layers, pad_id=tokenizer.pad_id,
    ).to(device)

    unet = UNet(n_channels=3, base_ch=args.base_ch).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule, device=device)

    logging.info(f'Network:\n'
                 f'\t{unet.n_channels} input channels\n'
                 f'\t{unet.base_ch} output channels (classes)\n'
                 f'\t{"Bilinear" if unet.bilinear else "Transposed conv"} upscaling')

    sample_text = "a boy with blue eye color, afro hair style"
    sample_prompt_ids = torch.as_tensor([tokenizer.encode(sample_text)], dtype=torch.long)

    train_model(
        unet=unet,
        text_encoder=text_encoder,
        diffusion=diffusion,
        tokenizer=tokenizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=args.run_name,
        dir_checkpoint=run_dir / "checkpoints",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_every=args.save_every,
        amp=args.amp,
        uncond_prob=args.uncond_prob,
        sample_prompt_ids=sample_prompt_ids,
        load_path=args.load if args.load else None,
        resume=args.resume,
        profile=args.profile,
    )