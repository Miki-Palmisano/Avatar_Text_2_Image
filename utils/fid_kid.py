"""
Valutazione quantitativa: FID/KID su test set e
su compositional-OOD, diversità tra seed per lo stesso prompt, parametri/
tempo di sampling.

Uso (un solo modello):
    python evaluate.py \
        --checkpoint runs/conditional/checkpoints/checkpoint_..._epoch100.pth \
        --tokenizer runs/conditional/tokenizer.json \
        --images_dir dataset --attribute_legend_path ... --image_attribute_path ... \
        --split_file dataset/splits.json --output_dir eval_conditional \
        --n_samples 200
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unet import UNet
from text_encoder import TextEncoder
from diffusion import GaussianDiffusion
from tokenizer import Tokenizer
from predict import tensor_to_image
from data_loading import parse_attribute_legend, parse_image_attributes, build_deterministic_caption, load_data, AvatarDataset


def load_model(checkpoint_path, tokenizer_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})

    tokenizer = Tokenizer.load(tokenizer_path)

    unet = UNet(n_channels=3, base_ch=train_args.get("base_ch", 64),
                text_dim=train_args.get("text_dim", 96)).to(device)
    unet.load_state_dict(checkpoint["unet_state"])
    unet.eval()

    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size,
                                     dim=train_args.get("text_dim", 96),
                                     pad_id=tokenizer.pad_id).to(device)
    text_encoder.load_state_dict(checkpoint["text_encoder_state"])
    text_encoder.eval()

    diffusion = GaussianDiffusion(timesteps=train_args.get("timesteps", 1000),
                                    schedule=train_args.get("schedule", "cosine"), device=device)
    image_size = train_args.get("image_size", 32)

    n_params = unet.num_params() + sum(p.numel() for p in text_encoder.parameters())
    return unet, text_encoder, diffusion, tokenizer, image_size, n_params


@torch.no_grad()
def generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size, seed=None):
    """Genera un batch di immagini, una per caption, in un'unica reverse loop batched."""
    input_ids = torch.stack([torch.as_tensor(tokenizer.encode(c), dtype=torch.long) for c in captions]).to(device)
    cond_mask = torch.ones(len(captions), device=device)
    text_hidden, _ = text_encoder(input_ids, cond_mask)
    pad_mask = input_ids.eq(tokenizer.pad_id)
    shape = (len(captions), 3, image_size, image_size)
    samples = diffusion.sample(unet, shape, text_hidden, pad_mask, device=device, seed=seed)
    return [tensor_to_image(s.cpu()) for s in samples]


def save_real_images(ids, id_to_path, images_dir, out_dir, image_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        img = load_data(Path(images_dir) / id_to_path[i])
        img = img.resize((image_size, image_size), resample=Image.BICUBIC)
        img.save(out_dir / f"{i}.png")


def compute_fid_kid(real_dir, gen_dir, device):
    """Richiede torch-fidelity: pip install torch-fidelity"""
    from torch_fidelity import calculate_metrics
    metrics = calculate_metrics(
        input1=str(real_dir), input2=str(gen_dir),
        fid=True, kid=True,
        cuda=(device.type == "cuda"),
        verbose=False,
        kid_subset_size=min(100, len(list(Path(real_dir).glob("*.png"))) - 1),
    )
    fid_score = metrics["frechet_inception_distance"]
    kid_score = metrics["kernel_inception_distance_mean"]
    return fid_score, kid_score


@torch.no_grad()
def diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size, n_seeds=8):
    """Genera n_seeds immagini per lo STESSO prompt, misura la diversità come
    deviazione standard media pixel-per-pixel tra le generazioni (proxy semplice,
    non richiede una rete di embedding aggiuntiva)."""
    imgs = []
    for seed in range(n_seeds):
        batch = generate_batch(unet, text_encoder, diffusion, tokenizer, [prompt], device, image_size, seed=seed)
        arr = np.asarray(batch[0], dtype=np.float32) / 255.0
        imgs.append(arr)
    stacked = np.stack(imgs, axis=0)  # (n_seeds, H, W, 3)
    pixel_std = stacked.std(axis=0).mean()  # media della deviazione standard per pixel/canale
    return pixel_std, imgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--attribute_legend_path", required=True)
    p.add_argument("--image_attribute_path", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_samples", type=int, default=200, help="Immagini generate per split (test / ood_compositional)")
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--diversity_prompts", nargs="*", default=None,
                    help="Prompt specifici per il test di diversità tra seed; default: alcuni presi dal test set")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet, text_encoder, diffusion, tokenizer, image_size, n_params = load_model(args.checkpoint, args.tokenizer, device)
    logging.info(f"Parametri totali: {n_params/1e6:.2f}M | image_size={image_size} | timesteps={diffusion.T}")

    legend = parse_attribute_legend(args.attribute_legend_path)
    metadata, id_to_path = parse_image_attributes(args.image_attribute_path, legend)
    splits = json.load(open(args.split_file))

    results = {"n_params": n_params, "image_size": image_size, "timesteps": diffusion.T}

    # --- FID/KID su test set (ordinario) e ood_compositional ---
    for split_name in ["test", "ood_compositional"]:
        ids = splits.get(split_name, [])
        if not ids:
            logging.warning(f"Split '{split_name}' vuoto o assente, salto.")
            continue
        ids = ids[: args.n_samples]

        real_dir = out_dir / split_name / "real"
        gen_dir = out_dir / split_name / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[{split_name}] Salvo {len(ids)} immagini reali di riferimento...")
        save_real_images(ids, id_to_path, args.images_dir, real_dir, image_size)

        logging.info(f"[{split_name}] Genero {len(ids)} immagini condizionate...")
        t0 = time.time()
        for i in range(0, len(ids), args.gen_batch_size):
            batch_ids = ids[i:i + args.gen_batch_size]
            captions = [build_deterministic_caption(metadata[j]) for j in batch_ids]
            imgs = generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size, seed=args.seed + i)
            for j, img in zip(batch_ids, imgs):
                img.save(gen_dir / f"{j}.png")
        gen_time = time.time() - t0
        sampling_time_per_image = gen_time / len(ids)

        logging.info(f"[{split_name}] Calcolo FID/KID...")
        fid_score, kid_score = compute_fid_kid(real_dir, gen_dir, device)
        results[split_name] = {
            "n_samples": len(ids),
            "fid": fid_score,
            "kid": kid_score,
            "sampling_time_per_image_sec": sampling_time_per_image,
        }
        logging.info(f"[{split_name}] FID={fid_score:.3f} | KID={kid_score:.5f} | {sampling_time_per_image:.2f}s/immagine")

    # --- Diversità tra seed ---
    diversity_prompts = args.diversity_prompts
    if not diversity_prompts:
        sample_ids = splits.get("test", [])[:3]
        diversity_prompts = [build_deterministic_caption(metadata[i]) for i in sample_ids]

    diversity_results = []
    for prompt in diversity_prompts:
        std, _ = diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size, n_seeds=8)
        diversity_results.append({"prompt": prompt, "pixel_std_across_seeds": float(std)})
        logging.info(f"Diversità (std pixel su 8 seed) per '{prompt[:50]}...': {std:.4f}")
    results["diversity_across_seeds"] = diversity_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Risultati salvati in {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()