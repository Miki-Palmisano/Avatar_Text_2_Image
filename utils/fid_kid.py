"""
Valutazione quantitativa: FID/KID su test set e
su compositional-OOD, diversità tra seed per lo stesso prompt, parametri/
tempo di sampling.

Uso (un solo modello):
    python3 utils/fid_kid.py \
        --checkpoint checkpoints/checkpoint_Conditional_Run_2Attention_Self_epoch110.pth \
        --tokenizer runs/tokenizer_sixcaption.json \
        --images_dir dataset/cartoonset100k \
        --attribute_legend_path dataset/cartoon_image_attributes_labels.csv \
        --image_attribute_path dataset/cartoon_image_attributes.csv \
        --split_file dataset/splits.json \
        --output_dir eval/conditional_2attention_self_2 \
        --n_samples 200

    python3 utils/fid_kid.py \
        --checkpoint checkpoints/checkpoint_UnConditional_Run_2_epoch100.pth \
        --images_dir dataset/cartoonset100k \
        --attribute_legend_path dataset/cartoon_image_attributes_labels.csv \
        --image_attribute_path dataset/cartoon_image_attributes.csv \
        --split_file dataset/splits.json \
        --output_dir eval/unconditional \
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
from data_loading import parse_attribute_legend, parse_image_attributes, build_deterministic_caption, load_data
from torch_fidelity import calculate_metrics


def load_model(checkpoint_path, tokenizer_path, device):
    """
    Ritorna anche `mode`, letta dal checkpoint:
      "text" -> condizionato dal testo (cond_mask=1)
      "null" -> allenato con uncond_prob=1.0 (stessa architettura, testo spento: cond_mask=0)
      "none" -> baseline --no_text: nessun text encoder / cross-attention
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})

    use_text = train_args.get("use_text", True)
    if not use_text:
        mode = "none"
    else:
        mode = "text"

    text_dim = train_args.get("text_dim") or 96   # None nei checkpoint no_text

    unet = UNet(n_channels=3, base_ch=train_args.get("base_ch", 64),
                text_dim=text_dim, use_text=use_text).to(device)
    unet.load_state_dict(checkpoint["unet_state"])
    unet.eval()

    tokenizer, text_encoder = None, None
    if use_text:
        tokenizer = Tokenizer.load(tokenizer_path)
        text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, dim=text_dim,
                                       pad_id=tokenizer.pad_id).to(device)
        text_encoder.load_state_dict(checkpoint["text_encoder_state"])
        text_encoder.eval()

    diffusion = GaussianDiffusion(timesteps=train_args.get("timesteps", 1000),
                                    schedule=train_args.get("schedule", "cosine"), device=device)
    image_size = train_args.get("image_size", 64)

    # i parametri del text encoder si contano solo se il modello li ha davvero
    n_params = unet.num_params()
    if text_encoder is not None:
        n_params += sum(p.numel() for p in text_encoder.parameters())
    return unet, text_encoder, diffusion, tokenizer, image_size, n_params, mode


@torch.no_grad()
def generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size, seed=None, mode="text"):
    """Genera un'immagine per ogni caption in un'unica reverse loop batched.
    In mode="none" le caption servono solo a contare quante immagini generare."""
    if mode == "none":
        text_hidden, pad_mask = None, None
    else:
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
def diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size, n_seeds=8, mode="text"):
    """Genera n_seeds immagini per lo STESSO prompt, misura la diversità come
    deviazione standard media pixel-per-pixel tra le generazioni (proxy semplice,
    non richiede una rete di embedding aggiuntiva)."""
    imgs = []
    for seed in range(n_seeds):
        batch = generate_batch(unet, text_encoder, diffusion, tokenizer, [prompt], device, image_size, seed=seed, mode=mode)
        arr = np.asarray(batch[0], dtype=np.float32) / 255.0
        imgs.append(arr)
    stacked = np.stack(imgs, axis=0)  # (n_seeds, H, W, 3)
    pixel_std = stacked.std(axis=0).mean()  # media della deviazione standard per pixel/canale
    return pixel_std, imgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None, help="Obbligatorio per i modelli condizionati; ignorato con --no_text")
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

    unet, text_encoder, diffusion, tokenizer, image_size, n_params, mode = load_model(args.checkpoint, args.tokenizer, device)
    if mode != "none" and not args.tokenizer:
        raise ValueError("Questo modello usa il testo: passa --tokenizer")
    logging.info(f"Modalità del modello: {mode}  (text=condizionato, null=testo spento, none=nessun percorso testuale)")
    logging.info(f"Parametri totali: {n_params/1e6:.2f}M | image_size={image_size} | timesteps={diffusion.T}")

    legend = parse_attribute_legend(args.attribute_legend_path)
    metadata, id_to_path = parse_image_attributes(args.image_attribute_path, legend)
    splits = json.load(open(args.split_file))

    results = {"mode": mode, "n_params": n_params, "image_size": image_size, "timesteps": diffusion.T}

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

        logging.info(f"[{split_name}] Genero {len(ids)} immagini ({mode})...")
        t0 = time.time()
        for i in range(0, len(ids), args.gen_batch_size):
            batch_ids = ids[i:i + args.gen_batch_size]
            captions = [build_deterministic_caption(metadata[j]) for j in batch_ids]
            imgs = generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size, seed=args.seed + i, mode=mode)
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
    if mode != "text":
        # il testo non conta: misurare 3 "prompt" darebbe 3 volte la stessa quantità
        diversity_prompts = diversity_prompts[:1]

    diversity_results = []
    for prompt in diversity_prompts:
        std, _ = diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size, n_seeds=8, mode=mode)
        diversity_results.append({"prompt": prompt, "pixel_std_across_seeds": float(std)})
        logging.info(f"Diversità (std pixel su 8 seed) per '{prompt[:50]}...': {std:.4f}")
    results["diversity_across_seeds"] = diversity_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Risultati salvati in {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()