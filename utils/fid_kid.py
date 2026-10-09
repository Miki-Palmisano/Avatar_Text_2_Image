"""
Valutazione quantitativa: FID/KID su test set e su compositional-OOD, diversità tra seed per lo
stesso prompt, parametri / tempo di sampling / memoria.

Uso (modello condizionato):
    python3 utils/fid_kid.py \
        --checkpoint checkpoints/checkpoint_Conditional_Run_epoch100.pth \
        --tokenizer runs/tokenizer.json \
        --images_dir dataset/cartoonset100k \
        --attribute_legend_path dataset/cartoon_image_attributes_labels.csv \
        --image_attribute_path dataset/cartoon_image_attributes.csv \
        --split_file dataset/splits.json \
        --output_dir eval/conditional \
        --n_samples 200

Baseline --no_text (nessun tokenizer):
    python3 utils/fid_kid.py \
        --checkpoint checkpoints/checkpoint_UnConditional_Run_epoch100.pth \
        --images_dir dataset/cartoonset100k ... --output_dir eval/unconditional

REGOLA PER IL CONFRONTO: tutti i modelli confrontati vanno valutati con gli STESSI --amp,
--gen_batch_size, --n_samples e --seed, sulla stessa macchina (tempo e memoria dipendono dall'hardware).
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import platform
import threading
import time

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


def text_encoder_kwargs(sd):
    """Parametri strutturali del text encoder ricavati dai pesi (vocabolario, dim, max_len, layer):
    non dipendono dai default del codice."""
    emb = sd["token_emb.weight"]
    pos = sd["pos_emb.pos_emb"] if "pos_emb.pos_emb" in sd else sd["pos_emb.pe"]
    n_layers = len({k.split(".")[2] for k in sd if k.startswith("encoder.layers.")})
    return dict(vocab_size=emb.shape[0], dim=emb.shape[1], max_len=pos.shape[1], n_layers=n_layers)


def load_model(checkpoint_path, tokenizer_path, device):
    """
    Ritorna anche `mode` ("text" = condizionato, "none" = baseline --no_text) e `meta`
    (checkpoint, epoca, pesi usati), da scrivere nel results.json.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})

    use_text = train_args.get("use_text", True)
    mode = "text" if use_text else "none"
    if use_text and not tokenizer_path:
        raise ValueError("Questo modello usa il testo: passa --tokenizer")

    text_dim = train_args.get("text_dim") or 96   # None nei checkpoint no_text

    unet = UNet(n_channels=3, base_ch=train_args.get("base_ch", 64),
                text_dim=text_dim, use_text=use_text).to(device)
    unet.load_state_dict(checkpoint["unet_state"])
    unet.eval()

    tokenizer, text_encoder = None, None
    if use_text:
        tokenizer = Tokenizer.load(tokenizer_path)
        text_encoder = TextEncoder(**text_encoder_kwargs(checkpoint["text_encoder_state"]),
                                   pad_id=tokenizer.pad_id).to(device)
        text_encoder.load_state_dict(checkpoint["text_encoder_state"])
        text_encoder.eval()

    diffusion = GaussianDiffusion(timesteps=train_args.get("timesteps", 1000),
                                  schedule=train_args.get("schedule", "cosine"), device=device)
    image_size = train_args.get("image_size", 64)

    n_params = unet.num_params()
    if text_encoder is not None:
        n_params += sum(p.numel() for p in text_encoder.parameters())

    meta = {"checkpoint": str(checkpoint_path), "epoch": checkpoint.get("epoch"),
            "weights": "raw"}
    return unet, text_encoder, diffusion, tokenizer, image_size, n_params, mode, meta


@torch.no_grad()
def generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size,
                   seed=None, mode="text", amp=False):
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
    # amp: stessa precisione del training (fp16, solo CUDA); lo stato x del campionamento resta in fp32
    with torch.autocast("cuda", enabled=(amp and device.type == "cuda")):
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
    n = min(len(list(Path(real_dir).glob("*.png"))), len(list(Path(gen_dir).glob("*.png"))))
    metrics = calculate_metrics(
        input1=str(real_dir), input2=str(gen_dir),
        fid=True, kid=True,
        cuda=(device.type == "cuda"),
        verbose=False,
        kid_subset_size=max(2, min(100, n - 1)),
    )
    return (metrics["frechet_inception_distance"],
            metrics["kernel_inception_distance_mean"],
            metrics["kernel_inception_distance_std"])


class PeakMemory:
    """Picco di memoria durante un blocco di codice.
    cuda: torch.cuda.max_memory_allocated() (picco reale, pesi + attivazioni)
    mps : lettura di current_allocated_memory ogni 10 ms (APPROSSIMATO, non esiste un contatore di picco)
    cpu : picco RSS del processo dall'avvio (sovrastima)"""
    def __init__(self, device):
        self.device, self.peak_mb, self.kind = device, None, None

    def _poll(self):
        while not self._stop.is_set():
            try:
                self._samples.append(torch.mps.current_allocated_memory())
            except Exception:
                pass
            time.sleep(0.01)

    def __enter__(self):
        d = self.device.type
        if d == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif d == "mps":
            torch.mps.synchronize()
            self._samples = [torch.mps.current_allocated_memory()]
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        d = self.device.type
        if d == "cuda":
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
            self.kind = "cuda max_memory_allocated (picco reale)"
        elif d == "mps":
            torch.mps.synchronize()
            self._stop.set(); self._thread.join()
            self.peak_mb = max(self._samples) / 2**20
            self.kind = "mps polling 10ms (approssimato)"
        else:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak_mb = rss / 2**20 if platform.system() == "Darwin" else rss / 1024
            self.kind = "cpu picco RSS del processo dall'avvio (sovrastima)"
        return False


def model_memory_mb(*modules):
    """Memoria dei soli pesi/buffer, in MB."""
    total = 0
    for m in modules:
        if m is not None:
            total += sum(p.numel() * p.element_size() for p in m.parameters())
            total += sum(b.numel() * b.element_size() for b in m.buffers())
    return total / 2**20


@torch.no_grad()
def diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size,
                           n_seeds=8, mode="text", amp=False):
    """Genera n_seeds immagini per lo STESSO prompt; diversità = deviazione standard media per pixel."""
    imgs = []
    for seed in range(n_seeds):
        batch = generate_batch(unet, text_encoder, diffusion, tokenizer, [prompt], device, image_size,
                               seed=seed, mode=mode, amp=amp)
        imgs.append(np.asarray(batch[0], dtype=np.float32) / 255.0)
    stacked = np.stack(imgs, axis=0)  # (n_seeds, H, W, 3)
    return stacked.std(axis=0).mean(), imgs


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
    p.add_argument("--gen_batch_size", type=int, default=32,
                   help="Immagini generate in parallelo: più alto = meno tempo per immagine. Va riportato col tempo")
    p.add_argument("--amp", action="store_true", help="Campiona in fp16 (solo CUDA). Stesso valore per TUTTI i modelli")
    p.add_argument("--diversity_prompts", nargs="*", default=None,
                   help="Prompt per la diversità tra seed; default: alcuni presi dal test set")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet, text_encoder, diffusion, tokenizer, image_size, n_params, mode, meta = load_model(
        args.checkpoint, args.tokenizer, device)
    logging.info(f"Modalità: {mode} | pesi: {meta['weights']} | epoca checkpoint: {meta['epoch']}")
    logging.info(f"Parametri totali: {n_params/1e6:.2f}M | image_size={image_size} | timesteps={diffusion.T}")

    legend = parse_attribute_legend(args.attribute_legend_path)
    metadata, id_to_path = parse_image_attributes(args.image_attribute_path, legend)
    splits = json.load(open(args.split_file))

    results = {**meta, "mode": mode, "n_params": n_params, "image_size": image_size, "timesteps": diffusion.T,
               "sampling_precision": "fp16 autocast" if (args.amp and device.type == "cuda") else "fp32",
               "gen_batch_size": args.gen_batch_size, "seed": args.seed, "device": str(device)}

    # --- Memoria: pesi + picco di campionamento con 1 immagine (riscalda anche la GPU) ---
    results["model_memory_mb"] = model_memory_mb(unet, text_encoder)
    probe_caption = [build_deterministic_caption(metadata[splits["test"][0]])] if mode == "text" else ["x"]
    with PeakMemory(device) as mem1:
        generate_batch(unet, text_encoder, diffusion, tokenizer, probe_caption, device, image_size,
                       seed=0, mode=mode, amp=args.amp)
    results["peak_memory_single_image_mb"] = mem1.peak_mb
    results["memory_measure"] = mem1.kind
    logging.info(f"Memoria: pesi {results['model_memory_mb']:.1f} MB | picco (1 immagine) {mem1.peak_mb:.1f} MB [{mem1.kind}]")

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
        with PeakMemory(device) as mem:
            for i in range(0, len(ids), args.gen_batch_size):
                batch_ids = ids[i:i + args.gen_batch_size]
                captions = [build_deterministic_caption(metadata[j]) for j in batch_ids]
                imgs = generate_batch(unet, text_encoder, diffusion, tokenizer, captions, device, image_size,
                                      seed=args.seed + i, mode=mode, amp=args.amp)
                for j, img in zip(batch_ids, imgs):
                    img.save(gen_dir / f"{j}.png")
        sampling_time_per_image = (time.time() - t0) / len(ids)

        logging.info(f"[{split_name}] Calcolo FID/KID...")
        fid_score, kid_score, kid_std = compute_fid_kid(real_dir, gen_dir, device)
        results[split_name] = {
            "n_samples": len(ids),
            "fid": fid_score,
            "kid": kid_score,
            "kid_std": kid_std,
            "sampling_time_per_image_sec": sampling_time_per_image,
            "peak_memory_mb": mem.peak_mb,          # al batch gen_batch_size
        }
        logging.info(f"[{split_name}] FID={fid_score:.3f} | KID={kid_score:.5f} ± {kid_std:.5f} | "
                     f"{sampling_time_per_image:.2f}s/immagine | picco memoria {mem.peak_mb:.0f} MB")

    # --- Diversità tra seed ---
    diversity_prompts = args.diversity_prompts
    if not diversity_prompts:
        sample_ids = splits.get("test", [])[:3]
        diversity_prompts = [build_deterministic_caption(metadata[i]) for i in sample_ids]
    if mode != "text":
        diversity_prompts = diversity_prompts[:1]   # senza testo il "prompt" non conta

    diversity_results = []
    for prompt in diversity_prompts:
        std, _ = diversity_across_seeds(unet, text_encoder, diffusion, tokenizer, prompt, device, image_size,
                                        n_seeds=8, mode=mode, amp=args.amp)
        diversity_results.append({"prompt": prompt, "pixel_std_across_seeds": float(std)})
        logging.info(f"Diversità (std pixel su 8 seed) per '{prompt[:50]}...': {std:.4f}")
    results["diversity_across_seeds"] = diversity_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Risultati salvati in {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()