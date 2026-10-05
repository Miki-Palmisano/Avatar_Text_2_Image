"""

python3 predict.py -m ./checkpoints/checkpoint_Conditional_Run_2_1_epoch100.pth -p "a avatar with red eye color and white hair color" --guidance-scale 2.0 -v

"""

import argparse
import logging
import re

import torch
from PIL import Image

from unet.unet_model import UNet
from text_encoder import TextEncoder
from diffusion import GaussianDiffusion
from tokenizer import Tokenizer


@torch.no_grad()
def sample_with_guidance(unet, diffusion, shape, text_hidden, text_pad_mask,
                          null_hidden, null_pad_mask, device, guidance_scale=1.0,
                          seed=None, clip_denoised=True):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    use_cfg = (guidance_scale != 1.0) and (text_hidden is not None)

    for t in reversed(range(diffusion.T)):
        batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)

        if not use_cfg:
            pred_noise = unet(x, batch_t, text_hidden, text_pad_mask)
        else:
            eps_cond = unet(x, batch_t, text_hidden, text_pad_mask)
            eps_uncond = unet(x, batch_t, null_hidden, null_pad_mask)
            pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        sqrt_ac_t = diffusion.sqrt_alphas_cumprod[t]
        sqrt_omac_t = diffusion.sqrt_one_minus_alphas_cumprod[t]

        if clip_denoised:
            x0_pred = ((x - sqrt_omac_t * pred_noise) / sqrt_ac_t).clamp(-1, 1)
            pred_noise = (x - sqrt_ac_t * x0_pred) / sqrt_omac_t

        beta_t = diffusion.betas[t]
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(diffusion.alphas[t])
        mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_omac_t * pred_noise)

        if t == 0:
            x = mean
        else:
            noise = torch.randn_like(x)
            std = torch.sqrt(diffusion.posterior_variance[t])
            x = mean + std * noise

    return x.clamp(-1, 1)

def warn_unknown_words(prompt, tokenizer):
    """Le parole fuori vocabolario diventano <unk> in silenzio: il modello non le "vede".
    Lo segnalo, perché è la causa più comune di 'il testo non influisce'."""
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    unknown = [w for w in words if w not in tokenizer.token2id]
    if unknown:
        logging.warning(f"Parole NON presenti nel vocabolario (trattate come <unk>): {sorted(set(unknown))}")


def generate_image(unet, text_encoder, diffusion, tokenizer, prompt, device,
                    image_size, guidance_scale=1.0, seed=None, text_mode='text'):
    """
    Genera una singola immagine (B=1).
    text_mode:
      "text"   -> condizionata dal prompt (cond_mask=1)
      "null"   -> modello allenato con uncond_prob=1.0: usa l'embedding null (cond_mask=0)
      "none"   -> baseline --no_text: nessun percorso testuale
    """
    unet.eval()

    if text_mode == 'none':
        shape = (1, 3, image_size, image_size)
        sample = sample_with_guidance(
            unet, diffusion, shape, None, None, None, None,
            device=device, guidance_scale=guidance_scale, seed=seed,
        )
        return sample[0].cpu()  # (3, H, W) in [-1, 1]

    text_encoder.eval()

    input_ids = torch.as_tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    pad_mask = input_ids.eq(tokenizer.pad_id)

    cond_mask = torch.ones(1, device=device)          # 1 = usa il testo vero
    uncond_mask = torch.zeros(1, device=device)        # 0 = forza l'embedding "null"

    text_hidden, _ = text_encoder(input_ids, cond_mask)
    null_hidden, _ = text_encoder(input_ids, uncond_mask)

    shape = (1, 3, image_size, image_size)
    sample = sample_with_guidance(
        unet, diffusion, shape, text_hidden, pad_mask, null_hidden, pad_mask,
        device=device, guidance_scale=guidance_scale, seed=seed,
    )
    return sample[0].cpu()  # (3, H, W) in [-1, 1]


def tensor_to_image(img: torch.Tensor) -> Image.Image:
    """[-1, 1] CHW -> PIL Image HWC uint8"""
    img = (img + 1) / 2                      # -> [0, 1]
    img = img.clamp(0, 1).mul(255).byte()
    img = img.permute(1, 2, 0).numpy()        # CHW -> HWC
    return Image.fromarray(img)


def get_args():
    parser = argparse.ArgumentParser(description='Generate avatar images from a text prompt')
    parser.add_argument('--model', '-m', required=True, metavar='FILE',
                         help='Path al checkpoint (.pt) salvato da train.py')
    parser.add_argument('--prompt', '-p', required=True,
                         help='Caption testuale, es: "a boy with blue eye color, short hair style"')
    parser.add_argument('--output', '-o', default=None, metavar='OUTPUT',
                         help='Nome file di output (default: derivato dal prompt)')
    parser.add_argument('--tokenizer', default=None,
                         help='Path al tokenizer.json; se omesso, prova a leggerlo dal checkpoint')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                         help='Classifier-free guidance scale (1.0 = nessuna guidance extra)')
    parser.add_argument('--viz', '-v', action='store_true', help='Mostra l\'immagine generata')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    logging.info(f'Loading checkpoint {args.model}')
    checkpoint = torch.load(args.model, map_location=device)
    train_args = checkpoint.get('args', {})  # iperparametri salvati da train.py

    use_text = train_args.get('use_text', True)  # False per la baseline --no_text
    uncond_trained = checkpoint.get('uncond_prob', 0.0) >= 1.0
    text_mode = "none" if not use_text else ("null" if uncond_trained else "text")
    logging.info(f'Modalità del checkpoint: {text_mode}')

    image_size = train_args.get('image_size', 64)
    text_dim = train_args.get('text_dim') or 96
    base_ch = train_args.get('base_ch', 64)
    timesteps = train_args.get('timesteps', 1000)
    schedule = train_args.get('schedule', 'cosine')

    unet = UNet(n_channels=3, base_ch=base_ch, text_dim=text_dim, use_text=use_text).to(device)
    unet.load_state_dict(checkpoint['unet_state'])

    tokenizer, text_encoder = None, None
    if use_text:
        if not args.prompt:
            raise ValueError('Questo checkpoint è condizionato dal testo: serve --prompt')

        if args.tokenizer:
            tokenizer = Tokenizer.load(args.tokenizer)
        elif 'tokenizer_vocab' in checkpoint:
            tokenizer = Tokenizer()
            tokenizer.token2id = checkpoint['tokenizer_vocab']
            tokenizer.id2token = {v: k for k, v in tokenizer.token2id.items()}
            tokenizer._fitted = True
        else:
            raise ValueError('Nessun tokenizer trovato: passa --tokenizer path/to/tokenizer.json')

        text_encoder = TextEncoder(
            vocab_size=tokenizer.vocab_size, dim=text_dim, pad_id=tokenizer.pad_id,
        ).to(device)
        text_encoder.load_state_dict(checkpoint['text_encoder_state'])
        warn_unknown_words(args.prompt, tokenizer)
        if text_mode == "null":
            logging.warning('Modello allenato con uncond_prob=1.0: il prompt viene ignorato.')
    elif args.prompt:
        logging.warning('Checkpoint --no_text: il prompt viene ignorato.')

    diffusion = GaussianDiffusion(timesteps=timesteps, schedule=schedule, device=device)

    logging.info('Model loaded!')
    logging.info(f'Generating: "{args.prompt}"' if use_text else 'Generating (unconditional)')

    image_tensor = generate_image(
        unet, text_encoder, diffusion, tokenizer, args.prompt, device,
        image_size=image_size, guidance_scale=args.guidance_scale, seed=args.seed,
        text_mode=text_mode,
    )
    result = tensor_to_image(image_tensor)

    out_path = args.output or f"{args.prompt[:40].strip().replace(' ', '_')}_OUT.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    logging.info(f'Image saved to {out_path}')

    if args.viz:
        result.show()