"""
DDPM machinery: noise schedule, forward (q) process, training loss,
and reverse (p) sampling loop. Implemented from scratch (Ho et al. 2020
formulation) — only tensor ops, no diffusers pipeline.
"""
import math
import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device="cpu"):
        self.T = timesteps
        betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # posterior q(x_{t-1} | x_t, x_0) variance
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward process: sample x_t given x_0 and timestep t (closed form)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def training_loss(self, model, x0, t, text_hidden, text_pad_mask=None):
        """Standard eps-prediction MSE loss."""
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t, text_hidden, text_pad_mask)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t: int, text_hidden, text_pad_mask=None, clip_denoised=True):
        batch_t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, batch_t, text_hidden, text_pad_mask)

        sqrt_ac_t = self.sqrt_alphas_cumprod[t]
        sqrt_omac_t = self.sqrt_one_minus_alphas_cumprod[t]

        # ricostruisci la stima di x0 e, se richiesto, vincolala a [-1, 1]
        # PRIMA di ricalcolare la mean del passo successivo
        x0_pred = (x_t - sqrt_omac_t * pred_noise) / sqrt_ac_t
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)

        # ricalcola pred_noise "corretto" implicito dal x0 clippato,
        # così la formula della mean resta coerente
        pred_noise = (x_t - sqrt_ac_t * x0_pred) / sqrt_omac_t

        beta_t = self.betas[t]
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t])
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_omac_t * pred_noise)

        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        std = torch.sqrt(self.posterior_variance[t])
        return mean + std * noise

    @torch.no_grad()
    def sample(self, model, shape, text_hidden, text_pad_mask=None, device="cpu", seed=None, clip_denoised=True):
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, text_hidden, text_pad_mask, clip_denoised=clip_denoised)
        return x.clamp(-1, 1)