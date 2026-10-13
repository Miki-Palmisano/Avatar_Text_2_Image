""" Full assembly of the parts to form the complete network """

from .unet_parts import *

def infer_unet_config(sd):
    """Per i checkpoint senza unet_cfg: legge la configurazione da chiavi e forme del state_dict.
    Non deducibili: n_heads (default 4) e bilinear (default True)."""
    has = lambda prefix: any(k.startswith(prefix) for k in sd)
    attn = [n for n, p in (("self", "self_attn."), ("mid", "mid_attn."), ("up", "up_attn.")) if has(p)]
    kv = sd.get("mid_attn.kv_proj.weight", sd.get("up_attn.kv_proj.weight"))
    return dict(base_ch=int(sd["inc.double_conv.0.weight"].shape[0]),
                time_dim=int(sd["time_mlp.0.weight"].shape[0]),
                text_dim=int(kv.shape[1]) if kv is not None else 96,
                use_text=("mid" in attn or "up" in attn),
                attn=attn)

class UNet(nn.Module):
    def __init__(self,
                 n_channels=3,
                 base_ch=64,
                 text_dim=96,
                 time_dim=256,
                 bilinear=True,
                 n_heads=4,
                 use_checkpointing=False,
                 use_text=True,
                 attn=("mid", "self")
        ):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.time_dim = time_dim
        self.base_ch = base_ch
        self.use_ckpt = use_checkpointing
        self.use_text = use_text

        self.has_self = "self" in attn
        self.has_mid = use_text and "mid" in attn
        self.has_up = use_text and "up" in attn
        self.attn = [n for n, on in (("self", self.has_self), ("mid", self.has_mid), ("up", self.has_up)) if on]
        self.cfg = dict(base_ch=base_ch, text_dim=text_dim, time_dim=time_dim, bilinear=bilinear,
                        n_heads=n_heads, use_text=use_text, attn=self.attn)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        factor = 2 if bilinear else 1

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.inc = DoubleConv(n_channels, c1, time_dim)  # x1: c1 canali
        self.down1 = Down(c1, c2, time_dim)  # x2: c2 canali
        self.down2 = Down(c2, c3 // factor, time_dim)  # x3 (bottleneck): c3 // factor canali

        if self.has_self:
            self.self_attn = SelfAttention(c3 // factor, n_heads)

        if self.has_mid:
            self.mid_attn = CrossAttention(c3 // factor, text_dim, n_heads)

        self.up1 = Up(c3, c2 // factor, time_dim, bilinear)  # concat: (c3//factor) + c2 = c3

        # ulteriore strato di crossattention, migliora il condizionamento del testo ma riduce del 20% le prestazioni
        if self.has_up:
            self.up_attn = CrossAttention(c2 // factor, text_dim, n_heads)

        self.up2 = Up(c2, c1, time_dim, bilinear)  # concat: (c2//factor) + c1 = c2

        self.outc = OutConv(c1, n_channels)  # combacia con l'output di up2

    @classmethod
    def from_checkpoint(cls, checkpoint, **overrides):
        """Costruisce la U-Net giusta per il checkpoint e ne carica i pesi (overrides: es. n_heads=8)."""
        sd = checkpoint["unet_state"]
        cfg = dict(infer_unet_config(sd))
        cfg.update(overrides)
        model = cls(**cfg)
        model.load_state_dict(sd)  # se i nomi dei moduli non coincidono, PyTorch elenca chiavi mancanti/inattese
        return model

    def forward(self, x, t, text_hidden, text_pad_mask):
        """
                x:              (B, 3, H, W) noisy image
                t:              (B,) integer timesteps
                text_hidden:    (B, T, text_dim) from text encoder
                text_pad_mask:  (B, T) bool, True at PAD positions
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))

        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)

        if self.has_self:
            x3 = self.self_attn(x3)

        if self.has_mid:
            x3 = self.mid_attn(x3, text_hidden, text_pad_mask)

        x = self.up1(x3, x2, t_emb)

        if self.has_up:
            x = self.up_attn(x, text_hidden, text_pad_mask)

        x = self.up2(x, x1, t_emb)

        logits = self.outc(x)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())