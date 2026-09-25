""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torch.utils.checkpoint import checkpoint

class UNet(nn.Module):
    def __init__(self, n_channels=3, base_ch=64, text_dim=96, time_dim=256,
                 bilinear=True, n_heads=4, use_checkpointing=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.time_dim = time_dim
        self.base_ch = base_ch
        self.use_ckpt = use_checkpointing

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        factor = 2 if bilinear else 1

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.inc = DoubleConv(n_channels, c1, time_dim)  # x1: c1 canali
        self.down1 = Down(c1, c2, time_dim)  # x2: c2 canali
        self.down2 = Down(c2, c3 // factor, time_dim)  # x3 (bottleneck): c3 // factor canali

        self.mid_attn = CrossAttention(c3 // factor, text_dim, n_heads)  # combacia con x3

        self.up1 = Up(c3, c2 // factor, time_dim, bilinear)  # concat: (c3//factor) + c2 = c3
        #self.up_attn = CrossAttention(c2 // factor, text_dim, n_heads)  # combacia con l'output di up1
        # ulteriore strato di crossattention, migliora il condizionamento del testo ma riduce del 20% le prestazioni

        self.up2 = Up(c2, c1, time_dim, bilinear)  # concat: (c2//factor) + c1 = c2

        self.outc = OutConv(c1, n_channels)  # combacia con l'output di up2

    def use_checkpointing(self):
        """Attiva il gradient checkpointing (risparmia VRAM, costa tempo extra)."""
        self.use_ckpt = True

    def forward(self, x, t, text_hidden, text_pad_mask):
        """
                x:              (B, 3, H, W) noisy image
                t:              (B,) integer timesteps
                text_hidden:    (B, T, text_dim) from your text encoder
                text_pad_mask:  (B, T) bool, True at PAD positions
                """
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))

        if self.use_ckpt and self.training:
            # use_reentrant=False è la modalità consigliata da PyTorch (più
            # robusta con BatchNorm/RNG rispetto al vecchio default reentrant)
            x1 = checkpoint(self.inc, x, t_emb, use_reentrant=False)
            x2 = checkpoint(self.down1, x1, t_emb, use_reentrant=False)
            x3 = checkpoint(self.down2, x2, t_emb, use_reentrant=False)

            x3 = checkpoint(self.mid_attn, x3, text_hidden, text_pad_mask, use_reentrant=False)

            x = checkpoint(self.up1, x3, x2, t_emb, use_reentrant=False)
            #x = checkpoint(self.up_attn, x, text_hidden, text_pad_mask, use_reentrant=False)
            x = checkpoint(self.up2, x, x1, t_emb, use_reentrant=False)
        else:
            x1 = self.inc(x, t_emb)
            x2 = self.down1(x1, t_emb)
            x3 = self.down2(x2, t_emb)

            x3 = self.mid_attn(x3, text_hidden, text_pad_mask)

            x = self.up1(x3, x2, t_emb)
            x = self.up_attn(x, text_hidden, text_pad_mask)
            x = self.up2(x, x1, t_emb)

        logits = self.outc(x)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())