"""Reliability-gated multimodal fusion network for solar flare prediction."""

import torch
import torch.nn as nn

EUV_CHANNELS = ("94", "131", "171", "193", "211", "304", "335", "1700")
IMG_SIZE = 64
WIDTH = 64
DROPOUT = 0.3


# Turns one modality's images into a small feature map. Three conv stages, each
# halving the resolution, so a 64x64 patch comes out as 8x8 with `width` channels.
# The magnetogram and the EUV stack get one of these each, with separate weights.
class Encoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1),
                nn.GroupNorm(8, co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1),
                nn.GroupNorm(8, co),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, width))

    def forward(self, x):
        return self.net(x)


# Decides how much to believe a modality, pixel by pixel. It looks at the feature
# map and predicts a mask between 0 and 1, then scales the features by it, so the
# network can ignore one input in part of the frame and keep it elsewhere.
# Returns both the gated features and the mask, since the mask is worth plotting.
class ReliabilityGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(width, width // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        trust = self.head(feat)
        return feat * trust, trust


# The whole flare predictor. Encodes both modalities, gates them, lets them attend
# to each other, then pools everything through a small transformer into one logit.
# `mode` picks the ablation: mag or euv run a single branch, concat fuses without
# the gate or the cross-attention, full is everything. One class covers all four so
# every variant trains with an identical backbone and parameter count.
class RGFusion(nn.Module):
    def __init__(self, width=WIDTH, n_euv=len(EUV_CHANNELS), dropout=DROPOUT):
        super().__init__()
        w = width
        self.mode = "full"
        self.enc_mag = Encoder(1, w)
        self.enc_euv = Encoder(n_euv, w)
        self.gate_mag = ReliabilityGate(w)
        self.gate_euv = ReliabilityGate(w)
        self.attn_m = nn.MultiheadAttention(w, 4, dropout=dropout, batch_first=True)
        self.attn_e = nn.MultiheadAttention(w, 4, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=w, nhead=4, dim_feedforward=2 * w,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, w))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(w, 1))
        self.last_trust = {}

    @staticmethod
    def _tok(feat):
        return feat.flatten(2).transpose(1, 2)

    def set_mode(self, mode):
        self.mode = mode
        return self

    def forward(self, mag, euv):
        use_mag = self.mode in ("mag", "concat", "full")
        use_euv = self.mode in ("euv", "concat", "full")
        fusion = self.mode in ("concat", "full")
        gate = self.mode == "full"
        cross = self.mode == "full"
        self.last_trust = {}

        tok_mag = tok_euv = None
        if use_mag:
            fm = self.enc_mag(mag)
            if gate:
                fm, tm = self.gate_mag(fm)
                self.last_trust["mag"] = tm.detach()
            tok_mag = self._tok(fm)
        if use_euv:
            fe = self.enc_euv(euv)
            if gate:
                fe, te = self.gate_euv(fe)
                self.last_trust["euv"] = te.detach()
            tok_euv = self._tok(fe)

        if cross and fusion:
            am, _ = self.attn_m(tok_mag, tok_euv, tok_euv)
            ae, _ = self.attn_e(tok_euv, tok_mag, tok_mag)
            tok_mag = tok_mag + am
            tok_euv = tok_euv + ae

        if fusion:
            tokens = torch.cat([tok_mag, tok_euv], dim=1)
        else:
            tokens = tok_mag if use_mag else tok_euv

        b = tokens.size(0)
        tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0]).squeeze(1)


def enable_dropout(model):
    """Reactivate dropout layers after eval() for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# Shows where a branch was looking when it made its call. Hooks one encoder's last
# conv block to grab activations on the way forward and gradients on the way back,
# then weights the channels by those gradients to get a heatmap. Call close() when
# done, otherwise the hooks stay attached to the model.
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.a = None
        self.g = None
        self._h1 = layer.register_forward_hook(self._fwd)
        self._h2 = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.a = o.detach()

    def _bwd(self, m, gi, go):
        self.g = go[0].detach()

    def __call__(self, mag, euv):
        self.model.eval()
        self.model.zero_grad()
        logit = self.model(mag, euv)
        logit.sum().backward()
        w = self.g.mean(dim=(2, 3), keepdim=True)
        raw = (w * self.a).sum(1)
        cam = torch.relu(raw)
        if float(cam.amax()) <= 1e-8:
            cam = raw.abs()
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return cam[0].cpu().numpy()

    def close(self):
        self._h1.remove()
        self._h2.remove()
