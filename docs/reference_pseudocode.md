# Reference pseudo-code

PyTorch-style pseudo-code for every module of C2D-ST, written from the specification in
[`architecture_spec.md`](architecture_spec.md). It is **not the training code**: it mirrors the
in-house implementation module by module but uses only plain PyTorch (no NATTEN, no flash-attn),
has not been used to produce the paper's numbers, and is not checkpoint-compatible with the
internal implementation: the `qkv` projections are unpacked per head as in the original, but
parameter names, buffer shapes and the compact RPB tables differ (see §2). Shapes were checked by
hand, not by running the code. The analytic parameter counts in
[`architecture_spec.md`](architecture_spec.md) refer to the specification with full-size RPB tables;
this code has 8,100 fewer RPB elements.

Conventions: `B` batch, `T` frames (after the stem, 20 ms), `F` mel bins on the current grid,
`C` channels. Grid tensors are channel-last `(B, T, F, C)`; the flattened "1D form" is `(B, T, F·C)`.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 1. Shared building blocks

```python
class LayerScale(nn.Module):                     # per-channel residual scaling, init 1e-5, no weight decay
    def __init__(self, dim, init=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))
    def forward(self, x):
        return x * self.gamma


def drop_path(x, p, training):                   # stochastic depth, per sample
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep) / keep
    return x * mask


class SwiGLU(nn.Module):                         # ffn_type: swiglu ; hidden = floor(units*2/3)
    def __init__(self, dim, units):              # units = 4*dim in the backbone, 576 in the 1D stage
        super().__init__()
        h = int(units * 2 // 3)
        self.fc1 = nn.Linear(dim, 2 * h)
        self.fc2 = nn.Linear(h, dim)
    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


EPS = 1e-6                                       # every LayerNorm / BatchNorm inside the encoder uses eps = 1e-6


class ChannelLastBN(nn.Module):                  # BatchNorm over the last (channel) dim of (B, ..., C)
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c, eps=EPS, momentum=0.1)
    def forward(self, x):
        shape = x.shape
        return self.bn(x.reshape(-1, shape[-1])).reshape(shape)


class Block(nn.Module):                          # pre-norm transformer block used everywhere
    def __init__(self, dim, attn, ffn, ls_init=1e-5, dpr=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=EPS), nn.LayerNorm(dim, eps=EPS)
        self.attn, self.ffn = attn, ffn
        self.ls1, self.ls2 = LayerScale(dim, ls_init), LayerScale(dim, ls_init)
        self.dpr = dpr
    def forward(self, x):
        x = x + drop_path(self.ls1(self.attn(self.norm1(x))), self.dpr, self.training)
        x = x + drop_path(self.ls2(self.ffn(self.norm2(x))), self.dpr, self.training)
        return x
```

Drop-path schedule (`drop_path_rate: 0.1`, `linear`): layer `ℓ` of the 24 attention layers
(16 backbone + 8 temporal, in forward order) gets `dpr_ℓ = 2·0.1·ℓ/23`.

---

## 2. 2D neighborhood attention with surrounding padding (`nattn2d_decay_rpb_v1`)

Every query attends to the `k×k` window centred on itself. The grid is zero-padded by `k//2` so
that border queries keep a full, centred window; keys that fall into the padding are masked out.
The implementation below uses `F.unfold` to materialise the neighbourhoods; the internal code does
the same with NATTEN's unfused `na2d_qk` / `na2d_av`. The computation is the same, but the RPB
table here is stored compactly with one entry per used offset (`k·k`), whereas the internal code
keeps NATTEN's full `(2k−1)×(2k−1)` table (2D) and `2k−1` (1D) per head and only ever indexes its
centre block — 8,100 fewer parameters in total for the main model, and a different tensor layout to
remap if weights were ever transferred.

```python
class NeighborhoodAttention2D(nn.Module):
    def __init__(self, dim, heads, k=7, value_rpe=True):
        super().__init__()
        self.h, self.dh, self.k, self.pad = heads, dim // heads, k, k // 2
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        # relative position bias for the k*k offsets of a centred window
        # (the internal code stores NATTEN's (2k-1)×(2k-1) table and uses only its centre k×k block)
        self.rpb = nn.Parameter(torch.zeros(heads, k * k))
        nn.init.trunc_normal_(self.rpb, std=0.02, a=-0.04, b=0.04)
        # value-side relative positional encoding (Table 3 ablation removes this)
        self.rpe = nn.Parameter(torch.zeros(heads, k * k, self.dh)) if value_rpe else None
        if value_rpe:
            nn.init.trunc_normal_(self.rpe, std=0.02, a=-0.04, b=0.04)

    def forward(self, x):                                   # x: (B, T, F, C)
        B, T, Fq, C = x.shape
        h, dh, k, P = self.h, self.dh, self.k, self.k * self.k
        q, kk, v = self.qkv(x).view(B, T, Fq, h, 3, dh).unbind(4)          # per-head [q|k|v] packing, each (B, T, F, h, dh)
        q = q * dh ** -0.5

        def as_image(t):                                     # (B, T, F, h, dh) -> (B*h, dh, T, F)
            return t.permute(0, 3, 4, 1, 2).reshape(B * h, dh, T, Fq)

        q_img = as_image(q)
        # surrounding padding + neighbourhood gathering: (B*h, dh, P, T, F)
        k_win = F.unfold(as_image(kk), k, padding=self.pad).view(B * h, dh, P, T, Fq)
        v_win = F.unfold(as_image(v),  k, padding=self.pad).view(B * h, dh, P, T, Fq)

        logits = (q_img.unsqueeze(2) * k_win).sum(1).view(B, h, P, T, Fq)   # <q_p, k_{p+o}>
        logits = logits + self.rpb.view(1, h, P, 1, 1)                      # + B[o]
        valid = F.unfold(x.new_ones(1, 1, T, Fq), k, padding=self.pad).view(1, 1, P, T, Fq)
        logits = logits.masked_fill(valid == 0, float("-inf"))              # mask padded keys
        attn = logits.softmax(dim=2)                                        # (B, h, P, T, F)

        out = (attn.view(B * h, 1, P, T, Fq) * v_win).sum(2).view(B, h, dh, T, Fq)   # Σ α·v
        if self.rpe is not None:
            out = out + torch.einsum("bhptf,hpd->bhdtf", attn, self.rpe)             # + Σ α·P[o]
        out = out.permute(0, 3, 4, 1, 2).reshape(B, T, Fq, C)
        return self.proj(out)
```

`nattn2d_rpb_v1` = `value_rpe=False`.

---

## 3. 1D neighborhood attention (`nattn1d_decay_rpb_v1`)

Same construction along time only; window 31, 9 heads of 16 in the final stage.

```python
class NeighborhoodAttention1D(nn.Module):
    def __init__(self, dim, heads, k=31, value_rpe=True):
        super().__init__()
        self.h, self.dh, self.k, self.pad = heads, dim // heads, k, k // 2
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros(heads, k))
        nn.init.trunc_normal_(self.rpb, std=0.02, a=-0.04, b=0.04)
        self.rpe = nn.Parameter(torch.zeros(heads, k, self.dh)) if value_rpe else None
        if value_rpe:
            nn.init.trunc_normal_(self.rpe, std=0.02, a=-0.04, b=0.04)

    def forward(self, x):                                   # x: (B, T, C)
        B, T, C = x.shape
        h, dh, k = self.h, self.dh, self.k
        q, kk, v = self.qkv(x).view(B, T, h, 3, dh).unbind(3)                # (B, T, h, dh)
        q = q * dh ** -0.5
        kk = F.pad(kk.permute(0, 2, 3, 1), (self.pad, self.pad)).unfold(3, k, 1)   # (B, h, dh, T, k)
        v  = F.pad(v.permute(0, 2, 3, 1),  (self.pad, self.pad)).unfold(3, k, 1)
        logits = (q.permute(0, 2, 1, 3).unsqueeze(-1) * kk.permute(0, 1, 3, 2, 4)).sum(3)   # (B, h, T, k)
        logits = logits + self.rpb.view(1, h, 1, k)
        valid = F.pad(x.new_ones(1, 1, 1, T), (self.pad, self.pad)).unfold(3, k, 1)[0, 0]   # (1, T, k)
        logits = logits.masked_fill(valid.unsqueeze(0) == 0, float("-inf"))
        attn = logits.softmax(-1)                                            # (B, h, T, k)
        out = torch.einsum("bhtk,bhdtk->bhtd", attn, v)                       # (B, h, T, dh)
        if self.rpe is not None:
            out = out + torch.einsum("bhtk,hkd->bhtd", attn, self.rpe)
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, T, C))
```

---

## 4. Rotary embedding with NTK length extrapolation

```python
def rope_freqs(dh, base=500.0):                  # fixed buffers ("static"), one set per axis
    return 1.0 / base ** (torch.arange(0, dh, 2).float() / dh)               # (dh/2,)


def ntk_extrapolate(freqs, L, L0, base, dh):     # inference only, time axis only, when L > L0
    if L <= L0:
        return freqs
    e = 1.0 + (1.0 + 2.0 / (dh - 2)) * math.log(L / L0) / math.log(base)
    scaled = freqs.sign() * freqs.abs() ** e                                # lower frequencies -> longer wavelengths
    return torch.where(freqs.abs() >= 1, freqs, scaled)


def apply_rope(x, freqs):                        # x: (B', L, h, dh) ; rotates adjacent channel pairs
    L = x.shape[1]
    ang = torch.arange(L, device=x.device).float()[:, None] * freqs[None, :]   # (L, dh/2)
    rot = torch.polar(torch.ones_like(ang), ang)                                # e^{i·ang}
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))         # (B', L, h, dh/2)
    return torch.view_as_real(xc * rot[None, :, None, :]).flatten(-2).type_as(x)


def attend(q, k, v):                             # q,k,v: (B', L, h, dh) -> (B', L, h*dh)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).flatten(2)
```

---

## 5. Axial attention (`axial_static_v1`)

Half of the channels attend along frequency (T folded into the batch), the other half along time
(F folded into the batch); each half has `n` heads of `d_h = C/(2n)` — with the stage table above,
`n` = 1/1/2/3/4 and `d_h` = 24/36/24/24/24. RoPE on both axes; NTK
extrapolation on the time axis only.

```python
class AxialAttention(nn.Module):
    def __init__(self, dim, heads_per_axis, base=500.0, extra_len=150):
        super().__init__()
        self.h, self.dh = heads_per_axis, dim // (2 * heads_per_axis)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("freqs_f", rope_freqs(self.dh, base))
        self.register_buffer("freqs_t", rope_freqs(self.dh, base))
        self.base, self.L0 = base, extra_len

    def _axis(self, qkv, along_time):            # qkv: (B, T, F, 3*dim/2)
        B, T, Fq, _ = qkv.shape
        if along_time:
            seq = qkv.transpose(1, 2).reshape(B * Fq, T, -1)                  # fold F into batch
            freqs = self.freqs_t
            if not self.training:
                freqs = ntk_extrapolate(freqs, T, self.L0, self.base, self.dh)
        else:
            seq = qkv.reshape(B * T, Fq, -1)                                  # fold T into batch
            freqs = self.freqs_f
        L = seq.shape[1]
        q, k, v = seq.view(seq.shape[0], L, self.h, 3, self.dh).unbind(3)     # (B', L, h, dh)
        out = attend(apply_rope(q, freqs), apply_rope(k, freqs), v)           # (B', L, dim/2)
        if along_time:
            return out.view(B, Fq, T, -1).transpose(1, 2)                     # (B, T, F, dim/2)
        return out.view(B, T, Fq, -1)

    def forward(self, x):                                                     # x: (B, T, F, C)
        qkv_f, qkv_t = self.qkv(x).chunk(2, dim=-1)
        return self.proj(torch.cat([self._axis(qkv_f, False), self._axis(qkv_t, True)], dim=-1))
```

**Time-only variant** (`axial_static_timeonly_v1`, Table 2): a single time branch over all `C`
channels with `2n` heads of the same `d_h` — `AxialAttention` with the frequency branch removed
and `self.h = 2 * heads_per_axis`; identical parameter count.

---

## 6. 1D global attention (`gattn1d_static_v1`)

```python
class GlobalAttention1D(nn.Module):              # final stage GA: 3 heads of 48, RoPE + NTK on time
    def __init__(self, dim, heads, base=500.0, extra_len=150):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("freqs", rope_freqs(self.dh, base))
        self.base, self.L0 = base, extra_len

    def forward(self, x):                                                     # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, self.h, 3, self.dh).unbind(3)
        freqs = self.freqs if self.training else ntk_extrapolate(self.freqs, T, self.L0, self.base, self.dh)
        return self.proj(attend(apply_rope(q, freqs), apply_rope(k, freqs), v))
```

---

## 7. Stage-wise aggregation and one backbone stage

```python
class WeightedSum(nn.Module):                    # softmax-weighted sum of N tensors, one weight per input per channel
    def __init__(self, n_inputs, channels):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_inputs, channels), requires_grad=n_inputs > 1)   # no weight decay
    def forward(self, xs):                       # list of N tensors (B, ..., channels)
        w = self.w.softmax(0)
        return sum(x * w[i] for i, x in enumerate(xs))


class Stage(nn.Module):
    """One spectral-temporal stage. Input: list of the stem output and all previous stage outputs,
    each in the flattened form (B, T, 2560). Output: this stage's result, also (B, T, 2560)."""
    def __init__(self, n_inputs, f, c, width, n_blocks, na_heads, ax_heads, dprs,
                 na_kernel=7, ffn_mult=4, extra_len=150, value_rpe=True):
        super().__init__()
        self.f, self.c = f, c
        self.agg = WeightedSum(n_inputs, c)                          # ws2d: weights per grid channel c
        self.norm_in = ChannelLastBN(c) if n_inputs > 1 else nn.Identity()
        self.proj_in = nn.Linear(c, width)                           # "2D projection"
        blocks = []
        for b in range(n_blocks):
            last = b == n_blocks - 1
            attn = (AxialAttention(width, ax_heads, extra_len=extra_len) if last
                    else NeighborhoodAttention2D(width, na_heads, na_kernel, value_rpe))
            blocks.append(Block(width, attn, SwiGLU(width, ffn_mult * width), dpr=dprs[b]))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = ChannelLastBN(width)
        self.proj_out = nn.Linear(width, c, bias=False)

    def forward(self, xs):
        B, T, _ = xs[0].shape
        grid = [x.view(B, T, self.f, self.c) for x in xs]           # reshape 1D form -> current grid
        x = self.proj_in(self.norm_in(self.agg(grid)))               # (B, T, f, width)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_out(self.norm_out(x))                          # (B, T, f, c)
        return x.flatten(2)                                          # back to 1D form (B, T, f*c)
```

Because `f·c = 2560` is constant, "frequency down-sampling" is just viewing the same 2560
numbers as a `(f/2, 2c)` grid in the next stage — there is no pooling layer.

---

## 8. The C2D-ST encoder

```python
class C2DST(nn.Module):
    # (freq stride, width, blocks, NA heads, axial heads per axis) -- stage_config s5_d2-2_s_16l_k7_div24_div48_g1
    STAGES = [(1, 48, 3, 2, 1), (2, 72, 3, 3, 1), (1, 96, 3, 4, 2), (1, 144, 3, 6, 3), (2, 192, 4, 8, 4)]

    def __init__(self, n_mels=80, stem_c=32, d1=144, n_blocks_1d=8, ga_every=4,
                 na1d_heads=9, ga_heads=3, ffn_units_1d=576, k1d=31,
                 drop_path=0.1, extra_len=150, value_rpe=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, stem_c, (3, 2), (1, 2), (1, 0), bias=False),
                                  nn.BatchNorm2d(stem_c, eps=EPS))      # (B,1,F,2T) -> (B,32,F,T)
        n_layers = sum(s[2] for s in self.STAGES) + n_blocks_1d          # 24
        dprs = [2 * drop_path * i / (n_layers - 1) for i in range(n_layers)]

        f, c, stages = n_mels, stem_c, []
        for i, (stride, width, nb, nh, ah) in enumerate(self.STAGES):
            f, c = f // stride, c * stride
            stages.append(Stage(i + 1, f, c, width, nb, nh, ah, dprs[:nb], extra_len=extra_len,
                                value_rpe=value_rpe))
            dprs = dprs[nb:]
        self.stages = nn.ModuleList(stages)

        fc = f * c                                                        # 2560
        self.fin_agg = WeightedSum(len(self.STAGES) + 1, fc)              # 6 inputs, per flattened channel
        self.fin_norm = ChannelLastBN(fc)
        self.fin_proj = nn.Linear(fc, d1)                                 # "1D projection" 2560 -> 144

        blocks1d = []
        for i in range(n_blocks_1d):
            attn = (GlobalAttention1D(d1, ga_heads, extra_len=extra_len) if (i + 1) % ga_every == 0
                    else NeighborhoodAttention1D(d1, na1d_heads, k1d, value_rpe))
            blocks1d.append(Block(d1, attn, SwiGLU(d1, ffn_units_1d), dpr=dprs[i]))
        self.blocks1d = nn.ModuleList(blocks1d)                           # NA NA NA GA NA NA NA GA
        self.out_dim = d1 * (n_blocks_1d + 1)                             # 1296

    def forward(self, mel):                       # mel: (B, 2T, 80) log-Mel, per-utterance MVN
        x = self.stem(mel.transpose(1, 2).unsqueeze(1))                  # (B, C, F, T)
        x = x.permute(0, 3, 2, 1).flatten(2)                              # (B, T, F*C) 1D form
        outs = [x]
        for stage in self.stages:
            outs.append(stage(outs))                                      # each stage sees all previous outputs
        x = self.fin_proj(self.fin_norm(self.fin_agg(outs)))              # (B, T, 144)
        feats = [x]
        for blk in self.blocks1d:
            x = blk(x)
            feats.append(x)                                               # multi-layer feature aggregation
        return torch.cat(feats, dim=-1).transpose(1, 2)                   # (B, 1296, T)
```

Initialisation (`init_style: v5`): PyTorch-default Kaiming-uniform (`a=√5`) for every conv/linear
weight and **zero biases** — the constructors above leave PyTorch's default uniform bias init in
place, so apply a reset loop over `nn.Conv2d/nn.Conv1d/nn.Linear` after building `C2DST` to match.
RPB/RPE trunc-normal(0.02), LayerScale 1e-5, aggregation weights 0. Pooling and projector are not
covered by that reset (PyTorch defaults).

---

## 9. Pooling, projector, loss

```python
class AttentiveStatsPooling(nn.Module):          # pooling: chn_attn_stat  (ECAPA-TDNN ASP)
    def __init__(self, c, bottleneck=128):
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(3 * c, bottleneck, 1), nn.ReLU(),
                                 nn.BatchNorm1d(bottleneck), nn.Conv1d(bottleneck, c, 1))
    def forward(self, x):                        # x: (B, C, T) -> (B, 2C)
        T = x.shape[-1]
        mu_g = x.mean(-1, keepdim=True).expand(-1, -1, T)
        sd_g = x.var(-1, keepdim=True).clamp(1e-4, 1e4).sqrt().expand(-1, -1, T)
        w = self.att(torch.cat([x, mu_g, sd_g], 1)).softmax(-1)
        mu = (x * w).sum(-1)
        sd = ((x ** 2 * w).sum(-1) - mu ** 2).clamp(1e-4, 1e4).sqrt()
        return torch.cat([mu, sd], 1)


class Projector(nn.Module):                      # projector: rawnet3 -> 192-d embedding
    def __init__(self, in_dim, emb_dim=192):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, emb_dim)
    def forward(self, x):
        return self.fc(self.bn(x))


class SphereFace2C(nn.Module):                   # loss: sphereface2, margin_type C
    def __init__(self, emb_dim, n_classes, scale=32.0, margin=0.2, t=3, lam=0.7, jt=False):
        super().__init__()
        rows = 3 * n_classes if jt else n_classes          # jt: hold the 3x-speaker pretrained anchors
        self.W = nn.Parameter(torch.empty(rows, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.bias = nn.Parameter(torch.zeros(1))
        self.n, self.s, self.m, self.t, self.lam, self.jt = n_classes, scale, margin, t, lam, jt

    def forward(self, emb, label):               # emb: (B, 192), label: (B,)
        W = self.W[: self.n] if self.jt else self.W          # jt: use only the speed-1.0 anchors
        cos = F.normalize(emb) @ F.normalize(W).t()          # (B, n)
        g = 2 * ((cos + 1) / 2) ** self.t - 1
        pos = self.lam * F.softplus(-(self.s * (g - self.m) + self.bias))
        neg = (1 - self.lam) * F.softplus(self.s * (g + self.m) + self.bias)
        onehot = F.one_hot(label, self.n).to(cos.dtype)
        return (onehot * pos + (1 - onehot) * neg).sum(1).mean()
```

Full model: `emb = Projector(2*1296)(AttentiveStatsPooling(1296)(C2DST()(mel)))`.

---

## 10. Ablation modules

### 10.1 ConvNeXt-style replacement (Table 4)

One NA transformer block is replaced by **two** of these (`convnext_expansion_rate: 3`), with the
same kernel (7×7 or 31) and the same drop-path rate as the block they replace.

```python
class ConvNeXtBlock2D(nn.Module):
    def __init__(self, c, k=7, expansion=3, ls_init=1e-5, dpr=0.0):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.LayerNorm(c, eps=EPS)
        self.pw1, self.pw2 = nn.Linear(c, expansion * c), nn.Linear(expansion * c, c)
        self.ls, self.dpr = LayerScale(c, ls_init), dpr
    def forward(self, x):                        # x: (B, T, F, C)
        y = self.dw(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.pw2(F.silu(self.pw1(self.norm(y))))
        return x + drop_path(self.ls(y), self.dpr, self.training)
# 1D version: nn.Conv1d(c, c, k, padding=k//2, groups=c) on (B, C, T)
```

### 10.2 Collapsed 1D transformer block (Table 2, "Axial → 1D")

Applied to the stage output in its flattened `(B, T, 2560)` form after the `N−1` NA blocks and
the output projection, replacing the axial block:

```python
class Stage1DTransformer(nn.Module):
    def __init__(self, in_dim, width, heads, dpr, extra_len=150):       # width = stage width, heads = 2*ax_heads
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.block = Block(width, GlobalAttention1D(width, heads, extra_len=extra_len),
                           SwiGLU(width, 4 * width), dpr=dpr)
        self.out_norm = nn.LayerNorm(width, eps=EPS)
        self.out_proj = nn.Linear(width, in_dim)
    def forward(self, x):                        # x: (B, T, 2560)
        return x + self.out_proj(self.out_norm(self.block(self.in_proj(x))))
```

The two `2560 ↔ width` projections per stage are the source of the +2.8 M parameters.
