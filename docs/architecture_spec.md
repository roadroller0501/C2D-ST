# C2D-ST architecture specification

This page specifies every component of C2D-ST in enough detail to re-implement it, and explains
what each key of the released YAML configs controls. Module-level PyTorch pseudo-code for the same
components is in [`reference_pseudocode.md`](reference_pseudocode.md). An analytic parameter count of
this specification agrees with the paper's Tables 2–4 to within rounding for every variant:

| Variant | Spec count | Paper |
|---|---|---|
| C2D-ST | 6.913 M | 6.92 M |
| Axial → 1D | 9.754 M | 9.76 M |
| Axial → NA | 6.944 M | 6.95 M |
| Time-only axial | 6.913 M | 6.92 M |
| RPB only | 6.823 M | 6.83 M |
| 2D NA → ConvNeXt | 6.967 M | 6.97 M |
| 1D NA → ConvNeXt | 6.936 M | 6.94 M |

(Counts exclude the loss anchors. For C2D-ST the exact count is 6,913,172 elements, of which 32 are
the frozen stage-1 aggregation weights. Every row is 0.003–0.007 M below the paper's rounded value;
the source of that small, roughly constant residual has not been identified. Every *difference*
between variants matches the paper to the reported precision.)

Where the 6.91 M of C2D-ST sit (analytic count of the specification below):

| Component | Params | Share |
|---|---|---|
| Stem (Conv2d 1→32 + BN) | 0.000 M | 0.0 % |
| Stage 1 (width 48, 3 blocks) | 0.094 M | 1.4 % |
| Stage 2 (width 72, 3 blocks) | 0.208 M | 3.0 % |
| Stage 3 (width 96, 3 blocks) | 0.360 M | 5.2 % |
| Stage 4 (width 144, 3 blocks) | 0.789 M | 11.4 % |
| Stage 5 (width 192, 4 blocks) | 1.865 M | 27.0 % |
| Final aggregation + BN + Linear 2560→144 | 0.389 M | 5.6 % |
| Final 1D temporal stage (8 blocks, width 144) | 2.040 M | 29.5 % |
| Attentive statistics pooling (on 1296-dim) | 0.665 M | 9.6 % |
| Projector (BN + Linear 2592→192) | 0.503 M | 7.3 % |
| **Total** | **6.913 M** | |

Each stage's count includes its aggregation weights, in/out norms and the two projections. The
2D backbone is 48 % of the model; the final 1D stage, despite its width of 144, is 30 % because
of its 8 blocks. Tensor shapes are written channel-last as
`(B, T, F, C)` unless stated otherwise; `B` batch, `T` frames, `F` mel bins, `C` channels.

---

## 1. Front-end (`frontend: melspec_torch_fix`)

```yaml
frontend_conf:
    preemp: true        # first-order pre-emphasis y[t] = x[t] − 0.97·x[t−1] (reflect-padded)
    n_fft: 512
    log: true           # log(mel + 1e-6)
    win_length: 400     # 25 ms @ 16 kHz, Hamming window
    hop_length: 160     # 10 ms
    f_min: 20
    f_max: 7600
    n_mels: 80          # HTK mel scale (torchaudio MelSpectrogram)
    normalize: mvn      # per-utterance, per-bin mean/variance normalisation over time
```

Computed in fp32 with no gradient via `torchaudio.transforms.MelSpectrogram` with its defaults
for the unlisted arguments: power spectrogram (`power = 2`), `center = True` with reflect padding,
no filter-bank normalisation (`norm = None`), HTK mel scale. No SpecAugment, no global CMVN.
Output `(B, 2T, 80)`.

---

## 2. Encoder (`encoder: redimnat1_encoder`)

### 2.1 Stem

```yaml
stem_size: 32
stem_kernel_size: [3, 2]     # (freq, time)
stem_stride:      [1, 2]
stem_padding:     [1, 0]
redim_stem_norm_in_type: bn
```

`Conv2d(1 → 32, k=(3,2), s=(1,2), p=(1,0), bias=False)` on `(B, 1, F=80, 2T)` followed by
BatchNorm2d. Frequency is preserved (80), **time is halved** (20 ms frames from here on). The
output is permuted to `(B, T, F, C)` and flattened to the *1D form* `(B, T, F·C = 2560)`. This
flattened form is what the list of stage outputs carries.

### 2.2 Stage table (`stage_config: s5_d2-2_s_16l_k7_div24_div48_g1`)

| Stage | freq stride | F | grid channels c (c·F = 2560) | transformer width d | blocks | NA heads | axial heads / axis | axial block |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 80 | 32 | 48 | 3 | 2 | 1 | 3rd (last) |
| 2 | 2 | 40 | 64 | 72 | 3 | 3 | 1 | 3rd |
| 3 | 1 | 40 | 64 | 96 | 3 | 4 | 2 | 3rd |
| 4 | 1 | 40 | 64 | 144 | 3 | 6 | 3 | 3rd |
| 5 | 2 | 20 | 128 | 192 | 4 | 8 | 4 | 4th |

16 layers total. NA heads = d/24. Axial heads per axis are 1/1/2/3/4 — the `div48` in the name is a
convention, not an exact formula: stage 2 (d = 72) has 1 head per axis with `d_h = 36`, all other
stages have `d_h = 24`. Name decoding: `s5` five stages,
`d2-2` two ↓2 frequency down-samplings (stages 2 and 5), `s` small width set, `16l` 16 layers,
`k7` 7×7 NA window, `div24`/`div48` head divisors, `g1` one global (axial) block per stage
(`g0` = none). Frequency down-sampling is ReDimNet-style: when F is halved, c is doubled by the
reshape of the flattened 2560-dim vector, so no parameters are involved.

### 2.3 One stage

Input: the list `[stem_out, stage_1_out, …, stage_{i−1}_out]`, each `(B, T, 2560)`.

1. **Stage-wise aggregation** (`redim_aggregation_type: ws2d`). Reshape every list element to
   `(B, T, F_i, c_i)` and take a softmax-weighted sum with learnable weights of shape
   `(N_i, c_i)` (one weight per input per channel, softmax over the N_i inputs; no weight decay).
   For stage 1 (N = 1) the weight is a frozen constant.
2. `redim_proj_norm_in_type: bn` — BatchNorm2d over c_i (skipped when N_i = 1).
3. **2D projection in**: `Linear(c_i → d)` applied per grid position.
4. **Transformer blocks** (pre-norm):

   ```
   x = x + DropPath( LS₁ · Attn( LN(x) ) )
   x = x + DropPath( LS₂ · FFN ( LN(x) ) )
   ```
   - `redim_block_norm_type: ln` — LayerNorm over d. **All LayerNorm and BatchNorm layers inside
     the encoder use `eps = 1e-6`** (BatchNorm momentum 0.1); the pooling and projector BatchNorms
     use the PyTorch default `eps = 1e-5`.
   - `ls_init_2d: 1e-5` — LayerScale, per-channel γ initialised to 1e-5, no weight decay.
   - `ffn_2d_type: swiglu`, `redim_mult: 4`, `activation_type: swish`:
     `FFN(x) = W₂ · ( SiLU(W₁ᵃx) ⊙ W₁ᵇx )`, hidden size `h = ⌊(4d)·2/3⌋`, `W₁ᵃ,W₁ᵇ ∈ ℝ^{h×d}`
     (implemented as one `Linear(d → 2h)` split in two), `W₂ ∈ ℝ^{d×h}`, all with bias.
   - `drop_path_rate: 0.1`, `drop_path_policy: linear`: layer ℓ ∈ {0,…,23} (16 backbone +
     8 temporal layers, in forward order) uses stochastic-depth rate `0.2·ℓ/23`.
   - Attention is 2D neighborhood attention (§2.4) except in the last block of the stage, which
     is axial attention (§2.5).
5. `redim_proj_norm_out_type: bn` — BatchNorm2d over d.
6. **2D projection out**: `Linear(d → c_i, bias=False)`; flatten back to `(B, T, 2560)` and
   append to the list.

### 2.4 2D neighborhood attention with surrounding padding (`na_type_2d: nattn2d_decay_rpb_v1`)

Window `k = 7×7`, heads `H = d/24`, `d_h = 24`, `qkv = Linear(d → 3d)`, `proj = Linear(d → d)`,
both with bias. Learnable parameters:

- relative position bias **RPB** `B ∈ ℝ^{H × 13 × 13}` (no weight decay), trunc-normal(0.02) init;
- value-side relative positional encoding **RPE** `P ∈ ℝ^{H × 49 × d_h}`, trunc-normal(0.02) init.

Pseudo-code (per head; `p` a grid position, `Ω(p)` the 49 offsets of the 7×7 window centred at p):

```
q, k, v = split(qkv(x))                        # (B, H, F, T, d_h) each
q = q / sqrt(d_h)
q, k, v = zero_pad(q, k, v, 3 on each side of F and of T)          # "surrounding padding"
logits[p, o] = <q_p, k_{p+o}> + B[o]           for o ∈ Ω(p)          # (B, H, F+6, T+6, 49)
logits[p, o] = −inf   if  p+o  lies in the padded border           # mask padded keys
α = softmax_o(logits)
out_p = Σ_o α[p, o] · ( v_{p+o} + P[o] )       # = Σ α·v  +  Σ α·P
out = crop border (back to F×T) → merge heads → proj
```

Differences from standard NAT: (i) the window is always centred at the query (stand-alone
self-attention style) and keys that fall in the zero-padded border are masked out, instead of
NAT's window shifting at the grid edges; (ii) the value-side term `Σ_o α[p,o]·P[o]`. The padding
mask is applied on the materialised `(…, 49)` logit tensor, which is why the implementation uses
the two-step unfused `na2d_qk`/`na2d_av` kernels. Since the window is small, the same result can
be obtained with an `unfold`-based implementation at higher memory cost.

`na_type_2d: nattn2d_rpb_v1` (RPB-only ablation, Table 3) is identical without `P`.

### 2.5 Axial attention (`ga_type_2d: axial_static_v1`)

`qkv = Linear(d → 3d)`, `proj = Linear(d → d)`, heads per axis `n = 1/1/2/3/4` for the five stages,
`d_h = d/(2n)` = 24/36/24/24/24.

```
packed_f, packed_t = split(qkv(x) into two halves)         # 3d/2 channels each
# inside each half: view as (n heads, 3·d_h) and split q/k/v per head
(q₁,k₁,v₁) = unpack(packed_f);  (q₂,k₂,v₂) = unpack(packed_t)   # d/2 per q, k, v
# frequency axis: fold T into batch
a₁ = Attn( RoPE(q₁), RoPE(k₁), v₁ )  over F  →  (B·T, F, d/2)
# time axis: fold F into batch
a₂ = Attn( RoPE(q₂), RoPE(k₂), v₂ )  over T  →  (B·F, T, d/2)
out = proj( concat(a₁, a₂) )                               # (B, T, F, d)
```

- Standard scaled-dot-product attention per axis (flash-attn or SDPA; no dropout, no bias).
- **RoPE** on q and k of both axes with base θ = 500, frequencies
  `ω_j = θ^{−2j/d_h}`, j = 0…d_h/2−1, fixed (not trained), separate buffers per axis.
- **NTK extrapolation on the time axis** (`extra_type: ntk`, `extra_len: L₀`). At inference
  only, if the sequence length `L > L₀`, the RoPE frequencies of the time axis with `|ω| < 1` are
  replaced by `sign(ω)·|ω|^e` with `e = 1 + (1 + 2/(d_h − 2)) · ln(L/L₀) / ln θ`
  (NTK-aware scaling written in the frequency domain). `L₀ = 150` for pretraining (3 s → 150
  frames after the stem) and `300` for LMFT (6 s). The frequency axis is never longer than at
  training time and is not rescaled.
- `window_size` unset → no local windowing.

`ga_type_2d: axial_static_timeonly_v1` (ablation): only the time branch, with `2n` heads of
the same `d_h` over all d channels — same width, head count and parameter count.

### 2.6 Backbone → temporal stage

After stage 5 the list has 6 tensors `(B, T, 2560)`. They are combined by a softmax-weighted sum
with learnable weights `(6, 2560)` (per input, per channel; no weight decay), followed by
`redim_fin_norm_type: bn` (BatchNorm1d over 2560) and `Linear(2560 → 144)`.

### 2.7 Final 1D temporal stage

```yaml
output_size_1d: 144
nattn_heads_1d: 9          # d_h = 16
gattn_heads_1d: 3          # d_h = 48
linear_units_1d: 576       # SwiGLU hidden = ⌊576·2/3⌋ = 384
kernel_size_1d: 31
num_blocks_1d: 8
gattn_freq_1d: 4           # blocks 4 and 8 are global → NA NA NA GA NA NA NA GA
nattn_type_1d: nattn1d_decay_rpb_v1
gattn_type_1d: gattn1d_static_v1
norm_1d_type: ln
ffn_1d_type: swiglu
ls_init_1d: 1e-5
fin_norm_1d_type: identity
mfa: true
mfa_freq: 1
```

Blocks have the same pre-norm / LayerScale / DropPath / SwiGLU structure as §2.3.

- **1D NA** (`nattn1d_decay_rpb_v1`): the 1D analogue of §2.4 — window 31, zero-pad 15 frames
  on each side, masked padded keys, RPB `B ∈ ℝ^{9×61}`, value-RPE `P ∈ ℝ^{9×31×16}`.
- **1D global attention** (`gattn1d_static_v1`): full self-attention over T with RoPE
  (θ = 500, fixed) and the same NTK time-axis extrapolation as §2.5 (`extra_len` 150 / 300).
- **Multi-layer feature aggregation**: the stage input (after `Linear(2560→144)`) and all 8
  block outputs are concatenated along channels → `(B, T, 9·144 = 1296)`. No final norm.

### 2.8 Initialisation (`init_style: v5`)

All `Conv1d/Conv2d/Linear` weights: Kaiming-uniform with `a = √5`, fan-in (PyTorch default);
biases 0. RPB/RPE: trunc-normal(std 0.02, ±0.04). LayerScale γ = 1e-5. Aggregation weights 0
(uniform after softmax). Loss anchors: Xavier-uniform.

---

## 3. Pooling, projector, loss

| Config | Specification |
|---|---|
| `pooling: chn_attn_stat` | ECAPA attentive statistics pooling on `(B, 1296, T)`: context = concat(x, mean_t(x), std_t(x)) → `Conv1d(3·1296 → 128, k=1)` → ReLU → BatchNorm1d → `Conv1d(128 → 1296, k=1)` → softmax over t → attention-weighted mean μ and std σ (the variance is clamped to [1e-4, 1e4] before the square root, both for the context statistics and for the pooled output) → concat → 2592-dim. |
| `projector: rawnet3`, `output_size: 192` | `BatchNorm1d(2592) → Linear(2592 → 192)` = speaker embedding. |
| `loss: sphereface2` | SphereFace2, C-type (CosFace-style additive margin on the cosine): `scale 32`, `margin 0.2` (pretrain) / `0.3` (LMFT), `t 3`, `lambda_ 0.7`, learnable scalar bias initialised to 0. Binary-classification formulation with `g(z) = 2·((z+1)/2)^t − 1`: loss = λ·softplus(−s·(g(cos_y) − m) − b) + (1−λ)·Σ_{j≠y} softplus(s·(g(cos_j) + m) + b), averaged over the batch. Anchors L2-normalised. |

**`jt: true` (LMFT only).** Pretraining is on the speed-perturbed set, so the pretrained anchor
matrix has `3 × 5994` rows (ordering: original speaker ids, then the 0.9× copies, then the 1.1×
copies). With `jt: true` the LMFT loss allocates a `(3·5994, 192)` matrix so that the pretrained
anchors load without shape mismatch, and uses only the first 5994 rows (speed-1.0 anchors) in
the forward pass; the other two thirds receive no gradient.

---

## 4. Preprocessor (`preprocessor: spkfull`)

```yaml
preprocessor_conf:
  target_duration: 3.0      # s; random crop, wrap-padded if the utterance is shorter (6.0 in LMFT)
  sample_rate: 16000
  num_eval: 5               # number of evenly spaced crops for validation scoring (3 in LMFT)
  noise_apply_prob: 0.5     # 0.0 in LMFT
  noise_info:               # [category prob, list, [n_min, n_max] files mixed, [snr_min, snr_max] dB]
  - [1.0, musan_speech, [4, 7], [13, 20]]
  - [1.0, musan_noise,  [1, 1], [0, 15]]
  - [1.0, musan_music,  [1, 1], [5, 15]]
  rir_apply_prob: 0.5       # 0.0 in LMFT
  rir_scp: rirs             # RIRS_NOISES simulated small + medium rooms (no large rooms)
```

Order per training sample: crop → (with p = 0.5) convolve with a random RIR (energy-normalised)
→ (with p = 0.5) pick one MUSAN category uniformly, mix `n` random files at a random SNR from the
category's range. Noise files shorter than the crop are wrap-padded, longer ones are randomly
offset.

---

## 5. Optimisation

```yaml
optim: adamw
optim_conf: {lr: 0.008, weight_decay: 0.05, amsgrad: false}
exclude_weight_decay: true
exclude_weight_decay_conf: {normalization_weight_decay: false, bias_weight_decay: false}
scheduler: CosineAnnealingWarmupRestartsFix
scheduler_conf:
  first_cycle_steps: 42660    # = total optimizer steps → one cosine cycle, no restart
  cycle_mult: 2.0             # unused
  max_lr: 0.008
  min_lr: 0.00000008
  warmup_steps: 4266          # 10 %: linear from min_lr to max_lr, then cosine to min_lr
  gamma: 1.0
```

Weight decay is not applied to normalisation parameters, biases, RPB, LayerScale γ and the
aggregation weights; it **is** applied to the value-side RPE tables and all other weights.
Gradient clipping is effectively disabled (`grad_clip: 9999`). There is **no dropout** anywhere
in the model (attention, projections, FFN); stochastic depth is the only regulariser besides
weight decay and data augmentation.

**Batching.** `batch_size` in the YAML is the *global* batch (split across GPUs); the effective
batch is `batch_size × accum_grad`. Steps per epoch = `num_iters_per_epoch / accum_grad`.

| | `batch_size` | `accum_grad` | effective | `num_iters_per_epoch` | steps/epoch | epochs | total steps | GPUs in the YAML |
|---|---|---|---|---|---|---|---|---|
| Pretrain | 64 | 24 | **1 536** | 51 187 (3 276 027 utts / 64) | 2 133 | 20 | 42 660 | 2 |
| LMFT | 64 | 6 | **384** | 17 062 (1 092 009 utts / 64) | 2 844 | 2 | 5 688 | 4 |

The YAMLs carry commented-out blocks for other GPU counts that keep the effective batch fixed.

Other flags: `use_amp: true`, `iterator_type: random` (random utterance sampling, one random
crop each), `drop_last_iter: true`, `seed: 0`. Training runs for the fixed number of epochs and
the final-epoch weights (`latest.pth`) are what is evaluated.
