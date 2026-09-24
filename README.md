# C2D-ST: Continuous 2D Spectral–Temporal Transformer for Speaker Verification

**Seongwook Ham\*, Thien-Phuc Doan** — Soongsil University, Republic of Korea
Interspeech 2026 (Sydney) · \*corresponding author · jsham90@gmail.com

> C2D-ST keeps the (C, F, T) spectral–temporal grid through every backbone stage, models global
> context with **axial attention along both frequency and time**, and confines temporal-only
> modeling to a single final stage. On VoxCeleb1-O/E/H it reaches **0.507 % average EER /
> 0.051 minDCF with 6.9 M parameters** — less than half the parameters of ReDimNet-B6 and a
> quarter of ECAPA2.

This page is the companion to the paper. It provides the **complete configuration** of every
system in the paper (training, large-margin fine-tuning, scoring), the ablation configurations,
and a component-level specification detailed enough to re-implement the model.

### Code availability

The training code is **not released**. It is an in-house fork of ESPnet-SPK whose neighborhood
attention relies on a specific NATTEN release that still exposes the *unfused* two-step
`na2d_qk` → `na2d_av` (and `na1d_qk` → `na1d_av`) functional API. We need the materialised
query–key attention logits to apply the surrounding-padding mask of PCF-NAT; this API has since
been removed in favour of fused kernels,
and the required version does not build against current PyTorch. Rather than publish code that
cannot be installed, we document the model precisely in [`docs/architecture_spec.md`](docs/architecture_spec.md),
provide plain-PyTorch reference pseudo-code for every module (neighborhood attention with
surrounding padding, axial attention with RoPE/NTK, stage aggregation, the full encoder, pooling
and loss) in [`docs/reference_pseudocode.md`](docs/reference_pseudocode.md), and release every
hyper-parameter as the original YAML files.

---

## Contents

| Path | What it is |
|---|---|
| [`configs/c2dst_pretrain.yaml`](configs/c2dst_pretrain.yaml) | 20-epoch pretraining (VoxCeleb2 dev + speed perturbation, 3 s crops) |
| [`configs/c2dst_lmft.yaml`](configs/c2dst_lmft.yaml) | 2-epoch large-margin fine-tuning (LMFT; 6 s crops, no augmentation) |
| [`configs/inference/`](configs/inference/) | Embedding extraction, cohort selection, AS-Norm and QMF configs used for all reported numbers |
| [`configs/ablations/`](configs/ablations/) | Pretrain + LMFT config pairs for every row of Tables 2–4 |
| [`docs/config_reference.md`](docs/config_reference.md) | **Start here to read the YAMLs**: every key and module name in the config files defined, plus the framework defaults the YAMLs rely on |
| [`docs/architecture_spec.md`](docs/architecture_spec.md) | Component-level specification of the model behind the config keys |
| [`docs/reference_pseudocode.md`](docs/reference_pseudocode.md) | Plain-PyTorch reference pseudo-code for all modules (no NATTEN / flash-attn dependency) |
| [`docs/training_and_scoring.md`](docs/training_and_scoring.md) | Data protocol, training schedule, embedding extraction, AS-Norm and QMF, exactly as run |
| [`docs/ablations.md`](docs/ablations.md) | Tables 2–4 ↔ config diffs and what each variant changes structurally |

The YAML files are in ESPnet-SPK format. Module names such as `redimnat1_encoder`,
`melspec_torch_fix` or `spkfull` refer to modules of the in-house fork; every key and every such
name is defined in [`docs/config_reference.md`](docs/config_reference.md), so the files can be read
without the code and the numbers transferred to any framework.

---

## Method in one figure

```
80-dim log-Mel (1, F=80, 2T)
        │  2D conv stem  k=(3,2) s=(1,2)  → (C=32, F=80, T)      # time ↓2, freq kept
        ▼
┌─ Spectral–temporal backbone: 5 stages · 16 transformer layers ───────────────────┐
│  stage i:  weighted-sum of {stem, stage_1..i-1} outputs (learnable softmax w)      │
│            → 2D proj → N″× Neighborhood-Attention block (7×7, RPB + value-RPE)     │
│            → 1× Axial-Attention block (RoPE; freq axis ∥ time axis, concat, proj) │
│            → 2D proj → reshape to (C·F, T), append to the stage list              │
│  widths 48/72/96/144/192 · blocks 3/3/3/3/4 · freq ↓2 at stages 2 and 5           │
└──────────────────────────────────────────────────────────────────────────────────┘
        │  weighted-sum of all 6 outputs → BN → Linear(2560→144)
        ▼
Final 1D temporal stage (width 144):  NA NA NA GA NA NA NA GA   (NA: 1D neighborhood, k=31; GA: global RoPE)
        │  concat input + 8 layer outputs (9×144 = 1296) → attentive statistics pooling → BN
        ▼
Linear → 192-d speaker embedding  →  SphereFace2 (C-type) loss
```

Every block is pre-norm (LayerNorm) with SwiGLU FFN, LayerScale (init 1e-5) and stochastic depth
(rate linear in depth, mean 0.1, max 0.2). See [`docs/architecture_spec.md`](docs/architecture_spec.md) for the
specification and [`docs/reference_pseudocode.md`](docs/reference_pseudocode.md) for module-level
pseudo-code.

---

## Results (VoxCeleb1, EER % / minDCF, P_target = 0.01)

All rows use AS-Norm. Baselines are cited from their papers.

| Model | Params | LMFT | QMF | Vox1-O | Vox1-E | Vox1-H | Avg EER | Avg minDCF |
|---|---|:-:|:-:|---|---|---|---|---|
| PCF-NAT 3×4 | 7.6 M | ✗ | ✗ | 0.54 / 0.050 | 0.70 / 0.079 | 1.30 / 0.133 | 0.847 | 0.087 |
| PCF-NAT 6×4 | 12.0 M | ✗ | ✗ | 0.44 / 0.047 | 0.66 / 0.071 | 1.20 / 0.128 | 0.767 | 0.082 |
| ReDimNet-B4 | 6.3 M | ✓ | ✗ | 0.44 / 0.042 | 0.64 / 0.067 | 1.17 / 0.111 | 0.750 | 0.073 |
| ReDimNet-B5 | 9.2 M | ✓ | ✗ | 0.39 / 0.037 | 0.59 / 0.057 | 1.05 / 0.095 | 0.677 | 0.063 |
| ReDimNet-B6 | 15.0 M | ✓ | ✗ | 0.37 / 0.030 | 0.53 / 0.051 | 1.00 / 0.097 | 0.633 | 0.059 |
| ECAPA2 | 27.1 M | ✓ | ✓ | 0.34 / 0.029 | 0.52 / 0.058 | 0.99 / 0.098 | 0.617 | 0.062 |
| **C2D-ST** | 6.9 M | ✗ | ✗ | 0.37 / 0.039 | 0.55 / 0.054 | 0.96 / 0.093 | 0.627 | 0.062 |
| + LMFT | 6.9 M | ✓ | ✗ | 0.31 / 0.034 | 0.44 / 0.045 | 0.80 / 0.077 | 0.517 | 0.052 |
| **+ LMFT + QMF** | 6.9 M | ✓ | ✓ | **0.29 / 0.034** | **0.44 / 0.044** | **0.79 / 0.076** | **0.507** | **0.051** |

Configuration behind each C2D-ST row:

| Row | Training | Scoring |
|---|---|---|
| C2D-ST | `configs/c2dst_pretrain.yaml`, final-epoch weights | cosine + AS-Norm |
| + LMFT | `c2dst_pretrain.yaml` → `configs/c2dst_lmft.yaml`, final-epoch weights | cosine + AS-Norm |
| + LMFT + QMF | same | cosine + AS-Norm + QMF |

### Ablations (EER %, all with LMFT + AS-Norm + QMF)

| Variant | Params | O | E | H | Avg | Config pair |
|---|---|---|---|---|---|---|
| **C2D-ST** | 6.92 M | 0.29 | 0.44 | 0.79 | **0.507** | `c2dst_*` |
| Axial → 1D temporal transformer | 9.76 M | 0.31 | 0.48 | 0.88 | 0.557 | `ablations/axial_to_1d_*` |
| Axial → Neighborhood attention | 6.95 M | 0.46 | 0.58 | 1.04 | 0.693 | `ablations/axial_to_na_*` |
| Time-only axial | 6.92 M | 0.31 | 0.45 | 0.84 | 0.533 | `ablations/time_only_axial_*` |
| 2D NA → ConvNeXt (backbone) | 6.97 M | 0.38 | 0.48 | 0.88 | 0.580 | `ablations/na2d_to_convnext_*` |
| 1D NA → ConvNeXt (final stage) | 6.94 M | 0.25 | 0.44 | 0.82 | 0.503 | `ablations/na1d_to_convnext_*` |
| RPB only (no value-side RPE) | 6.83 M | 0.27 | 0.45 | 0.82 | 0.513 | `ablations/rpb_only_*` |

See [`docs/ablations.md`](docs/ablations.md) for the exact diff of each variant against the full model.

---

## Training recipe at a glance

| | Pretraining | LMFT |
|---|---|---|
| Data | VoxCeleb2 dev, speed perturbation ×{0.9, 1.0, 1.1} (17 982 pseudo-speakers) | VoxCeleb2 dev, no speed perturbation (5 994 speakers) |
| Crop | random 3 s | random 6 s |
| Augmentation | MUSAN (speech/noise/music) p = 0.5, simulated RIR p = 0.5 | none |
| Epochs | 20 | 2 |
| Effective batch | 1 536 | 384 |
| Optimizer | AdamW, wd 0.05 (no wd on norms, biases, RPB, LayerScale, aggregation weights) | same |
| LR | cosine, peak 8e-3, 10 % linear warm-up, single cycle, min 8e-8 | cosine, peak 3e-4, 10 % warm-up, min 3e-8 |
| Loss | SphereFace2 (C-type), s = 32, m = 0.2, t = 3, λ = 0.7 | m = 0.3; anchors initialised from the pretrained matrix (speed-1.0 rows) |
| Init | PyTorch-default Kaiming-uniform for all conv/linear (`init_style: v5`) | pretrained final-epoch weights |
| Precision | mixed (AMP) | mixed (AMP) |
| Weights evaluated | final epoch (`latest.pth`) | final epoch (`latest.pth`) |

Scoring: full-length utterances, cosine scoring, AS-Norm with a speaker-averaged VoxCeleb2-dev
cohort (top-500 adaptive), QMF = logistic regression on {score, log-duration max/min, embedding
L1/L2-norm max/min} trained on 6 × 10 000 VoxCeleb2-dev trials. No SNR-based quality features.
Details in [`docs/training_and_scoring.md`](docs/training_and_scoring.md).

---

## Re-implementing from this page

What the page fully specifies: the model (every layer, shape and initialisation —
`docs/architecture_spec.md`, `docs/reference_pseudocode.md`), the training schedule and
augmentation, and the scoring pipeline including the exact AS-Norm and QMF procedures
(`docs/training_and_scoring.md`). An analytic parameter count of the specification matches the
paper's parameter column for all seven systems.

What you have to supply yourself: VoxCeleb1/2 audio at 16 kHz (VoxCeleb2 converted from m4a),
the cleaned VoxCeleb1 trial lists, MUSAN and RIRS_NOISES, sox for speed perturbation, and a
training loop that implements the batching / accumulation / mixed-precision settings described in
`docs/config_reference.md` §2.6–2.7.

What is inherently not reproducible bit-exactly: GPU non-determinism, the random cohort-utterance
and QMF-trial selection (no fixed seed), and the exact parameter layout of our checkpoints.
Expect small deviations from the reported EERs rather than identical numbers.

## Software used (for the record)

- Python 3, PyTorch 2.x with mixed precision; `scaled_dot_product_attention` as the attention fallback
- NATTEN with the unfused functional API (`na2d_qk`/`na2d_av`, `na1d_qk`/`na1d_av`); fused NA disabled
- flash-attn for axial / global attention when available (numerically equivalent to the SDPA fallback)
- ESPnet-SPK pipeline (data preparation, speed perturbation, training loop, scoring scripts), in-house fork

---

## Citation

```bibtex
@inproceedings{ham2026c2dst,
  title     = {Continuous 2D Spectral--Temporal Transformer for Speaker Verification},
  author    = {Ham, Seongwook and Doan, Thien-Phuc},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  address   = {Sydney, Australia}
}
```

## License

The configuration files and documentation in this repository are released under the
[Apache License 2.0](LICENSE).
