# Training and scoring protocol

Everything below is what was actually run for the paper, described so that it can be reproduced
in any framework. The pipeline followed the ESPnet-SPK VoxCeleb recipe (data preparation →
speed perturbation → training → embedding extraction → cosine scoring → AS-Norm → QMF → metrics);
the released YAMLs are its configuration files.

## 1. Data

| Set | Use | Details |
|---|---|---|
| VoxCeleb2 dev | training (both phases), AS-Norm cohort, QMF training trials | 5 994 speakers, 1 092 009 utterances, 16 kHz |
| VoxCeleb2 dev × speed {0.9, 1.0, 1.1} | pretraining only | sox `speed` effect (resampling: tempo **and** pitch change); each perturbed copy is a **new speaker** → 17 982 classes, 3 276 027 utterances. Perturbed ids are prefixed `sp0.9-` / `sp1.1-`, so in sorted order the original speakers come first |
| VoxCeleb1-O | test (also the recipe's validation list) | `veri_test2.txt` (cleaned) |
| VoxCeleb1-E / -H | test | `list_test_all2.txt`, `list_test_hard2.txt` (cleaned) |
| MUSAN | noise augmentation (pretraining) | speech / noise / music lists |
| RIRS_NOISES | reverberation augmentation (pretraining) | simulated small + medium rooms only |

Audio is stored as 16 kHz flac. VoxCeleb2 (distributed as m4a) must be converted to 16 kHz
wav/flac beforehand; the data-preparation script only lists `.wav`/`.flac` files. **No duration
filtering** is applied to the training utterances.

## 2. Phase 1 — pretraining (`configs/c2dst_pretrain.yaml`)

- Speed-perturbed VoxCeleb2 dev, random 3 s crops, MUSAN (p 0.5) + RIR (p 0.5) — §4 of the spec.
- 20 epochs, effective batch 1 536, AdamW (wd 0.05), single cosine cycle, peak lr 8e-3, 10 %
  linear warm-up, min lr 8e-8, AMP.
- Batching: each epoch is one random permutation of all training utterances (re-seeded with the
  epoch index), cut into global batches of 64; the last incomplete batch is dropped. One random
  crop per utterance per epoch.
- SphereFace2 C-type, s 32, m 0.2, t 3, λ 0.7, 17 982 anchors.
- RoPE reference length `extra_len: 150` frames (3 s after the stem's ↓2).
- The weights after epoch 20 (`latest.pth`) are used. Scored with cosine + AS-Norm they give the
  **"C2D-ST" row** of Table 1 (0.37 / 0.55 / 0.96, avg EER 0.627).

## 3. Phase 2 — large-margin fine-tuning (`configs/c2dst_lmft.yaml`)

Initialised from the phase-1 final weights (`init_param`), including the loss anchors (see
`jt: true` in the spec §3). Changes with respect to phase 1:

| Key | Pretrain | LMFT |
|---|---|---|
| training set | VoxCeleb2 dev × speed | VoxCeleb2 dev (5 994 speakers) |
| `target_duration` | 3.0 s | **6.0 s** |
| `noise_apply_prob` / `rir_apply_prob` | 0.5 / 0.5 | **0.0 / 0.0** |
| `margin` | 0.2 | **0.3** |
| `jt` | – | **true** |
| `max_epoch` | 20 | **2** |
| `lr` / `max_lr` | 8e-3 | **3e-4** |
| `min_lr` | 8e-8 | 3e-8 |
| `first_cycle_steps` / `warmup_steps` | 42 660 / 4 266 | **5 688 / 569** |
| `batch_size × accum_grad` | 64 × 24 = 1 536 | 64 × 6 = **384** |
| `extra_len` | 150 | **300** |
| `num_eval` (validation crops) | 5 | 3 |

Everything else (architecture, optimizer, weight-decay exclusions, AMP) is unchanged. The active
batch block in the YAML is the 4-GPU one; framework defaults not written in the YAML are listed in
`config_reference.md` §4.

Final-epoch weights: cosine + AS-Norm → **"+ LMFT" row** (0.31 / 0.44 / 0.80, avg 0.517);
cosine + AS-Norm + QMF → **"+ LMFT + QMF" row** (0.29 / 0.44 / 0.79, avg 0.507).

## 4. Embedding extraction (`configs/inference/decode_full.yaml`)

```yaml
target_duration: -2560   # negative → whole utterance (cap 2560 s); no cropping, no padding
num_eval: 1
average_embd: true
valid_batch_size: 1      # one utterance per forward pass (variable length)
```

One embedding per full-length utterance for test, cohort and QMF-training utterances alike. The
model is in eval mode: no augmentation, BatchNorm in inference mode, and the NTK RoPE
extrapolation is active whenever the utterance exceeds 150 (phase-1 model) or 300 (LMFT model)
frames.

## 5. Scoring

### 5.1 Raw score

`s = <ê, t̂>`, the cosine similarity between the L2-normalised enrolment and test embeddings.

### 5.2 AS-Norm (`decode_cohort_full.yaml` + `asnorm_full.yaml`)

```yaml
# cohort selection
num_cohort_spk: 5994          # all VoxCeleb2-dev speakers (first 5994 lines of spk2utt)
num_utt_per_spk: 10           # utterances per speaker, chosen at random (no fixed seed)
utt_select_sec: 0             # no minimum duration
# normalisation
average_spk: true             # one cohort vector per speaker
adaptive_cohort_size: 500
```

Cohort vector of speaker j: `c_j = mean_k ĉ_{j,k}` — the mean of that speaker's **L2-normalised**
utterance embeddings, **not re-normalised** after averaging (so ‖c_j‖ ≤ 1). For a trial (e, t):

```
S_e = top-500 of { <ê, c_j> : j = 1…5994 },   μ_e = mean(S_e),  σ_e = std(S_e)   (unbiased std)
S_t = top-500 of { <t̂, c_j> },                 μ_t, σ_t likewise
s'  = ½ · ( (s − μ_e)/σ_e + (s − μ_t)/σ_t )
```

### 5.3 QMF (`decode_qmf_full.yaml`)

```yaml
qmf_dur_thresh: 6                  # "short" < 6 s ≤ "long"
qmf_num_trial_per_condition: 10000
```

**Training trials** are built from VoxCeleb2 dev, excluding utterances shorter than 2 s and the
utterances used in the AS-Norm cohort; speakers need ≥ 2 usable utterances in a duration class to
be eligible. Six conditions × 10 000 trials are sampled at random (no fixed seed; exact duplicates
dropped):

| Condition | Target (same speaker) | Non-target (different speakers) |
|---|---|---|
| short – short | 10 000 | 10 000 |
| long – long | 10 000 | 10 000 |
| long – short | 10 000 | 10 000 |

These trials are scored exactly like test trials (cosine, then AS-Norm with the same cohort).

**Features** (7-dimensional, no standardisation):

```
[ s',  ln d_max,  ln d_min,  ‖e‖₁ max, ‖e‖₁ min,  ‖e‖₂ max, ‖e‖₂ min ]
```

`d` = the two utterance durations clipped to [2, 40] s; norms are of the two **un-normalised**
embeddings; max/min over the pair. **Model**: scikit-learn `LogisticRegression(solver="lbfgs",
max_iter=1000, random_state=0)` with default L2 regularisation (C = 1). The calibrated score of a
test trial is the predicted target probability `P(target | features)`. No SNR-based quality
features are used — this is the only departure from the DKU-MSXF VoxSRC-2023 QMF setup.

### 5.4 Metrics

EER (%) and minDCF with `P_target = 0.01`, `C_miss = C_fa = 1` on VoxCeleb1-O / E / H. "Avg" is
the unweighted mean over the three lists.

## 6. Ablations

Every ablation is a full pretrain → LMFT → AS-Norm + QMF pass with the same settings; only the
architectural keys differ. Config pairs and diffs are in `ablations.md`.

## 7. Notes

- Where the paper's wording is looser than the implementation, this page follows the
  implementation: (i) the stage-wise aggregation weights are learnable **per input and per
  channel** (softmax over inputs), not one scalar per input as the paper's "scalar weights"
  suggests; (ii) the paper's "initial learning rate" (0.008 / 0.0003) is the **peak** of the
  warm-up-then-cosine schedule, the schedule starts at `min_lr`; (iii) the AS-Norm cohort is
  speaker-averaged (5 994 vectors) before the top-500 selection, which the paper does not spell out.
- All numbers are from single runs, `seed: 0`; run-to-run variance was not measured.
- Two scoring steps are randomised without a fixed seed: the choice of 10 cohort utterances per
  speaker and the sampling of QMF training trials. Their effect on EER is expected to be small
  but has not been quantified.
- Every reported number is the final-epoch checkpoint (`latest.pth`) of a run of fixed length
  (20 epochs, then 2).
- Parameter counts (6.92 M etc.) exclude the loss anchors.
