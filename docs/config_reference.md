# Config reference — every key explained

The YAML files in `configs/` are the configuration files of the in-house ESPnet-SPK fork,
unchanged except for two cosmetic edits: the header comment of each file, and the experiment-
directory names inside `init_param`, which were renamed to match the descriptive file names used
here (§5). Because the code is not released, this page defines **every key and every string value**
that appears in them, so that the files can be read without the code. Architecture details are in
[`architecture_spec.md`](architecture_spec.md); the training/scoring protocol in
[`training_and_scoring.md`](training_and_scoring.md).

There are two kinds of files:

| Kind | Files | Notes |
|---|---|---|
| Training YAML | `c2dst_pretrain.yaml`, `c2dst_lmft.yaml`, `ablations/*` | What was written by hand. Everything not listed takes the framework default (§4). |
| Inference YAML | `inference/*.yaml` | Read by the embedding-extraction, cohort-selection, AS-Norm and QMF steps. §3. |

Paths such as `dump/raw/...` and `exp/...` are recipe-internal locations (prepared data lists and
experiment directories) and carry no information beyond what is described here.

---

## 1. String values (module selectors)

Several keys select a module by name. These are the names used in the released files:

| Key | Value | Meaning |
|---|---|---|
| `frontend` | `melspec_torch_fix` | Log-Mel front-end with pre-emphasis and per-utterance MVN (spec §1). |
| `encoder` | `redimnat1_encoder` | The C2D-ST encoder (spec §2). |
| | `redimnat4_encoder` | Same encoder with the axial block replaced by a collapsed 1D transformer block (Table 2, "Axial → 1D"). |
| | `redimnat5_encoder` | Same encoder plus the two ConvNeXt switches `use_convnext_na_2d` / `use_convnext_nattn_1d` (Table 4). |
| `pooling` | `chn_attn_stat` | ECAPA-style attentive statistics pooling (spec §3). |
| `projector` | `rawnet3` | `BatchNorm1d → Linear(→192)` embedding head. |
| `preprocessor` | `spkfull` | Speaker-task preprocessor: random crop, MUSAN / RIR augmentation, label mapping (spec §4). |
| `loss` | `sphereface2` | SphereFace2 with additive (C-type) margin (spec §3). |
| `optim` | `adamw` | `torch.optim.AdamW`. |
| `scheduler` | `CosineAnnealingWarmupRestartsFix` | Linear warm-up then cosine decay, stepped per optimizer update (spec §5). "Restarts" are never triggered in our settings. |
| `na_type_2d` | `nattn2d_decay_rpb_v1` | 2D neighborhood attention, surrounding padding, RPB **and** value-side RPE (spec §2.4). |
| | `nattn2d_rpb_v1` | Same without the value-side RPE (Table 3). |
| `ga_type_2d` | `axial_static_v1` | Axial attention over frequency ∥ time with fixed-frequency RoPE (spec §2.5). |
| | `axial_static_timeonly_v1` | Time-axis-only variant with doubled heads (Table 2). |
| `nattn_type_1d` | `nattn1d_decay_rpb_v1` / `nattn1d_rpb_v1` | 1D counterparts of the 2D NA modules (spec §2.7). |
| `gattn_type_1d` | `gattn1d_static_v1` | Full self-attention over time with fixed-frequency RoPE (spec §2.7). |
| `ffn_2d_type`, `ffn_1d_type` | `swiglu` | Gated FFN `W₂(SiLU(W₁ᵃx) ⊙ W₁ᵇx)`, hidden `⌊units·2/3⌋`. |
| `redim_*_norm_type`, `norm_1d_type`, `fin_norm_1d_type` | `bn` / `ln` / `identity` | BatchNorm over channels (channel-last), LayerNorm over channels, or no norm. |
| `redim_aggregation_type` | `ws2d` | Softmax-weighted sum of stage outputs with one weight per input per grid channel, performed on the 2D grid (spec §2.3). |
| `activation_type` | `swish` | SiLU, used inside the FFNs and ConvNeXt blocks. |
| `init_style` | `v5` | PyTorch-default Kaiming-uniform for conv/linear, zero bias (spec §2.8). |
| `extra_type` | `ntk` | NTK-aware RoPE frequency rescaling at inference for time sequences longer than `extra_len` (spec §2.5). |
| `drop_path_policy` | `linear` | Stochastic-depth rate grows linearly with depth from 0 to `2·drop_path_rate`. |
| `normalize` (frontend) | `mvn` | Per-utterance mean/variance normalisation of each log-Mel bin over time. |
| `stage_config` | see §2.3 | Encoded stage table. |

---

## 2. Short training YAML

### 2.1 Front-end

```yaml
frontend: melspec_torch_fix
frontend_conf:
    preemp: true       # apply first-order pre-emphasis (0.97) before the STFT
    n_fft: 512
    log: true          # log(mel + 1e-6)
    win_length: 400    # samples (25 ms)
    hop_length: 160    # samples (10 ms)
    f_min: 20          # Hz
    f_max: 7600        # Hz
    n_mels: 80
    normalize: mvn
```

### 2.2 Encoder — generic keys

| Key | Value | Meaning |
|---|---|---|
| `stem_size` | 32 | Output channels of the stem conv. |
| `stem_kernel_size` / `stem_stride` / `stem_padding` | `[3,2]` / `[1,2]` / `[1,0]` | Stem Conv2d hyper-parameters as (frequency, time). Time is down-sampled by 2. |
| `redim_stem_norm_in_type` | `bn` | Norm after the stem conv (BatchNorm2d). |
| `stage_config` | string | Stage table, §2.3. |
| `na_type_2d`, `ga_type_2d` | names | Local / global attention modules of the backbone (§1). |
| `ffn_2d_type` | `swiglu` | FFN type in backbone blocks. |
| `redim_mult` | 4 | FFN expansion in the backbone: `units = 4·width`. |
| `ls_init_2d` | 1e-5 | LayerScale initial value in backbone blocks (`null` = no LayerScale). |
| `redim_proj_norm_in_type` | `bn` | Norm applied to the aggregated grid before the 2D projection in (skipped for stage 1). |
| `redim_proj_norm_out_type` | `bn` | Norm applied after the last block, before the 2D projection out. |
| `redim_post_post_norm` | `false` | If true, the out-norm would be applied *after* the projection out instead of before; also decides which projection carries a bias. `false` = norm before projection, projection out without bias. |
| `redim_fin_norm_type` | `bn` | Norm on the final aggregated 2560-dim vector before the 1D projection. |
| `redim_block_norm_type` | `ln` | Norm inside backbone transformer blocks. |
| `redim_aggregation_type` | `ws2d` | Stage-wise aggregation method (§1). |
| `output_size_1d` | 144 | Width of the final temporal stage. |
| `nattn_heads_1d` / `gattn_heads_1d` | 9 / 3 | Heads of 1D neighborhood / global attention. |
| `linear_units_1d` | 576 | FFN units of the 1D blocks (SwiGLU hidden = 384). |
| `kernel_size_1d` | 31 | 1D neighborhood window. |
| `num_blocks_1d` | 8 | Number of 1D blocks. |
| `gattn_freq_1d` | 4 | Every `gattn_freq_1d`-th 1D block is global attention (blocks 4 and 8). |
| `mfa` / `mfa_freq` | `true` / 1 | Multi-layer feature aggregation: concatenate the stage input and every `mfa_freq`-th block output (all 8) before pooling. |
| `nattn_type_1d`, `gattn_type_1d` | names | 1D attention modules (§1). |
| `norm_1d_type`, `ffn_1d_type`, `ls_init_1d` | `ln`, `swiglu`, 1e-5 | 1D block norm / FFN / LayerScale. |
| `fin_norm_1d_type` | `identity` | Norm on the 1296-dim MFA output (none). |
| `drop_path_rate` / `drop_path_policy` | 0.1 / `linear` | Stochastic depth over the 24 attention blocks. |
| `activation_type` | `swish` | FFN activation. |
| `init_style` | `v5` | Weight initialisation (§1). |
| `extra_type` / `extra_len` | `ntk` / 150 or 300 | RoPE extrapolation; `extra_len` = training sequence length in frames after the stem (3 s → 150, 6 s → 300). |
| `use_convnext_na_2d` | `true` (`na2d_to_convnext_*` only) | `redimnat5_encoder`: replace each backbone NA block by two ConvNeXt-style blocks. |
| `use_convnext_nattn_1d` | `true` (`na1d_to_convnext_*` only) | `redimnat5_encoder`: replace each 1D NA block by two ConvNeXt-style blocks. |

Keys **not** present in the YAML keep their defaults and matter for re-implementation:
`bias_2d = bias_1d = true` (all attention/FFN linears have bias), `window_size = null` (no
local window in axial/global attention), `input_diversity = false`, `pcf_groups_1d = null`,
`final_proj_size = null` (no extra projection after MFA), `convnext_expansion_rate = 3`.

### 2.3 `stage_config` grammar

`s5_d2-2_s_16l_k7_div24_div48_g1` selects the table below (the string is a lookup key, not parsed):

| Token | Meaning |
|---|---|
| `s5` | 5 stages |
| `d2-2` | two ↓2 frequency down-samplings, placed at stages 2 and 5 |
| `s` | width set: 48, 72, 96, 144, 192 |
| `16l` | 16 backbone blocks: 3, 3, 3, 3, 4 per stage |
| `k7` | 7×7 neighborhood window |
| `div24`, `div48` | NA heads = width/24, axial heads per axis = width/48 |
| `g1` / `g0` | one axial block per stage, placed last / no axial block (all blocks NA) |

Full table: spec §2.2.

### 2.4 Pooling / projector / loss

```yaml
pooling: chn_attn_stat          # pooling_conf: {}  (bottleneck 128 is the default)
projector: rawnet3
projector_conf:
  output_size: 192              # embedding dimension
loss: sphereface2
loss_conf:
  margin: 0.2                   # C-type additive cosine margin; 0.3 in LMFT
  scale: 32                     # logit scale s
  t: 3                          # exponent of g(z) = 2·((z+1)/2)^t − 1
  lambda_: 0.7                  # weight of the positive term (1−λ for negatives)
  jt: true                      # LMFT only: allocate 3×n_speakers anchors so the speed-perturbed
                                #   pretrained anchor matrix loads; use only the first n (speed 1.0)
```

`model_conf: {extract_feats_in_collect_stats: false}` — framework flag; the statistics-collection
stage does not run the front-end.

### 2.5 Preprocessor

```yaml
preprocessor: spkfull
preprocessor_conf:
  target_duration: 3.0          # training crop length in seconds (6.0 in LMFT); wrap-padded if shorter
  sample_rate: 16000
  num_eval: 5                   # validation only: number of evenly spaced crops per trial utterance (3 in LMFT)
  noise_apply_prob: 0.5         # probability of adding MUSAN noise to a crop (0.0 in LMFT)
  noise_info:                   # one entry per MUSAN category:
  - [1.0, <speech list>, [4, 7], [13, 20]]   # [selection weight, list, [min,max] files mixed, [min,max] SNR dB]
  - [1.0, <noise list>,  [1, 1], [0, 15]]
  - [1.0, <music list>,  [1, 1], [5, 15]]
  rir_apply_prob: 0.5           # probability of convolving with a random simulated RIR (0.0 in LMFT)
  rir_scp: <rir list>           # RIRS_NOISES small + medium rooms
```

### 2.6 Training loop

| Key | Value | Meaning |
|---|---|---|
| `init_param` | path | LMFT only: checkpoint whose parameters initialise the model (all parameters, including loss anchors). |
| `max_epoch` | 20 / 2 | Number of epochs. |
| `num_iters_per_epoch` | 51 187 / 17 062 | Mini-batches per epoch (= utterances / `batch_size`, i.e. one pass over the set). |
| `batch_size` | 64 | **Global** mini-batch (utterances), split across GPUs. |
| `accum_grad` | 24 / 6 | Gradient accumulation steps. Effective batch = `batch_size × accum_grad` = 1 536 / 384. |
| `valid_batch_size` | 16 / 32 | Trials per validation mini-batch. |
| `num_workers` | 8 | Data-loader workers. |
| `iterator_type` | `random` | Sample utterances uniformly at random each epoch (one random crop each). |
| `valid_iterator_type` | `sequence` | Validation trials in file order. |
| `shuffle_within_batch` | `false` | Keep the sampler's order inside a batch. |
| `drop_last_iter` | `true` | Drop the last incomplete batch. |
| `use_amp` | `true` | Mixed-precision (autocast + GradScaler). |
| `grad_clip` | 9999 | L2 gradient-norm clipping threshold (effectively off). |
| `keep_nbest_models` | 3 | Framework checkpoint-retention option (recipe default). The final-epoch checkpoint (`latest.pth`) is the one evaluated. |
| `best_model_criterion` | `[[valid, eer, min]]` | Framework checkpoint-ranking option (recipe default); the validation set is the VoxCeleb1-O trial list. Not what is evaluated — see `latest.pth` above. |
| `num_att_plot` | 0 | Attention plots disabled. |
| `log_interval` | 100 | Log every 100 mini-batches. |
| `cudnn_benchmark` / `cudnn_deterministic` | `true` / `false` | cuDNN autotuning on, determinism off. |

The commented blocks `#1gpu … #8gpu` in the YAMLs give the (`batch_size`, `accum_grad`,
`num_iters_per_epoch`) triple for other GPU counts with the same effective batch; only one block
is active.

### 2.7 Optimizer and schedule

```yaml
optim: adamw
optim_conf:
  lr: 0.008                     # placeholder; the scheduler overrides it every step
  weight_decay: 0.05
  amsgrad: false
exclude_weight_decay: true      # build two param groups: decayed / not decayed
exclude_weight_decay_conf:
  normalization_weight_decay: false   # no decay on norm weights & biases
  bias_weight_decay: false            # no decay on any bias
                                # additionally no decay on RPB, LayerScale γ, aggregation weights
scheduler: CosineAnnealingWarmupRestartsFix
scheduler_conf:
  first_cycle_steps: 42660      # length of the (only) cycle in optimizer steps = epochs × iters/epoch ÷ accum_grad
  cycle_mult: 2.0               # length multiplier for subsequent cycles (never reached)
  max_lr: 0.008                 # peak learning rate
  min_lr: 0.00000008            # lr at step 0 and at the end of the cycle
  warmup_steps: 4266            # linear warm-up min_lr → max_lr over the first 10 % of steps
  gamma: 1.0                    # peak-lr decay per cycle (unused)
```

---

## 3. Inference YAML

| File | Key | Value | Meaning |
|---|---|---|---|
| `decode_full.yaml` | `target_duration` | −2560 | Negative → use the whole utterance (up to 2560 s); positive would mean fixed-length crops. |
| | `num_eval` | 1 | Number of crops per utterance (irrelevant for full-length). |
| | `average_embd` | `true` | Average the crop embeddings into one per utterance. |
| | `valid_batch_size` / `num_workers` | 1 / 1 | One utterance per forward pass (variable length). |
| `decode_cohort_full.yaml` | same extraction keys | | Applied to cohort utterances. |
| | `num_cohort_spk` | 5994 | Number of VoxCeleb2-dev speakers used as cohort (all). |
| | `num_utt_per_spk` | 10 | Utterances selected per cohort speaker. |
| | `utt_select_sec` | 0 | Minimum duration for a cohort utterance (none). |
| | `per_spk_select_order` | `shuffle` | Pick the utterances at random (alternatives: longest / shortest first). |
| `asnorm_full.yaml` | `adaptive_cohort_size` | 500 | Top-k cohort scores used for the mean/std in adaptive s-norm. |
| | `average_spk` | `true` | Average each speaker's cohort embeddings into one vector before scoring. |
| `decode_qmf_full.yaml` | extraction keys | as above | Applied to QMF-training utterances. |
| | `qmf_dur_thresh` | 6 | Duration (s) separating "short" and "long" utterances when building QMF training trials. |
| | `qmf_num_trial_per_condition` | 10000 | Trials generated per condition: target and non-target × {short–short, long–long, long–short} = 6 × 10 000. |

---

## 4. Framework defaults and launcher values

The trainer fills in every option not written in the YAML. Those that carry information about
how the runs were executed are:

| Key | Value | Meaning |
|---|---|---|
| `ngpu`, `dist_world_size`, `multiprocessing_distributed`, `dist_backend` | 1, 4, true, nccl | 4-GPU single-node DDP (`ngpu` is per process). |
| `seed` | 0 | Random seed. |
| `spk_num` | 5994 | Number of training speakers (17 982 for the speed-perturbed pretraining set). |
| `spk2utt` | path | Speaker → utterance list used to build the label map. |
| `train_data_path_and_name_and_type` | `wav.scp` (speech, sound), `utt2spk` (spk_labels, text) | Training inputs: waveform + speaker id. |
| `valid_data_path_and_name_and_type` | `trial.scp`, `trial2.scp`, `trial_label` | Validation = VoxCeleb1-O trial pairs with 0/1 labels. |
| `train_shape_file`, `valid_shape_file` | paths | Per-utterance sample counts (from the statistics stage). |
| `batch_type` | `folded` | Length-bucketed batching with `fold_length: [120000]` samples; irrelevant here because training crops are fixed-length. |
| `sort_in_batch`, `sort_batch` | `descending` | Ordering inside/among batches (no effect for fixed-length crops). |
| `train_dtype` | float32 | Master weights in fp32 (AMP handles the fp16 compute). |
| `grad_clip_type` | 2.0 | L2 norm for `grad_clip`. |
| `resume` | true | Resume from the last checkpoint if present. |
| `save_strategy` | all | Save full checkpoints. |
| `version` | 202402 | ESPnet base version of the fork. |
| `target_duration: 3.0`, `num_eval: 10`, `rir_scp: ''` (top level) | | Task-level defaults that are **overridden** by `preprocessor_conf`. |
| `input_size: null` | | Encoder input size is taken from the front-end (80). |
| `specaug`, `normalize` | null | No SpecAugment, no global feature normalisation. |
| `metric` | eer | Validation metric. |

Everything else is an ESPnet trainer option left at its default or set by the launcher; none of
them affects the model or the reported results:
adapters (`use_adapter`, `adapter`, `adapter_conf`), partial loading/freezing (`pretrain_path`,
`ignore_init_mismatch`, `freeze_param`, `unfreeze_param`, `init`), per-module lr/wd scaling
(`lr_scale_param`, `wd_scale_param`), chunk iterator (`chunk_length`, `chunk_shift_ratio`,
`chunk_default_fs`, `chunk_excluded_key_prefixes`, `num_cache_chunks`), caching (`max_cache_size`,
`max_cache_fd`, `valid_max_cache_size`), alternative batching (`batch_bins`, `valid_batch_bins`,
`valid_batch_type`, `multiple_iterator`, `allow_variable_data_keys`, `allow_multi_rates`),
stopping/averaging (`patience`, `early_stopping_criterion`, `val_scheduler_criterion`,
`nbest_averaging_interval`), statistics stage (`collect_stats`, `write_collected_feats`),
logging (`use_tensorboard`, `use_matplotlib`, `create_graph_in_tensorboard`, `use_wandb`,
`wandb_project`, `wandb_id`, `wandb_entity`, `wandb_name`, `wandb_model_log_interval`,
`log_level`, `print_config`), debugging (`detect_anomaly`, `dry_run`, `no_forward_run`,
`grad_noise`, `cudnn_enabled`, `torch_deterministic_algorithms`), distributed launcher
(`dist_init_method`, `dist_launcher`, `dist_master_addr`, `dist_master_port`, `dist_rank`,
`local_rank`, `distributed`, `sharded_ddp`, `unused_parameters`), empty sub-configs
(`specaug_conf`, `normalize_conf`, `pooling_conf`), and bookkeeping (`use_preprocessor`,
`output_dir`, `config`, `required`). `encoder_conf`, `frontend_conf`, `preprocessor_conf`,
`loss_conf`, `projector_conf`, `optim_conf`, `scheduler_conf` are the containers for the keys of §2.

---

## 5. `init_param` paths

Each LMFT YAML initialises from its own pretraining run through `init_param`. The recipe names an
experiment directory `exp/spk_<config basename>_<feats_type>[_sp]`, with `_sp` appended when speed
perturbation is on, so the paths read

```
c2dst_lmft.yaml             init_param: exp/spk_c2dst_pretrain_raw_sp/latest.pth
ablations/<variant>_lmft.yaml   init_param: exp/spk_<variant>_pretrain_raw_sp/latest.pth
```

`latest.pth` is the final-epoch checkpoint. These directory names were substituted for the
in-house experiment names when the files were copied here; nothing else in the YAMLs was changed.
