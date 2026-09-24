# Ablations (Tables 2–4) ↔ configs

All ablation variants share the full recipe: 20-epoch pretraining with speed perturbation and
augmentation → 2-epoch LMFT (6 s, no augmentation) → AS-Norm + QMF. Each variant therefore has a
**pretrain / LMFT config pair** in `configs/ablations/`; the only differences from the C2D-ST
pair (`configs/c2dst_pretrain.yaml`, `configs/c2dst_lmft.yaml`) are listed below. The LMFT YAML of
each variant differs from `c2dst_lmft.yaml` only in the same encoder keys and in `init_param`,
which points at the variant's own pretraining run.

| Variant | Table | Params | Avg EER | Config pair |
|---|---|---|---|---|
| C2D-ST (full) | 2/3/4 | 6.92 M | 0.507 | `c2dst_{pretrain,lmft}.yaml` |
| Axial → 1D | 2 | 9.76 M | 0.557 | `axial_to_1d_{pretrain,lmft}.yaml` |
| Axial → NA | 2 | 6.95 M | 0.693 | `axial_to_na_{pretrain,lmft}.yaml` |
| Time-only axial | 2 | 6.92 M | 0.533 | `time_only_axial_{pretrain,lmft}.yaml` |
| RPB only | 3 | 6.83 M | 0.513 | `rpb_only_{pretrain,lmft}.yaml` |
| 2D NA → ConvNeXt | 4 | 6.97 M | 0.580 | `na2d_to_convnext_{pretrain,lmft}.yaml` |
| 1D NA → ConvNeXt | 4 | 6.94 M | 0.503 | `na1d_to_convnext_{pretrain,lmft}.yaml` |

---

## Table 2 — attention structure

### Axial → 1D temporal transformer (`axial_to_1d_*`)

```diff
-encoder: redimnat1_encoder
+encoder: redimnat4_encoder
```

Structural change (ReDimNet-style hybrid): in every stage the axial block is removed, the
remaining `N−1` neighborhood-attention blocks run on the 2D grid as before, the stage output is
projected and flattened to the collapsed 1D form `(B, T, c·F = 2560)`, and **one 1D transformer
block with pre/post projections** is applied there:

```
y = in_proj(x)                                  # Linear(2560 → d),  d = stage width (48…192)
y = y + LS·GlobalAttn_RoPE( LN(y) )             # 2·(axial heads per axis) heads → same head count as the axial block
y = y + LS·SwiGLU( LN(y) )                      # hidden ⌊4d·2/3⌋
x = x + out_proj( LN(y) )                       # Linear(d → 2560), residual to the collapsed input
```

Width `d` and total head count match the axial block it replaces; the two projections between
2560 and `d` per stage account for the +2.8 M parameters. Global attention is the same
RoPE/NTK module as the final stage's GA (`gattn1d_static_v1`), with the 1D-stage norm and FFN
types.

### Axial → Neighborhood attention (`axial_to_na_*`)

```diff
-  stage_config: 's5_d2-2_s_16l_k7_div24_div48_g1'
+  stage_config: 's5_d2-2_s_16l_k7_div24_div48_g0'
```

`g0`: no axial block in any stage; all 16 backbone blocks are 7×7 neighborhood attention. Widths
and heads unchanged. (This pretrain YAML is written for 4 GPUs, `batch_size 96 × accum_grad 16`;
the effective batch is still 1 536, and its LMFT uses `48 × 8 = 384`.)

### Time-only axial (`time_only_axial_*`)

```diff
-  ga_type_2d: 'axial_static_v1'
+  ga_type_2d: 'axial_static_timeonly_v1'
```

The frequency branch is removed; the time branch uses all `d` channels with `2n` heads of the
same `d_h = d/(2n)` — 36 in stage 2, 24 elsewhere — instead of `n` heads over `d/2`. Feature dimension, number of heads and
parameter count are identical to the two-axis block; RoPE + NTK extrapolation on time as before.

---

## Table 3 — positional encoding in neighborhood attention

### RPB only (`rpb_only_*`)

```diff
-  na_type_2d: 'nattn2d_decay_rpb_v1'
+  na_type_2d: 'nattn2d_rpb_v1'
-  nattn_type_1d: "nattn1d_decay_rpb_v1"
+  nattn_type_1d: "nattn1d_rpb_v1"
```

Removes the value-side relative positional encoding `P` (the `Σ_o α[p,o]·P[o]` term) from **both**
the 2D backbone NA (`H×49×24` per layer) and the 1D final-stage NA (`9×31×16` per layer); the
relative position bias `B` is kept. Nothing else changes (−0.09 M parameters).

---

## Table 4 — attention vs. convolution

Both variants use `encoder: redimnat5_encoder`, which is the C2D-ST encoder plus two switches.
Each selected neighborhood-attention transformer block (attention + FFN) is replaced by **two**
ConvNeXt-style blocks:

```
x = x + DropPath( LS · W₂ · SiLU( W₁ · LN( DWConv_k(x) ) ) )      # W₁: d→3d, W₂: 3d→d, DWConv: depthwise, groups = d
```

Kernel size equals that of the attention it replaces (7×7 in 2D, 31 in 1D),
`convnext_expansion_rate: 3`, LayerScale init 1e-5, and both blocks of a pair use the
drop-path rate of the layer they replace. Two blocks with expansion 3 keep the parameter count
close to one attention block.

### 2D NA → ConvNeXt (`na2d_to_convnext_*`)

```diff
-encoder: redimnat1_encoder
+encoder: redimnat5_encoder
+  use_convnext_na_2d: true
```

All 11 backbone neighborhood-attention blocks become ConvNeXt pairs; the 5 axial blocks and the
final 1D stage are unchanged.

### 1D NA → ConvNeXt (`na1d_to_convnext_*`)

```diff
-encoder: redimnat1_encoder
+encoder: redimnat5_encoder
+  use_convnext_nattn_1d: true
```

The six 1D neighborhood-attention layers of the final stage become ConvNeXt pairs (k = 31); the
two global-attention layers and the whole 2D backbone are unchanged.
