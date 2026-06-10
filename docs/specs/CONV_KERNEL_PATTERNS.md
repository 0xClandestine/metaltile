# Convolution Module — Authoring Patterns

This document is the canonical guide for writing and extending kernels in
`metaltile-std/src/convolution/`. Read it before adding, porting, or refactoring
any file here.

---

## Philosophy: library, not copy-paste

Every kernel in this module is DSL code — it is compiled to IR, run through the
standard optimisation passes, and emitted as MSL. The goal is the same one that
motivates a library API: **one place of truth**, **named abstractions**, and
**mechanically verifiable correctness** via the in-source test suite.

The two tools that make this possible are `#[kernel(variants(...))]` for
compile-time constant specialisation, and cross-kernel calling for shared
sub-expressions. Between them they cover the two dominant sources of repetition
in convolution code: format families (int2/int3/int4/…) and shared decode logic
(e2m1, e8m0, nibble unpack, …).

---

## The direct-convolution skeleton

All dense convolutions — 1D, 2D, 3D, depthwise, grouped — share the same
four-step skeleton:

```
1. Index decode:   flat output index → spatial coordinates (n, oc, oh, ow, …)
2. Anchor:         receptive-field start in the *padded* input frame
3. Field walk:     loop over (ic, kd?, kh, kw); for each tap:
                     a. bounds-check  →  valid: bool
                     b. clamped-load  →  pix_m = select(valid, pix, 0.0)
                     c. weight load / decode  →  wt
                     d. accumulate  →  acc += pix_m * wt
4. Store:          out[idx] = acc.cast::<T>()
```

This is intentionally not extracted into a single super-kernel — the loop bounds
differ per dimensionality and padding rules differ per format. What *is* shared
is the **tap loop body (steps 3b–3d)** and the **weight decode sub-expression**;
those are the right extraction points.

### Rule: always accumulate in f32

All accumulation is in `f32` regardless of `T`. Cast input loads at the point of
use with `.cast::<f32>()`; cast the final store back with `.cast::<T>()`. This is
identical to MLX's direct-conv approach and avoids catastrophic cancellation in
bf16.

### Rule: padded-frame indexing

Receptive-field anchors are computed in the **padded** input frame so every index
stays a non-negative u32. A real pixel at padded coordinate `p` sits at unpadded
`p - pad`, valid iff `pad <= p < pad + extent`. No i32 arithmetic.

```rust
let ph0 = oh * stride_h;
for ky in range(0u32, kh, 1u32) {
    let ph = ph0 + ky;                        // padded coordinate
    let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
    let ih = select(row_ok, ph - pad_h, 0u32); // clamped unpadded coordinate
    // ... ih * in_w + iw ...
}
```

For dilated convs (3D, grouped 2D, dilated 1D), the tap offset is multiplied
by dilation: `ph = ph0 + ky * dilation_h`.

For transposed convs (adjoint form), the tap is inverted: `ip = (opp - kx*dilation) / stride`,
valid only when `opp >= kx*dilation`, remainder is zero, and `ip < in_len`.

---

## Compile-time specialisation with `#[kernel(variants(...))]`

Use `variants` whenever a family of kernels:
- Has the **same signature** (parameter names and types)
- Has the **same body structure**
- Differs only in **integer compile-time constants**

### Pattern A — fixed spatial configuration (e.g. patch sizes)

```rust
// conv2d_patch14 and conv2d_patch16 bake KH/KW/SH/SW so the receptive-field
// loops unroll completely. The suffix produces the kernel name.
#[kernel(variants(
    KH = [14u32, 16u32],
    KW = [14u32, 16u32],
    SH = [14u32, 16u32],
    SW = [14u32, 16u32],
    suffix = "patch{KH}"
))]
pub fn conv2d<T>(
    ...
    #[constexpr] kh: u32,
    ...
) {
    ...
    for ky in range(0u32, KH, 1u32) { // KH is the baked constant; kh is the runtime constexpr
        for kx in range(0u32, KW, 1u32) {
            ...
        }
    }
    ...
}
// Produces: conv2d_patch14, conv2d_patch16
```

**Lists are zipped, not a cartesian product.** `KH=[14,16], KW=[14,16]` yields
two variants, not four. For four combos, repeat: `KH=[14,14,16,16],
KW=[14,16,14,16]`.

### Pattern B — integer quantisation bit-width

The `int_conv2d_f32` macro_rules! families (`int2…int6`) are canonical candidates
for `variants`. Because `half = 1 << (BITS-1)` and `full = (1 << BITS) as f32`
are derivable from `BITS` alone inside the body, no float-literal variants are
needed — the float values are computed at substitution time inside the body:

```rust
// Before (macro_rules!):
//   int_conv2d_f32!(mt_int2_conv2d, 2u32, 2u32, 4.0f32);
//   int_conv2d_f32!(mt_int3_conv2d, 3u32, 4u32, 8.0f32);
//   ...

// After (variants):
#[kernel(variants(BITS = [2u32, 3u32, 4u32, 5u32, 6u32], suffix = "int{BITS}_conv2d"))]
pub fn mt_intN_conv2d_f32<T>(..., #[constexpr] block_size: u32) {
    ...
    let words_per_row = contraction * BITS / 32u32;   // BITS substituted at compile time
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    ...
    let elem = select(q >= half, qf - full, qf);       // sign-extend
    ...
}
// Produces: mt_int2_conv2d_f32, mt_int3_conv2d_f32, …, mt_int6_conv2d_f32
```

The same pattern applies to the `int_conv2d_e8m0` family (E8M0 scale decode
instead of raw f32).

### When *not* to use variants

- Kernels with **different signatures** (e.g. `mt_mxfp4_conv2d` takes
  `scales: Tensor<u8>` while `mt_fp8_e5m2_conv2d` takes `scales: Tensor<f32>` —
  different types, different kernels).
- Kernels with **structurally different bodies** (different if-branches, different
  decode logic that can't be expressed as arithmetic on integer substitution
  variables).

In those cases, write separate `#[kernel]` functions. Clearly name each one after
its format so the inventory is self-documenting.

---

## Cross-kernel calling for shared decode primitives

Cross-kernel calling (`let result = my_primitive(value);` inside a `#[kernel]`
body) is the right tool for a decode sub-expression that:
- Appears **identically** in multiple kernels across files (e.g. E2M1, E8M0, byte
  unpack)
- Takes one or a few **scalar arguments** already in registers
- Returns a **scalar result** consumed immediately

Extracting a decode step into a named primitive serves the library goal: every
caller names the operation it is performing (`mt_decode_e2m1`, `mt_decode_e8m0`,
…) rather than inlining the bit-manipulation logic and inviting drift.

### Primitive conventions

Decode primitives live in `convolution/primitives.rs`. A primitive:
- Is a `#[kernel]` with no tensor parameters (all arguments are scalar values)
- Takes the already-fetched raw bits and any per-element scale, and returns `f32`
- Is named `mt_decode_<format>` or `mt_unpack_<format>`

```rust
// convolution/primitives.rs

/// Decode an E2M1 nibble to f32 (mxfp4 / nvfp4 weight elements).
/// Input: the 4-bit code in the low nibble of `nib`.
#[kernel]
pub fn mt_decode_e2m1(nib: u32) -> f32 { ... }

/// Decode an E8M0 byte to an f32 scale: 2^(byte - 127).
#[kernel]
pub fn mt_decode_e8m0(byte: u8) -> f32 {
    let b = byte.cast::<f32>();
    exp2(b - 127.0f32)
}

/// Unpack BITS bits at `bit_off` from a packed u32 word pair (lo, hi).
/// Returns the raw unsigned code.
#[kernel]
pub fn mt_unpack_nbit(w0: u32, w1: u32, bit_off: u32, bits: u32) -> u32 { ... }
```

Calling a primitive inside a convolution kernel:

```rust
// In mt_mxfp4_conv2d:
let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
let elem = mt_decode_e2m1(nib);               // cross-kernel call
let scale = mt_decode_e8m0(load(scales[...])); // cross-kernel call
acc = acc + pix_m * (elem * scale);
```

The `KernelInlinePass` inlines both primitives at codegen time — the MSL output
is identical to hand-inlined code with zero overhead.

### When *not* to use cross-kernel calling

- The sub-expression appears in only **one kernel** — inline it directly.
- The sub-expression involves **tensor arguments** (buffer loads with strided
  indices) — those cannot be mapped via `KernelCallArg::Value`; use `Tensor`
  args or a separate kernel instead.
- The logic is format-unique and **unlikely to be reused** — naming is still
  valuable but a private `fn` comment is enough.

---

## File layout

```
convolution/
  mod.rs                          — module declarations
  primitives.rs                   — shared decode/unpack primitives (#[kernel])

  // ── 1D dense / dilated / transposed ─────────────────────────────
  audio_conv1d.rs                 — dense strided 1D (audio patch-embed)
  audio_conv1d_block_scaled.rs    — quantized-weight twins of audio_conv1d
  fishspeech_conv1d_block_scaled.rs — quantized dilated 1D (FishSpeech ResBlock)
  conv1d_dilated_transpose.rs     — dilated 1D + transposed (upsampling) 1D
  conv1d_transpose_depthwise.rs   — depthwise transposed 1D (StyleTTS2/Kokoro pool)
  conv1d_causal_step_silu_cast_many.rs — streaming causal 1D + SiLU (GDN)

  // ── 2D dense / grouped / MMA / quantized ────────────────────────
  conv2d.rs                       — patch14/16 (variants), generic, grouped (dilated)
  conv2d_block_scaled.rs          — quantized-weight twins of conv2d (all formats)
  conv2d_mma.rs                   — MMA-tiled 2D (implicit-GEMM, stride=1/pad=0)
  conv2d_mma_block_scaled.rs      — quantized-weight MMA-tiled 2D

  // ── 3D dense / grouped / MMA / quantized ────────────────────────
  conv3d.rs                       — generic, grouped (dilated)
  conv3d_block_scaled.rs          — quantized-weight twins of conv3d
  conv3d_mma.rs                   — MMA-tiled 3D
  conv3d_mma_block_scaled.rs      — quantized-weight MMA-tiled 3D

  // ── Depthwise 2D ────────────────────────────────────────────────
  depthwise_conv2d.rs             — NCHW depthwise 2D (k, stride, pad, dilation)
  depthwise_conv2d_block_scaled.rs — quantized-weight depthwise 2D
  depthwise_conv2d_nhwc.rs        — NHWC depthwise 2D (k_h, k_w, FastVLM)

  // ── Winograd fast convolution ───────────────────────────────────
  winograd_conv.rs                — F(2×2,3×3) + split filter-transform path

  // ── Steel implicit-GEMM stubs (not yet in DSL) ──────────────────
  steel_conv/
    mod.rs                        — re-exports
    steel_conv.rs                 — 2D steel conv stub
    steel_conv_3d.rs              — 3D steel conv stub
    steel_conv_general.rs         — grouped/dilated steel conv stub

  // ── Stale placeholder (not declared in mod.rs) ──────────────────
  mlx_conv_stub.rs                — old metaltile-bench stub; not compiled
```

### File assignment rules

| Kernel shape | File | Dispatch mode |
|---|---|---|
| 1D strided, dense | `audio_conv1d.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 1D dilated, dense | `conv1d_dilated_transpose.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 1D transposed (upsampling) | `conv1d_dilated_transpose.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 1D depthwise transposed | `conv1d_transpose_depthwise.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 1D streaming causal + SiLU | `conv1d_causal_step_silu_cast_many.rs` | Grid1D over channels (`[conv_dim]`) |
| 1D quantized (audio stem) | `audio_conv1d_block_scaled.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 1D quantized dilated (FishSpeech) | `fishspeech_conv1d_block_scaled.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 2D strided/padded, dense | `conv2d.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 2D grouped / dilated | `conv2d.rs` (Grouped) | Grid3D, `grid_1d(n_out, 256)` |
| 2D MMA-tiled (implicit-GEMM) | `conv2d_mma.rs` | Reduction, `grid_3d(..., [128,1,1])` |
| 2D quantized (all formats) | `conv2d_block_scaled.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 2D quantized MMA-tiled | `conv2d_mma_block_scaled.rs` | Reduction, `grid_3d(..., [128,1,1])` |
| 3D strided/padded, dense | `conv3d.rs` | Grid3D, `grid_1d(n_out, 256)` |
| 3D grouped / dilated | `conv3d.rs` (Grouped) | Grid3D, `grid_1d(n_out, 256)` |
| 3D quantized | `conv3d_block_scaled.rs` | Grid3D, `grid_1d(n_out, 256)` |
| Depthwise NCHW | `depthwise_conv2d.rs` | Grid3D, `grid_1d(n_out, 256)` |
| Depthwise NHWC | `depthwise_conv2d_nhwc.rs` | Grid3D, `grid_1d(n_out, 256)` |
| Depthwise quantized | `depthwise_conv2d_block_scaled.rs` | Grid3D, `grid_1d(n_out, 256)` |
| Winograd F(2×2,3×3) tile | `winograd_conv.rs` | Grid3D, `grid_1d(n_tiles, 64)` |
| Winograd filter transform | `winograd_conv.rs` | Grid3D, `grid_1d(n_filt, 64)` |

### Kernel inventory

| Kernel name | Location | Description |
|---|---|---|
| `conv2d_patch14` | `conv2d.rs` | 14×14 stride-14 patch-embed (Qwen-VL / SigLIP) |
| `conv2d_patch16` | `conv2d.rs` | 16×16 stride-16 patch-embed (CLIP / Gemma-VL) |
| `conv2d_generic` | `conv2d.rs` | Generic 2D conv (strides, padding) |
| `conv2d_grouped` | `conv2d.rs` | General 2D conv (dilation, groups) |
| `conv2d_mma` | `conv2d_mma.rs` | MMA-tiled 2D conv (stride=1, pad=0) |
| `mt_mxfp4_conv2d` | `conv2d_block_scaled.rs` | MXFP4 (E2M1+E8M0) quantized 2D |
| `mt_nvfp4_conv2d` | `conv2d_block_scaled.rs` | NVFP4 (E2M1+E4M3×global) 2D |
| `mt_fp4_conv2d` | `conv2d_block_scaled.rs` | Legacy FP4 (E2M1+FP32 scale) 2D |
| `mt_mxfp8_e4m3_conv2d` | `conv2d_block_scaled.rs` | MXFP8 E4M3 (E8M0 scale) 2D |
| `mt_mxfp8_e5m2_conv2d` | `conv2d_block_scaled.rs` | MXFP8 E5M2 (E8M0 scale) 2D |
| `mt_fp8_e5m2_conv2d` | `conv2d_block_scaled.rs` | Legacy FP8 E5M2 (FP32 scale) 2D |
| `mt_nvfp8_conv2d` | `conv2d_block_scaled.rs` | NVFP8 (E4M3+FP32 scale) 2D; also fp8_e4m3 |
| `mt_int8_conv2d` | `conv2d_block_scaled.rs` | Symmetric int8 (FP32 scale) 2D |
| `mt_int2_conv2d` … `mt_int6_conv2d` | `conv2d_block_scaled.rs` | Sub-byte int (FP32 scale) 2D |
| `mt_mxint2_conv2d` … `mt_mxint6_conv2d` | `conv2d_block_scaled.rs` | Sub-byte int (E8M0 scale) 2D |
| `mt_mxint8_conv2d` | `conv2d_block_scaled.rs` | MXINT8 (E8M0 scale) 2D |
| `mt_nvfp8_f16_conv2d` | `conv2d_block_scaled.rs` | NVFP8 with FP16 scale 2D |
| `mt_fp4_f16_conv2d` | `conv2d_block_scaled.rs` | FP4 with FP16 scale 2D |
| `mt_fp8_e5m2_f16_conv2d` | `conv2d_block_scaled.rs` | FP8 E5M2 with FP16 scale 2D |
| `mt_int8_f16_conv2d` | `conv2d_block_scaled.rs` | Symmetric int8 with FP16 scale 2D |
| `mt_int2_f16_conv2d` … `mt_int6_f16_conv2d` | `conv2d_block_scaled.rs` | Sub-byte int (FP16 scale) 2D |
| `mt_mxfp4_conv2d_mma` | `conv2d_mma_block_scaled.rs` | MXFP4 MMA-tiled 2D |
| `mt_nvfp4_conv2d_mma` | `conv2d_mma_block_scaled.rs` | NVFP4 MMA-tiled 2D |
| `mt_fp4_conv2d_mma` | `conv2d_mma_block_scaled.rs` | Legacy FP4 MMA-tiled 2D |
| `mt_mxfp8_e4m3_conv2d_mma` | `conv2d_mma_block_scaled.rs` | MXFP8 E4M3 MMA-tiled 2D |
| `mt_mxfp8_e5m2_conv2d_mma` | `conv2d_mma_block_scaled.rs` | MXFP8 E5M2 MMA-tiled 2D |
| `conv3d_generic` | `conv3d.rs` | Generic 3D conv (strides, padding) |
| `conv3d_grouped` | `conv3d.rs` | General 3D conv (dilation, groups) |
| `depthwise_conv2d` | `depthwise_conv2d.rs` | NCHW depthwise 2D |
| `depthwise_conv2d_nhwc` | `depthwise_conv2d_nhwc.rs` | NHWC depthwise 2D (k_h/k_w) |
| `audio_conv1d` | `audio_conv1d.rs` | Dense strided 1D (audio stem) |
| `ffai_conv1d_causal_step_silu_cast_many` | `conv1d_causal_step_silu_cast_many.rs` | Streaming causal 1D + SiLU, K=4 fixed |
| `conv1d_dilated` | `conv1d_dilated_transpose.rs` | Dilated 1D conv (ResBlock) |
| `conv1d_transpose` | `conv1d_dilated_transpose.rs` | Transposed 1D (upsampling) |
| `ffai_conv1d_transpose_depthwise` | `conv1d_transpose_depthwise.rs` | Depthwise transposed 1D |
| `winograd_conv2d_3x3` | `winograd_conv.rs` | Winograd F(2×2,3×3) single-pass |
| `winograd_filter_transform_3x3` | `winograd_conv.rs` | Pre-transform 3×3→4×4 filters |
| `winograd_conv2d_3x3_split` | `winograd_conv.rs` | Winograd with pre-transformed filters |

**Single kernel or small family (≤ 3 closely related variants):** keep in the
same file as the primary kernel.  
**Large format family (4+ variants sharing the same body):** use `variants(...)`;
do not create separate files per bit-width.

---

## The block-scaled (quantized-weight) convolution pattern

Every quantized-weight convolution in this module follows a single unified
pattern. The filter `[out_ch, in_ch, kd?, kh, kw]` is flattened to a 2-D matrix
`[out_ch, C]` where `C = in_ch · kd? · kh · kw` — the per-output-channel
contraction — block-scaled along `C`. Only the weight is quantized; the
per-channel `bias` stays `T`.

### Quantized weight layouts by format

| Format | Code tensor | Scale tensor | Decode pattern |
|---|---|---|---|
| mxfp4 | `Tensor<u32>` `[out_ch, C/8]` — 8 nibbles/word | `Tensor<u8>` `[out_ch, C/block_size]` — E8M0 | `e2m1_decode(nib) * exp2(sbyte - 127)` |
| nvfp4 | `Tensor<u32>` `[out_ch, C/8]` — 8 nibbles/word | `Tensor<u8>` `[out_ch, C/block_size]` — E4M3 micro | `e2m1_decode(nib) * e4m3_decode(micro) * global` |
| fp4 | `Tensor<u32>` `[out_ch, C/8]` — 8 nibbles/word | `Tensor<f32>` `[out_ch, C/block_size]` — raw f32 | `e2m1_decode(nib) * scale` |
| fp4_f16 | `Tensor<u32>` `[out_ch, C/8]` | `Tensor<f16>` `[out_ch, C/block_size]` | `e2m1_decode(nib) * scale.cast::<f32>()` |
| mxfp8_e4m3 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<u8>` `[out_ch, C/block_size]` — E8M0 | `e4m3_decode(byte) * exp2(sbyte - 127)` |
| mxfp8_e5m2 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<u8>` `[out_ch, C/block_size]` — E8M0 | `e5m2_decode(byte) * exp2(sbyte - 127)` |
| nvfp8 / fp8_e4m3 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<f32>` `[out_ch, C/block_size]` — raw f32 | `e4m3_decode(byte) * scale` |
| fp8_e5m2 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<f32>` `[out_ch, C/block_size]` — raw f32 | `e5m2_decode(byte) * scale` |
| int8 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<f32>` `[out_ch, C/block_size]` — raw f32 | `int8_decode(byte) * scale` |
| mxint8 | `Tensor<u8>` `[out_ch, C]` — 1 byte/tap | `Tensor<u8>` `[out_ch, C/block_size]` — E8M0 | `int8_decode(byte) * exp2(sbyte - 127)` |
| int2…int6 | `Tensor<u32>` `[out_ch, C·bits/32]` — tight bit-stream | `Tensor<f32>` `[out_ch, C/block_size]` | straddle-aware unpack + sign-extend × scale |
| mxint2…mxint6 | `Tensor<u32>` `[out_ch, C·bits/32]` — tight bit-stream | `Tensor<u8>` `[out_ch, C/block_size]` — E8M0 | straddle-aware unpack + sign-extend × `exp2(sbyte-127)` |

### Straddle-aware sub-byte decode (int2–6 / mxint2–6)

Sub-byte integer formats pack codes LSB-first into u32 words **per output-channel
row**. Row `oc` starts at word `oc · (C·bits / 32)`. For contraction index `col`
the code sits at bit `col·bits`. A straddle-aware two-word read extracts it:

```rust
let bit_off = col * bits;
let word_idx = bit_off / 32u32;
let bit_in_w = bit_off & 31u32;
let bits_in_w0 = 32u32 - bit_in_w;
let lo_bits = select(bits_in_w0 >= bits, bits, bits_in_w0);
let spill = bits - lo_bits;
let w0 = load(weight[w_row_word + word_idx]);
let w1 = load(weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)]);
let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
let q = lo | hi;
let qf = q.cast::<f32>();
let elem = select(q >= half, qf - full, qf); // sign-extend
```

### FP16-scale twins

Every FP32-scaled quantized conv has a `_f16` twin where `scales` is
`Tensor<f16>` read with `.cast::<f32>()`. The per-element decode, weight
indexing, receptive-field walk, and accumulation are **identical** to the FP32
twin — only the scale load changes. This matches the host `f16_scale_decode`
path.

### Block-size constraints

- `C` must be a multiple of `block_size` for all formats.
- For 4-bit formats, `block_size` must be a multiple of 8 (pack alignment).
- For MMA (simdgroup-matrix) variants, `C` must also be a multiple of 32 (the
  MMA K-tile).

### Per-file scope of quantized kernels

Each dimensionality (1D, 2D, 3D, depthwise) has its own `*_block_scaled.rs`
file. Every file covers the **complete format superset** — mxfp4, nvfp4, fp4,
mxfp8_e4m3, mxfp8_e5m2, fp8_e5m2, nvfp8, int8, int2–6 (FP32 scale), mxint2–6
(E8M0 scale), mxint8, plus FP16-scale twins. When adding a new format, add it
to **all** `*_block_scaled.rs` files that cover that dimensionality.

---

## The MMA-tiled (implicit-GEMM) convolution pattern

MMA-tiled convolutions (the `*_mma.rs` / `*_mma_block_scaled.rs` files) replace
the direct-conv skeleton with an implicit-im2col GEMM. They follow a structure
shared with `mt_qmm_mma` in the quantization module.

### Tile geometry

| Parameter | Value |
|---|---|
| threads per group (tpg) | 128 = 4 simdgroups × 32 lanes |
| BM (oc-axis tile) | 32 |
| BN (pixel-axis tile) | 32 |
| BK (K-tile) | 32 |
| Warp grid | 2×2 (sm = sg/2, sn = sg%2) |
| Sub-tile per SG | 16×16 = 4 × 8×8 frags (c_f00..c_f11) |
| TG memory A | `[32 × 36]` = 1152 elements, skew stride 36 |
| TG memory B | `[32 × 36]` = 1152 elements, skew stride 36 |
| Grid | `[out_ch/32, n_pixels/32, 1]` |
| Dispatch mode | Reduction |
| Constraints | `out_ch % 32 == 0`, `n_pixels % 32 == 0`, `BK % 32 == 0`, stride=1, dilation=1, pad=0 |

### MMA inner loop (4 × 4 k-inner)

The inner loop unrolls 4 k-inner steps. Each step:
1. Loads 2 8×8 A-frags from `as` via `threadgroup_load` + `simdgroup_elem_store`
2. Loads 2 8×8 B-frags from `bs`
3. Issues 4 `simdgroup_matmul` calls: `c_f00 += a_f0·b_f0`, `c_f01 += a_f0·b_f1`,
   `c_f11 += a_f1·b_f1`, `c_f10 += a_f1·b_f0`

This 4×4=16 MMA calls per SG per K-block of 32. The A/B loads use the 8×8 frag
lane mapping from `mt_qmm_mma`:

```rust
let qid = lane / 4u32;
let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
let fn1 = fn0 + 1u32;
```

### Quantized MMA B-load

In `*_mma_block_scaled.rs`, only the cooperative B-load differs from the dense
MMA kernel. Each lane dequantizes 8 contiguous K-elements for its oc-row on the
fly and stores the result into `bs`. The A-load (implicit im2col gather) and the
MMA inner loop are copied **verbatim** from the dense MMA kernel.

### K-tail masking

When `total_k = in_ch·kh·kw` is not a multiple of 32 (common: e.g. ViT-patch14
has `total_k = 3·14·14 = 588`), the K-loop steps by 32 and the final iteration
reads past `total_k`. The A/B coop loads use `select(kt < total_k, load, 0)` to
zero-fill the tail; the clamped index keeps the gather in range. Both
contributors are zero, so the partial-K MMA accumulator stays correct.

---

## The Winograd F(2×2,3×3) pattern

Winograd is a 2.25× multiply reduction vs direct 3×3 conv. The module provides
three kernels:

### `winograd_conv2d_3x3` (single kernel)

Computes the entire F(2×2,3×3) algorithm per output tile:
1. **Input transform** `V = Bᵀ·d·B` — row-mix then col-mix of the 4×4 input tile
2. **Filter transform** `U = G·g·Gᵀ` — row-mix then col-mix of the 3×3 filter
   (recomputed per tile — redundant but simple)
3. **Accumulate** `M += U ⊙ V` across input channels
4. **Output transform** `Y = Aᵀ·M·A` — row-mix then col-mix, add bias, store 2×2 tile

Dispatch: Grid3D, one thread per 2×2 tile over `batch·out_ch·tiles_h·tiles_w`.
Output dims must be even.

### `winograd_filter_transform_3x3` + `winograd_conv2d_3x3_split` (two-pass)

The filter transform is hoisted into a separate kernel. `winograd_filter_transform_3x3`
pre-transforms every `(oc, ic)` 3×3 filter into a 4×4 `U` (one thread per filter,
Grid3D `grid_1d(out_ch·in_ch, 64)`). `winograd_conv2d_3x3_split` loads those 16
precomputed values instead of the 9 raw taps + transform, eliminating the
O(tiles) redundant transform work.

### Transform matrices (F(2×2, 3×3))

```
Bᵀ = ⎡ 1  0 -1  0 ⎤   G = ⎡ 1     0    0   ⎤   Aᵀ = ⎡ 1  1  1  0 ⎤
     ⎢ 0  1  1  0 ⎥       ⎢ ½     ½    ½   ⎥        ⎣ 0  1 -1 -1 ⎦
     ⎢ 0 -1  1  0 ⎥       ⎢ ½    -½    ½   ⎥
     ⎣ 0  1  0 -1 ⎦       ⎣ 0     0    1   ⎦
```

All transforms are built from adds and shifts (no multiplies except the three
½ factors in G and three ½ factors in Gᵀ).

---

## The causal streaming conv1d + activation fusion pattern

The `ffai_conv1d_causal_step_silu_cast_many` kernel collapses a T-length sweep
of per-token `conv1d_causal_step` + `mt_silu_cast_to_f32` into one dispatch per
channel.

### Key architecture decisions

- **K=4 is hardcoded.** Three explicit state scalars (`s0, s1, s2`) keep the K-1
  rolling window in registers for the duration of the channel's T-sweep. No
  round-trip through device memory between tokens.
- **Grid**: `[conv_dim]` threads, one per channel. Each thread sequentially sweeps
  T tokens.
- **Bandwidth save**: ~30× less conv-state traffic vs per-token dispatch
  (24 KB → 24 KB per channel sweep for Qwen3.6-A3B prefill).
- **Weight convention**: `w[0..K-2]` weights state slots (oldest→newest),
  `w[K-1]` weights the current input — matches `conv1d_causal_step`.

When adding a K≠4 variant, create a new file with explicit scalar count matching
the kernel size.

---

## Adding a new convolution kernel

### Step 1 — identify which skeleton it fits

| Shape | Skeleton | File |
|---|---|---|
| 1D strided/padded, dense | `audio_conv1d` body (1 spatial loop, NCL) | `audio_conv1d.rs` |
| 1D dilated/padded, dense | `conv1d_dilated` body (dilation in tap index) | `conv1d_dilated_transpose.rs` |
| 1D transposed (upsampling) | `conv1d_transpose` body (adjoint gather form) | `conv1d_dilated_transpose.rs` |
| 1D depthwise transposed | `ffai_conv1d_transpose_depthwise` body (per-channel) | `conv1d_transpose_depthwise.rs` |
| 1D causal streaming + activation | `ffai_conv1d_causal_step_silu_cast_many` body | `conv1d_causal_step_silu_cast_many.rs` |
| 1D quantized (audio stem) | block-scaled body + decode primitive | `audio_conv1d_block_scaled.rs` |
| 1D quantized dilated | block-scaled body + decode (with dilation) | `fishspeech_conv1d_block_scaled.rs` |
| 2D strided/padded, dense | `conv2d_generic` / `conv2d_grouped` skeleton | `conv2d.rs` |
| 2D quantized weight | block-scaled body + decode primitive | `conv2d_block_scaled.rs` |
| 2D SIMD-matrix tiled | `conv2d_mma` skeleton (implicit im2col) | `conv2d_mma.rs` |
| 2D quantized + MMA | MMA skeleton + dequant B-load | `conv2d_mma_block_scaled.rs` |
| 2D Winograd 3×3 | `winograd_conv2d_3x3` skeleton (F(2×2,3×3)) | `winograd_conv.rs` |
| 3D strided/padded, dense | add a `D` loop to the 2D skeleton | `conv3d.rs` |
| 3D quantized | block-scaled body + decode (3D) | `conv3d_block_scaled.rs` |
| Depthwise (groups==channels) | `depthwise_conv2d` body (NCHW) | `depthwise_conv2d.rs` |
| Depthwise NHWC | `depthwise_conv2d_nhwc` body (channel-last) | `depthwise_conv2d_nhwc.rs` |
| Depthwise quantized | block-scaled body + decode (depthwise) | `depthwise_conv2d_block_scaled.rs` |

### Step 2 — choose the right tool

```
New kernel differs from existing only in compile-time integer constants?
  └── YES → add to an existing #[kernel(variants(...))] list, or start one
  └── NO  →
        Shares a decode sub-expression with another kernel?
          └── YES → extract/use a primitive in primitives.rs
          └── NO  → write a standalone #[kernel] with a clear name

Is it a new quantized format (e.g. fp6, nvfp6)?
  └── Add the format's decode + scale to ALL *_block_scaled.rs files
       plus the MMA-block-scaled files for 2D/3D.

Is it a new activation fusion (e.g. conv + SiLU + cast to f32)?
  └── Follow the causal streaming pattern: one channel per thread,
       state in registers, sequential T-sweep, single dispatch.
```

### Step 3 — write the kernel

**Direct-conv skeleton:** Follow the four-step skeleton:
1. Decode the flat output index into spatial coordinates first.
2. Compute receptive-field anchors from the spatial coordinates in the padded frame.
3. Walk the field; clamp-and-mask each padding tap.
4. Load the weight element; call a decode primitive if applicable.
5. Accumulate in `f32`; store `.cast::<T>()`.

**MMA skeleton:** Follow the tile geometry above. Copy the A-load and MMA inner
loop from the dense MMA kernel; replace only the B-load for quantized variants.

**Winograd skeleton:** Follow the F(2×2,3×3) transforms. Precompute validity
flags per row/column before the input tile load loop.

**Causal streaming skeleton:** One thread per channel. Load state once at start,
sweep T sequentially, write state once at end.

Keep the body under ~60 lines for direct-conv kernels. If it grows beyond that,
the body likely contains logic that belongs in a primitive or should be split
into a separate dispatch.

### Step 4 — write the test

Every kernel requires at least one `#[test_kernel]` covering:
- A non-trivial input with `batch > 1`
- All three dtypes (`f32, f16, bf16`) via `dtypes = [f32, f16, bf16]`
- A reference CPU oracle (a plain Rust `naive_*` function)

Use the existing oracles where possible:
- `naive_conv2d` in `conv2d.rs` (also usable by block-scaled via dequant)
- `naive_depthwise_conv2d` in `depthwise_conv2d.rs`
- `naive_conv1d` in `audio_conv1d.rs`
- `naive_conv3d` in `conv3d.rs`
- `naive_conv2d_mma` in `conv2d_mma.rs` (pixel-major output)
- `naive_conv3x3` in `winograd_conv.rs`
- `cpu_reference` in `conv1d_causal_step_silu_cast_many.rs`
- `naive_dilated` / `naive_transpose` in `conv1d_dilated_transpose.rs`
- `naive` in `conv1d_transpose_depthwise.rs`

For block-scaled kernels, the oracle is `naive_*` run over the **dequantized**
filter using `quant::format::QFormat::dequant`.

**Tolerances:** `tol = [1e-3, 8e-3, 4e-2]` for `f32/f16/bf16` for direct convs.
Winograd split path: `tol = [1e-3, 8e-3, 8e-2]` (more rounding via the
intermediate U buffer). Depthwise transpose: `tol = [1e-4, 1e-2, 5e-2]`.

**Dispatch modes:**
- Direct convs (Grid3D): `.grid_1d(n_out, 256)`
- MMA convs (Reduction): `.grid_3d((out_ch/32), (n_pixels/32), 1, [128,1,1])`
- Winograd: `.grid_1d(n_tiles, 64)`
- Filter transform: `.grid_1d(n_filt, 64)`
- Causal streaming: `.grid_3d(conv_dim, 1, 1, [1,1,1])`

### Step 5 — write the bench

Every kernel requires a `#[bench]` using a realistic inference shape (not a
toy). Set `.bytes_moved(...)` and `.flops(...)` so the bench harness can report
GB/s and TFLOP/s alongside the MT% figure.

**FLOP counting:** Each output element does a multiply-add per tap — `2` flops
per `(in_ch · kd? · kh · kw)` tap.

```
For dense convs (groups=1):
  flops = 2 · N · Co · Do? · Ho · Wo · Ci · kd? · kh · kw

For depthwise convs (groups=ch):
  flops = 2 · N · ch · Ho · Wo · kh · kw   (Ci=1 per group, omitted)

For MMA convs:
  flops = 2 · N · Co · Ho · Wo · Ci · kh · kw   (same; stride=1, pad=0)

For transposed convs (depthwise):
  flops = n_out · k · 2   (k taps per output element)
```

`bytes_moved` should count the output stream as the stable proxy (input/weight
reuse makes a precise count shape-dependent).

### Step 6 — declare in mod.rs

Add `pub mod <file_stem>;` in alphabetical order. No re-exports needed; the
inventory macros register kernels globally. Files not compiled (e.g.
`mlx_conv_stub.rs`) are intentionally not declared, with a comment explaining
why.

---

## What `macro_rules!` is for and when NOT to use it in this module

`macro_rules!` inside `#[kernel]` bodies is **silently dropped by the DSL
parser** — do not use it there.

At the file level, `macro_rules!` *can* be used to stamp out `#[kernel]`
functions, but it is now always inferior to `#[kernel(variants(...))]`:

| | `macro_rules!` | `#[kernel(variants(...))]` |
|---|---|---|
| Visible in docs | Macro items, confusing | `fn` items per variant |
| Error spans | Point at macro call site | Point at `#[kernel]` attr |
| DSL integration | No | Yes — standard pipeline |
| Suffix derivation | Manual | Automatic or templated |

**All existing `macro_rules!` invocations in this module are migration targets.**
When touching a file that uses them, convert the macro to `variants` as part of
the same change.

Current `macro_rules!` usage in the module (migration targets):
- `conv2d_block_scaled.rs` — `int_conv2d_f32!` (6 invocations), `int_conv2d_e8m0!` (6), `int_conv2d_f16!` (6)
- `conv2d_mma_block_scaled.rs` — similar int-family macros
- `audio_conv1d_block_scaled.rs` — similar int-family macros
- `fishspeech_conv1d_block_scaled.rs` — similar int-family macros
- `depthwise_conv2d_block_scaled.rs` — similar int-family macros
- `conv3d_block_scaled.rs` — similar int-family macros

---

## Naming conventions

| Pattern | Examples |
|---|---|
| Dense float kernel (2D) | `conv2d_generic`, `conv2d_grouped` |
| Fixed-constant variant | `conv2d_patch14`, `conv2d_patch16` |
| 3D variants | `conv3d_generic`, `conv3d_grouped` |
| Depthwise | `depthwise_conv2d`, `depthwise_conv2d_nhwc` |
| 1D audio stem | `audio_conv1d` |
| 1D dilated / transpose | `conv1d_dilated`, `conv1d_transpose` |
| 1D depthwise transpose | `ffai_conv1d_transpose_depthwise` |
| Causal streaming | `ffai_conv1d_causal_step_silu_cast_many` |
| MMA-tiled (dense) | `conv2d_mma` |
| Block-scaled / quantised | `mt_mxfp4_conv2d`, `mt_int4_conv2d_f32` |
| MMA + quantised | `mt_mxfp4_conv2d_mma` |
| FP16-scale twin | `mt_nvfp8_f16_conv2d`, `mt_int4_f16_conv2d` |
| Winograd | `winograd_conv2d_3x3`, `winograd_filter_transform_3x3`, `winograd_conv2d_3x3_split` |
| Decode primitive | `mt_decode_e2m1`, `mt_decode_e8m0`, `mt_unpack_nbit` |

Format qualifiers follow the pattern `mt_<format>_<op>`:
`mxfp4`, `nvfp4`, `fp4`, `mxfp8_e4m3`, `mxfp8_e5m2`, `fp8_e5m2`, `nvfp8`,
`int8`, `mxint8`, `int4` (= 4-bit symmetric), `intN` (N-bit symmetric).

---

## Steel implicit-GEMM stubs (not yet implemented in DSL)

The `steel_conv/` directory contains documentation stubs for the MLX steel
implicit-GEMM 2D/3D convolution paths:

- `steel_conv.rs` — 2D implicit-GEMM conv (im2col + tiled GEMM via simdgroup
  matrix ops)
- `steel_conv_3d.rs` — 3D implicit-GEMM conv (volumetric im2col)
- `steel_conv_general.rs` — 2D general implicit-GEMM (strides, dilation, groups)

These are **NOT YET IMPLEMENTED** in the `#[kernel]` DSL because they require
simdgroup matrix operations (same blocker as `steel_gemm_fused`) and
`MLXConvParams`-driven im2col index arithmetic. The DSL has neither simdgroup
matmul primitives (beyond the MMA conv kernels in this module, which use direct
`simdgroup_matmul` calls) nor im2col index-descriptor primitives.

The `mlx_conv_stub.rs` file is a stale placeholder from the old metaltile-bench
crate that references `crate::runner` from metaltile-cli and does not compile.
Kept for future-work notes but intentionally not declared in `mod.rs`.