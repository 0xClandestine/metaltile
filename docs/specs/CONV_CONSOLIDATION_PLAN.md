# Convolution Module Consolidation Plan

**Current:** 20 files, 24,213 lines
**Target:** 5 files + `fused/`, ~1,625 lines
**Reduction:** ~93%

---

## Target structure

```
convolution/
  mod.rs              (~25 lines)
  primitives.rs       (~100 lines)   weight decode + unpack sub-expressions
  conv1d.rs           (~300 lines)   all 1D convolutions
  conv2d.rs           (~500 lines)   all 2D convolutions (direct + depthwise + MMA)
  conv3d.rs           (~400 lines)   all 3D convolutions (direct + MMA)
  fused/
    winograd.rs       (~200 lines)   Winograd algorithm (different structure)
    causal_silu.rs    (~100 lines)   streaming causal conv1d + silu cast
  steel_conv/         (unchanged)    MLX implicit-GEMM path
```

---

## The key insight: outer loop is free, decode is everything

Every kernel in this module follows the same skeleton regardless of format:

```
index_decode  →  anchor  →  field_walk { pad_mask; decode_weight; accumulate }  →  store
```

The outer loop and accumulation are byte-for-byte identical across all 80+ kernels.
The **only** line that differs between formats is `decode_weight`.

This means the actual variable content across all block-scaled files is:

| What varies | Mechanism |
|---|---|
| Spatial dimensionality (1D/2D/3D/depthwise) | separate `#[kernel]` — different loop depth |
| Weight element decode (e2m1, e8m0, intN, …) | cross-kernel call to a primitive in `primitives.rs` |
| Integer decode constants (bit-width, sign thresholds) | `#[kernel(variants(...))]` with compile-time `if` |
| Weight/scale tensor type | separate `variants` group per `(weight_type, scale_type)` pair |

All 7 block-scaled files collapse to sections within `conv1d.rs`, `conv2d.rs`,
`conv3d.rs` once this structure is applied.

---

## Format inventory

Every block-scaled file contains the same 4 format families, each present across
all dimensionalities and dispatch paths (direct / depthwise / MMA):

| Family | Formats | weight | scale | output | macro? |
|---|---|---|---|---|---|
| **u32×u8** | mxfp4, nvfp4, mxint2–6 | `Tensor<u32>` | `Tensor<u8>` | `Tensor<T>` | `int_*_e8m0!` (intN) |
| **u32×f32** | fp4, int2–6 | `Tensor<u32>` | `Tensor<f32>` | `Tensor<T>` | `int_*_f32!` (intN) |
| **u8×u8** | mxfp8\_e4m3, mxfp8\_e5m2, nvfp8, mxint8 | `Tensor<u8>` | `Tensor<u8>` | `Tensor<T>` | — |
| **u8×f32** | fp8\_e5m2, int8 | `Tensor<u8>` | `Tensor<f32>` | `Tensor<T>` | — |
| **f16 output** | nvfp8\_f16, fp4\_f16, fp8\_e5m2\_f16, int8\_f16, int2–6\_f16 | varies | varies | `Tensor<f16>` | `int_*_f16!` (intN) |

The `_f16` family is a mixed-precision output variant where the output tensor is
pinned to `f16` instead of generic `T`. It was absent from the original plan.

### Signature groups (for `variants` blocks)

Within a given dimensionality, kernels can only share a `variants` block when their
Rust signatures are identical. With type params (`WT = [u32, u8]`), the weight and
scale types can now vary across variants — so the 4 core families can collapse to 2
groups when their body logic is compatible, using `FMT` as a discriminant:

| Group | WT | ST | Formats | `FMT` values |
|---|---|---|---|---|
| **1** | `Tensor<WT>` | `Tensor<ST>` (u8/f32) | all 4 core families combined | 0=mxfp4, 1=nvfp4, 2=fp4, 3=mxfp8\_e4m3, 4=mxfp8\_e5m2, 5=fp8\_e5m2, 6=nvfp8, 7=int8, 8=mxint8, 9–13=intN×f32, 14–18=mxintN×E8M0 |
| **2** | `Tensor<WT>` | `Tensor<ST>` + global | f16 output family | same FMT range, output pinned |

In practice, combining all 4 core families into one `variants` block requires the
body to distinguish u32 vs u8 weight loads via `if FMT == ... { }` compile-time
guards. This is valid as long as the decode branches are `FMT`-gated (not
`WT`-gated — type params can't be used in compile-time `if` conditions).

If the body difference between families is large, use a tighter grouping:

| Group | WT | ST | Formats |
|---|---|---|---|
| **A** | `Tensor<u32>` | `Tensor<u8>` | mxfp4, nvfp4, mxint2–6 |
| **B** | `Tensor<u32>` | `Tensor<f32>` | fp4, int2–6 |
| **C** | `Tensor<u8>` | `Tensor<u8>` | mxfp8\_e4m3, mxfp8\_e5m2, nvfp8, mxint8 |
| **D** | `Tensor<u8>` | `Tensor<f32>` | fp8\_e5m2, int8 |
| **E** | `Tensor<WT>` | `Tensor<ST>` + global | all f16-output variants |

The right choice depends on how much body logic is shared across weight types.
Start with fine-grained groups (A–E); merge later if the bodies are truly parallel.

---

## The three tools, applied

### 1. Decode primitives (cross-kernel calling)

Define once in `primitives.rs`, call from every conv that uses the format.
`KernelInlinePass` inlines them at codegen — zero overhead.

```rust
// primitives.rs

/// Extract a 4-bit nibble (E2M1 / mxfp4 code) and decode to f32.
#[kernel]
pub fn mt_decode_e2m1(nib: u32) -> f32 { ... }

/// Decode an E8M0 byte to f32 scale: 2^(byte - 127).
#[kernel]
pub fn mt_decode_e8m0(raw: u8) -> f32 {
    exp2(raw.cast::<f32>() - 127.0f32)
}

/// Decode an E4M3 byte to f32.
#[kernel]
pub fn mt_decode_e4m3(raw: u32) -> f32 { ... }

/// Decode an E5M2 byte to f32.
#[kernel]
pub fn mt_decode_e5m2(raw: u32) -> f32 { ... }

/// Straddle-aware N-bit unpack: extract BITS bits at contiguous bit offset `col*BITS`
/// from a (word0, word1) pair. Returns the raw unsigned code.
#[kernel]
pub fn mt_unpack_nbit(w0: u32, w1: u32, bit_in_w: u32, lo_bits: u32, spill: u32) -> u32 { ... }
```

### 2. `#[kernel(variants(...))]` collapsing format families

#### How the naming works

The function name is a **throwaway base prefix** — it is always discarded after
renaming.  The actual kernel name is `"{base}_{suffix}"`.  With type params the
suffix template encodes the types automatically:

```rust
// suffix = "conv2d_{WT}_{ST}"  →  auto-stringifies types:
//   WT=u32, ST=u8  →  suffix "conv2d_u32_u8"  →  full name "mt_conv2d_u32_u8"
//   WT=u8,  ST=f32 →  suffix "conv2d_u8_f32"  →  full name "mt_conv2d_u8_f32"
```

The function body is written with `Tensor<WT>` / `Tensor<ST>` so no types appear
in the source name.  The `FMT` integer discriminant drives `compile-time if` to
select the decode path; unused branches are stripped before body parsing.

#### Example — all u32-weight formats in one block

```rust
// "mt" is the base prefix (discarded).
// WT/ST carry the tensor types; FMT + BITS drive the decode logic.
// Kernel names: mt_conv2d_u32_u8  (FMT 0,1,2,3,4)
//               mt_conv2d_u32_f32 (FMT 5,6,7,8,9)

#[kernel(variants(
    FMT  = [0u32, 1u32, 2u32, 3u32, 4u32,   5u32, 6u32, 7u32, 8u32, 9u32 ],
    BITS = [4u32, 4u32, 2u32, 3u32, 4u32,   2u32, 3u32, 4u32, 5u32, 6u32 ],
    WT   = [u32,  u32,  u32,  u32,  u32,    u32,  u32,  u32,  u32,  u32  ],
    ST   = [u8,   u8,   u8,   u8,   u8,     f32,  f32,  f32,  f32,  f32  ],
    suffix = "conv2d_{WT}_{ST}",
))]
pub fn mt<T>(
    input: Tensor<T>, weight: Tensor<WT>, scales: Tensor<ST>,
    bias: Tensor<T>,  out: Tensor<T>,
    #[constexpr] ... // geometry constexprs
) {
    // index decode (identical for all formats — not shown) ...

    let elem = if FMT == 0u32 {
        // mxfp4: 4-bit E2M1 nibble × E8M0 block scale
        let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
        mt_decode_e2m1(nib) * mt_decode_e8m0(load(scales[w_row_blk + col / block_size]))
    } else if FMT == 1u32 {
        // nvfp4: 4-bit E2M1 nibble × E4M3 micro-scale × global
        let nib = (load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
        mt_decode_e2m1(nib) * mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>()) * global
    } else {
        // FMT 2-4 = mxintN (E8M0 scale), FMT 5-9 = intN (f32 scale)
        // BITS driven by compile-time substitution; half/full derived from BITS.
        let half = 1u32 << (BITS - 1u32);
        let full = (1u32 << BITS).cast::<f32>();
        let words_per_row = contraction * BITS / 32u32;
        let bit_off  = col * BITS;
        let word_idx = bit_off / 32u32;
        let bit_in_w = bit_off & 31u32;
        let lo_bits  = select(32u32 - bit_in_w >= BITS, BITS, 32u32 - bit_in_w);
        let spill    = BITS - lo_bits;
        let q = mt_unpack_nbit(
            load(weight[oc * words_per_row + word_idx]),
            load(weight[oc * words_per_row + select(spill > 0u32, word_idx + 1u32, word_idx)]),
            bit_in_w, lo_bits, spill,
        );
        let qf = q.cast::<f32>();
        let elem_q = select(q >= half, qf - full, qf);
        // ST=u8 → E8M0 scale; ST=f32 → direct load.  Both load from Tensor<ST>.
        // The compile-time if on FMT selects which path; the body is type-consistent
        // because both branches call mt_decode_e8m0(x.cast::<f32>()) or load(x).
        if FMT < 5u32 {
            elem_q * mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
        } else {
            elem_q * load(scales[w_row_blk + col / block_size]).cast::<f32>()
        }
    };
    acc = acc + pix_m * elem;
    // store ...
}
```

The identical `FMT`/`BITS`/`WT`/`ST` block is used for `conv3d`, `conv1d`,
`depthwise_conv2d` — only the index decode and field-walk loop depth differ.

### 3. Merge by dimensionality

Once each format variant is one body, there is nothing preventing all formats
for a given dimensionality from living in the same file:

- `conv2d.rs` = dense direct-conv (patch variants + generic + grouped) + depthwise + all block-scaled groups A–G + MMA + MMA block-scaled
- `conv3d.rs` = dense + grouped + all block-scaled + MMA  
- `conv1d.rs` = `audio_conv1d` + all block-scaled 1D formats + transpose depthwise + dilated transpose

---

## Phase-by-phase migration

### Phase 1 — Extract `primitives.rs` (no behaviour change)

Create `primitives.rs` and declare it in `mod.rs`.

Extract every decode helper that appears in more than one file into a `#[kernel]`:

| Primitive | Currently inlined in |
|---|---|
| `mt_decode_e2m1` | mxfp4 kernels in all 6 files |
| `mt_decode_e8m0` | mxfp4/mxfp8/mxintN kernels (the `exp2(sbits - 127)` line) |
| `mt_decode_e4m3` | mxfp8\_e4m3 + nvfp4 |
| `mt_decode_e5m2` | mxfp8\_e5m2 + fp8\_e5m2 |
| `mt_decode_nvfp8` | nvfp8 format |
| `mt_unpack_nbit` | all int2–int6 straddle-aware paths |

After this phase every decode site in the existing files becomes a cross-kernel
call; bodies shrink but files are not yet merged.

**Test gate:** `cargo test -p metaltile-std` must stay green.

### Phase 2 — Replace `macro_rules!` with `#[kernel(variants(...))]`

Every file has **three** `macro_rules!` families that expand 5 variants each
(`BITS = [2, 3, 4, 5, 6]`):

| Macro family | Scale type | Output type | Present in |
|---|---|---|---|
| `int_*_f32!` | `Tensor<f32>` | `Tensor<T>` | all 6 block-scaled files |
| `int_*_e8m0!` | `Tensor<u8>` | `Tensor<T>` | all 6 block-scaled files |
| `int_*_f16!` | `Tensor<f32>` | `Tensor<f16>` | all 6 block-scaled files |

Each macro takes `(kernel_name, BITS, HALF, FULL)` — e.g.
`int_conv2d_f32!(mt_int4_conv2d, 4u32, 8u32, 16.0f32)`. All 3 arguments are
derivable from `BITS` at substitution time:

```rust
let half = 1u32 << (BITS - 1u32);       // 8u32 when BITS=4
let full = (1u32 << BITS).cast::<f32>(); // 16.0f32 when BITS=4
```

No float-literal variants needed. The `int_*_f16!` family shares the same decode
body as `int_*_f32!`; only the output store differs (`.cast::<f16>()`).

Since all three macro families share the same `BITS` range and the only difference
is scale type and output type, they can be collapsed into **one** variants block
using `ST` and `OUT` type params:

```rust
// "mt" base + suffix → mt_conv2d_u32_u8, mt_conv2d_u32_f32, mt_conv2d_u32_f16
#[kernel(variants(
    BITS = [2u32, 3u32, 4u32, 5u32, 6u32,  2u32, 3u32, 4u32, 5u32, 6u32,  2u32, 3u32, 4u32, 5u32, 6u32],
    ST   = [u8,   u8,   u8,   u8,   u8,    f32,  f32,  f32,  f32,  f32,   f32,  f32,  f32,  f32,  f32 ],
    OUT  = [T,    T,    T,    T,    T,      T,    T,    T,    T,    T,     f16,  f16,  f16,  f16,  f16 ],
    FMT  = [0u32, 0u32, 0u32, 0u32, 0u32,  1u32, 1u32, 1u32, 1u32, 1u32,  2u32, 2u32, 2u32, 2u32, 2u32],
    suffix = "conv2d_{ST}_{OUT}",   // e.g. conv2d_u8_T, conv2d_f32_T, conv2d_f32_f16
))]
pub fn mt<T>(
    ..., weight: Tensor<u32>, scales: Tensor<ST>, out: Tensor<OUT>, ...
) {
    // FMT == 0 → E8M0 scale (ST=u8), FMT == 1/2 → f32 scale
    let scale = if FMT == 0u32 {
        mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
    } else {
        load(scales[w_row_blk + col / block_size]).cast::<f32>()
    };
    acc = acc + pix_m * (elem_q * scale);
    store(out[idx], acc.cast::<OUT>());
}
```

**Test gate:** same kernel names must appear in the inventory (suffix template must
produce the same base names as the old macro invocations).

### Phase 3 — Collapse fixed-format kernels and merge by dimensionality

The fixed-format kernels (mxfp4, nvfp4, fp4, mxfp8, fp8, nvfp8, int8, mxint8)
have unique names that don't fit the `{WT}_{ST}` auto-naming and aren't part of a
BITS family. They can still benefit from primitive extraction (Phase 1) which
reduces each from ~80 lines to ~30. After that they live alongside the variants
blocks in the merged dimensionality files.

Phase 3 and 4 run together — merge each dimensionality's files and collapse format
families in the same pass:

| Source files | Target | Lines before | Lines after |
|---|---|---|---|
| conv2d + conv2d_block_scaled + conv2d_mma + conv2d_mma_block_scaled + depthwise* | `conv2d.rs` | 10,363 | ~500 |
| conv3d + conv3d_block_scaled + conv3d_mma + conv3d_mma_block_scaled | `conv3d.rs` | 8,438 | ~400 |
| audio_conv1d + audio_conv1d_block_scaled + fishspeech_conv1d_block_scaled + conv1d_dilated_transpose + conv1d_transpose_depthwise | `conv1d.rs` | 4,208 | ~300 |

**Test gate:** `cargo test -p metaltile-std` green; `cargo run -p metaltile-bench --bin dump_msl -- --filter conv` output diffs only in whitespace/comments.

### Phase 4 — Merge by dimensionality

After Phase 3, each old `_block_scaled.rs` file is small enough to inline into
its dimensionality file. Merge in order:

```
conv2d.rs   ← conv2d_block_scaled.rs + conv2d_mma.rs + conv2d_mma_block_scaled.rs
             ← depthwise_conv2d.rs + depthwise_conv2d_block_scaled.rs + depthwise_conv2d_nhwc.rs
conv3d.rs   ← conv3d_block_scaled.rs + conv3d_mma.rs + conv3d_mma_block_scaled.rs
conv1d.rs   ← audio_conv1d.rs + audio_conv1d_block_scaled.rs
             ← conv1d_transpose_depthwise.rs + conv1d_dilated_transpose.rs
             ← fishspeech_conv1d_block_scaled.rs
```

Internal section ordering within each merged file:
```
§ Dense / float
§ Block-scaled — group A+C (u32 weight, u8 scale)
§ Block-scaled — group B   (u32 weight, f32 scale)
§ Block-scaled — group D   (u8 weight, u8 scale)
§ Block-scaled — group E   (u8 weight, f32 scale)
§ Block-scaled — groups F+G (mixed-precision output)
§ MMA (dense)
§ MMA block-scaled
§ Depthwise (conv2d only)
§ Tests
§ Benches
```

Update `mod.rs` to remove the merged files.

**Test gate:** `cargo test -p metaltile-std` green; no inventory name regressions.

### Phase 5 — Move fused/special into `fused/`

| Current file | New location | Reason |
|---|---|---|
| `winograd_conv.rs` | `fused/winograd.rs` | Different algorithm (Winograd transform, not direct-conv) |
| `conv1d_causal_step_silu_cast_many.rs` | `fused/causal_silu.rs` | Streaming decode + fused activation — special dispatch |

These files get no LOC reduction in this phase; the move is purely structural.
Update `mod.rs` and `fused/mod.rs`.

---

## LOC projection

| File | Current sources | Current LOC | After |
|---|---|---|---|
| `primitives.rs` | (new) | 0 | ~100 |
| `conv1d.rs` | audio_conv1d + audio_conv1d_block_scaled + conv1d_transpose_depthwise + conv1d_dilated_transpose + fishspeech_conv1d_block_scaled | 219 + 1,640 + 177 + 413 + 1,759 = **4,208** | ~300 |
| `conv2d.rs` | conv2d + conv2d_block_scaled + conv2d_mma + conv2d_mma_block_scaled + depthwise_conv2d + depthwise_conv2d_block_scaled + depthwise_conv2d_nhwc | 646 + 2,210 + 467 + 4,700 + 262 + 1,811 + 267 = **10,363** | ~500 |
| `conv3d.rs` | conv3d + conv3d_block_scaled + conv3d_mma + conv3d_mma_block_scaled | 626 + 2,608 + 462 + 4,742 = **8,438** | ~400 |
| `fused/winograd.rs` | winograd_conv | 887 | ~200 |
| `fused/causal_silu.rs` | conv1d_causal_step_silu_cast_many | 263 | ~100 |
| `mod.rs` | mod.rs | 27 | ~25 |
| **Total** | | **24,213** | **~1,625** |

**~93% LOC reduction** with no change in generated MSL.

---

## DSL capabilities (all unlocked)

All three needed DSL features are fully implemented in `variants.rs`:

| Feature | Status | How used here |
|---|---|---|
| `#[kernel(variants(...))]` with compile-time `if` | ✅ | `FMT`/`BITS` integer discriminants |
| Cross-kernel calling (`let x = mt_decode_e2m1(nib)`) | ✅ | Decode primitives in `primitives.rs` |
| Type params (`WT = [u32, u8]`) | ✅ | `Tensor<WT>` / `Tensor<ST>` for weight/scale types |
| Float literal params (`SCALE = [0.5f32, 1.0f32]`) | ✅ | Not needed here (derive from integer `BITS`) |

**One remaining rule: compile-time `if` conditions are integer-only.**
Type and float params cannot appear in `if` conditions — they fall through as
runtime. Use an integer `FMT` discriminant to gate decode branches:

```rust
// ✅ works — FMT is an integer param
let elem = if FMT == 0u32 { mt_decode_e2m1(nib) } else { ... };

// ❌ does NOT evaluate at substitution time — becomes a runtime if
let elem = if WT == u32 { ... } else { ... };
```

**`constexpr` idents in bodies are also runtime** — `if some_constexpr == 0u32`
passes through unchanged since the body parser hasn't run yet at substitution time.

---

## What this does NOT change

- Generated MSL output — identical post-consolidation (same IR, same passes)
- Kernel inventory names — suffix templates must reproduce existing names exactly
- `steel_conv/` — left unchanged; it uses the steel implicit-GEMM path, not direct-conv
- `metaltile-core`, `metaltile-codegen`, all other crates — zero changes
