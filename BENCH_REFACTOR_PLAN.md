# Bench Runner Refactor Plan — Universal Generic Runner

## Goal
`run_generic` is the **only runner**. `BenchDispatch` has only `Generic`. Every op is
expressed entirely through data+fn-pointers in `BenchSpec`/`ShapeSpec`. The `bench_kernel`
macro generates everything — no hand-written dispatch code in `run_spec.rs` ever again.

---

## Why Custom Runners Exist Today — Real Root Causes

Every custom runner comes down to one or more of these four blockers:

| Blocker | Affected ops |
|---|---|
| **B1** Multi-value packed buffer (rope strides `[d, h*d, 1]`, steel params struct) | rope, strided_copy, steel_gemm |
| **B2** Computed scalar (scale=`1/√d`, grid_x=`d/2npg`, base=`log2(10000)`) | rope, attention, sdpa_vector |
| **B3** Extra shape dimensions (rope has h, l, d, npg; ShapeSpec only has n, b) | rope, strided_copy, sdpa_vector, steel_gemm |
| **B4** Custom correctness logic (is-sorted?, bit-exact vs CPU, vs MLX with different dtype) | sort, random, fp_quantized, quantized_mat_vec |

None of these require a custom runner — they all require **fn-pointer hooks** in `ShapeSpec`.

---

## The Design: Fn-Pointer Hooks

### New ShapeSpec fields

```rust
pub struct ShapeSpec {
    // ── existing ────────────────────────────────────────────────────────────
    pub label: &'static str,
    pub n: usize,
    pub b: usize,
    pub check_n: usize,
    pub check_b: usize,
    pub mode: KernelMode,
    pub tpg: usize,
    pub grid: DispatchGrid,
    pub tensor_bufs: &'static [TensorBufSpec],
    pub scalar_bufs: &'static [ScalarBufSpec],
    pub cexprs: &'static [(&'static str, Dim)],
    pub out_elems: Dim,
    pub reads: usize,
    pub bytes_fn: fn(usize, usize, usize, usize, usize) -> usize,
    pub mlx_args: Option<&'static [MlxArg]>,
    pub mlx_grid: Option<DispatchGrid>,
    pub mlx_tpg: usize,

    // ── NEW: extra shape dimensions ──────────────────────────────────────────
    /// Up to 8 class-specific extra shape values (e.g. rope's l, n_per_group;
    /// steel_gemm's bm, bn, K). Access by stable index defined per class.
    pub extra: [usize; 8],

    // ── NEW: fn-pointer overrides (None = use standard logic) ────────────────

    /// Custom MT buffer builder.
    /// Signature: fn(runner, shape, is_bench) → Vec<GpuBuffer>
    /// is_bench=false  →  use check_n/check_b sizes
    /// is_bench=true   →  use n/b sizes
    /// When Some, replaces the tensor_bufs + scalar_bufs path entirely.
    pub mt_bufs_fn: Option<fn(&GpuRunner, &ShapeSpec, bool, DType) -> Vec<GpuBuffer>>,

    /// Index of the output buffer in the mt_bufs_fn result (default: auto from params).
    pub mt_out_idx: Option<usize>,

    /// Custom MT grid. fn(shape, is_bench, tpg) → [usize; 3]
    pub mt_grid_fn: Option<fn(&ShapeSpec, bool, usize) -> [usize; 3]>,

    /// Custom MLX buffer builder.  fn(runner, shape, is_bench) → Vec<GpuBuffer>
    /// When Some, replaces the mlx_args path entirely.
    pub mlx_bufs_fn: Option<fn(&GpuRunner, &ShapeSpec, bool, DType) -> Vec<GpuBuffer>>,

    /// Index of the output buffer in the mlx_bufs_fn result.
    pub mlx_out_idx: usize,

    /// Custom MLX grid. fn(shape, is_bench, tpg) → [usize; 3]
    pub mlx_grid_fn: Option<fn(&ShapeSpec, bool, usize) -> [usize; 3]>,

    /// n elements to read back from the output buffer.
    /// When Some, overrides the standard out_elems.resolve() calculation.
    pub out_n_fn: Option<fn(&ShapeSpec, bool) -> usize>,
}
```

### New BenchSpec fields

```rust
pub struct BenchSpec {
    // ── existing ────────────────────────────────────────────────────────────
    pub op: &'static str,
    pub subop: &'static str,
    pub kernel_name: &'static str,
    pub kernel_ir: fn(DType) -> Kernel,
    pub dtypes: &'static [DType],
    pub tol: f32,
    pub mlx_src: Option<&'static str>,
    pub mlx_pattern: Option<&'static str>,
    pub shapes: &'static [ShapeSpec],
    pub dispatch: BenchDispatch,
    pub kernel_mode: Option<KernelMode>,

    // ── NEW ──────────────────────────────────────────────────────────────────

    /// Custom MLX compile fn. fn(runner, src, dt) → Option<CompiledKernel>
    /// When Some, replaces the standard compile(src, mlx_pattern) call.
    /// Use for bool-constants (rope, attention) and dynamic kernel names.
    pub mlx_compile_fn: Option<fn(&GpuRunner, &str, DType) -> Option<CompiledKernel>>,

    /// Custom correctness check. fn(ref_vals, mt_vals, tol) → EquivResult
    /// When None, uses check_equiv (element-wise abs error + cosine sim).
    /// Use for sort ("is sorted?"), random (bit-exact vs CPU), etc.
    pub check_fn: Option<fn(&[f32], &[f32], f32) -> EquivResult>,
}
```

### BenchDispatch collapses

```rust
pub enum BenchDispatch {
    Generic,  // the only variant that remains
}
```

All current variants (`Sort`, `Scan`, `ArgReduce`, `Rope`, `Attention`, `SdpaVector`,
`StridedCopy`, `Random`, `FpQuantized`, `QuantizedMatVec`, `AffineDequantize`,
`AffineQuantize`, `SteelGemm`) are **deleted**. Their logic moves into fn-pointer
implementations generated by the macro.

---

## How `run_generic` Changes

One pass for check, one for bench, calling fn pointers when present:

```rust
pub fn run(spec: &BenchSpec, runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    run_generic(spec, runner, dt, &OpBench::new(spec.op, "GB/s"))
}

fn run_generic(spec, runner, dt, bench) -> Vec<OpResult> {
    // compile MT — unchanged
    let mk = ...;

    // compile MLX — now uses mlx_compile_fn when set
    let mlx_k = if let Some(cfn) = spec.mlx_compile_fn {
        spec.mlx_src.and_then(|src| cfn(runner, src, dt))
    } else {
        compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, tn)
    };

    for shape in spec.shapes {
        // --- check pass ---
        let mt_bufs = build_mt_bufs(runner, shape, /*is_bench=*/false, dt);
        let mt_grid = resolve_mt_grid(shape, /*is_bench=*/false);
        let n_out   = resolve_out_n(shape, /*is_bench=*/false);
        let mt_out_idx = resolve_mt_out_idx(spec, shape);
        let mt_vals = run_and_read(runner, &mk, &mt_bufs, mt_out_idx, mt_grid, shape.tpg, n_out, dt);

        let equiv = if let Some(mlx_k) = &mlx_k {
            let mlx_bufs = build_mlx_bufs(runner, shape, /*is_bench=*/false, dt);
            let mlx_grid = resolve_mlx_grid(shape, /*is_bench=*/false);
            let mlx_vals = run_and_read(runner, mlx_k, &mlx_bufs, shape.mlx_out_idx, mlx_grid, ..., n_out, dt);
            if let Some(cfn) = spec.check_fn { cfn(&mlx_vals, &mt_vals, spec.tol) }
            else { check_equiv(&mlx_vals, &mt_vals, spec.tol) }
        } else {
            EquivResult { passed: true, n_checked: 0, .. }
        };

        // --- bench pass ---
        let mt_bufs_p  = build_mt_bufs(runner, shape, /*is_bench=*/true, dt);
        let mt_grid_p  = resolve_mt_grid(shape, /*is_bench=*/true);
        let n_out_p    = resolve_out_n(shape, /*is_bench=*/true);
        let bytes      = resolve_bytes(spec, shape, dt, n_out_p);
        let mt_perf    = bench_gbps(runner, &mk, &refs(&mt_bufs_p), mt_grid_p, [shape.tpg,1,1], bytes);
        let ref_perf   = mlx_k.as_ref().and_then(|k| bench_mlx(runner, k, shape, is_bench=true, bytes));

        results.push(bench.result_sub_timed(...));
    }
}

// These three helpers check for fn-ptr, fall back to data path:
fn build_mt_bufs(runner, shape, is_bench, dt) -> Vec<GpuBuffer> {
    if let Some(f) = shape.mt_bufs_fn { f(runner, shape, is_bench, dt) }
    else { build_from_tensor_scalar_bufs(runner, shape, is_bench, dt) }
}
fn build_mlx_bufs(runner, shape, is_bench, dt) -> Vec<GpuBuffer> {
    if let Some(f) = shape.mlx_bufs_fn { f(runner, shape, is_bench, dt) }
    else { build_from_mlx_args(runner, shape, is_bench, dt) }
}
fn resolve_mt_grid(shape, is_bench) -> [usize; 3] {
    if let Some(f) = shape.mt_grid_fn { f(shape, is_bench, shape.tpg) }
    else {
        let (n, b) = if is_bench { (shape.n, shape.b) } else { (shape.check_n, shape.check_b) };
        shape.grid.eval(n, b, shape.tpg)
    }
}
```

`run_spec.rs` becomes ~150 lines. No custom dispatch arms.

---

## How the Macro Generates Fn Pointers

The key insight: fn pointers (unlike closures) cannot capture variables, but they **can**
read constants baked in at code-generation time. The macro emits named functions that
embed class-specific constants as `const` items, making them valid `fn` pointers.

### Example — rope

Macro input (after defaults):
```rust
#[bench_kernel(op="rope", class=Rope, tol=0.01, mlx="rope_{tn}", metal_file="rope.metal")]
```

Macro emits (alongside the kernel):
```rust
// Rope extra dims layout:
//   extra[0] = l (sequence length)
//   extra[1] = n_per_group

fn _bench_mt_rope_mt_bufs(
    runner: &::metaltile_std::runner::GpuRunner,
    shape: &::metaltile_std::spec::ShapeSpec,
    is_bench: bool,
    dt: ::metaltile_std::bench_types::DType,
) -> Vec<::metaltile_std::runner::GpuBuffer> {
    let d   = shape.n;              // n = head_dim d
    let h   = shape.b;              // b = n_heads h
    let l   = if is_bench { shape.extra[0] } else { 4 }; // l_check=4
    let npg = shape.extra[1];
    let n_elems = h * l * d;
    let in_f16: Vec<u16> = (0..n_elems).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
    let inp = runner.buffer_f16(&in_f16);
    let out = runner.buffer_zeros(n_elems * 2);
    let h_stride     = runner.buffer_u32(d as u32);
    let seq_stride   = runner.buffer_u32((h * d) as u32);
    let grid_x       = runner.buffer_u32((d / (2 * npg)) as u32);
    let base         = runner.buffer_f32_scalar((10000f32).log2());
    vec![inp, out, h_stride, seq_stride, grid_x, base]
}

fn _bench_mt_rope_mlx_bufs(
    runner: &::metaltile_std::runner::GpuRunner,
    shape: &::metaltile_std::spec::ShapeSpec,
    is_bench: bool,
    dt: ::metaltile_std::bench_types::DType,
) -> Vec<::metaltile_std::runner::GpuBuffer> {
    let d   = shape.n;
    let h   = shape.b;
    let l   = if is_bench { shape.extra[0] } else { 4 };
    let npg = shape.extra[1];
    let n_elems = h * l * d;
    let in_f16: Vec<u16> = (0..n_elems).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
    let inp   = runner.buffer_f16(&in_f16);
    let out   = runner.buffer_zeros(n_elems * 2);
    let strides: Vec<u8> = [d as i64, (h * d) as i64, 1i64]
        .iter().flat_map(|v| v.to_le_bytes()).collect();
    let strides_buf      = runner.buffer_bytes(&strides);
    let offset_arr       = runner.buffer_i32(0);
    let scale_buf        = runner.buffer_f32_scalar(1.0);
    let offset_stride    = runner.buffer_i64(1);
    let n_head_buf       = runner.buffer_i32(h as i32);
    let dummy            = runner.buffer_zeros(4);
    let base             = runner.buffer_f32_scalar((10000f32).log2());
    vec![inp, out, offset_arr, scale_buf, strides_buf, strides_buf.clone(),
         offset_stride, n_head_buf, dummy, dummy.clone(), base]
}
// mlx_out_idx = 1  (out is index 1 in mlx_bufs)

fn _bench_mt_rope_mlx_compile(
    runner: &::metaltile_std::runner::GpuRunner,
    src: &str,
    dt: ::metaltile_std::bench_types::DType,
) -> Option<::metaltile_std::runner::CompiledKernel> {
    use ::metaltile_std::bench_types::DType;
    let name = match dt {
        DType::F16 => "rope_float16",
        DType::F32 => "rope_float32",
        DType::BF16 => "rope_bfloat16",
        _ => return None,
    };
    runner.compile_with_bool_constants(src, name, &[(1,true),(2,false),(3,false)]).ok()
}

fn _bench_mt_rope_mt_grid(shape: &ShapeSpec, is_bench: bool, tpg: usize) -> [usize; 3] {
    let d   = shape.n;
    let h   = shape.b;
    let l   = if is_bench { shape.extra[0] } else { 4 };
    let npg = shape.extra[1];
    [d / (2 * npg), l, h / npg]
}

fn _bench_mt_rope_mlx_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let d   = shape.n;
    let h   = shape.b;
    let l   = if is_bench { shape.extra[0] } else { 4 };
    let npg = shape.extra[1];
    [d / (2 * npg), l, h / npg]  // same grid shape for MLX rope
}

fn _bench_mt_rope_out_n(shape: &ShapeSpec, is_bench: bool) -> usize {
    let h = shape.b;
    let d = shape.n;
    let l = if is_bench { shape.extra[0] } else { 4 };
    h * l * d
}

inventory::submit! { ::metaltile_std::spec::BenchSpec {
    op: "rope",
    subop: "rope",
    kernel_name: "mt_rope",
    kernel_ir: mt_rope::kernel_ir_for,
    dtypes: ::metaltile_std::bench_types::FLOAT_DTYPES,
    tol: 0.01,
    mlx_src: Some(include_str!(concat!(env!("OUT_DIR"), "/metal/rope.metal"))),
    mlx_pattern: Some("rope_{tn}"),
    dispatch: ::metaltile_std::spec::BenchDispatch::Generic,
    kernel_mode: Some(::metaltile_core::ir::KernelMode::Grid3D),
    mlx_compile_fn: Some(_bench_mt_rope_mlx_compile),
    check_fn: None,
    shapes: &[::metaltile_std::spec::ShapeSpec {
        label: "B1H32L512D128 f16",
        n: 128,      // d
        b: 32,       // h
        check_n: 128,
        check_b: 32,
        extra: [512, 4, 0, 0, 0, 0, 0, 0],  // [l, n_per_group, ...]
        mode: ::metaltile_core::ir::KernelMode::Grid3D,
        tpg: 1,
        grid: ::metaltile_std::spec::DispatchGrid::Single,  // overridden by mt_grid_fn
        tensor_bufs: &[],        // overridden by mt_bufs_fn
        scalar_bufs: &[],
        cexprs: &[],
        out_elems: ::metaltile_std::spec::Dim::One,  // overridden by out_n_fn
        reads: 2,
        bytes_fn: rope_bytes,    // custom bytes calc
        mlx_args: None,          // overridden by mlx_bufs_fn
        mlx_grid: None,
        mlx_tpg: 1,
        mt_bufs_fn:  Some(_bench_mt_rope_mt_bufs),
        mt_out_idx:  Some(1),
        mt_grid_fn:  Some(_bench_mt_rope_mt_grid),
        mlx_bufs_fn: Some(_bench_mt_rope_mlx_bufs),
        mlx_out_idx: 1,
        mlx_grid_fn: Some(_bench_mt_rope_mlx_grid),
        out_n_fn:    Some(_bench_mt_rope_out_n),
    }],
}}
```

The op file `rope.rs` stays exactly as it is today. Zero hand-written runner code.

---

## Fn-Pointer Mapping Per Class

| Class | mt_bufs_fn | mlx_bufs_fn | mlx_compile_fn | mt_grid_fn | check_fn |
|---|---|---|---|---|---|
| Unary / Binary / AllReduce / RowReduce / Arange / BinaryTwo / Select / RowNorm / MatVec / MatVecMasked | None (tensor_bufs path) | None (mlx_args path) | None | None | None |
| Sort | mt_bufs_fn (Descending init) | mlx_bufs_fn (I32 args) | None | None | check_fn ("is sorted?") |
| Scan | None (tensor_bufs) | mlx_bufs_fn (U64 N) | None | None | None |
| ArgReduce | None | mlx_bufs_fn (ndim/stride args) | None | mlx_grid_fn ([tpg,1,1]) | None |
| Random | mt_bufs_fn (u32 output) | None (no MLX ref) | None | None | check_fn (bit-exact CPU) |
| FpQuantized | None | mlx_bufs_fn (matching dispatch) | None | mlx_grid_fn | check_fn (CPU FP4) |
| MatVecMasked (quantized) | None | mlx_bufs_fn (f16 scales/biases) | None | None | None |
| Rope | mt_bufs_fn | mlx_bufs_fn (packed strides) | mlx_compile_fn (bool consts) | mt_grid_fn | None |
| Attention | mt_bufs_fn | mlx_bufs_fn | mlx_compile_fn (bool consts) | None | None |
| SdpaVector | mt_bufs_fn | mlx_bufs_fn | mlx_compile_fn (bool consts) | None | None |
| StridedCopy | mt_bufs_fn (shape+strides) | mlx_bufs_fn (i64 strides) | None | None | None |
| AffineDequantize | None | mlx_bufs_fn | None | mlx_grid_fn (pack-factor grid) | None |
| AffineQuantize | None | mlx_bufs_fn | None | mlx_grid_fn ([n_groups,1,1]) | None |
| SteelGemm | None | mlx_bufs_fn (params struct) | mlx_compile_fn (bool consts) | None | None |

---

## extra_dims Layout Per Class

| Class | extra[0] | extra[1] | extra[2] | extra[3] |
|---|---|---|---|---|
| Rope | l (seq len) | n_per_group | — | — |
| Attention | head_dim (128) | — | — | — |
| SdpaVector | head_dim | gqa_factor | n_kv_heads | — |
| StridedCopy | pad | — | — | — |
| AffineDequantize | bits | group_size | n_groups | batch |
| AffineQuantize | bits | group_size | n_groups | batch |
| QuantizedMatVec (class K) | group_size | — | — | — |
| SteelGemm | bm | bn | K | check_K |
| Sort | — | — | — | — |
| Rope | l | n_per_group | — | — |

---

## Macro Changes

### Simplified `BenchArgs`

After this refactor, `BenchArgs` no longer needs separate dispatch-variant fields.
All class-specific params feed into `extra_dims` and `ShapeSpec.n/.b`:

```rust
pub struct BenchArgs {
    // always required
    pub op: LitStr,
    pub tol: LitFloat,

    // optional with smart defaults
    pub subop:       Option<LitStr>,     // default: same as op
    pub class:       Option<ClassKind>,  // default: auto (Generic)
    pub dtypes:      Option<Expr>,       // default: FLOAT_DTYPES
    pub mlx:         Option<LitStr>,
    pub metal_file:  Option<LitStr>,

    // shape — all optional, defaults per class
    pub n:    Option<LitInt>,
    pub b:    Option<LitInt>,
    pub tpg:  Option<LitInt>,

    // class-specific extra dims — map to extra[0..7]
    // (reuses existing field names; macro maps them to extra[] by class)
    pub l:           Option<LitInt>,  // rope: seq len
    pub d:           Option<LitInt>,  // rope: head_dim
    pub n_per_group: Option<LitInt>,  // rope, sdpa
    pub n_kv:        Option<LitInt>,
    pub n_heads:     Option<LitInt>,
    pub gqa_factor:  Option<LitInt>,
    pub pad:         Option<LitInt>,  // strided_copy
    pub bits:        Option<LitInt>,  // affine quant
    pub group_size:  Option<LitInt>,
    pub n_groups:    Option<LitInt>,
    pub batch:       Option<LitInt>,
    pub bm:          Option<LitInt>,  // steel_gemm
    pub bn:          Option<LitInt>,
    // ... (same set as today, but ALL optional)

    // existing simple-class fields
    pub input:       InputKind,
    pub reads:       Option<LitInt>,
    pub out_elements: Option<LitInt>,
    // ...
}
```

### What the macro generates per class

- **Simple classes** (Unary, Binary, etc.): same as today — pure data, no fn ptrs.
- **Complex classes** (Rope, Attention, Sort, etc.): generates named fn helpers
  (`_bench_{fn_name}_mt_bufs`, `_bench_{fn_name}_mlx_bufs`, etc.) with class
  constants baked in from macro args, then references them in `ShapeSpec`.
- **`ClassKind::Custom`** (new, for one-off ops): developer writes the fn helpers
  manually and passes their names: `mt_bufs=my_fn, mlx_bufs=my_fn, ...`. The macro
  just wires them into the ShapeSpec.

---

## Required params after refactor

```
Always required:    op, tol
Always optional:    subop (default=op), class (default=Generic), dtypes,
                    mlx, metal_file, n, b, tpg, ...all shape params
```

Minimum working bench annotation for a simple new op:
```rust
#[bench_kernel(op="my_op", tol=1e-4, mlx="my_mlx_{tn}", metal_file="my.metal")]
#[kernel]
pub fn mt_my_op<T>(inp: Tensor<T>, out: Tensor<T>) { ... }
```

For rope (with all defaults):
```rust
#[bench_kernel(op="rope", class=Rope, tol=0.01, mlx="rope_{tn}", metal_file="rope.metal")]
```

---

## Files Changed

| File | Change |
|---|---|
| `spec.rs` | Add `extra: [usize; 8]` + 7 fn-ptr fields to `ShapeSpec`; add 2 fn-ptr fields to `BenchSpec`; delete all `BenchDispatch` variants except `Generic` |
| `run_spec.rs` | Replace all `run_*` functions with `run_generic` using the fn-ptr hooks; ~150 lines total |
| `macros/src/lib.rs` | `BenchArgs` all fields `Option`; `generate_submit` emits named fn helpers per complex class; delete `BenchDispatch` variant emission except Generic |
| `mlx/rope.rs` | Drop all shape params from `#[bench_kernel]`, add `class=Rope` with defaults |
| `mlx/strided.rs` | Same |
| `mlx/sort.rs` | Same |
| `mlx/scan.rs` | Same |
| `mlx/arg_reduce.rs` | Same |
| `mlx/random.rs` | Same |
| `mlx/fp_quantized.rs` | Same |
| `mlx/quantized.rs` | Same |
| `mlx/scaled_dot_product_attention.rs` | Same |
| `mlx/sdpa_vector.rs` | Same |
| `mlx/steel/gemm/steel_gemm_fused.rs` | Same |
| `mlx/fp_quantized_nax.rs`, `mlx/quantized_nax.rs`, etc. | Same pattern |

---

## Migration Order

```
Phase 0 — Add the scaffolding (no behaviour change)
  0-a  Add extra, mt_bufs_fn, mt_out_idx, mt_grid_fn,
       mlx_bufs_fn, mlx_out_idx, mlx_grid_fn, out_n_fn
       to ShapeSpec (all None / default values).
  0-b  Add mlx_compile_fn, check_fn to BenchSpec (both None).
  0-c  Update run_generic to call fn ptrs when Some, existing path when None.
       Suite stays green — no fn ptrs set yet.

Phase 1 — Migrate ops one by one (each leaves suite green)
  1-a  scan          → mlx_bufs_fn (U64 N), remove BenchDispatch::Scan
  1-b  sort          → mt_bufs_fn (Descending), mlx_bufs_fn (I32 args),
                        check_fn ("is sorted?"), remove BenchDispatch::Sort
  1-c  arg_reduce    → mlx_bufs_fn (ndim/stride args), mlx_grid_fn ([tpg,1,1]),
                        remove BenchDispatch::ArgReduce
  1-d  random        → check_fn (bit-exact CPU), remove BenchDispatch::Random
  1-e  fp_quantized  → mlx_bufs_fn + mlx_grid_fn + check_fn (CPU FP4),
                        remove BenchDispatch::FpQuantized
  1-f  strided_copy  → mt_bufs_fn + mlx_bufs_fn, remove BenchDispatch::StridedCopy
  1-g  rope          → mt_bufs_fn + mlx_bufs_fn + mlx_compile_fn + mt_grid_fn,
                        remove BenchDispatch::Rope
  1-h  attention     → mt_bufs_fn + mlx_bufs_fn + mlx_compile_fn,
                        remove BenchDispatch::Attention
  1-i  sdpa_vector   → same pattern, remove BenchDispatch::SdpaVector
  1-j  quantized_mat_vec → mlx_bufs_fn (f16 scales), remove BenchDispatch::QuantizedMatVec
  1-k  affine_dequantize → mlx_bufs_fn + mlx_grid_fn, remove BenchDispatch::AffineDequantize
  1-l  affine_quantize   → mlx_bufs_fn + mlx_grid_fn, remove BenchDispatch::AffineQuantize
  1-m  steel_gemm    → mlx_bufs_fn (params struct) + mlx_compile_fn,
                        remove BenchDispatch::SteelGemm

Phase 2 — Macro DX (after all dispatch variants gone)
  2-a  Make ALL BenchArgs fields Option with per-class defaults.
  2-b  Macro emits named fn helpers for complex classes instead of
       BenchDispatch variant tokens.
  2-c  Update op files: drop all now-redundant params.
  2-d  Delete BenchDispatch entirely — spec.rs uses Generic only.

Phase 3 — Cleanup
  3-a  Delete ClassKind variants for all complex classes that now use fn ptrs.
       Only the simple/Generic ones remain: Unary, Binary, AllReduce, RowReduce,
       Arange, BinaryTwo, Select, RowNorm, MatVec, MatVecMasked + new Custom.
  3-b  run_spec.rs: confirm only run_generic remains. Delete file-level dead code.
  3-c  cargo test + tile bench — suite green.
```

---

## End State

```
run_spec.rs     ~150 lines, exports pub fn run() → run_generic()
spec.rs         ShapeSpec with extra[8] + 7 fn-ptr fields, BenchSpec + 2 fn-ptr fields,
                BenchDispatch::Generic (single variant)
macros/lib.rs   BenchArgs all-optional, generate_submit emits fn helpers per class
op files        #[bench_kernel(op="X", class=Y, tol=Z, mlx="...", metal_file="...")]
                — 3-5 params for any op, class-specific extras only when non-default
```

Zero custom runners. Every op goes through `run_generic`. New ops require zero changes
to `run_spec.rs` — ever.
