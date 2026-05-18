//! GPU runner: single generic runner with class-level fn-pointer hooks.
//!
//! All ops go through `run_generic`. Complex ops (rope, sort, attention, …)
//! provide fn-pointer implementations in this file; the `#[bench_kernel]` macro
//! wires them up via `ShapeSpec` / `BenchSpec` fields.
//!
//! Adding a new kernel never requires changes to this file — the macro provides
//! everything the generic runner needs.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{dtype::DType, ir::KernelMode, ir::Kernel};

use crate::{
    bench_types::{
        DtypeCtx,
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        check_equiv,
        check_equiv_with,
    },
    runner::{
        GpuBuffer,
        GpuRunner,
        bench_gbps,
        bench_gbps_only,
        buffer_typed,
        read_typed,
        run_typed_once,
        zeros_typed,
    },
    spec::{BenchDispatch, BenchSpec, MlxArg, ScalarBufSpec, ShapeSpec},
};

// ── Public entry point ───────────────────────────────────────────────────

pub fn run(spec: &BenchSpec, runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let bench = OpBench::new(spec.op, "GB/s");
    match &spec.dispatch {
        BenchDispatch::Generic => run_generic(spec, runner, dt, &bench),
        BenchDispatch::Scan { shapes, tpg } => run_legacy_scan(spec, runner, dt, &bench, shapes, *tpg),
        BenchDispatch::Attention { shapes, tpg } => run_legacy_attention(spec, runner, dt, &bench, shapes, *tpg),
        BenchDispatch::QuantizedMatVec { shapes, group_size, tpg } => run_legacy_quantized_mat_vec(spec, runner, dt, &bench, shapes, *group_size, *tpg),
        BenchDispatch::SdpaVector2Pass {
            head_dim, n_kv, n_q_heads, gqa_factor, batch, blocks,
            pass2_kernel_name, pass2_kernel_ir,
        } => run_sdpa_vector_2pass(
            spec, runner, dt, &bench,
            *head_dim, *n_kv, *n_q_heads, *gqa_factor, *batch, *blocks,
            pass2_kernel_name, *pass2_kernel_ir,
        ),
    }
}

// ── MSL generation ────────────────────────────────────────────────────────

fn msl_elementwise(spec: &BenchSpec, dt: DType) -> Option<String> {
    MslGenerator::default().generate(&(spec.kernel_ir)(dt)).ok()
}
fn msl_reduction(spec: &BenchSpec, dt: DType) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).ok()
}
fn msl_grid3d(spec: &BenchSpec, dt: DType) -> Option<String> {
    let mut k = (spec.kernel_ir)(dt);
    k.mode = KernelMode::Grid3D;
    MslGenerator::default().generate(&k).ok()
}
fn msl_for_mode(spec: &BenchSpec, dt: DType, mode: KernelMode) -> Option<String> {
    match mode {
        KernelMode::Elementwise => msl_elementwise(spec, dt),
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D =>
            msl_reduction(spec, dt),
        KernelMode::Grid3D => msl_grid3d(spec, dt),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn mlx_name(pat: &str, tn: &str) -> String { pat.replace("{tn}", tn) }
fn compile_mt(runner: &GpuRunner, msl: &str, name: &str) -> Option<crate::runner::CompiledKernel> {
    match runner.compile(msl, name) {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("[error] compile '{}': {}", name, e);
            None
        },
    }
}
fn compile_mlx(
    runner: &GpuRunner,
    src: Option<&str>,
    pat: Option<&str>,
    tn: &str,
) -> Option<crate::runner::CompiledKernel> {
    let src = src?;
    let pat = pat?;
    runner.compile(src, &mlx_name(pat, tn)).ok()
}

// ── Generic runner (the ONLY runner) ─────────────────────────────────────

fn run_generic(spec: &BenchSpec, runner: &GpuRunner, dt: DType, bench: &OpBench) -> Vec<OpResult> {
    let mut compiled: std::collections::HashMap<u8, crate::runner::CompiledKernel> =
        std::collections::HashMap::new();
    let mode_key = |m: KernelMode| match m {
        KernelMode::Elementwise => 0u8,
        KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D => 1,
        KernelMode::Grid3D => 2,
    };

    let mut results = Vec::new();
    let mlx_compiled: Option<crate::runner::CompiledKernel> = if let Some(cfn) =
        spec.mlx_compile_fn
    {
        spec.mlx_src.and_then(|src| cfn(runner, src, dt))
    } else {
        let ctx0 = DtypeCtx::reduce(dt);
        compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, ctx0.tn)
    };

    for shape in spec.shapes {
        let ctx = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D =>
                DtypeCtx::reduce(dt),
            _ => DtypeCtx::elementwise(dt),
        };
        let mk = match compiled.entry(mode_key(shape.mode)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let msl = match msl_for_mode(spec, dt, shape.mode) {
                    Some(s) => s,
                    None => continue,
                };
                match compile_mt(runner, &msl, spec.kernel_name) {
                    Some(k) => e.insert(k),
                    None => continue,
                }
            },
        };

        let kernel = (spec.kernel_ir)(dt);
        let params: Vec<_> = kernel.params.iter().collect();
        let check_n = shape.check_n;
        let check_b = match shape.mode {
            KernelMode::Reduction | KernelMode::Tile2D | KernelMode::SimdGroup2D => 1,
            _ => shape.check_b,
        };
        let primary_out_idx = params.iter().position(|p| p.is_output);

        // ── Check pass: build buffers, run MT, compare against MLX ref ──
        let check_bufs: Vec<GpuBuffer> = if let Some(f) = shape.mt_bufs_fn {
            f(runner, shape, false, dt)
        } else {
            let mut bufs = Vec::new();
            for buf_spec in shape.tensor_bufs {
                let count = buf_spec.count.resolve(check_n, check_b);
                let init_data = buf_spec.init.generate(count);
                let param_dt = buf_spec.dtype_override.unwrap_or(dt);
                bufs.push(buffer_typed(runner, &init_data, param_dt));
            }
            for &sb in shape.scalar_bufs {
                bufs.push(scalar_buf(runner, sb, check_n, check_b));
            }
            bufs
        };

        let out_idx =
            shape.mt_out_idx.unwrap_or_else(|| primary_out_idx.unwrap_or(0));
        let out_count_check = shape
            .out_n_fn
            .map(|f| f(shape, false))
            .unwrap_or_else(|| shape.out_elems.resolve(check_n, check_b).max(1));
        let check_grid = shape
            .mt_grid_fn
            .map(|f| f(shape, false, shape.tpg))
            .unwrap_or_else(|| shape.grid.eval(check_n, check_b, shape.tpg));
        let check_refs: Vec<&GpuBuffer> = check_bufs.iter().collect();
        let mt_vals = run_typed_once(
            runner,
            mk,
            &check_refs,
            &check_bufs[out_idx],
            out_count_check,
            check_grid,
            [shape.tpg, 1, 1],
            dt,
        );

        // Correctness: compare MT against MLX reference on check shapes.
        let has_mlx_bufs = shape.mlx_bufs_fn.is_some() || shape.mlx_args.is_some();
        let equiv = if has_mlx_bufs {
            if let Some(rk) = &mlx_compiled {
                let mlx_tpg_check =
                    if shape.mlx_tpg > 0 { shape.mlx_tpg } else { shape.tpg };
                let (mlx_check_bufs, mlx_out_idx_c) = if let Some(f) = shape.mlx_bufs_fn {
                    (f(runner, shape, false, dt), shape.mlx_out_idx)
                } else {
                    let mlx_args = shape.mlx_args.unwrap();
                    let bufs = mlx_args
                        .iter()
                        .map(|arg| mlx_buf(runner, arg, shape, check_n, check_b, dt))
                        .collect();
                    let idx = mlx_args
                        .iter()
                        .position(|arg| matches!(arg, MlxArg::FreshOut(_)))
                        .unwrap_or(1);
                    (bufs, idx)
                };
                let mlx_grid_check = shape
                    .mlx_grid_fn
                    .map(|f| f(shape, false, mlx_tpg_check))
                    .unwrap_or_else(|| {
                        shape.mlx_grid.unwrap_or(shape.grid).eval(
                            check_n,
                            check_b,
                            mlx_tpg_check,
                        )
                    });
                let mlx_out_buf = &mlx_check_bufs[mlx_out_idx_c];
                let mlx_refs: Vec<&GpuBuffer> = mlx_check_bufs.iter().collect();
                let mlx_vals = run_typed_once(
                    runner,
                    rk,
                    &mlx_refs,
                    mlx_out_buf,
                    out_count_check,
                    mlx_grid_check,
                    [mlx_tpg_check, 1, 1],
                    dt,
                );
                spec.check_fn
                    .map(|f| f(&mlx_vals, &mt_vals, spec.tol, check_n))
                    .unwrap_or_else(|| check_equiv(&mlx_vals, &mt_vals, spec.tol))
            } else {
                EquivResult { n_checked: 0, max_abs_err: 0.0, cosine_sim: 0.0, passed: true }
            }
        } else {
            EquivResult { n_checked: 0, max_abs_err: 0.0, cosine_sim: 0.0, passed: true }
        };

        // ── Bench pass: build perf buffers, bench MT, optional MLX ref ──
        let n = shape.n;
        let b = shape.b;
        let perf_bufs: Vec<GpuBuffer> = if let Some(f) = shape.mt_bufs_fn {
            f(runner, shape, true, dt)
        } else {
            let mut bufs = Vec::new();
            for buf_spec in shape.tensor_bufs {
                let count = buf_spec.count.resolve(n, b);
                let init_data = buf_spec.init.generate(count);
                let param_dt = buf_spec.dtype_override.unwrap_or(dt);
                bufs.push(buffer_typed(runner, &init_data, param_dt));
            }
            for &sb in shape.scalar_bufs {
                bufs.push(scalar_buf(runner, sb, n, b));
            }
            bufs
        };

        let perf_grid = shape
            .mt_grid_fn
            .map(|f| f(shape, true, shape.tpg))
            .unwrap_or_else(|| shape.grid.eval(n, b, shape.tpg));
        let out_count_perf = shape
            .out_n_fn
            .map(|f| f(shape, true))
            .unwrap_or_else(|| shape.out_elems.resolve(n, b).max(1));
        let bytes = (shape.bytes_fn)(n, b, shape.reads, out_count_perf, ctx.eb) as f64;
        let perf_refs: Vec<&GpuBuffer> = perf_bufs.iter().collect();
        let (mt_perf_val, mt_stats) =
            match bench_gbps(runner, mk, &perf_refs, perf_grid, [shape.tpg, 1, 1], bytes) {
                Some((p, t)) => (Some(p), Some(t)),
                None => (None, None),
            };

        let has_mlx_perf = shape.mlx_bufs_fn.is_some() || shape.mlx_args.is_some();
        let (ref_perf_val, ref_stats) = if has_mlx_perf {
            let mlx_tpg = if shape.mlx_tpg > 0 { shape.mlx_tpg } else { shape.tpg };
            let mlx_grid = shape
                .mlx_grid_fn
                .map(|f| f(shape, true, mlx_tpg))
                .unwrap_or_else(|| shape.mlx_grid.unwrap_or(shape.grid).eval(n, b, mlx_tpg));
            mlx_compiled
                .as_ref()
                .map(|rk| {
                    let mlx_bufs: Vec<GpuBuffer> = if let Some(f) = shape.mlx_bufs_fn {
                        f(runner, shape, true, dt)
                    } else {
                        shape
                            .mlx_args
                            .unwrap()
                            .iter()
                            .map(|arg| mlx_buf(runner, arg, shape, n, b, dt))
                            .collect()
                    };
                    let mlx_refs: Vec<&GpuBuffer> = mlx_bufs.iter().collect();
                    match bench_gbps(runner, rk, &mlx_refs, mlx_grid, [mlx_tpg, 1, 1], bytes) {
                        Some((p, t)) => (Some(p), Some(t)),
                        None => (None, None),
                    }
                })
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        results.push(bench.result_sub_timed(
            Some(spec.subop),
            format!("{} {}", shape.label, ctx.label),
            ref_perf_val,
            mt_perf_val,
            Some(equiv),
            mt_stats,
            ref_stats,
        ));
    }
    results
}

fn scalar_buf(runner: &GpuRunner, sb: ScalarBufSpec, n: usize, b: usize) -> GpuBuffer {
    match sb {
        ScalarBufSpec::U32N => runner.buffer_u32(n as u32),
        ScalarBufSpec::U32B => runner.buffer_u32(b as u32),
        ScalarBufSpec::U64N => runner.buffer_u64(n as u64),
        ScalarBufSpec::U64B => runner.buffer_u64(b as u64),
        ScalarBufSpec::I64B => runner.buffer_i64(b as i64),
    }
}

fn mlx_buf(
    runner: &GpuRunner,
    arg: &MlxArg,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    dt: DType,
) -> GpuBuffer {
    match arg {
        MlxArg::TensorBuf(i) => {
            let spec = &shape.tensor_bufs[*i];
            let count = spec.count.resolve(n, b);
            let init_data = spec.init.generate(count);
            let param_dt = spec.dtype_override.unwrap_or(dt);
            buffer_typed(runner, &init_data, param_dt)
        },
        MlxArg::FreshOut(i) => {
            let spec = &shape.tensor_bufs[*i];
            let count = spec.count.resolve(n, b);
            let param_dt = spec.dtype_override.unwrap_or(dt);
            zeros_typed(runner, count, param_dt)
        },
        MlxArg::U32N => runner.buffer_u32(n as u32),
        MlxArg::U64N => runner.buffer_u64(n as u64),
        MlxArg::U64B => runner.buffer_u64(b as u64),
        MlxArg::I64B => runner.buffer_i64(b as i64),
        MlxArg::Zeros8 => runner.buffer_zeros(8),
        MlxArg::BoolAltN => runner
            .buffer_bytes(&(0..n).map(|i| if i % 2 == 0 { 1u8 } else { 0u8 }).collect::<Vec<_>>()),
        MlxArg::U32V(v) => runner.buffer_u32(*v),
    }
}

// ── f16 helpers ──────────────────────────────────────────────────────────

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 { sign }
    else if exp >= 31 { sign | 0x7c00 }
    else { sign | ((exp as u16) << 10) | mant as u16 }
}

fn f16_vec(vals: &[f32]) -> Vec<u16> {
    vals.iter().map(|&v| f32_to_f16(v)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// CLASS-LEVEL FN-POINTER FUNCTIONS
// Each function reads params from `shape.n`, `shape.b`, `shape.extra[]`.
// The macro wires them into `ShapeSpec` / `BenchSpec` based on `class=`.
// ═══════════════════════════════════════════════════════════════════════════

// ── Sort ──────────────────────────────────────────────────────────────────

pub fn sort_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let (b, n) = if is_bench { (shape.b, shape.n) } else { (4, shape.check_n) };
    let data: Vec<f32> = (0..b * n).rev().map(|i| i as f32).collect();
    let inp = buffer_typed(runner, &data, dt);
    let out = zeros_typed(runner, b * n, dt);
    let n_buf = runner.buffer_u32(n as u32);
    vec![inp, out, n_buf]
}

pub fn sort_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let n = if is_bench { shape.n } else { shape.check_n };
    let b = if is_bench { shape.b } else { 4 };
    let data: Vec<f32> = (0..b * n).rev().map(|i| i as f32).collect();
    let inp = buffer_typed(runner, &data, dt);
    let out = zeros_typed(runner, b * n, dt);
    let size = runner.buffer_i32(n as i32);
    let stride1_a = runner.buffer_i32(1);
    let stride1_b = runner.buffer_i32(1);
    let stride_n_a = runner.buffer_i32(n as i32);
    let stride_n_b = runner.buffer_i32(n as i32);
    vec![inp, out, size, stride1_a, stride1_b, stride_n_a, stride_n_b]
}

pub fn sort_check_fn(_ref_vals: &[f32], mt_vals: &[f32], tol: f32, n: usize) -> EquivResult {
    let n_bad: usize = mt_vals
        .chunks(n)
        .map(|chunk| chunk.windows(2).filter(|w| w[0] > w[1] + tol).count())
        .sum();
    EquivResult {
        n_checked: mt_vals.len(),
        max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
        cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
        passed: n_bad == 0,
    }
}

pub fn sort_mt_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let rows = if is_bench { shape.b } else { 4 };
    [rows, 1, 1]
}

// ── Scan ──────────────────────────────────────────────────────────────────

pub fn scan_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let (rows, n) = if is_bench { (shape.b, shape.n) } else { (4, 256) };
    let inp_vals: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
    let inp = buffer_typed(runner, &inp_vals, DType::F32);
    let out = zeros_typed(runner, rows * n, DType::F32);
    let ns = runner.buffer_u64(n as u64);
    vec![inp, out, ns]
}

// ── ArgReduce ─────────────────────────────────────────────────────────────

pub fn arg_reduce_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let n = if is_bench { shape.n } else { shape.check_n };
    let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1009) as f32 * 0.001).collect();
    let inp = buffer_typed(runner, &vals, DType::F32);
    let out = runner.buffer_zeros(4);
    let dummy_a = runner.buffer_u32(0);
    let dummy_b = runner.buffer_u32(0);
    let dummy_c = runner.buffer_u32(0);
    let ndim = runner.buffer_u64(0);
    let ax_stride = runner.buffer_i64(1);
    let ax_size = runner.buffer_u64(n as u64);
    vec![inp, out, dummy_a, dummy_b, dummy_c, ndim, ax_stride, ax_size]
}

pub fn arg_reduce_mlx_grid(_shape: &ShapeSpec, _is_bench: bool, tpg: usize) -> [usize; 3] {
    [tpg, 1, 1]
}

pub fn bytes_single(n: usize, _b: usize, _reads: usize, _out: usize, _eb: usize) -> usize { n * 4 }
pub fn bytes_double(n: usize, _b: usize, _reads: usize, _out: usize, _eb: usize) -> usize { n * 8 }
pub fn bytes_io(_n: usize, _b: usize, reads: usize, out: usize, eb: usize) -> usize { out * eb * reads }

// ── Random ────────────────────────────────────────────────────────────────

pub fn random_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let n = if is_bench { shape.n } else { 1024 };
    let out = runner.buffer_zeros(n * 4);
    let n_buf = runner.buffer_u32(n as u32);
    vec![out, n_buf]
}

pub fn random_check_fn(_ref_vals: &[f32], mt_vals: &[f32], _tol: f32, _n: usize) -> EquivResult {
    let ref_vals: Vec<u32> = (0..mt_vals.len() as u32)
        .map(|gid| {
            let mut s = gid + 1;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect();
    let mt_u32: Vec<u32> = mt_vals.iter().map(|f| f.to_bits()).collect();
    let n_bad = ref_vals.iter().zip(&mt_u32).filter(|(a, b)| a != b).count();
    EquivResult {
        n_checked: mt_vals.len(),
        max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
        cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
        passed: n_bad == 0,
    }
}

// ── FpQuantized ───────────────────────────────────────────────────────────

pub fn fp_quantized_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let n = if is_bench { shape.n } else { 1024 };
    let data: Vec<f32> = (0..n).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
    let inp = buffer_typed(runner, &data, DType::F32);
    let out = zeros_typed(runner, n, DType::F32);
    vec![inp, out]
}

pub fn fp_quantized_mlx_grid(_shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let n = if is_bench { 1_048_576usize } else { 1024 };
    [1, n / 32, 1]
}

pub fn fp_quantized_check_fn(ref_vals: &[f32], mt_vals: &[f32], _tol: f32, _n: usize) -> EquivResult {
    // ref_vals = CPU-computed FP4 reference
    check_equiv_with(ref_vals, mt_vals, EquivTolerance::new(0.5, 0.99))
}

// ── QuantizedMatVec ───────────────────────────────────────────────────────

pub fn quantized_mat_vec_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let m = if is_bench { shape.b } else { 4 };
    let k = if is_bench { shape.n } else { shape.extra[0] }; // group_size as check_k
    let group_size = shape.extra[0];
    let _gs_per_row = k / group_size;
    let w_elems = m * k / 8;
    let sb_elems = m * k / group_size;
    let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
    let scale_f16: Vec<u8> =
        (0..sb_elems * 2).map(|i| if i % 2 == 0 { 0x66 } else { 0x2E }).collect();
    let bias_f16 = vec![0u8; sb_elems * 2];
    let x_f16: Vec<u8> = (0..k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();
    let w_buf = runner.buffer_bytes(&w_data);
    let scales_buf = runner.buffer_bytes(&scale_f16);
    let biases_buf = runner.buffer_bytes(&bias_f16);
    let x_buf = runner.buffer_bytes(&x_f16);
    let y_buf = runner.buffer_zeros(m * 2);
    let in_size = runner.buffer_i32(k as i32);
    let out_size = runner.buffer_i32(m as i32);
    let batch_zero_a = runner.buffer_i32(0);
    let batch_zero_b = runner.buffer_i32(0);
    let zero_a = runner.buffer_zeros(8);
    let zero_b = runner.buffer_zeros(8);
    let zero_c = runner.buffer_zeros(8);
    let zero_d = runner.buffer_zeros(8);
    let zero_e = runner.buffer_zeros(8);
    let zero_f = runner.buffer_zeros(8);
    let batch_zero_c = runner.buffer_i32(0);
    vec![
        w_buf, scales_buf, biases_buf, x_buf, y_buf,
        in_size, out_size, batch_zero_a, batch_zero_b,
        zero_a, zero_b, batch_zero_c,
        zero_c, zero_d, zero_e, zero_f,
    ]
}

// ── Rope ──────────────────────────────────────────────────────────────────

pub fn rope_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let d = shape.n;                // head_dim
    let h = shape.b;                // n_heads
    let l = if is_bench { shape.extra[0] } else { 4 };
    let npg = shape.extra[1];
    let n_elems = h * l * d;
    let gx = d / (2 * npg);
    let in_f16 = f16_vec(&(0..n_elems).map(|i| i as f32 * 0.001).collect::<Vec<_>>());
    let inp = runner.buffer_f16(&in_f16);
    let out = runner.buffer_zeros(n_elems * 2);
    let h_stride = runner.buffer_u32(d as u32);
    let seq_stride = runner.buffer_u32((h * d) as u32);
    let grid_x = runner.buffer_u32(gx as u32);
    let base = runner.buffer_f32_scalar((10000f32).log2());
    vec![inp, out, h_stride, seq_stride, grid_x, base]
}

pub fn rope_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, _dt: DType)
    -> Vec<GpuBuffer>
{
    let d = shape.n;
    let h = shape.b;
    let l = if is_bench { shape.extra[0] } else { 4 };
    let n_elems = h * l * d;
    let in_f16 = f16_vec(&(0..n_elems).map(|i| i as f32 * 0.001).collect::<Vec<_>>());
    let inp = runner.buffer_f16(&in_f16);
    let out = runner.buffer_zeros(n_elems * 2);
    let strides_bytes: Vec<u8> =
        [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let strides_buf_a = runner.buffer_bytes(&strides_bytes);
    let strides_buf_b = runner.buffer_bytes(&strides_bytes);
    let offset_arr = runner.buffer_i32(0);
    let scale_buf = runner.buffer_f32_scalar(1.0);
    let offset_stride = runner.buffer_i64(1);
    let n_head_buf = runner.buffer_i32(h as i32);
    let dummy_a = runner.buffer_zeros(4);
    let dummy_b = runner.buffer_zeros(4);
    let base = runner.buffer_f32_scalar((10000f32).log2());
    vec![inp, out, offset_arr, scale_buf, strides_buf_a, strides_buf_b,
         offset_stride, n_head_buf, dummy_a, dummy_b, base]
}

pub fn rope_mlx_compile(runner: &GpuRunner, src: &str, dt: DType)
    -> Option<crate::runner::CompiledKernel>
{
    let name = match dt {
        DType::F16 => "rope_float16",
        DType::F32 => "rope_float32",
        DType::BF16 => "rope_bfloat16",
        _ => return None,
    };
    runner.compile_with_bool_constants(src, name, &[(1, true), (2, false), (3, false)]).ok()
}

pub fn rope_mt_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let d = shape.n;
    let h = shape.b;
    let l = if is_bench { shape.extra[0] } else { 4 };
    let npg = shape.extra[1];
    [d / (2 * npg), l, h / npg]
}

pub fn rope_mlx_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    rope_mt_grid(shape, is_bench, _tpg)
}

pub fn rope_out_n(shape: &ShapeSpec, is_bench: bool) -> usize {
    let h = shape.b;
    let d = shape.n;
    let l = if is_bench { shape.extra[0] } else { 4 };
    h * l * d
}

// ── Attention ─────────────────────────────────────────────────────────────

pub fn attention_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, _is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let d = shape.extra[0];       // head_dim
    let h = shape.n;              // n_heads (or 2 for check)
    let n_kv = shape.b;           // n_kv (or 64 for check)
    let scale = 1.0_f32 / (d as f32).sqrt();
    let max_n = h * n_kv * d;
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let q_buf = buffer_typed(runner, &vals[..h * d], dt);
    let k_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
    let v_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
    let out_buf = zeros_typed(runner, h * d, dt);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);
    vec![q_buf, k_buf, v_buf, out_buf, n_buf, sc_buf]
}

pub fn attention_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, _is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let d = shape.extra[0];
    let h = shape.n;
    let n_kv = shape.b;
    let scale = 1.0_f32 / (d as f32).sqrt();
    let max_n = h * n_kv * d;
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let q_buf = buffer_typed(runner, &vals[..h * d], dt);
    let k_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
    let v_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
    let out = zeros_typed(runner, h * d, dt);
    let gqa = runner.buffer_i32(1);
    let n_i32 = runner.buffer_i32(n_kv as i32);
    let khs = runner.buffer_u64((n_kv * d) as u64);
    let kss = runner.buffer_u64(d as u64);
    let sc_buf = runner.buffer_f32_scalar(scale);
    let khs2 = runner.buffer_u64((n_kv * d) as u64);
    let kss2 = runner.buffer_u64(d as u64);
    vec![q_buf, k_buf, v_buf, out, gqa, n_i32, khs, kss, khs2, kss2, sc_buf]
}

pub fn attention_mlx_compile(runner: &GpuRunner, src: &str, dt: DType)
    -> Option<crate::runner::CompiledKernel>
{
    let name = match dt {
        DType::F32 => "sdpa_vector_float_128_128",
        DType::F16 => "sdpa_vector_float16_t_128_128",
        _ => return None,
    };
    const FCS: &[(usize, bool)] =
        &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
    runner.compile_with_bool_constants(src, name, FCS).ok()
}

// ── SdpaVector ────────────────────────────────────────────────────────────

pub fn sdpa_vector_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, _is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let head_dim = shape.extra[0];
    let gqa_factor = shape.extra[1];
    let n_q_heads = shape.n;
    let n_kv = shape.b;
    let n_kv_heads = n_q_heads / gqa_factor;
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
    let hd_buf = runner.buffer_u32(head_dim as u32);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);
    vec![q_buf, k_buf, v_buf, out_buf, hd_buf, n_buf, gqa_buf, sc_buf]
}

pub fn sdpa_vector_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, _is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let head_dim = shape.extra[0];
    let gqa_factor = shape.extra[1];
    let n_q_heads = shape.n;
    let n_kv = shape.b;
    let n_kv_heads = n_q_heads / gqa_factor;
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let out = zeros_typed(runner, n_q_heads * head_dim, dt);
    let gqa = runner.buffer_i32(gqa_factor as i32);
    let n_i32 = runner.buffer_i32(n_kv as i32);
    let khs = runner.buffer_u64((n_kv * head_dim) as u64);
    let kss = runner.buffer_u64(head_dim as u64);
    let sc_buf = runner.buffer_f32_scalar(scale);
    let khs2 = runner.buffer_u64((n_kv * head_dim) as u64);
    let kss2 = runner.buffer_u64(head_dim as u64);
    vec![q_buf, k_buf, v_buf, out, gqa, n_i32, khs, kss, khs2, kss2, sc_buf]
}

pub fn sdpa_vector_mlx_compile(runner: &GpuRunner, src: &str, dt: DType)
    -> Option<crate::runner::CompiledKernel>
{
    const FCS: &[(usize, bool)] = &[
        (20, false), (21, false), (22, false), (23, false), (24, false), (25, false),
    ];
    let dims = [128, 128]; // head_dim hardcoded to 128
    let name = match dt {
        DType::F32 => format!("sdpa_vector_float_{}_{}", dims[0], dims[1]),
        DType::F16 => format!("sdpa_vector_float16_t_{}_{}", dims[0], dims[1]),
        DType::BF16 => format!("sdpa_vector_bfloat16_t_{}_{}", dims[0], dims[1]),
        _ => return None,
    };
    runner.compile_with_bool_constants(src, &name, FCS).ok()
}

// ── StridedCopy ───────────────────────────────────────────────────────────

pub fn strided_copy_mt_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let pad = shape.extra[0];
    let (m, n) = if is_bench { (shape.n, shape.b) } else { (8, 16) };
    let src_stride = n + pad;
    let src_vals: Vec<f32> = (0..m * src_stride)
        .map(|i| {
            let row = i / src_stride;
            let col = i % src_stride;
            if col < n { (row * n + col) as f32 + 1.0 } else { -999.0 }
        })
        .collect();
    let src_buf = buffer_typed(runner, &src_vals, dt);
    let src_shape = runner.buffer_bytes(
        &[m as u32, n as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let src_strides = runner.buffer_bytes(
        &[src_stride as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let cols_buf = runner.buffer_u32(n as u32);
    let out = zeros_typed(runner, m * n, dt);
    vec![src_buf, src_shape, src_strides, out, cols_buf]
}

pub fn strided_copy_mlx_bufs(runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType)
    -> Vec<GpuBuffer>
{
    let pad = shape.extra[0];
    let (m, n) = if is_bench { (shape.n, shape.b) } else { (8, 16) };
    let src_stride = n + pad;
    let src_vals: Vec<f32> = (0..m * src_stride).map(|i| (i % 256) as f32 * 0.01).collect();
    let src_buf = buffer_typed(runner, &src_vals, dt);
    let out = zeros_typed(runner, m * n, dt);
    let strides_i64 = runner.buffer_bytes(
        &[src_stride as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    vec![src_buf, out, strides_i64]
}

pub fn strided_copy_mt_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let (m, n) = if is_bench { (shape.n, shape.b) } else { (8, 16) };
    [m, n, 1]
}

pub fn strided_copy_mlx_grid(shape: &ShapeSpec, is_bench: bool, _tpg: usize) -> [usize; 3] {
    let (m, n) = if is_bench { (shape.n, shape.b) } else { (8, 16) };
    [n, m, 1]
}

// ── AffineDequantize ──────────────────────────────────────────────────────

fn affine_pack_factor(bits: usize) -> usize {
    match bits { 3..=5 => 8, 6 | 8 => 4, _ => panic!("unsupported bits={bits}") }
}
fn affine_bytes_per_pack(bits: usize) -> usize {
    match bits { 3 => 3, 4 => 4, 5 => 5, 6 => 3, 8 => 4, _ => panic!("unsupported bits={bits}") }
}
fn affine_mlx_pack_factor(bits: usize) -> usize {
    match bits { 3 => 8, 4 => 2, 5 => 8, 6 => 4, 8 => 1, _ => panic!("unsupported bits={bits}") }
}

pub fn affine_dequantize_mlx_bufs(
    runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType,
) -> Vec<GpuBuffer> {
    let bits = shape.extra[0];
    let group_size = shape.extra[1];
    let n_groups = if is_bench { shape.extra[2] } else { 4 };   // check: 4 groups
    let batch = if is_bench { shape.extra[3] } else { 1 };
    let n_total_groups = n_groups * batch;
    let n_elem = n_total_groups * group_size;
    let pack_factor = affine_pack_factor(bits);
    let bytes_per_pack = affine_bytes_per_pack(bits);
    let n_packs = n_elem / pack_factor;
    let weight_bytes_needed = n_packs * bytes_per_pack;
    let weight_u32s = weight_bytes_needed.div_ceil(4) + 1;
    let w_bytes: Vec<u8> =
        (0..weight_u32s * 4).map(|i| ((i as u32).wrapping_mul(0x0103_5b1d) ^ 0xa5) as u8).collect();
    let scales_f32: Vec<f32> = (0..n_total_groups).map(|i| 0.01 + (i % 7) as f32 * 0.005).collect();
    let biases_f32: Vec<f32> = (0..n_total_groups).map(|i| -0.1 + (i % 5) as f32 * 0.02).collect();
    let w_buf = runner.buffer_bytes(&w_bytes);
    let scales_buf = buffer_typed(runner, &scales_f32, dt);
    let biases_buf = buffer_typed(runner, &biases_f32, dt);
    let out_buf = zeros_typed(runner, n_elem, dt);
    vec![w_buf, scales_buf, biases_buf, out_buf]
}

pub fn affine_dequantize_mlx_grid(
    shape: &ShapeSpec, is_bench: bool, tpg: usize,
) -> [usize; 3] {
    let bits = shape.extra[0];
    let group_size = shape.extra[1];
    let n_groups = if is_bench { shape.extra[2] } else { 4 };
    let batch = if is_bench { shape.extra[3] } else { 1 };
    let n_total_groups = n_groups * batch;
    let n_elem = n_total_groups * group_size;
    let mlx_pf = affine_mlx_pack_factor(bits);
    let mlx_n_packs = n_elem / mlx_pf;
    [mlx_n_packs.div_ceil(tpg), 1, 1]
}

// ── AffineQuantize ────────────────────────────────────────────────────────

pub fn affine_quantize_mlx_bufs(
    runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType,
) -> Vec<GpuBuffer> {
    let bits = shape.extra[0];
    let group_size = shape.extra[1];
    let n_groups = if is_bench { shape.extra[2] } else { 4 };
    let batch = if is_bench { shape.extra[3] } else { 1 };
    let n_total_groups = n_groups * batch;
    let n_elem = n_total_groups * group_size;
    let pack_factor = 32 / bits;
    let n_packs = n_elem / pack_factor;
    let w_f32: Vec<f32> = (0..n_elem).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    let w_buf = buffer_typed(runner, &w_f32, dt);
    let packed_buf = runner.buffer_zeros(n_packs * 4);
    let scales_buf = zeros_typed(runner, n_total_groups, dt);
    let biases_buf = zeros_typed(runner, n_total_groups, dt);
    vec![w_buf, packed_buf, scales_buf, biases_buf]
}

pub fn affine_quantize_mt_grid(shape: &ShapeSpec, _is_bench: bool, _tpg: usize) -> [usize; 3] {
    let n_groups = shape.n;
    [n_groups, 1, 1]
}

pub fn affine_quantize_mlx_grid(
    _shape: &ShapeSpec, _is_bench: bool, _tpg: usize,
) -> [usize; 3] {
    [1, 1, 1]
}

// ── SteelGemm ─────────────────────────────────────────────────────────────

pub fn steel_gemm_mlx_bufs(
    runner: &GpuRunner, shape: &ShapeSpec, is_bench: bool, dt: DType,
) -> Vec<GpuBuffer> {
    let bm = shape.extra[0];
    let bn = shape.extra[1];
    let (m, n, k) = if is_bench {
        (shape.n, shape.b, shape.extra[2])
    } else {
        (shape.check_n, shape.check_b, shape.extra[3])
    };
    let a_buf = buffer_typed(runner, &vec![1.0f32; m * k], dt);
    let b_buf = buffer_typed(runner, &vec![1.0f32; k * n], dt);
    let d_buf = zeros_typed(runner, m * n, dt);
    // Build GEMMParams struct
    let lda = k as i32;
    let ldb = n as i32;
    let ldd = n as i32;
    let mut v = Vec::with_capacity(72);
    v.extend_from_slice(&(m as i32).to_le_bytes());
    v.extend_from_slice(&(n as i32).to_le_bytes());
    v.extend_from_slice(&(k as i32).to_le_bytes());
    v.extend_from_slice(&lda.to_le_bytes());
    v.extend_from_slice(&ldb.to_le_bytes());
    v.extend_from_slice(&ldd.to_le_bytes());
    v.extend_from_slice(&((n / bn) as i32).to_le_bytes()); // tiles_n
    v.extend_from_slice(&((m / bm) as i32).to_le_bytes()); // tiles_m
    v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_a
    v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_b
    v.extend_from_slice(&0i64.to_le_bytes()); // batch_stride_d
    v.extend_from_slice(&0i32.to_le_bytes()); // swizzle_log
    v.extend_from_slice(&((k / 16) as i32).to_le_bytes()); // gemm_k_iterations
    v.extend_from_slice(&0i32.to_le_bytes()); // batch_ndim
    let params_buf = runner.buffer_bytes(&v);
    let addmm_buf = runner.buffer_zeros(32);
    let batch_shape_buf = runner.buffer_zeros(4);
    let batch_strides_buf = runner.buffer_zeros(8);
    let d_buf2 = zeros_typed(runner, m * n, dt);
    vec![a_buf, b_buf, d_buf, d_buf2, params_buf, addmm_buf, batch_shape_buf, batch_strides_buf]
}

pub fn steel_gemm_mlx_compile(runner: &GpuRunner, src: &str, dt: DType)
    -> Option<crate::runner::CompiledKernel>
{
    use crate::bench_types::DtypeCtx;
    let name = DtypeCtx::elementwise(dt).tn;
    runner
        .compile_with_bool_constants(src, name, &[
            (10, false), (100, false), (110, false),
            (200, true), (201, true), (202, true),
        ])
        .ok()
}

// ── SdpaVector2Pass (legacy special case) ────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sdpa_vector_2pass(
    spec: &BenchSpec,
    runner: &GpuRunner,
    dt: DType,
    bench: &OpBench,
    head_dim: usize,
    n_kv: usize,
    n_q_heads: usize,
    gqa_factor: usize,
    _batch: usize,
    blocks: usize,
    pass2_kernel_name: &str,
    pass2_kernel_ir: fn(DType) -> Kernel,
) -> Vec<OpResult> {
    assert_eq!(head_dim, 128, "sdpa_decode_2pass hardcodes head_dim=128");
    assert!(n_q_heads.is_multiple_of(gqa_factor), "n_q_heads must be divisible by gqa_factor");
    assert!(blocks.is_multiple_of(32), "blocks must be a multiple of 32 (pass-2 reducer)");
    let n_kv_heads = n_q_heads / gqa_factor;
    let gqa_factor_u = gqa_factor;
    let ctx = DtypeCtx::elementwise(dt);

    let p1_msl = match msl_reduction(spec, dt) {
        Some(s) => s,
        None => return vec![],
    };
    let p1_mk = match compile_mt(runner, &p1_msl, spec.kernel_name) {
        Some(k) => k,
        None => return vec![],
    };
    let mut p2_kernel = pass2_kernel_ir(dt);
    p2_kernel.mode = KernelMode::Reduction;
    let p2_msl = match MslGenerator::default().generate(&p2_kernel) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let p2_mk = match compile_mt(runner, &p2_msl, pass2_kernel_name) {
        Some(k) => k,
        None => return vec![],
    };

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
    let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
    let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
    let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);

    let partial_o = runner.buffer_zeros(n_q_heads * blocks * head_dim * ctx.eb);
    let partial_max = runner.buffer_zeros(n_q_heads * blocks * 4);
    let partial_sum = runner.buffer_zeros(n_q_heads * blocks * 4);
    let mt_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);

    let hd_buf = runner.buffer_u32(head_dim as u32);
    let n_buf = runner.buffer_u32(n_kv as u32);
    let kvs_buf = runner.buffer_u32(n_kv as u32);
    let gqa_buf = runner.buffer_u32(gqa_factor as u32);
    let blocks_buf = runner.buffer_u32(blocks as u32);
    let sc_buf = runner.buffer_f32_scalar(scale);

    let p1_bufs: Vec<&GpuBuffer> = vec![
        &q_buf, &k_buf, &v_buf,
        &partial_o, &partial_max, &partial_sum,
        &hd_buf, &n_buf, &kvs_buf, &gqa_buf, &blocks_buf, &sc_buf,
    ];
    let p1_grid = [n_kv_heads, blocks, 1];
    let p1_tpg = [32, gqa_factor_u, 1];
    let p2_bufs: Vec<&GpuBuffer> =
        vec![&partial_o, &partial_max, &partial_sum, &mt_out_buf, &hd_buf, &blocks_buf];
    let p2_grid = [n_q_heads, 1, 1];
    let p2_tpg = [1024, 1, 1];

    runner.measure(&p1_mk, &p1_bufs, p1_grid, p1_tpg, 0, 1);
    runner.measure(&p2_mk, &p2_bufs, p2_grid, p2_tpg, 0, 1);
    let mt_out = read_typed(runner, &mt_out_buf, n_q_heads * head_dim, dt);

    const REF_FCS: &[(usize, bool)] =
        &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
    let ref_name: Option<String> = match dt {
        DType::F32 => Some(format!("sdpa_vector_float_{head_dim}_{head_dim}")),
        DType::F16 => Some(format!("sdpa_vector_float16_t_{head_dim}_{head_dim}")),
        DType::BF16 => Some(format!("sdpa_vector_bfloat16_t_{head_dim}_{head_dim}")),
        _ => None,
    };
    let rk = ref_name.as_ref().and_then(|name| {
        spec.mlx_src.and_then(|src| runner.compile_with_bool_constants(src, name, REF_FCS).ok())
    });

    let equiv = rk.as_ref().map(|rk| {
        let gqa = runner.buffer_i32(gqa_factor as i32);
        let n_i32 = runner.buffer_i32(n_kv as i32);
        let khs = runner.buffer_u64((n_kv * head_dim) as u64);
        let kss = runner.buffer_u64(head_dim as u64);
        let mlx_out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
        runner.measure(
            rk,
            &[&q_buf, &k_buf, &v_buf, &mlx_out_buf, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
            [n_q_heads, 1, 1],
            [1024, 1, 1],
            0,
            1,
        );
        let ref_out = read_typed(runner, &mlx_out_buf, n_q_heads * head_dim, dt);
        check_equiv_with(&ref_out, &mt_out, EquivTolerance::new(spec.tol, 0.999))
    });

    let bytes = ((n_q_heads * head_dim + 2 * n_kv_heads * n_kv * head_dim + n_q_heads * head_dim)
        * ctx.eb) as f64;
    let p1_perf = bench_gbps_only(runner, &p1_mk, &p1_bufs, p1_grid, p1_tpg, bytes);
    let p2_perf = bench_gbps_only(runner, &p2_mk, &p2_bufs, p2_grid, p2_tpg, bytes);
    let mt_perf = match (p1_perf, p2_perf) {
        (Some(p1), Some(p2)) if p1 > 0.0 && p2 > 0.0 => {
            let gb = bytes / 1.0e9;
            let p1_s = gb / p1;
            let p2_s = gb / p2;
            Some(gb / (p1_s + p2_s))
        },
        _ => None,
    };
    let ref_perf = rk
        .as_ref()
        .map(|rk| {
            let gqa = runner.buffer_i32(gqa_factor as i32);
            let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * head_dim) as u64);
            let kss = runner.buffer_u64(head_dim as u64);
            let out = zeros_typed(runner, n_q_heads * head_dim, dt);
            bench_gbps_only(
                runner,
                rk,
                &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [n_q_heads, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        })
        .unwrap_or(None);

    let label = format!(
        "H={n_q_heads} N={n_kv} D={head_dim} gqa={gqa_factor} blocks={blocks} {}",
        ctx.label
    );
    vec![bench.result_sub(Some(spec.subop), label, ref_perf, mt_perf, equiv)]
}

// ── Legacy shape-iterating wrappers (scan, attention, quantized_mat_vec) ─
// These classes have dynamic shapes from compile-time arrays.
// They will be migrated to fn-pointers in a follow-up.

fn run_legacy_scan(spec: &BenchSpec, runner: &GpuRunner, _dt: DType, bench: &OpBench,
    shapes: &[(usize, usize)], tpg: usize) -> Vec<OpResult>
{
    let msl = match msl_reduction(spec, DType::F32) {
        Some(s) => s, None => return vec![],
    };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) {
        Some(k) => k, None => return vec![],
    };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "float32");
    let mut results = Vec::new();
    for &(rows, n) in shapes {
        let check_rows = 4usize; let check_n = 256usize;
        let inp_vals: Vec<f32> = (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
        let ref_out: Vec<f32> = {
            let mut out = vec![0.0f32; check_rows * check_n];
            for r in 0..check_rows {
                let mut acc = 0.0f32;
                for c in 0..check_n { acc += inp_vals[r * check_n + c]; out[r * check_n + c] = acc; }
            }
            out
        };
        let inp_c = buffer_typed(runner, &inp_vals[..check_rows * check_n], DType::F32);
        let out_c = zeros_typed(runner, check_rows * check_n, DType::F32);
        let ns_c = runner.buffer_u32(check_n as u32);
        let mt_chk = run_typed_once(runner, &mk, &[&inp_c, &out_c, &ns_c], &out_c,
            check_rows * check_n, [1, check_rows, 1], [tpg, 1, 1], DType::F32);
        let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(spec.tol, 0.5));
        let inp_buf = buffer_typed(runner, &inp_vals, DType::F32);
        let bytes = (rows * n * 8) as f64;
        let ns_u64 = runner.buffer_u64(n as u64); let ns_u32 = runner.buffer_u32(n as u32);
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps_only(runner, rk, &[&inp_buf, &out, &ns_u64], [1, rows, 1], [tpg, 1, 1], bytes)
        });
        let mt_perf = {
            let out = zeros_typed(runner, rows * n, DType::F32);
            bench_gbps_only(runner, &mk, &[&inp_buf, &out, &ns_u32], [1, rows, 1], [tpg, 1, 1], bytes)
        };
        results.push(bench.result_sub(Some(spec.subop), format!("B={rows} N={n} f32"), ref_perf, mt_perf, Some(equiv)));
    }
    results
}

fn run_legacy_attention(spec: &BenchSpec, runner: &GpuRunner, dt: DType, bench: &OpBench,
    shapes: &[(usize, usize, usize)], tpg: usize) -> Vec<OpResult>
{
    let ctx = DtypeCtx::elementwise(dt);
    let msl = match msl_reduction(spec, dt) { Some(s) => s, None => return vec![] };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) { Some(k) => k, None => return vec![] };
    const REF_FCS: &[(usize, bool)] =
        &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
    let ref_name: Option<&str> = match dt {
        DType::F32 => Some("sdpa_vector_float_128_128"),
        DType::F16 => Some("sdpa_vector_float16_t_128_128"), _ => None,
    };
    let rk = ref_name.and_then(|n| spec.mlx_src.and_then(|s| runner.compile_with_bool_constants(s, n, REF_FCS).ok()));
    let mut results = Vec::new();
    for &(h, n_kv, d) in shapes {
        let scale = 1.0_f32 / (d as f32).sqrt(); let ch = 2usize; let cn = 64usize;
        let cq: Vec<f32> = (0..ch * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let ck_: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let cv: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let ref_out: Vec<f32> = {
            let mut out = vec![0.0f32; ch * d];
            for head in 0..ch {
                let q_base = head * d; let kv_base = head * cn * d;
                let mut scores = vec![0.0f32; cn]; let mut max_score = f32::NEG_INFINITY;
                for t in 0..cn {
                    let base = kv_base + t * d;
                    let qk: f32 = (0..d).map(|e| cq[q_base + e] * ck_[base + e]).sum::<f32>() * scale;
                    scores[t] = qk; max_score = max_score.max(qk);
                }
                let mut sum = 0.0f32; let mut o = vec![0.0f32; d];
                for t in 0..cn {
                    let w = (scores[t] - max_score).exp(); sum += w;
                    for e in 0..d { o[e] += w * cv[kv_base + t * d + e]; }
                }
                let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
                for e in 0..d { out[q_base + e] = o[e] * inv; }
            }
            out
        };
        let q_b = buffer_typed(runner, &cq, dt); let k_b = buffer_typed(runner, &ck_, dt);
        let v_b = buffer_typed(runner, &cv, dt); let out_b = zeros_typed(runner, ch * d, dt);
        let n_b = runner.buffer_u32(cn as u32); let sc_b = runner.buffer_f32_scalar(scale);
        runner.measure(&mk, &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b], [ch, 1, 1], [tpg, 1, 1], 0, 1);
        let mt_chk = read_typed(runner, &out_b, ch * d, dt);
        let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(spec.tol, 0.999));
        let vals: Vec<f32> = (0..h * n_kv * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let bytes = (h * n_kv * d * ctx.eb * 2 + h * d * ctx.eb * 2) as f64;
        let q_buf = buffer_typed(runner, &vals[..h * d], dt);
        let k_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
        let v_buf = buffer_typed(runner, &vals[..h * n_kv * d], dt);
        let n_buf = runner.buffer_u32(n_kv as u32); let sc_buf = runner.buffer_f32_scalar(scale);
        let ref_perf = rk.as_ref().and_then(|rk| {
            let gqa = runner.buffer_i32(1); let n_i32 = runner.buffer_i32(n_kv as i32);
            let khs = runner.buffer_u64((n_kv * d) as u64); let kss = runner.buffer_u64(d as u64);
            let out = zeros_typed(runner, h * d, dt);
            bench_gbps_only(runner, rk, &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                [h, 1, 1], [1024, 1, 1], bytes)
        });
        let mt_perf = {
            let out = zeros_typed(runner, h * d, dt);
            bench_gbps_only(runner, &mk, &[&q_buf, &k_buf, &v_buf, &out, &n_buf, &sc_buf],
                [h, 1, 1], [tpg, 1, 1], bytes)
        };
        results.push(bench.result_sub(Some(spec.subop),
            format!("H={h} N={n_kv} D={d} {}", ctx.label), ref_perf, mt_perf, Some(equiv)));
    }
    results
}

fn run_legacy_quantized_mat_vec(spec: &BenchSpec, runner: &GpuRunner, _dt: DType, bench: &OpBench,
    shapes: &[(usize, usize)], group_size: usize, tpg: usize) -> Vec<OpResult>
{
    let msl = match msl_reduction(spec, DType::F32) { Some(s) => s, None => return vec![] };
    let mk = match compile_mt(runner, &msl, spec.kernel_name) { Some(k) => k, None => return vec![] };
    let ref_kernel = compile_mlx(runner, spec.mlx_src, spec.mlx_pattern, "");
    let mut results = Vec::new();
    for &(m, k) in shapes {
        let w_elems = m * k / 8; let sb_elems = m * k / group_size; let gs_per_row = k / group_size;
        let cm = 4usize; let ck = group_size;
        let w_check: Vec<u32> = (0..cm * ck / 8).map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 { v |= ((i as u32 + bit) & 0xF) << (bit * 4); }
            v
        }).collect();
        let s_check = vec![0.1f32; cm]; let b_check = vec![0.0f32; cm]; let x_check = vec![1.0f32; ck];
        let ref_out: Vec<f32> = (0..cm).map(|row| {
            let mut acc = 0.0f32;
            for g in 0..1usize {
                let s = s_check[row + g]; let bias = b_check[row + g];
                for p in 0..8usize {
                    let packed = w_check[row * ck / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let int4_val = ((packed >> (bit * 4)) & 0xF) as f32;
                        acc += (s * int4_val + bias) * x_check[g * ck + p * 8 + bit as usize];
                    }
                }
            }
            acc
        }).collect();
        let w_bytes: Vec<u8> = w_check.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_buf_c = runner.buffer_bytes(&w_bytes); let s_buf_c = runner.buffer_f32(&s_check);
        let b_buf_c = runner.buffer_f32(&b_check); let x_buf_c = runner.buffer_f32(&x_check);
        let out_c = runner.buffer_zeros(cm * 4);
        let k_buf_c = runner.buffer_u32(ck as u32); let gpr_buf_c = runner.buffer_u32(1u32);
        runner.measure(&mk, &[&w_buf_c, &s_buf_c, &b_buf_c, &x_buf_c, &out_c, &k_buf_c, &gpr_buf_c],
            [cm, 1, 1], [tpg, 1, 1], 0, 1);
        let mt_out_c = runner.read_f32_slice(&out_c, cm);
        let n_bad = ref_out.iter().zip(mt_out_c.iter()).filter(|(r, m)| (*r - *m).abs() > 1e-3).count();
        let equiv = EquivResult {
            n_checked: cm, max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 }, passed: n_bad == 0,
        };
        let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
        let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
        let biases_f32 = vec![0.0f32; sb_elems];
        let x_f32: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();
        let w_mt_buf = runner.buffer_bytes(&w_data); let s_mt_buf = runner.buffer_f32(&scales_f32);
        let b_mt_buf = runner.buffer_f32(&biases_f32); let x_mt_buf = runner.buffer_f32(&x_f32);
        let k_buf = runner.buffer_u32(k as u32); let gpr_buf = runner.buffer_u32(gs_per_row as u32);
        let bytes_mt = (m * k / 2 + sb_elems * 4 * 2 + k * 4 + m * 4) as f64;
        let mt_perf = {
            let out_buf = runner.buffer_zeros(m * 4);
            bench_gbps_only(runner, &mk, &[&w_mt_buf, &s_mt_buf, &b_mt_buf, &x_mt_buf, &out_buf, &k_buf, &gpr_buf],
                [m, 1, 1], [tpg, 1, 1], bytes_mt)
        };
        const ROWS_PER_TG: usize = 8;
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let scale_f16: Vec<u8> = (0..sb_elems * 2).map(|i| if i % 2 == 0 { 0x66 } else { 0x2E }).collect();
            let bias_f16 = vec![0u8; sb_elems * 2];
            let x_f16: Vec<u8> = (0..k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();
            let scales_f16_buf = runner.buffer_bytes(&scale_f16);
            let biases_f16_buf = runner.buffer_bytes(&bias_f16); let x_f16_buf = runner.buffer_bytes(&x_f16);
            let in_size = runner.buffer_i32(k as i32); let out_size = runner.buffer_i32(m as i32);
            let batch_zero = runner.buffer_i32(0); let zero = runner.buffer_zeros(8);
            let y_buf = runner.buffer_zeros(m * 2);
            let bytes_f16 = (m * k / 2 + sb_elems * 2 * 2 + k * 2 + m * 2) as f64;
            bench_gbps_only(runner, rk, &[&w_mt_buf, &scales_f16_buf, &biases_f16_buf, &x_f16_buf, &y_buf,
                &in_size, &out_size, &batch_zero, &zero, &zero, &batch_zero, &zero, &zero, &zero, &zero],
                [1, m / ROWS_PER_TG, 1], [64, 1, 1], bytes_f16)
        });
        results.push(bench.result_sub(Some(spec.subop),
            format!("M={m} K={k} f32 gs{group_size} b4"), ref_perf, mt_perf, Some(equiv)));
    }
    results
}
