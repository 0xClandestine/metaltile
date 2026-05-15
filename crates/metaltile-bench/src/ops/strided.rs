//! Strided-tensor benchmark — #[kernel] DSL with #[strided] params vs MLX metal/copy.metal
//!
//! MLX kernel: copy_g_nd2 (copy.metal)
//!   Params: (src: device T*, dst: device T*, src_strides: constant int64_t*, index: uint2)
//!   Grid: [N, M, 1] × [1, 1, 1]  (one thread per output element)
//!   Algorithm: dst[row*N + col] = src[row*src_strides[0] + col*src_strides[1]]
//!
//! MetalTile: mt_strided_copy — same algorithm with #[strided] attribute on src.
//!   KernelMode::Elementwise, Grid3D dispatch [N, M, 1] × [1, 1, 1]
//!
//! The test uses a non-contiguous view: a sub-matrix of M×N taken from a M×(N+PAD) buffer,
//! so src_strides = [N+PAD, 1] while the logical shape is M×N.

use metaltile::{core::ir::KernelMode, kernel};
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{
        DType,
        DtypeCtx,
        OpBench,
        OpResult,
        bench_all_dtypes,
        bench_gbps,
        buffer_typed,
        check_equiv,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/copy.metal");

const BENCH: OpBench = OpBench::new("strided_copy", "GB/s");
// M rows × N cols, padded row stride = N+PAD (non-contiguous source)
const M: usize = 1024;
const N: usize = 4096;
const PAD: usize = 128; // extra elements per row making source non-contiguous
const TPG: usize = 1; // copy_g_nd2 uses one thread per element

/// Strided copy: dst[row, col] = src[row, col] where src has a non-unit row stride.
/// The #[strided] attribute causes codegen to emit {src}_strides[d] for index computation.
#[kernel]
pub fn mt_strided_copy<T>(#[strided] src: Tensor<T>, out: Tensor<T>, #[constexpr] cols: u32) {
    let row = program_id::<0>();
    let col = program_id::<1>();
    let flat_out = row * cols + col;
    let val = load(src[(row, col)]);
    store(out[flat_out], val);
}

fn strided_copy_msl_for(dt: DType) -> String {
    let mut k = mt_strided_copy::kernel_ir_for(dt);
    k.mode = KernelMode::Grid3D;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[strided_copy {dt:?}]: {e}");
        String::new()
    })
}

pub fn bench_strided(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_strided_for)
}

fn bench_strided_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let msl = strided_copy_msl_for(dt);
    let mk = runner.compile(&msl, "mt_strided_copy").ok();

    // copy_g_nd2 takes src_strides as a constant int64_t* buffer.
    // Buffer layout: [stride_for_dim0, stride_for_dim1] = [N+PAD, 1]
    let ref_name = format!("copy_g_nd2{tn}{tn}");
    let rk = runner.compile(SRC, &ref_name).ok();

    // ── Correctness ──────────────────────────────────────────────────────────
    // Small check: 8 rows × 16 cols, padded stride = 16+4 = 20
    const CM: usize = 8;
    const CN: usize = 16;
    const CP: usize = 4;
    let src_stride = CN + CP;

    // Source buffer: CM × (CN+CP) filled with recognisable values; only CM×CN are read.
    let src_vals: Vec<f32> = (0..CM * src_stride)
        .map(|i| {
            let row = i / src_stride;
            let col = i % src_stride;
            if col < CN { (row * CN + col) as f32 + 1.0 } else { -999.0 }
        })
        .collect();

    // Expected output: row-major CM×CN block.
    let expected: Vec<f32> = (0..CM * CN).map(|i| i as f32 + 1.0).collect();

    let src_buf = buffer_typed(runner, &src_vals, dt);
    // MLX ref uses int64_t strides; MT kernel uses uint strides (slot 1=shape, slot 2=strides)
    let strides_buf = runner.buffer_bytes(
        &[src_stride as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let src_shape_check = runner.buffer_bytes(
        &[CM as u32, CN as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let src_strides_check = runner.buffer_bytes(
        &[src_stride as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let cols_buf = runner.buffer_u32(CN as u32);

    let ref_equiv = rk.as_ref().map(|rk| {
        let out = zeros_typed(runner, CM * CN, dt);
        run_typed_once(
            runner,
            rk,
            &[&src_buf, &out, &strides_buf],
            &out,
            CM * CN,
            [CN, CM, 1],
            [TPG, TPG, 1],
            dt,
        )
    });

    // MT kernel slot layout: [src, src_shape, src_strides, out, cols]
    let mt_check_small = mk.as_ref().map(|mk| {
        let out = zeros_typed(runner, CM * CN, dt);
        run_typed_once(
            runner,
            mk,
            &[&src_buf, &src_shape_check, &src_strides_check, &out, &cols_buf],
            &out,
            CM * CN,
            [CM, CN, 1],
            [1, 1, 1],
            dt,
        )
    });

    let equiv = match mt_check_small {
        Some(got) => check_equiv(&expected, &got, tol),
        None => {
            return vec![BENCH.nyi(format!("M={M} N={N}+{PAD} {dlabel}"), None)];
        },
    };
    let _ = ref_equiv; // suppress unused warning

    // ── Throughput ───────────────────────────────────────────────────────────
    // Full M×N copy from a M×(N+PAD) source.
    let full_src: Vec<f32> = (0..M * (N + PAD)).map(|i| (i % 256) as f32 * 0.01).collect();
    let full_src_buf = buffer_typed(runner, &full_src, dt);
    let full_strides = runner.buffer_bytes(
        &[(N + PAD) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_src_shape = runner.buffer_bytes(
        &[M as u32, N as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_src_strides = runner.buffer_bytes(
        &[(N + PAD) as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    );
    let full_cols = runner.buffer_u32(N as u32);
    let bytes = (M * N * eb * 2) as f64; // 1 read + 1 write

    let ref_perf = rk.as_ref().and_then(|rk| {
        let out = zeros_typed(runner, M * N, dt);
        bench_gbps(
            runner,
            rk,
            &[&full_src_buf, &out, &full_strides],
            [N, M, 1],
            [TPG, TPG, 1],
            bytes,
        )
    });

    let mt_perf = mk.as_ref().and_then(|mk| {
        let out = zeros_typed(runner, M * N, dt);
        bench_gbps(
            runner,
            mk,
            &[&full_src_buf, &full_src_shape, &full_src_strides, &out, &full_cols],
            [M, N, 1],
            [1, 1, 1],
            bytes,
        )
    });

    let shape = format!("M={M} N={N}+{PAD} {dlabel}");
    vec![BENCH.result(shape, ref_perf, mt_perf, Some(equiv))]
}

crate::bench_tests!(msl_fn: strided_copy_msl_for, kernel_name: "mt_strided_copy");

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "strided",
        mt_kernel: "mt_strided_copy".into(),
        metal_file: "copy.metal",
        ref_spec: RefSpec::Format("copy_g_nd2{tn}{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
