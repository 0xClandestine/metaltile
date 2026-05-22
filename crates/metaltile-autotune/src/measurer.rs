//! Candidate measurement strategies.
//!
//! Each `BenchDispatch` variant gets its own measurer: it owns the
//! pre-allocated buffer set + the dispatch geometry for the shape, and
//! exposes a uniform `CandidateMeasurer::measure` hook the search
//! orchestrator drives.

use metaltile::autotune::{PsoReflection, TuneConfig};
use metaltile_codegen::msl::{MslConfig, MslGenerator};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_std::{
    runner::{GpuBuffer, GpuRunner, buffer_typed, zeros_typed},
    spec::{BenchSpec, ScalarBufSpec, ShapeSpec},
};

use crate::{budget::BenchBudget, cost::tune_to_schedule, util::median_us};

/// Strategy for measuring one candidate config. The real impls are the
/// per-dispatch measurers below; tests inject a fake to cover the
/// orchestration (outcome flipping, fallback counting, no-measurer
/// path) without needing a Metal device.
pub(crate) trait CandidateMeasurer {
    /// Returns `(elapsed_us, pso_reflection)`. Reflection populated when
    /// the measurer compiled a PSO (the common case); `None` only for
    /// future variants that bypass PSO compilation entirely.
    fn measure(
        &self,
        cfg: &TuneConfig,
        budget: BenchBudget,
    ) -> Result<(f64, Option<PsoReflection>), String>;
}

pub(crate) struct GpuMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    mode: KernelMode,
    shape: &'a ShapeSpec,
    n: usize,
    b: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> GpuMeasurer<'a> {
    /// Build the buffer set and freeze the dispatch (`n`, `b`) for this
    /// shape. Allocates once — the candidate sweep reuses it across
    /// every config. `n_override` lets `--shapes` retarget the same
    /// spec at multiple cache buckets in one run; when `None`, falls
    /// back to `spec.shapes[0].n`.
    pub(crate) fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        mode: KernelMode,
        n_override: Option<usize>,
    ) -> Self {
        let shape = &spec.shapes[0];
        let n = n_override.unwrap_or(shape.n);
        let b = shape.b.max(1);
        let bufs = build_generic_buffers(runner, shape, n, b, dt);
        Self { runner, spec, dt, mode, shape, n, b, bufs }
    }
}

impl CandidateMeasurer for GpuMeasurer<'_> {
    fn measure(
        &self,
        cfg: &TuneConfig,
        budget: BenchBudget,
    ) -> Result<(f64, Option<PsoReflection>), String> {
        measure_generic(
            self.runner,
            self.spec,
            self.dt,
            self.mode,
            self.shape,
            self.n,
            self.b,
            cfg,
            &self.bufs,
            budget,
        )
    }
}

/// Measurer for `BenchDispatch::SdpaVector` (`mt_sdpa_vector`).
/// Mirrors `run_spec::run_sdpa_vector`'s buffer recipe + dispatch
/// geometry: 8 buffers (Q, K, V, out, head_dim, n_kv, gqa, scale) and
/// one threadgroup per Q head at the kernel's hardcoded TPG. The
/// candidate config's `threads.0` is ignored — the `SdpaVector`
/// family clamps it to the hardcoded value, so the measurer trusts
/// the dispatch params and not the sweep.
pub(crate) struct SdpaVectorMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    n_q_heads: usize,
    tpg: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> SdpaVectorMeasurer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        tpg: usize,
    ) -> Result<Self, String> {
        if !n_q_heads.is_multiple_of(gqa_factor) {
            return Err(format!(
                "SdpaVector: n_q_heads ({n_q_heads}) must be divisible by gqa_factor ({gqa_factor})",
            ));
        }
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Same value pattern run_sdpa_vector uses — modest range, no
        // zero shortcut, no NaN landmines. Sized to the largest tensor
        // (K/V at head_dim × n_kv × n_kv_heads) and aliased into the
        // smaller tensors via prefix slicing.
        let max_n = (n_kv_heads * n_kv * head_dim).max(n_q_heads * head_dim);
        let vals: Vec<f32> = (0..max_n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let q_buf = buffer_typed(runner, &vals[..n_q_heads * head_dim], dt);
        let k_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
        let v_buf = buffer_typed(runner, &vals[..n_kv_heads * n_kv * head_dim], dt);
        let out_buf = zeros_typed(runner, n_q_heads * head_dim, dt);
        let hd_buf = runner.buffer_u32(head_dim as u32);
        let n_buf = runner.buffer_u32(n_kv as u32);
        let gqa_buf = runner.buffer_u32(gqa_factor as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);

        Ok(Self {
            runner,
            spec,
            dt,
            n_q_heads,
            tpg,
            bufs: vec![q_buf, k_buf, v_buf, out_buf, hd_buf, n_buf, gqa_buf, sc_buf],
        })
    }
}

impl CandidateMeasurer for SdpaVectorMeasurer<'_> {
    fn measure(
        &self,
        cfg: &TuneConfig,
        budget: BenchBudget,
    ) -> Result<(f64, Option<PsoReflection>), String> {
        let (warmup, iters) = budget.iters();

        let mut k = (self.spec.kernel_ir)(self.dt);
        k.mode = KernelMode::Reduction;

        // Pin `expected_tpg` to the kernel's hardcoded TPG, NOT
        // `cfg.threads.0`. The SdpaVector family already keeps
        // `threads = (1024, 1, 1)`, but be defensive: a future config
        // generator drift mustn't silently miscompile us.
        let msl_cfg = MslConfig { expected_tpg: Some(self.tpg as u32), ..MslConfig::default() };
        let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
        let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
        let compiled = self
            .runner
            .compile(&msl, self.spec.kernel_name)
            .map_err(|e| format!("compile: {e}"))?;
        let reflection = self.runner.pso_reflection(&compiled);

        let buf_refs: Vec<&GpuBuffer> = self.bufs.iter().collect();
        let tgs = [self.n_q_heads, 1, 1];
        let tpg_arr = [self.tpg, 1, 1];

        self.runner.flush_slc();
        let samples = self.runner.measure(&compiled, &buf_refs, tgs, tpg_arr, warmup, iters);
        median_us(samples).map(|us| (us, Some(reflection)))
    }
}

/// Measurer for `BenchDispatch::SdpaPrefill` (`mt_sdpa_prefill*`).
/// Mirrors `run_spec::run_sdpa_prefill`'s 10-buffer recipe + dispatch
/// geometry (`[q_tiles, n_q_heads, batch]` TGs at the kernel's
/// per-spec TPG). The kernel's `bfloat_reinterpret_cast` flag is set
/// the same way run_spec does — keeps the f32→bf16 MFA reinterpret
/// active for the MMA kernels so their compile + dispatch match
/// production behavior.
pub(crate) struct SdpaPrefillMeasurer<'a> {
    runner: &'a GpuRunner,
    spec: &'a BenchSpec,
    dt: DType,
    q_tiles: usize,
    n_q_heads: usize,
    batch: usize,
    tpg: usize,
    bufs: Vec<GpuBuffer>,
}

impl<'a> SdpaPrefillMeasurer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        runner: &'a GpuRunner,
        spec: &'a BenchSpec,
        dt: DType,
        head_dim: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        q_len: usize,
        k_len: usize,
        bq: usize,
        tpg: usize,
    ) -> Result<Self, String> {
        if !n_q_heads.is_multiple_of(gqa_factor) {
            return Err(format!(
                "SdpaPrefill: n_q_heads ({n_q_heads}) must be divisible by gqa_factor ({gqa_factor})",
            ));
        }
        if !q_len.is_multiple_of(bq) {
            return Err(format!("SdpaPrefill: q_len ({q_len}) must be a multiple of bq ({bq})",));
        }
        let n_kv_heads = n_q_heads / gqa_factor;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let qsz = batch * n_q_heads * q_len * head_dim;
        let kvsz = batch * n_kv_heads * k_len * head_dim;
        let vals: Vec<f32> = (0..qsz.max(kvsz)).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        let q_buf = buffer_typed(runner, &vals[..qsz], dt);
        let k_buf = buffer_typed(runner, &vals[..kvsz], dt);
        let v_buf = buffer_typed(runner, &vals[..kvsz], dt);
        let out_buf = zeros_typed(runner, qsz, dt);
        let q_len_buf = runner.buffer_u32(q_len as u32);
        let k_len_buf = runner.buffer_u32(k_len as u32);
        let gqa_buf = runner.buffer_u32(gqa_factor as u32);
        let n_q_heads_buf = runner.buffer_u32(n_q_heads as u32);
        let n_kv_heads_buf = runner.buffer_u32(n_kv_heads as u32);
        let sc_buf = runner.buffer_f32_scalar(scale);

        Ok(Self {
            runner,
            spec,
            dt,
            q_tiles: q_len / bq,
            n_q_heads,
            batch,
            tpg,
            bufs: vec![
                q_buf,
                k_buf,
                v_buf,
                out_buf,
                q_len_buf,
                k_len_buf,
                gqa_buf,
                n_q_heads_buf,
                n_kv_heads_buf,
                sc_buf,
            ],
        })
    }
}

impl CandidateMeasurer for SdpaPrefillMeasurer<'_> {
    fn measure(
        &self,
        cfg: &TuneConfig,
        budget: BenchBudget,
    ) -> Result<(f64, Option<PsoReflection>), String> {
        let (warmup, iters) = budget.iters();

        let mut k = (self.spec.kernel_ir)(self.dt);
        // SimdGroup2D mode: the kernel reads tgid_{x,y,z} directly.
        // Match the dispatch geometry the run_spec path uses.
        k.mode = KernelMode::SimdGroup2D;
        // run_spec::run_sdpa_prefill sets this unconditionally for
        // every SdpaPrefill spec — the MMA kernels need it for the
        // f32→bf16 narrowing cast, and the non-MMA variant tolerates
        // it as a no-op. Keep the autotune path bit-identical.
        k.bfloat_reinterpret_cast = true;

        // Pin `expected_tpg` to the dispatch's TPG, not the family's
        // placeholder `cfg.threads.0` — different prefill kernels
        // use different per-tile TPGs, and the kernel constexprs are
        // baked around the dispatch value.
        let msl_cfg = MslConfig { expected_tpg: Some(self.tpg as u32), ..MslConfig::default() };
        let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
        let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
        let compiled = self
            .runner
            .compile(&msl, self.spec.kernel_name)
            .map_err(|e| format!("compile: {e}"))?;
        let reflection = self.runner.pso_reflection(&compiled);

        let buf_refs: Vec<&GpuBuffer> = self.bufs.iter().collect();
        let tgs = [self.q_tiles, self.n_q_heads, self.batch];
        let tpg_arr = [self.tpg, 1, 1];

        self.runner.flush_slc();
        let samples = self.runner.measure(&compiled, &buf_refs, tgs, tpg_arr, warmup, iters);
        median_us(samples).map(|us| (us, Some(reflection)))
    }
}

/// Either measurer impl, picked per dispatch type at construction.
/// Holds the buffers + dispatch geometry the measurer needs. Add a
/// new arm here when wiring more `BenchDispatch` kinds under
/// `--measure`.
pub(crate) enum AnyMeasurer<'a> {
    Generic(GpuMeasurer<'a>),
    SdpaVector(SdpaVectorMeasurer<'a>),
    SdpaPrefill(SdpaPrefillMeasurer<'a>),
}

impl AnyMeasurer<'_> {
    pub(crate) fn as_dyn(&self) -> &dyn CandidateMeasurer {
        match self {
            AnyMeasurer::Generic(m) => m,
            AnyMeasurer::SdpaVector(m) => m,
            AnyMeasurer::SdpaPrefill(m) => m,
        }
    }
}

/// Compile + dispatch the kernel under `cfg` and return median
/// elapsed_us across the budget's iters after its warmup. Only handles
/// `BenchDispatch::Generic` shape recipes; other dispatch arms have
/// per-kernel buffer setup and need follow-up wiring.
///
/// `bufs` is the pre-allocated buffer set for `shape` — held across
/// every config in the sweep so we don't pay the round-trip-to-VRAM
/// cost between candidates.
#[allow(clippy::too_many_arguments)]
fn measure_generic(
    runner: &GpuRunner,
    spec: &BenchSpec,
    dt: DType,
    mode: KernelMode,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    cfg: &TuneConfig,
    bufs: &[GpuBuffer],
    budget: BenchBudget,
) -> Result<(f64, Option<PsoReflection>), String> {
    let (warmup, iters) = budget.iters();

    let mut k = (spec.kernel_ir)(dt);
    k.mode = mode;

    let msl_cfg = MslConfig { expected_tpg: Some(cfg.threads.0), ..MslConfig::default() };
    let generator = MslGenerator::new(msl_cfg).with_schedule_override(tune_to_schedule(cfg));
    let msl = generator.generate(&k).map_err(|e| format!("MSL gen: {e}"))?;
    let compiled = runner.compile(&msl, spec.kernel_name).map_err(|e| format!("compile: {e}"))?;
    let reflection = runner.pso_reflection(&compiled);

    let buf_refs: Vec<&GpuBuffer> = bufs.iter().collect();

    let tpg_x = cfg.threads.0 as usize;
    let tgs = shape.grid.eval(n, b, tpg_x);
    let tpg = [tpg_x, cfg.threads.1 as usize, cfg.threads.2 as usize];

    // Flush SLC before each kernel's bench burst so DVFS stays at peak
    // — same hygiene `tile bench` uses.
    runner.flush_slc();
    let samples = runner.measure(&compiled, &buf_refs, tgs, tpg, warmup, iters);
    median_us(samples).map(|us| (us, Some(reflection)))
}

fn build_generic_buffers(
    runner: &GpuRunner,
    shape: &ShapeSpec,
    n: usize,
    b: usize,
    dt: DType,
) -> Vec<GpuBuffer> {
    let mut out: Vec<GpuBuffer> =
        Vec::with_capacity(shape.tensor_bufs.len() + shape.scalar_bufs.len());
    for buf_spec in shape.tensor_bufs {
        let count = buf_spec.count.resolve(n, b);
        let param_dt = buf_spec.dtype_override.unwrap_or(dt);
        // Honor the spec's BufInit — the authors picked these patterns
        // because zero-paths legitimately hit short-circuits (fmul-by-0,
        // softmax(0), NaN-guards-never-trigger) that bias the bench
        // away from what the kernel actually does in production.
        // `BufInit::Zeros` still flows through this path cheaply.
        let init_data = buf_spec.init.generate(count);
        out.push(buffer_typed(runner, &init_data, param_dt));
    }
    for &sb in shape.scalar_bufs {
        out.push(match sb {
            ScalarBufSpec::U32N => runner.buffer_u32(n as u32),
            ScalarBufSpec::U32B => runner.buffer_u32(b as u32),
            ScalarBufSpec::U64N => runner.buffer_u64(n as u64),
            ScalarBufSpec::U64B => runner.buffer_u64(b as u64),
            ScalarBufSpec::I64B => runner.buffer_i64(b as i64),
        });
    }
    let _ = zeros_typed; // re-exported for symmetry with run_spec; not needed here
    out
}
