//! `BenchSpec::run()` — executes one benchmark (correctness + perf) for a given dtype.
//!
//! All the boilerplate that used to live in individual op files is consolidated here.
//! Each `BenchClass` variant maps to a private `run_*` method that knows the
//! dispatch geometry, buffer layout, and reference kernel calling convention.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::ir::KernelMode;

use crate::{
    ops::{
        DType,
        DtypeCtx,
        OpBench,
        OpResult,
        bench_gbps,
        buffer_typed,
        check_equiv,
        quantize_roundtrip,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
    spec::{
        ALL_REDUCE_N,
        ALL_REDUCE_N_CHECK,
        ALL_REDUCE_TPG,
        BINARY_TPG,
        BenchClass,
        BenchSpec,
        ELEMENTWISE_N_BENCH,
        ELEMENTWISE_N_CHECK,
        ELEMENTWISE_TPG,
        InputGen,
        ROW_REDUCE_CHECK_B,
        ROW_REDUCE_CHECK_N,
        ROW_REDUCE_TPG,
    },
};

impl BenchSpec {
    /// Run correctness check + performance measurement for `dt`.
    /// Returns `None` only if the kernel failed to compile (expected for NYI ops).
    pub fn run(&self, runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
        let bench = OpBench::new(self.op, "GB/s");
        match &self.class {
            BenchClass::Unary { cpu, inputs, mlx_src, mlx_pattern } =>
                self.run_unary(runner, dt, &bench, *cpu, *inputs, mlx_src, mlx_pattern),
            BenchClass::Binary {
                cpu,
                inputs_a,
                inputs_b,
                ref_n_per_thread,
                mlx_src,
                mlx_pattern,
            } => self.run_binary(
                runner,
                dt,
                &bench,
                *cpu,
                *inputs_a,
                *inputs_b,
                *ref_n_per_thread,
                mlx_src,
                mlx_pattern,
            ),
            BenchClass::AllReduce { cpu, mlx_src, mlx_pattern } =>
                self.run_all_reduce(runner, dt, &bench, *cpu, mlx_src, mlx_pattern),
            BenchClass::RowReduce { shapes, cpu, mlx_src, mlx_pattern } =>
                self.run_row_reduce(runner, dt, &bench, shapes, *cpu, mlx_src, mlx_pattern),
        }
    }

    // ── MSL generation helpers ────────────────────────────────────────────────

    fn msl_elementwise(&self, dt: DType) -> Option<String> {
        MslGenerator::default().generate(&(self.kernel_ir)(dt)).ok()
    }

    fn msl_reduction(&self, dt: DType) -> Option<String> {
        let mut k = (self.kernel_ir)(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).ok()
    }

    fn mlx_name(pat: &str, tn: &str) -> String { pat.replace("{tn}", tn) }

    // ── Unary ─────────────────────────────────────────────────────────────────

    fn run_unary(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        cpu: fn(f32) -> f32,
        inputs: InputGen,
        mlx_src: &Option<&'static str>,
        mlx_pattern: &Option<&'static str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [ELEMENTWISE_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 2) as f64; // 1 read + 1 write

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match runner.compile(&msl, self.kernel_name).ok() {
            Some(k) => k,
            None => return vec![],
        };

        // Correctness: quantize inputs so CPU ref matches GPU arithmetic
        let nc = ELEMENTWISE_N_CHECK;
        let check_in = inputs.generate(nc);
        let check_q = quantize_roundtrip(&check_in, dt);
        let cpu_ref: Vec<f32> = check_q.iter().copied().map(cpu).collect();
        let in_buf = buffer_typed(runner, &check_in, dt);
        let out_buf = zeros_typed(runner, nc, dt);
        let mt_vals = run_typed_once(
            runner,
            &mk,
            &[&in_buf, &out_buf],
            &out_buf,
            nc,
            [nc.div_ceil(ELEMENTWISE_TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(&cpu_ref, &mt_vals, self.tol);

        // MT perf
        let inp = buffer_typed(runner, &InputGen::Half.generate(nb), dt);
        let out_mt = zeros_typed(runner, nb, dt);
        let tgs = [nb.div_ceil(ELEMENTWISE_TPG), 1, 1];
        let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt], tgs, tpg, bytes);

        // MLX ref perf
        let ref_perf = mlx_src.zip(*mlx_pattern).and_then(|(src, pat)| {
            let rk = runner.compile(src, &Self::mlx_name(pat, ctx.tn)).ok()?;
            let out_ref = zeros_typed(runner, nb, dt);
            let sz = runner.buffer_u32(nb as u32);
            bench_gbps(runner, &rk, &[&inp, &out_ref, &sz], tgs, tpg, bytes)
        });

        vec![bench.result_sub(
            Some(self.subop),
            format!("N={nb} {}", ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── Binary ────────────────────────────────────────────────────────────────

    fn run_binary(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        cpu: fn(f32, f32) -> f32,
        inputs_a: InputGen,
        inputs_b: InputGen,
        ref_n_per_thread: usize,
        mlx_src: &Option<&'static str>,
        mlx_pattern: &Option<&'static str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [BINARY_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 3) as f64; // 2 reads + 1 write

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match runner.compile(&msl, self.kernel_name).ok() {
            Some(k) => k,
            None => return vec![],
        };

        // Correctness
        let nc = ELEMENTWISE_N_CHECK;
        let a_in = inputs_a.generate(nc);
        let b_in = inputs_b.generate(nc);
        let a_q = quantize_roundtrip(&a_in, dt);
        let b_q = quantize_roundtrip(&b_in, dt);
        let cpu_ref: Vec<f32> = a_q.iter().zip(&b_q).map(|(&a, &b)| cpu(a, b)).collect();
        let a_buf = buffer_typed(runner, &a_in, dt);
        let b_buf = buffer_typed(runner, &b_in, dt);
        let o_buf = zeros_typed(runner, nc, dt);
        let n_buf = runner.buffer_u32(nc as u32);
        let mt_vals = run_typed_once(
            runner,
            &mk,
            &[&a_buf, &b_buf, &o_buf, &n_buf],
            &o_buf,
            nc,
            [nc.div_ceil(BINARY_TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(&cpu_ref, &mt_vals, self.tol);

        // Perf buffers (large)
        let a_lb = buffer_typed(runner, &inputs_a.generate(nb), dt);
        let b_lb = buffer_typed(runner, &inputs_b.generate(nb), dt);

        // MT perf
        let out_mt = zeros_typed(runner, nb, dt);
        let n_perf = runner.buffer_u32(nb as u32);
        let mt_perf = bench_gbps(
            runner,
            &mk,
            &[&a_lb, &b_lb, &out_mt, &n_perf],
            [nb.div_ceil(BINARY_TPG), 1, 1],
            tpg,
            bytes,
        );

        // MLX ref perf — uses N_PER_THREAD=2 so grid = N/(npt*tpg)
        let ref_perf = mlx_src.zip(*mlx_pattern).and_then(|(src, pat)| {
            let rk = runner.compile(src, &Self::mlx_name(pat, ctx.tn)).ok()?;
            let out_ref = zeros_typed(runner, nb, dt);
            let sz = runner.buffer_u32(nb as u32);
            bench_gbps(
                runner,
                &rk,
                &[&a_lb, &b_lb, &out_ref, &sz],
                [nb / (ref_n_per_thread * BINARY_TPG), 1, 1],
                tpg,
                bytes,
            )
        });

        vec![bench.result_sub(
            Some(self.subop),
            format!("N={nb} {}", ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── AllReduce ─────────────────────────────────────────────────────────────

    fn run_all_reduce(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        cpu: fn(&[f32]) -> f32,
        mlx_src: &Option<&'static str>,
        mlx_pattern: &Option<&'static str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::reduce(dt);
        let tpg = [ALL_REDUCE_TPG, 1, 1];
        let nb = ALL_REDUCE_N;
        let bytes = (nb * ctx.eb) as f64;

        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match runner.compile(&msl, self.kernel_name).ok() {
            Some(k) => k,
            None => return vec![],
        };

        // Correctness
        let nc = ALL_REDUCE_N_CHECK;
        let inp_vals: Vec<f32> = (0..nc).map(|i| 0.25 + (i % 19) as f32 * 0.03125).collect();
        let inp_chk = buffer_typed(runner, &inp_vals, dt);
        let mt_ns = runner.buffer_u32(nc as u32);
        let chk_out = zeros_typed(runner, 1, dt);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&inp_chk, &chk_out, &mt_ns],
            &chk_out,
            1,
            [1, 1, 1],
            tpg,
            dt,
        );
        // Compare vs MLX ref or CPU fallback
        let ref_chk: Vec<f32>;
        let ref_kernel_chk;
        let equiv = if let Some((src, pat)) = mlx_src.zip(*mlx_pattern) {
            let fn_name = Self::mlx_name(pat, ctx.tn);
            if let Some(rk) = runner.compile(src, &fn_name).ok() {
                let ri_sz = runner.buffer_u64(nc as u64);
                let rr_sz = runner.buffer_u64(nc as u64);
                let rk_out = zeros_typed(runner, 1, dt);
                ref_kernel_chk = Some(run_typed_once(
                    runner,
                    &rk,
                    &[&inp_chk, &rk_out, &ri_sz, &rr_sz],
                    &rk_out,
                    1,
                    [1, 1, 1],
                    tpg,
                    dt,
                ));
                check_equiv(ref_kernel_chk.as_ref().unwrap(), &mt_chk, self.tol)
            } else {
                ref_chk = vec![cpu(&inp_vals)];
                check_equiv(&ref_chk, &mt_chk, self.tol)
            }
        } else {
            ref_chk = vec![cpu(&inp_vals)];
            check_equiv(&ref_chk, &mt_chk, self.tol)
        };

        // Perf
        let inp = buffer_typed(runner, &vec![1.0f32 / nb as f32; nb], dt);
        let ns = runner.buffer_u32(nb as u32);
        let out_mt = zeros_typed(runner, 1, dt);
        let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt, &ns], [1, 1, 1], tpg, bytes);

        let ref_perf = mlx_src.zip(*mlx_pattern).and_then(|(src, pat)| {
            let rk = runner.compile(src, &Self::mlx_name(pat, ctx.tn)).ok()?;
            let ri_sz = runner.buffer_u64(nb as u64);
            let rr_sz = runner.buffer_u64(nb as u64);
            let out_ref = zeros_typed(runner, 1, dt);
            bench_gbps(runner, &rk, &[&inp, &out_ref, &ri_sz, &rr_sz], [1, 1, 1], tpg, bytes)
        });

        vec![bench.result_sub(
            Some(self.subop),
            format!("N={}M {}", nb / 1_000_000, ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── RowReduce ─────────────────────────────────────────────────────────────

    fn run_row_reduce(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        cpu: fn(&[f32]) -> f32,
        mlx_src: &Option<&'static str>,
        mlx_pattern: &Option<&'static str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::reduce(dt);
        let tpg = [ROW_REDUCE_TPG, 1, 1];

        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match runner.compile(&msl, self.kernel_name).ok() {
            Some(k) => k,
            None => return vec![],
        };

        let ref_kernel = mlx_src
            .zip(*mlx_pattern)
            .and_then(|(src, pat)| runner.compile(src, &Self::mlx_name(pat, ctx.tn)).ok());

        let mut results = Vec::new();

        for &(b, n) in shapes {
            // Correctness
            let cb = ROW_REDUCE_CHECK_B;
            let cn = ROW_REDUCE_CHECK_N;
            let inp_vals: Vec<f32> = (0..cb * cn)
                .map(|i| 0.25 + (i / cn) as f32 * 0.0625 + (i % 13) as f32 * 0.03125)
                .collect();
            let inp_chk = buffer_typed(runner, &inp_vals, dt);
            let mt_ns = runner.buffer_u32(cn as u32);
            let chk_out = zeros_typed(runner, cb, dt);
            let mt_chk = run_typed_once(
                runner,
                &mk,
                &[&inp_chk, &chk_out, &mt_ns],
                &chk_out,
                cb,
                [cb, 1, 1],
                tpg,
                dt,
            );

            let equiv = if let Some(rk) = &ref_kernel {
                let ref_red = runner.buffer_u64(cn as u64);
                let ref_osz = runner.buffer_i64(cb as i64);
                let rk_out = zeros_typed(runner, cb, dt);
                let rk_chk = run_typed_once(
                    runner,
                    rk,
                    &[&inp_chk, &rk_out, &ref_red, &ref_osz],
                    &rk_out,
                    cb,
                    [1, cb, 1],
                    tpg,
                    dt,
                );
                check_equiv(&rk_chk, &mt_chk, self.tol)
            } else {
                let cpu_ref: Vec<f32> =
                    (0..cb).map(|row| cpu(&inp_vals[row * cn..(row + 1) * cn])).collect();
                check_equiv(&cpu_ref, &mt_chk, self.tol)
            };

            // Perf
            let inp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
            let bytes = (b * n * ctx.eb) as f64;
            let ns = runner.buffer_u32(n as u32);
            let out_mt = zeros_typed(runner, b, dt);
            let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt, &ns], [b, 1, 1], tpg, bytes);

            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let ref_red = runner.buffer_u64(n as u64);
                let ref_osz = runner.buffer_i64(b as i64);
                let out_ref = zeros_typed(runner, b, dt);
                bench_gbps(runner, rk, &[&inp, &out_ref, &ref_red, &ref_osz], [1, b, 1], tpg, bytes)
            });

            results.push(bench.result_sub(
                Some(self.subop),
                format!("B={b} N={n} {}", ctx.label),
                ref_perf,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }
}
