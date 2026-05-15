//! `BenchSpec::run()` — executes one benchmark (correctness + perf) for a given dtype.
//!
//! All boilerplate consolidated here. Correctness via the CPU interpreter
//! (metaltile_interp), perf via GPU.

use std::collections::BTreeMap;

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{constexpr::ConstExprValues, ir::KernelMode};
use metaltile_interp::{Interpreter, TensorData};

use crate::{
    ops::{
        DType,
        DtypeCtx,
        EquivResult,
        EquivTolerance,
        OpBench,
        OpResult,
        bench_gbps,
        buffer_typed,
        check_equiv,
        check_equiv_with,
        quantize_roundtrip,
        run_typed_once,
        zeros_typed,
    },
    runner::{GpuBuffer, GpuRunner},
    spec::{
        ALL_REDUCE_N,
        ALL_REDUCE_N_CHECK,
        ALL_REDUCE_TPG,
        ARANGE_N,
        ARANGE_N_CHECK,
        ARANGE_TPG,
        BINARY_TPG,
        BINARY_TWO_TPG,
        BenchClass,
        BenchSpec,
        ELEMENTWISE_N_BENCH,
        ELEMENTWISE_N_CHECK,
        ELEMENTWISE_TPG,
        ExtraInput,
        InputGen,
        ROW_REDUCE_CHECK_B,
        ROW_REDUCE_CHECK_N,
        ROW_REDUCE_TPG,
        SELECT_TPG,
    },
};

impl BenchSpec {
    pub fn run(&self, runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
        let bench = OpBench::new(self.op, "GB/s");
        match &self.class {
            BenchClass::Unary { inputs, mlx_src, mlx_pattern } =>
                self.run_unary(runner, dt, &bench, *inputs, mlx_src, mlx_pattern),
            BenchClass::Binary { inputs_a, inputs_b, ref_n_per_thread, mlx_src, mlx_pattern } =>
                self.run_binary(
                    runner,
                    dt,
                    &bench,
                    *inputs_a,
                    *inputs_b,
                    *ref_n_per_thread,
                    mlx_src,
                    mlx_pattern,
                ),
            BenchClass::AllReduce { mlx_src, mlx_pattern } =>
                self.run_all_reduce(runner, dt, &bench, mlx_src, mlx_pattern),
            BenchClass::RowReduce { shapes, mlx_src, mlx_pattern } =>
                self.run_row_reduce(runner, dt, &bench, shapes, mlx_src, mlx_pattern),
            BenchClass::Arange { start, step, mlx_src, mlx_pattern } =>
                self.run_arange(runner, dt, &bench, *start, *step, mlx_src, mlx_pattern),
            BenchClass::BinaryTwo { inputs_a, inputs_b } =>
                self.run_binary_two(runner, dt, &bench, *inputs_a, *inputs_b),
            BenchClass::Select { mlx_src, mlx_pattern } =>
                self.run_select(runner, dt, &bench, mlx_src, mlx_pattern),
            BenchClass::RowNorm {
                shapes,
                tpg,
                reads,
                out_elements,
                extra,
                mlx_src,
                mlx_pattern,
                mlx_extra_slots,
            } => self.run_row_norm(
                runner,
                dt,
                &bench,
                shapes,
                *tpg,
                *reads,
                *out_elements,
                extra,
                mlx_src,
                mlx_pattern,
                *mlx_extra_slots,
            ),
            BenchClass::Sort { b, n, tpg, mlx_src, mlx_pattern } =>
                self.run_sort(runner, dt, &bench, *b, *n, *tpg, mlx_src, mlx_pattern),
            BenchClass::Scan { shapes, tpg, mlx_src, mlx_pattern } =>
                self.run_scan(runner, dt, &bench, shapes, *tpg, mlx_src, mlx_pattern),
            BenchClass::ArgReduce { n, check_n, tpg, mlx_src, mlx_pattern } =>
                self.run_arg_reduce(runner, dt, &bench, *n, *check_n, *tpg, mlx_src, mlx_pattern),
            BenchClass::Random { n, tpg, mlx_src, mlx_pattern } =>
                self.run_random(runner, dt, &bench, *n, *tpg, mlx_src, mlx_pattern),
            BenchClass::FpQuantized { n, tpg, mlx_src, mlx_pattern } =>
                self.run_fp_quantized(runner, dt, &bench, *n, *tpg, mlx_src, mlx_pattern),
            BenchClass::MatVec { shapes, tpg, mlx_src, mlx_pattern } =>
                self.run_mat_vec(runner, dt, &bench, shapes, *tpg, mlx_src, mlx_pattern),
            BenchClass::MatVecMasked { shapes, tpg } =>
                self.run_mat_vec_masked(runner, dt, &bench, shapes, *tpg),
            BenchClass::QuantizedMatVec { shapes, group_size, tpg, mlx_src, mlx_pattern } => self
                .run_quantized_mat_vec(
                    runner,
                    dt,
                    &bench,
                    shapes,
                    *group_size,
                    *tpg,
                    mlx_src,
                    mlx_pattern,
                ),
            BenchClass::Rope { b, h, l, d, n_per_group, mlx_src } =>
                self.run_rope(runner, dt, &bench, *b, *h, *l, *d, *n_per_group, mlx_src),
            BenchClass::Attention { shapes, tpg, mlx_src } =>
                self.run_attention(runner, dt, &bench, shapes, *tpg, mlx_src),
            BenchClass::StridedCopy { m, n, pad, mlx_src, mlx_pattern } =>
                self.run_strided_copy(runner, dt, &bench, *m, *n, *pad, mlx_src, mlx_pattern),
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn msl_elementwise(&self, dt: DType) -> Option<String> {
        MslGenerator::default().generate(&(self.kernel_ir)(dt)).ok()
    }
    fn msl_reduction(&self, dt: DType) -> Option<String> {
        let mut k = (self.kernel_ir)(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).ok()
    }
    fn msl_grid3d(&self, dt: DType) -> Option<String> {
        let mut k = (self.kernel_ir)(dt);
        k.mode = KernelMode::Grid3D;
        MslGenerator::default().generate(&k).ok()
    }
    fn mlx_name(pat: &str, tn: &str) -> String { pat.replace("{tn}", tn) }
    fn compile_mt(
        runner: &GpuRunner,
        msl: &str,
        name: &str,
    ) -> Option<crate::runner::CompiledKernel> {
        runner.compile(msl, name).ok()
    }
    fn compile_mlx(
        runner: &GpuRunner,
        src: &Option<&str>,
        pat: &Option<&str>,
        tn: &str,
    ) -> Option<crate::runner::CompiledKernel> {
        let src = (*src)?;
        let pat = (*pat)?;
        runner.compile(src, &Self::mlx_name(pat, tn)).ok()
    }
    fn td(dt: DType, shape: &[usize], data: &[f32]) -> TensorData {
        let mut td = TensorData::zeros(shape, dt);
        for (i, &v) in data.iter().enumerate() {
            td.write_scalar(i, v as f64);
        }
        td
    }
    fn constexprs(vals: &[(&str, usize)]) -> ConstExprValues {
        let mut cv = ConstExprValues::new();
        for (k, v) in vals {
            cv.insert(k.to_string(), *v);
        }
        cv
    }
    fn interp(
        kernel: &metaltile_core::ir::Kernel,
        inputs: BTreeMap<String, TensorData>,
        cv: ConstExprValues,
        mode: InterpMode,
    ) -> Option<BTreeMap<String, Vec<f32>>> {
        let mut interp = Interpreter::new(inputs, cv);
        let result = match mode {
            InterpMode::Elementwise(n) => interp.run_grid(kernel, n),
            InterpMode::Reduction(rows) => interp.run_grid_reduction(kernel, rows),
            InterpMode::Grid3D(x, y, z) => interp.run_grid_3d(kernel, x, y, z),
        }
        .ok()?;
        let mut out = BTreeMap::new();
        for (name, td) in &result.outputs {
            out.insert(
                name.clone(),
                (0..td.num_elements()).map(|i| td.read_scalar(i) as f32).collect(),
            );
        }
        Some(out)
    }

    // ── Unary ─────────────────────────────────────────────────────────────────

    fn run_unary(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        inputs: InputGen,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [ELEMENTWISE_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 2) as f64;

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        // Correctness via interpreter
        let nc = ELEMENTWISE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let check_in = inputs.generate(nc);
        let inp_name = match kernel.params.iter().find(|p| !p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let out_name = match kernel.params.iter().find(|p| p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let cv = Self::constexprs(&[]);
        let mut inp_map = BTreeMap::new();
        inp_map.insert(inp_name.clone(), Self::td(dt, &[nc], &check_in));
        inp_map.insert(out_name.clone(), TensorData::zeros(&[nc], dt));
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Elementwise(nc)) {
            Some(o) => o,
            None => return vec![],
        };
        let interp_vals = match interp_out.get(&out_name) {
            Some(v) => v,
            None => return vec![],
        };

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
        let equiv = check_equiv(interp_vals, &mt_vals, self.tol);

        let inp = buffer_typed(runner, &InputGen::Half.generate(nb), dt);
        let out_mt = zeros_typed(runner, nb, dt);
        let tgs = [nb.div_ceil(ELEMENTWISE_TPG), 1, 1];
        let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt], tgs, tpg, bytes);
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn).and_then(|rk| {
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
        inputs_a: InputGen,
        inputs_b: InputGen,
        ref_n_per_thread: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [BINARY_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 3) as f64;

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let nc = ELEMENTWISE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let a_in = inputs_a.generate(nc);
        let b_in = inputs_b.generate(nc);
        let inp_params: Vec<_> = kernel.params.iter().filter(|p| !p.is_output).collect();
        let out_name = match kernel.params.iter().find(|p| p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let cv = Self::constexprs(&[]);
        let mut inp_map = BTreeMap::new();
        if inp_params.len() >= 2 {
            inp_map.insert(inp_params[0].name.clone(), Self::td(dt, &[nc], &a_in));
            inp_map.insert(inp_params[1].name.clone(), Self::td(dt, &[nc], &b_in));
        }
        inp_map.insert(out_name.clone(), TensorData::zeros(&[nc], dt));
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Elementwise(nc)) {
            Some(o) => o,
            None => return vec![],
        };
        let interp_vals = match interp_out.get(&out_name) {
            Some(v) => v,
            None => return vec![],
        };

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
        let equiv = check_equiv(interp_vals, &mt_vals, self.tol);

        let a_lb = buffer_typed(runner, &inputs_a.generate(nb), dt);
        let b_lb = buffer_typed(runner, &inputs_b.generate(nb), dt);
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
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn).and_then(|rk| {
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
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::reduce(dt);
        let tpg = [ALL_REDUCE_TPG, 1, 1];
        let nb = ALL_REDUCE_N;
        let bytes = (nb * ctx.eb) as f64;

        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let nc = ALL_REDUCE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let inp_vals: Vec<f32> = (0..nc).map(|i| 0.25 + (i % 19) as f32 * 0.03125).collect();
        let inp_name = match kernel.params.iter().find(|p| !p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let out_name = match kernel.params.iter().find(|p| p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let cv = Self::constexprs(&[("n", nc)]);
        let mut inp_map = BTreeMap::new();
        inp_map.insert(inp_name.clone(), Self::td(dt, &[nc], &inp_vals));
        inp_map.insert(out_name.clone(), TensorData::zeros(&[1], dt));
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Reduction(1)) {
            Some(o) => o,
            None => return vec![],
        };
        let interp_vals = match interp_out.get(&out_name) {
            Some(v) => v,
            None => return vec![],
        };

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
        let equiv = check_equiv(interp_vals, &mt_chk, self.tol);

        let inp = buffer_typed(runner, &vec![1.0f32 / nb as f32; nb], dt);
        let ns = runner.buffer_u32(nb as u32);
        let out_mt = zeros_typed(runner, 1, dt);
        let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt, &ns], [1, 1, 1], tpg, bytes);
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn).and_then(|rk| {
            let ri = runner.buffer_u64(nb as u64);
            let rr = runner.buffer_u64(nb as u64);
            let out = zeros_typed(runner, 1, dt);
            bench_gbps(runner, &rk, &[&inp, &out, &ri, &rr], [1, 1, 1], tpg, bytes)
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
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::reduce(dt);
        let tpg = [ROW_REDUCE_TPG, 1, 1];
        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn);
        let mut results = Vec::new();
        for &(b, n) in shapes {
            let cb = ROW_REDUCE_CHECK_B;
            let cn = ROW_REDUCE_CHECK_N;
            let kernel = (self.kernel_ir)(dt);
            let inp_vals: Vec<f32> = (0..cb * cn)
                .map(|i| 0.25 + (i / cn) as f32 * 0.0625 + (i % 13) as f32 * 0.03125)
                .collect();
            let inp_name = match kernel.params.iter().find(|p| !p.is_output) {
                Some(p) => p.name.clone(),
                None => return vec![],
            };
            let out_name = match kernel.params.iter().find(|p| p.is_output) {
                Some(p) => p.name.clone(),
                None => return vec![],
            };
            let cv = Self::constexprs(&[("n", cn)]);
            let mut inp_map = BTreeMap::new();
            inp_map.insert(inp_name.clone(), Self::td(dt, &[cb * cn], &inp_vals));
            inp_map.insert(out_name.clone(), TensorData::zeros(&[cb], dt));
            let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Reduction(cb)) {
                Some(o) => o,
                None => return vec![],
            };
            let interp_vals = match interp_out.get(&out_name) {
                Some(v) => v,
                None => return vec![],
            };

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
            let equiv = check_equiv(interp_vals, &mt_chk, self.tol);

            let inp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
            let bytes = (b * n * ctx.eb) as f64;
            let ns = runner.buffer_u32(n as u32);
            let out_mt = zeros_typed(runner, b, dt);
            let mt_perf = bench_gbps(runner, &mk, &[&inp, &out_mt, &ns], [b, 1, 1], tpg, bytes);
            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let rr = runner.buffer_u64(n as u64);
                let ro = runner.buffer_i64(b as i64);
                let out = zeros_typed(runner, b, dt);
                bench_gbps(runner, rk, &[&inp, &out, &rr, &ro], [1, b, 1], tpg, bytes)
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

    // ── Arange ────────────────────────────────────────────────────────────────

    fn run_arange(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        start: f32,
        step: f32,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [ARANGE_TPG, 1, 1];
        let nb = ARANGE_N;
        let bytes = (nb * ctx.eb) as f64;

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let nc = ARANGE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let out_name = match kernel.params.iter().find(|p| p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let start_name = kernel
            .params
            .iter()
            .find(|p| !p.is_output && p.name.contains("start"))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "start".into());
        let step_name = kernel
            .params
            .iter()
            .find(|p| !p.is_output && p.name.contains("step"))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "step".into());
        let cv = Self::constexprs(&[("n", nc)]);
        let mut inp_map = BTreeMap::new();
        inp_map.insert(start_name.clone(), Self::td(dt, &[1], &[start]));
        if step_name != start_name {
            // avoid duplicate key (arange kernel may have start==step tensor name? unlikely)
            inp_map.entry(step_name).or_insert_with(|| Self::td(dt, &[1], &[step]));
        }
        inp_map.insert(out_name.clone(), TensorData::zeros(&[nc], dt));
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Elementwise(nc)) {
            Some(o) => o,
            None => return vec![],
        };
        let interp_vals = match interp_out.get(&out_name) {
            Some(v) => v,
            None => return vec![],
        };

        let s_buf = buffer_typed(runner, &[start], dt);
        let st_buf = buffer_typed(runner, &[step], dt);
        let out_buf = zeros_typed(runner, nc, dt);
        let n_buf = runner.buffer_u32(nc as u32);
        let mt_vals = run_typed_once(
            runner,
            &mk,
            &[&out_buf, &s_buf, &st_buf, &n_buf],
            &out_buf,
            nc,
            [nc.div_ceil(ARANGE_TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(interp_vals, &mt_vals, self.tol);

        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn).and_then(|rk| {
            let rs = runner.buffer_f32_scalar(start);
            let rst = runner.buffer_f32_scalar(step);
            let ro = runner.buffer_zeros(nb * 4);
            bench_gbps(runner, &rk, &[&rs, &rst, &ro], [nb.div_ceil(ARANGE_TPG), 1, 1], tpg, bytes)
        });
        let mt_start = buffer_typed(runner, &[start], dt);
        let mt_step = buffer_typed(runner, &[step], dt);
        let mt_out = zeros_typed(runner, nb, dt);
        let mt_n = runner.buffer_u32(nb as u32);
        let mt_perf = bench_gbps(
            runner,
            &mk,
            &[&mt_out, &mt_start, &mt_step, &mt_n],
            [nb.div_ceil(ARANGE_TPG), 1, 1],
            tpg,
            bytes,
        );
        vec![bench.result_sub(
            Some(self.subop),
            format!("N={} {}", nb, ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── BinaryTwo ─────────────────────────────────────────────────────────────

    fn run_binary_two(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        inputs_a: InputGen,
        inputs_b: InputGen,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [BINARY_TWO_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 4) as f64;

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let nc = ELEMENTWISE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let a_f32 = inputs_a.generate(nc);
        let b_f32 = inputs_b.generate(nc);
        let inp_params: Vec<_> = kernel.params.iter().filter(|p| !p.is_output).collect();
        let out_params: Vec<_> = kernel.params.iter().filter(|p| p.is_output).collect();
        let cv = Self::constexprs(&[]);
        let mut inp_map = BTreeMap::new();
        if inp_params.len() >= 2 {
            inp_map.insert(inp_params[0].name.clone(), Self::td(dt, &[nc], &a_f32));
            inp_map.insert(inp_params[1].name.clone(), Self::td(dt, &[nc], &b_f32));
        }
        for op in &out_params {
            inp_map.insert(op.name.clone(), TensorData::zeros(&[nc], dt));
        }
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Elementwise(nc)) {
            Some(o) => o,
            None => return vec![],
        };

        let a_buf = buffer_typed(runner, &a_f32, dt);
        let b_buf = buffer_typed(runner, &b_f32, dt);
        let c_buf = zeros_typed(runner, nc, dt);
        let d_buf = zeros_typed(runner, nc, dt);
        let mt_c = run_typed_once(
            runner,
            &mk,
            &[&a_buf, &b_buf, &c_buf, &d_buf],
            &c_buf,
            nc,
            [nc.div_ceil(BINARY_TWO_TPG), 1, 1],
            tpg,
            dt,
        );
        let mt_d = run_typed_once(
            runner,
            &mk,
            &[&a_buf, &b_buf, &c_buf, &d_buf],
            &d_buf,
            nc,
            [nc.div_ceil(BINARY_TWO_TPG), 1, 1],
            tpg,
            dt,
        );
        let name0 = out_params.first().map(|p| &p.name);
        let name1 = out_params.get(1).map(|p| &p.name);
        let eq_c = name0.and_then(|n| interp_out.get(n)).map(|v| check_equiv(v, &mt_c, self.tol));
        let eq_d = name1
            .or(name0)
            .and_then(|n| interp_out.get(n))
            .map(|v| check_equiv(v, &mt_d, self.tol));
        let equiv = match (eq_c, eq_d) {
            (Some(c), Some(d)) =>
                if c.max_abs_err > d.max_abs_err {
                    c
                } else {
                    d
                },
            (Some(c), _) => c,
            (_, Some(d)) => d,
            _ => return vec![],
        };

        let a_lb = buffer_typed(runner, &inputs_a.generate(nb), dt);
        let b_lb = buffer_typed(runner, &inputs_b.generate(nb), dt);
        let c_lb = zeros_typed(runner, nb, dt);
        let d_lb = zeros_typed(runner, nb, dt);
        let mt_perf = bench_gbps(
            runner,
            &mk,
            &[&a_lb, &b_lb, &c_lb, &d_lb],
            [nb.div_ceil(BINARY_TWO_TPG), 1, 1],
            tpg,
            bytes,
        );
        vec![bench.result_sub(
            Some(self.subop),
            format!("N={} {}", nb, ctx.label),
            None,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── Select ────────────────────────────────────────────────────────────────

    fn run_select(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tpg = [SELECT_TPG, 1, 1];
        let nb = ELEMENTWISE_N_BENCH;
        let bytes = (nb * ctx.eb * 4) as f64;

        let msl = match self.msl_elementwise(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let nc = ELEMENTWISE_N_CHECK;
        let kernel = (self.kernel_ir)(dt);
        let cond_f32: Vec<f32> = (0..nc).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
        let true_f32: Vec<f32> = (0..nc).map(|i| 1.0 + i as f32 * 0.01).collect();
        let false_f32: Vec<f32> = (0..nc).map(|i| -2.0 - i as f32 * 0.02).collect();
        let inp_params: Vec<_> = kernel.params.iter().filter(|p| !p.is_output).collect();
        let out_name = match kernel.params.iter().find(|p| p.is_output) {
            Some(p) => p.name.clone(),
            None => return vec![],
        };
        let cv = Self::constexprs(&[]);
        let mut inp_map = BTreeMap::new();
        if inp_params.len() >= 3 {
            inp_map.insert(inp_params[0].name.clone(), Self::td(dt, &[nc], &cond_f32));
            inp_map.insert(inp_params[1].name.clone(), Self::td(dt, &[nc], &true_f32));
            inp_map.insert(inp_params[2].name.clone(), Self::td(dt, &[nc], &false_f32));
        }
        inp_map.insert(out_name.clone(), TensorData::zeros(&[nc], dt));
        let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Elementwise(nc)) {
            Some(o) => o,
            None => return vec![],
        };
        let interp_vals = match interp_out.get(&out_name) {
            Some(v) => v,
            None => return vec![],
        };

        let mt_cond = buffer_typed(runner, &cond_f32, dt);
        let mt_true = buffer_typed(runner, &true_f32, dt);
        let mt_false = buffer_typed(runner, &false_f32, dt);
        let mt_out = zeros_typed(runner, nc, dt);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&mt_cond, &mt_true, &mt_false, &mt_out],
            &mt_out,
            nc,
            [nc.div_ceil(SELECT_TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(interp_vals, &mt_chk, self.tol);

        let cond_bool_perf: Vec<u8> = (0..nb).map(|i| if i % 2 == 0 { 1u8 } else { 0u8 }).collect();
        let true_perf = buffer_typed(runner, &vec![1.0f32; nb], dt);
        let false_perf = buffer_typed(runner, &vec![-1.0f32; nb], dt);
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn).and_then(|rk| {
            let rc = runner.buffer_bytes(&cond_bool_perf);
            let rs = runner.buffer_u32(nb as u32);
            let out = zeros_typed(runner, nb, dt);
            bench_gbps(
                runner,
                &rk,
                &[&rc, &true_perf, &false_perf, &out, &rs],
                [nb.div_ceil(SELECT_TPG), 1, 1],
                tpg,
                bytes,
            )
        });
        let mt_cond_perf = buffer_typed(
            runner,
            &(0..nb).map(|i| if i % 2 == 0 { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>(),
            dt,
        );
        let mt_perf = {
            let out = zeros_typed(runner, nb, dt);
            bench_gbps(
                runner,
                &mk,
                &[&mt_cond_perf, &true_perf, &false_perf, &out],
                [nb.div_ceil(SELECT_TPG), 1, 1],
                tpg,
                bytes,
            )
        };
        vec![bench.result_sub(
            Some(self.subop),
            format!("N={} {}", nb, ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── RowNorm ──────────────────────────────────────────────────────────────

    fn run_row_norm(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        tpg: usize,
        reads: usize,
        out_elements: usize,
        extra: &[ExtraInput],
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
        mlx_extra_slots: usize,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::reduce(dt);
        let tpg_arr = [tpg, 1, 1];
        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn);

        let mut results = Vec::new();
        for &(b, n) in shapes {
            let kernel = (self.kernel_ir)(dt);
            let inp_vals: Vec<f32> = (0..b * n)
                .map(|i| 0.25 + (i / n) as f32 * 0.0625 + (i % 13) as f32 * 0.03125)
                .collect();
            let out_size = b * out_elements;
            let cv = Self::constexprs(&[("n", n)]);
            let inp_name = match kernel.params.iter().find(|p| {
                !p.is_output
                    && !p.name.contains('w')
                    && !p.name.contains('b')
                    && !p.name.contains("eps")
            }) {
                Some(p) => p.name.clone(),
                None => return vec![],
            };
            let out_name = match kernel.params.iter().find(|p| p.is_output) {
                Some(p) => p.name.clone(),
                None => return vec![],
            };
            let mut inp_map = BTreeMap::new();
            inp_map.insert(inp_name.clone(), Self::td(dt, &[b * n], &inp_vals));
            for e in extra {
                match e {
                    ExtraInput::WeightPerCol { val } => {
                        if let Some(wn) = kernel
                            .params
                            .iter()
                            .find(|p| p.name.contains('w'))
                            .map(|p| p.name.clone())
                        {
                            inp_map.entry(wn).or_insert_with(|| Self::td(dt, &[n], &vec![*val; n]));
                        }
                    },
                    ExtraInput::BiasPerCol { val } => {
                        if let Some(bn) = kernel
                            .params
                            .iter()
                            .find(|p| p.name.contains('b') && !p.name.contains("eps"))
                            .map(|p| p.name.clone())
                        {
                            inp_map.entry(bn).or_insert_with(|| Self::td(dt, &[n], &vec![*val; n]));
                        }
                    },
                    ExtraInput::ScalarF32 { val } => {
                        if let Some(en) = kernel
                            .params
                            .iter()
                            .find(|p| p.name.contains("eps"))
                            .map(|p| p.name.clone())
                        {
                            inp_map
                                .entry(en)
                                .or_insert_with(|| Self::td(DType::F32, &[1], &[*val]));
                        }
                    },
                }
            }
            inp_map.entry(out_name.clone()).or_insert_with(|| TensorData::zeros(&[out_size], dt));
            let interp_out = match Self::interp(&kernel, inp_map, cv, InterpMode::Reduction(b)) {
                Some(o) => o,
                None => return vec![],
            };
            let interp_vals = match interp_out.get(&out_name) {
                Some(v) => v,
                None => return vec![],
            };

            let build_extras = |runner: &GpuRunner, n: usize, dt: DType| -> Vec<GpuBuffer> {
                extra
                    .iter()
                    .map(|e| match e {
                        ExtraInput::WeightPerCol { val } =>
                            buffer_typed(runner, &vec![*val; n], dt),
                        ExtraInput::BiasPerCol { val } => buffer_typed(runner, &vec![*val; n], dt),
                        ExtraInput::ScalarF32 { val } => runner.buffer_f32_scalar(*val),
                    })
                    .collect()
            };

            let inp = buffer_typed(runner, &inp_vals, dt);
            let out_mt = zeros_typed(runner, out_size, dt);
            let mt_n = runner.buffer_u32(n as u32);
            let extra_bufs = build_extras(runner, n, dt);
            let mut mt_bufs: Vec<&GpuBuffer> = vec![&inp, &out_mt, &mt_n];
            mt_bufs.extend(extra_bufs.iter());
            let mt_chk =
                run_typed_once(runner, &mk, &mt_bufs, &out_mt, out_size, [b, 1, 1], tpg_arr, dt);
            let equiv = check_equiv(interp_vals, &mt_chk, self.tol);

            let inp_perf = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
            let out_mt_perf = zeros_typed(runner, out_size, dt);
            let extra_perf = build_extras(runner, n, dt);
            let bytes = (b * n * ctx.eb * reads + out_size * ctx.eb) as f64;
            let mt_perf = {
                let mut bufs: Vec<&GpuBuffer> = vec![&inp_perf, &out_mt_perf, &mt_n];
                bufs.extend(extra_perf.iter());
                bench_gbps(runner, &mk, &bufs, [b, 1, 1], tpg_arr, bytes)
            };
            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let out = zeros_typed(runner, out_size, dt);
                let rn = runner.buffer_u64(n as u64);
                let ro = runner.buffer_i64(b as i64);
                let mut bufs: Vec<&GpuBuffer> = vec![&inp_perf, &out, &rn, &ro];
                let dummy: Vec<GpuBuffer> =
                    (0..mlx_extra_slots).map(|_| runner.buffer_u64(0)).collect();
                bufs.extend(dummy.iter());
                bufs.extend(extra_perf.iter());
                bench_gbps(runner, rk, &bufs, [b, 1, 1], tpg_arr, bytes)
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
    // ── Sort ──────────────────────────────────────────────────────────────────

    fn run_sort(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        b: usize,
        n: usize,
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_reduction(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, "float32");

        let check_b = 4usize;
        let check_data: Vec<f32> = (0..check_b * n).map(|i| (check_b * n - i) as f32).collect();
        let ref_out = {
            let mut out = check_data.clone();
            for chunk in out.chunks_mut(n) {
                chunk.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            out
        };
        let inp_c = buffer_typed(runner, &check_data, DType::F32);
        let n_buf_c = runner.buffer_u32(n as u32);
        let out_c = zeros_typed(runner, check_b * n, DType::F32);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&inp_c, &out_c, &n_buf_c],
            &out_c,
            check_b * n,
            [check_b, 1, 1],
            [tpg, 1, 1],
            DType::F32,
        );
        let n_bad = ref_out.iter().zip(&mt_chk).filter(|(a, b)| a != b).count();
        let equiv = EquivResult {
            n_checked: check_b * n,
            max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
            passed: n_bad == 0,
        };

        let data: Vec<f32> = (0..b * n).map(|i| (b * n - i) as f32).collect();
        let inp = buffer_typed(runner, &data, DType::F32);
        let bytes = (b * n * 4 * 2) as f64;
        let n_buf = runner.buffer_u32(n as u32);

        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, b * n, DType::F32);
            let size = runner.buffer_i32(n as i32);
            let stride1 = runner.buffer_i32(1i32);
            let stride_n = runner.buffer_i32(n as i32);
            bench_gbps(
                runner,
                rk,
                &[&inp, &out, &size, &stride1, &stride1, &stride_n, &stride_n],
                [b, 1, 1],
                [tpg, 1, 1],
                bytes,
            )
        });
        let mt_perf = {
            let out = zeros_typed(runner, b * n, DType::F32);
            bench_gbps(runner, &mk, &[&inp, &out, &n_buf], [b, 1, 1], [tpg, 1, 1], bytes)
        };
        vec![bench.result_sub(
            Some(self.subop),
            format!("B={b} N={n} f32"),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── Scan ──────────────────────────────────────────────────────────────────

    fn run_scan(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_reduction(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, "float32");

        let mut results = Vec::new();
        for &(rows, n) in shapes {
            let check_rows = 4usize;
            let check_n = 256usize;
            let inp_vals: Vec<f32> =
                (0..rows * n).map(|i| ((i % 31) as f32 - 15.0) * 0.0625).collect();
            let ref_out: Vec<f32> = {
                let mut out = vec![0.0f32; check_rows * check_n];
                for r in 0..check_rows {
                    let mut acc = 0.0f32;
                    for c in 0..check_n {
                        acc += inp_vals[r * check_n + c];
                        out[r * check_n + c] = acc;
                    }
                }
                out
            };
            let inp_c = buffer_typed(runner, &inp_vals[..check_rows * check_n], DType::F32);
            let out_c = zeros_typed(runner, check_rows * check_n, DType::F32);
            let ns_c = runner.buffer_u32(check_n as u32);
            let mt_chk = run_typed_once(
                runner,
                &mk,
                &[&inp_c, &out_c, &ns_c],
                &out_c,
                check_rows * check_n,
                [1, check_rows, 1],
                [tpg, 1, 1],
                DType::F32,
            );
            let equiv = check_equiv(&ref_out, &mt_chk, self.tol);

            let inp_buf = buffer_typed(runner, &inp_vals, DType::F32);
            let bytes = (rows * n * 8) as f64;
            let ns_u64 = runner.buffer_u64(n as u64);
            let ns_u32 = runner.buffer_u32(n as u32);
            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let out = zeros_typed(runner, rows * n, DType::F32);
                bench_gbps(runner, rk, &[&inp_buf, &out, &ns_u64], [1, rows, 1], [tpg, 1, 1], bytes)
            });
            let mt_perf = {
                let out = zeros_typed(runner, rows * n, DType::F32);
                bench_gbps(
                    runner,
                    &mk,
                    &[&inp_buf, &out, &ns_u32],
                    [1, rows, 1],
                    [tpg, 1, 1],
                    bytes,
                )
            };
            results.push(bench.result_sub(
                Some(self.subop),
                format!("B={rows} N={n} f32"),
                ref_perf,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }

    // ── ArgReduce ─────────────────────────────────────────────────────────────

    fn run_arg_reduce(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        n: usize,
        check_n: usize,
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_reduction(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, "float32");

        let check_vals: Vec<f32> = (0..check_n).map(|i| ((i * 7 + 3) % 97) as f32 * 0.1).collect();
        let expected: f32 = {
            let mut best = f32::NEG_INFINITY;
            let mut idx = 0usize;
            for (i, &v) in check_vals.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = i;
                }
            }
            idx as f32
        };
        let inp_c = buffer_typed(runner, &check_vals, DType::F32);
        let out_c = zeros_typed(runner, 1, DType::F32);
        let ns_c = runner.buffer_u32(check_n as u32);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&inp_c, &out_c, &ns_c],
            &out_c,
            1,
            [1, 1, 1],
            [tpg, 1, 1],
            DType::F32,
        );
        let equiv = check_equiv(&[expected], &mt_chk, 0.5);

        let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1009) as f32 * 0.001).collect();
        let inp = buffer_typed(runner, &vals, DType::F32);
        let bytes = (n * 4) as f64;
        let ns = runner.buffer_u32(n as u32);
        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = runner.buffer_zeros(4);
            let dummy = runner.buffer_u32(0u32);
            let ndim = runner.buffer_u64(0u64);
            let ax_stride = runner.buffer_i64(1i64);
            let ax_size = runner.buffer_u64(n as u64);
            bench_gbps(
                runner,
                rk,
                &[&inp, &out, &dummy, &dummy, &dummy, &ndim, &ax_stride, &ax_size],
                [tpg, 1, 1],
                [tpg, 1, 1],
                bytes,
            )
        });
        let mt_out = zeros_typed(runner, 1, DType::F32);
        let mt_perf = bench_gbps(runner, &mk, &[&inp, &mt_out, &ns], [1, 1, 1], [tpg, 1, 1], bytes);
        vec![bench.result_sub(
            Some(self.subop),
            format!("N={n} f32"),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── Random ────────────────────────────────────────────────────────────────

    fn run_random(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        n: usize,
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_elementwise(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let check_n = 1024usize;
        let ref_vals: Vec<u32> = (0..check_n as u32)
            .map(|gid| {
                let mut s = gid + 1;
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
            .collect();
        let n_buf_c = runner.buffer_u32(check_n as u32);
        let check_out = runner.buffer_zeros(check_n * 4);
        runner.measure(
            &mk,
            &[&check_out, &n_buf_c],
            [check_n.div_ceil(tpg), 1, 1],
            [tpg, 1, 1],
            0,
            1,
        );
        let raw = runner.read_f32_slice(&check_out, check_n);
        let mt_vals: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
        let n_bad = ref_vals.iter().zip(&mt_vals).filter(|(a, b)| a != b).count();
        let equiv = EquivResult {
            n_checked: check_n,
            max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
            passed: n_bad == 0,
        };

        let bytes = (n * 4) as f64;
        let n_buf = runner.buffer_u32(n as u32);
        let mt_out = runner.buffer_zeros(n * 4);
        let mt_perf = bench_gbps(
            runner,
            &mk,
            &[&mt_out, &n_buf],
            [n.div_ceil(tpg), 1, 1],
            [tpg, 1, 1],
            bytes,
        );

        // MLX rbitsc uses completely different PRNG and dispatch, just measure if available
        let num_keys = 1024usize;
        let bytes_per_key = 4096usize;
        let half_size = bytes_per_key / 8;
        let total = num_keys * bytes_per_key / 4;
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, "").and_then(|rk| {
            let key_data: Vec<u8> = (0..num_keys * 2 * 4).map(|i| i as u8).collect();
            let keys_buf = runner.buffer_bytes(&key_data);
            let ref_out_buf = runner.buffer_zeros(num_keys * bytes_per_key);
            let odd_buf = runner.buffer_bytes(std::slice::from_ref(&(false as u8)));
            let bpk_buf = runner.buffer_bytes(&(bytes_per_key as u32).to_le_bytes());
            bench_gbps(
                runner,
                &rk,
                &[&keys_buf, &ref_out_buf, &odd_buf, &bpk_buf],
                [num_keys, 1, 1],
                [1, half_size, 1],
                (total * 4) as f64,
            )
        });
        vec![bench.result_sub(
            Some(self.subop),
            format!("{}M u32", n / (1024 * 1024)),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── FpQuantized ───────────────────────────────────────────────────────────

    fn run_fp_quantized(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        n: usize,
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_elementwise(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };

        let data: Vec<f32> = (0..n).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
        let check_n = 1024usize;
        let ref_out: Vec<f32> = data[..check_n]
            .chunks(32)
            .flat_map(|group| {
                let max_abs = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let inv_scale = if max_abs > 0.0 { 6.0 / max_abs } else { 0.0 };
                let scale = max_abs / 6.0;
                group.iter().map(move |&x| {
                    let norm = x.abs() * inv_scale;
                    let q = if norm < 0.25 {
                        0.0
                    } else if norm < 0.75 {
                        0.5
                    } else if norm < 1.25 {
                        1.0
                    } else if norm < 1.75 {
                        1.5
                    } else if norm < 2.5 {
                        2.0
                    } else if norm < 3.5 {
                        3.0
                    } else if norm < 5.0 {
                        4.0
                    } else {
                        6.0
                    };
                    let sign = if x < 0.0 { -1.0 } else { 1.0 };
                    sign * q * scale
                })
            })
            .collect();
        let inp_c = buffer_typed(runner, &data[..check_n], DType::F32);
        let out_c = zeros_typed(runner, check_n, DType::F32);
        let n_buf_c = runner.buffer_u32(check_n as u32);
        runner.measure(&mk, &[&inp_c, &out_c, &n_buf_c], [check_n / tpg, 1, 1], [tpg, 1, 1], 0, 1);
        let mt_out_c = runner.read_f32_slice(&out_c, check_n);
        let equiv = check_equiv_with(&ref_out, &mt_out_c, EquivTolerance::new(0.5, 0.99));

        let inp = buffer_typed(runner, &data, DType::F32);
        let n_buf = runner.buffer_u32(n as u32);
        let bytes = (n * 4 * 2) as f64;
        let ref_perf = Self::compile_mlx(runner, mlx_src, mlx_pattern, "").and_then(|rk| {
            let out = zeros_typed(runner, n, DType::F32);
            bench_gbps(runner, &rk, &[&inp, &out], [1, n / 32, 1], [32, 1, 1], bytes)
        });
        let mt_perf = {
            let out = zeros_typed(runner, n, DType::F32);
            bench_gbps(runner, &mk, &[&inp, &out, &n_buf], [n / tpg, 1, 1], [tpg, 1, 1], bytes)
        };
        vec![bench.result_sub(
            Some(self.subop),
            format!("N={}M f32 gs32", n / (1024 * 1024)),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }

    // ── MatVec ────────────────────────────────────────────────────────────────

    fn run_mat_vec(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tol = self.tol.max(1e-2f32);
        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn);
        let mut results = Vec::new();
        for &(m, k) in shapes {
            let cm = 64usize;
            let ck = 256usize;
            let sm: Vec<f32> = (0..cm * ck).map(|i| (i % 16) as f32 * 0.01).collect();
            let sv: Vec<f32> = (0..ck).map(|i| (i % 8) as f32 * 0.01).collect();
            let sm_q = quantize_roundtrip(&sm, dt);
            let sv_q = quantize_roundtrip(&sv, dt);
            let ref_out: Vec<f32> = (0..cm)
                .map(|row| (0..ck).map(|col| sm_q[row * ck + col] * sv_q[col]).sum())
                .collect();
            let mat_b = buffer_typed(runner, &sm, dt);
            let vec_b = buffer_typed(runner, &sv, dt);
            let out_b = zeros_typed(runner, cm, dt);
            let k_b = runner.buffer_u32(ck as u32);
            let mt_vals = run_typed_once(
                runner,
                &mk,
                &[&mat_b, &vec_b, &out_b, &k_b],
                &out_b,
                cm,
                [cm, 1, 1],
                [tpg, 1, 1],
                dt,
            );
            let equiv = check_equiv(&ref_out, &mt_vals, tol);

            let mat_vals: Vec<f32> = (0..m * k).map(|i| (i % 16) as f32 * 0.01).collect();
            let vec_vals: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01).collect();
            let mat_buf = buffer_typed(runner, &mat_vals, dt);
            let vec_buf = buffer_typed(runner, &vec_vals, dt);
            let k_buf = runner.buffer_u32(k as u32);
            let bytes = (m * k * ctx.eb + k * ctx.eb + m * ctx.eb) as f64;

            // MLX gemv ref has 15 params: mat, vec, bias, out, in_vec_size, out_vec_size, mat_ld,
            // alpha, beta, batch_ndim, <4 empty batching ptrs>, bias_stride
            const REF_BM: usize = 4;
            const REF_TM: usize = 4;
            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let out_r = runner.buffer_zeros(m * ctx.eb);
                let bias_r = runner.buffer_zeros(m * ctx.eb);
                let zero_buf = runner.buffer_zeros(8);
                let in_vec_size = runner.buffer_i32(k as i32);
                let out_vec_size = runner.buffer_i32(m as i32);
                let mat_ld = runner.buffer_i32(k as i32);
                let alpha = runner.buffer_f32_scalar(1.0f32);
                let beta = runner.buffer_f32_scalar(0.0f32);
                let batch_ndim = runner.buffer_i32(0i32);
                let bias_stride = runner.buffer_i32(1i32);
                bench_gbps(
                    runner,
                    rk,
                    &[
                        &mat_buf,
                        &vec_buf,
                        &bias_r,
                        &out_r,
                        &in_vec_size,
                        &out_vec_size,
                        &mat_ld,
                        &alpha,
                        &beta,
                        &batch_ndim,
                        &zero_buf,
                        &zero_buf,
                        &zero_buf,
                        &zero_buf,
                        &bias_stride,
                    ],
                    [m / (REF_BM * REF_TM), 1, 1],
                    [REF_BM * 32, 1, 1],
                    bytes,
                )
            });
            let mt_perf = {
                let out_buf = zeros_typed(runner, m, dt);
                bench_gbps(
                    runner,
                    &mk,
                    &[&mat_buf, &vec_buf, &out_buf, &k_buf],
                    [m, 1, 1],
                    [tpg, 1, 1],
                    bytes,
                )
            };
            results.push(bench.result_sub(
                Some(self.subop),
                format!("M={m} K={k} {}", ctx.label),
                ref_perf,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }

    // ── MatVecMasked ──────────────────────────────────────────────────────────

    fn run_mat_vec_masked(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        tpg: usize,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let tol = self.tol.max(1e-2f32);
        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let mut results = Vec::new();
        for &(m, k) in shapes {
            let cm = 64usize;
            let ck = 256usize;
            let sm: Vec<f32> = (0..cm * ck).map(|i| (i % 13) as f32 * 0.01).collect();
            let sv: Vec<f32> = (0..ck).map(|i| (i % 7) as f32 * 0.01).collect();
            let mask_vals: Vec<f32> = (0..ck).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
            let sm_q = quantize_roundtrip(&sm, dt);
            let sv_q = quantize_roundtrip(&sv, dt);
            let ref_out: Vec<f32> = (0..cm)
                .map(|row| {
                    (0..ck)
                        .filter(|&col| mask_vals[col] != 0.0)
                        .map(|col| sm_q[row * ck + col] * sv_q[col])
                        .sum()
                })
                .collect();
            let mat_b = buffer_typed(runner, &sm, dt);
            let vec_b = buffer_typed(runner, &sv, dt);
            let mask_b = buffer_typed(runner, &mask_vals, dt);
            let out_b = zeros_typed(runner, cm, dt);
            let k_b = runner.buffer_u32(ck as u32);
            let mt_vals = run_typed_once(
                runner,
                &mk,
                &[&mat_b, &vec_b, &mask_b, &out_b, &k_b],
                &out_b,
                cm,
                [cm, 1, 1],
                [tpg, 1, 1],
                dt,
            );
            let equiv = check_equiv(&ref_out, &mt_vals, tol);

            let mat_vals: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
            let vec_vals: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.01).collect();
            let mask_perf: Vec<f32> = (0..k).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
            let mat_buf = buffer_typed(runner, &mat_vals, dt);
            let vec_buf = buffer_typed(runner, &vec_vals, dt);
            let mask_buf = buffer_typed(runner, &mask_perf, dt);
            let k_buf = runner.buffer_u32(k as u32);
            let bytes = (m * k * ctx.eb + k * ctx.eb * 2 + m * ctx.eb) as f64;
            let mt_perf = {
                let out = zeros_typed(runner, m, dt);
                bench_gbps(
                    runner,
                    &mk,
                    &[&mat_buf, &vec_buf, &mask_buf, &out, &k_buf],
                    [m, 1, 1],
                    [tpg, 1, 1],
                    bytes,
                )
            };
            results.push(bench.result_sub(
                Some(self.subop),
                format!("M={m} K={k} {}", ctx.label),
                None,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }

    // ── QuantizedMatVec ───────────────────────────────────────────────────────

    fn run_quantized_mat_vec(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize)],
        group_size: usize,
        tpg: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_reduction(DType::F32) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, "");
        let mut results = Vec::new();
        for &(m, k) in shapes {
            let w_elems = m * k / 8;
            let sb_elems = m * k / group_size;
            let gs_per_row = k / group_size;
            // Correctness check: M=4 rows, K=group_size (one group per row)
            let cm = 4usize;
            let ck = group_size;
            let w_check: Vec<u32> = (0..cm * ck / 8)
                .map(|i| {
                    let mut v = 0u32;
                    for bit in 0..8u32 {
                        v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                    }
                    v
                })
                .collect();
            let s_check = vec![0.1f32; cm];
            let b_check = vec![0.0f32; cm];
            let x_check = vec![1.0f32; ck];
            let ref_out: Vec<f32> = (0..cm)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for g in 0..1usize {
                        let s = s_check[row * 1 + g];
                        let bias = b_check[row * 1 + g];
                        for p in 0..8usize {
                            let packed = w_check[row * ck / 8 + g * 8 + p];
                            for bit in 0..8u32 {
                                let int4_val = ((packed >> (bit * 4)) & 0xF) as f32;
                                acc +=
                                    (s * int4_val + bias) * x_check[g * ck + p * 8 + bit as usize];
                            }
                        }
                    }
                    acc
                })
                .collect();
            let w_bytes: Vec<u8> = w_check.iter().flat_map(|v| v.to_le_bytes()).collect();
            let w_buf_c = runner.buffer_bytes(&w_bytes);
            let s_buf_c = runner.buffer_f32(&s_check);
            let b_buf_c = runner.buffer_f32(&b_check);
            let x_buf_c = runner.buffer_f32(&x_check);
            let out_c = runner.buffer_zeros(cm * 4);
            let k_buf_c = runner.buffer_u32(ck as u32);
            let gpr_buf_c = runner.buffer_u32(1u32);
            runner.measure(
                &mk,
                &[&w_buf_c, &s_buf_c, &b_buf_c, &x_buf_c, &out_c, &k_buf_c, &gpr_buf_c],
                [cm, 1, 1],
                [tpg, 1, 1],
                0,
                1,
            );
            let mt_out_c = runner.read_f32_slice(&out_c, cm);
            let n_bad =
                ref_out.iter().zip(mt_out_c.iter()).filter(|(r, m)| (*r - *m).abs() > 1e-3).count();
            let equiv = EquivResult {
                n_checked: cm,
                max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
                cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
                passed: n_bad == 0,
            };

            let w_data: Vec<u8> = (0..w_elems * 4).map(|i| (i % 256) as u8).collect();
            let scales_f32: Vec<f32> = (0..sb_elems).map(|_| 0.05f32).collect();
            let biases_f32 = vec![0.0f32; sb_elems];
            let x_f32: Vec<f32> = (0..k).map(|i| (i % 8) as f32 * 0.01 + 0.5).collect();
            let w_mt_buf = runner.buffer_bytes(&w_data);
            let s_mt_buf = runner.buffer_f32(&scales_f32);
            let b_mt_buf = runner.buffer_f32(&biases_f32);
            let x_mt_buf = runner.buffer_f32(&x_f32);
            let k_buf = runner.buffer_u32(k as u32);
            let gpr_buf = runner.buffer_u32(gs_per_row as u32);
            let bytes_mt = (m * k / 2 + sb_elems * 4 * 2 + k * 4 + m * 4) as f64;
            let mt_perf = {
                let out_buf = runner.buffer_zeros(m * 4);
                bench_gbps(
                    runner,
                    &mk,
                    &[&w_mt_buf, &s_mt_buf, &b_mt_buf, &x_mt_buf, &out_buf, &k_buf, &gpr_buf],
                    [m, 1, 1],
                    [tpg, 1, 1],
                    bytes_mt,
                )
            };
            // MLX ref uses f16 data (different dtype)
            const ROWS_PER_TG: usize = 8;
            let ref_perf = ref_kernel.as_ref().and_then(|rk| {
                let scale_f16: Vec<u8> =
                    (0..sb_elems * 2).map(|i| if i % 2 == 0 { 0x66 } else { 0x2E }).collect();
                let bias_f16 = vec![0u8; sb_elems * 2];
                let x_f16: Vec<u8> =
                    (0..k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();
                let scales_f16_buf = runner.buffer_bytes(&scale_f16);
                let biases_f16_buf = runner.buffer_bytes(&bias_f16);
                let x_f16_buf = runner.buffer_bytes(&x_f16);
                let in_size = runner.buffer_i32(k as i32);
                let out_size = runner.buffer_i32(m as i32);
                let batch_zero = runner.buffer_i32(0i32);
                let zero = runner.buffer_zeros(8);
                let y_buf = runner.buffer_zeros(m * 2);
                let bytes_f16 = (m * k / 2 + sb_elems * 2 * 2 + k * 2 + m * 2) as f64;
                bench_gbps(
                    runner,
                    rk,
                    &[
                        &w_mt_buf,
                        &scales_f16_buf,
                        &biases_f16_buf,
                        &x_f16_buf,
                        &y_buf,
                        &in_size,
                        &out_size,
                        &batch_zero,
                        &zero,
                        &zero,
                        &batch_zero,
                        &zero,
                        &zero,
                        &zero,
                        &zero,
                    ],
                    [1, m / ROWS_PER_TG, 1],
                    [64, 1, 1],
                    bytes_f16,
                )
            });
            results.push(bench.result_sub(
                Some(self.subop),
                format!("M={m} K={k} f32 gs{group_size} b4"),
                ref_perf,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }

    // ── Rope ──────────────────────────────────────────────────────────────────

    fn run_rope(
        &self,
        runner: &GpuRunner,
        _dt: DType,
        bench: &OpBench,
        b: usize,
        h: usize,
        l: usize,
        d: usize,
        n_per_group: usize,
        mlx_src: &Option<&str>,
    ) -> Vec<OpResult> {
        let msl = match self.msl_grid3d(DType::F16) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let rk = mlx_src.and_then(|src| {
            runner
                .compile_with_bool_constants(src, "rope_float16", &[
                    (1, true),
                    (2, false),
                    (3, false),
                ])
                .ok()
        });

        let gx = d / (2 * n_per_group);
        let gy = l;
        let gz = h / n_per_group;
        let n_elems = b * l * h * d;

        let f32_to_f16 = |v: f32| -> u16 {
            let bits = v.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let mant = (bits >> 13) & 0x3ff;
            if exp <= 0 {
                sign
            } else if exp >= 31 {
                sign | 0x7c00
            } else {
                sign | ((exp as u16) << 10) | mant as u16
            }
        };
        let in_f16: Vec<u16> = (0..n_elems).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
        let inp = runner.buffer_f16(&in_f16);
        let base_val = (10000f32).log2();

        // Correctness: compare MT vs MLX ref on small L_CHECK=4 sub-problem
        let equiv = rk.as_ref().map(|rk| {
            let mk = &mk;
            let l_check = 4usize;
            let n_check = b * l_check * h * d;
            let check_f16: Vec<u16> = (0..n_check).map(|i| f32_to_f16(i as f32 * 0.001)).collect();
            let inp_c = runner.buffer_f16(&check_f16);
            let ref_out_c = runner.buffer_zeros(n_check * 2);
            let mt_out_c = runner.buffer_zeros(n_check * 2);

            // MLX ref params: (in, out, offset[B], scale, strides[3], out_strides[3], offset_stride, n_head, dummy, dummy, base)
            let strides_bytes: Vec<u8> =
                [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
            let strides_buf = runner.buffer_bytes(&strides_bytes);
            let offset_arr = runner.buffer_i32(0i32);
            let scale_buf = runner.buffer_f32_scalar(1.0f32);
            let offset_stride_buf = runner.buffer_i64(1i64);
            let n_head_buf = runner.buffer_i32(h as i32);
            let dummy = runner.buffer_zeros(4);
            let base_buf = runner.buffer_f32_scalar(base_val);
            runner.measure(
                rk,
                &[
                    &inp_c,
                    &ref_out_c,
                    &offset_arr,
                    &scale_buf,
                    &strides_buf,
                    &strides_buf,
                    &offset_stride_buf,
                    &n_head_buf,
                    &dummy,
                    &dummy,
                    &base_buf,
                ],
                [gx, l_check, gz],
                [1, 1, 1],
                0,
                1,
            );
            let ref_vals = runner.read_f16_slice(&ref_out_c, n_check);

            // MT params: (inp, out, h_stride, seq_stride, grid_x, base)
            let mt_h_stride = runner.buffer_u32(d as u32);
            let mt_seq_stride = runner.buffer_u32((h * d) as u32);
            let mt_grid_x = runner.buffer_u32(gx as u32);
            let mt_base = runner.buffer_f32_scalar(base_val);
            runner.measure(
                mk,
                &[&inp_c, &mt_out_c, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
                [gx, l_check, gz],
                [1, 1, 1],
                0,
                1,
            );
            let mt_vals = runner.read_f16_slice(&mt_out_c, n_check);
            check_equiv(&ref_vals, &mt_vals, self.tol)
        });

        let strides_bytes: Vec<u8> =
            [d as i64, (h * d) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let strides_buf = runner.buffer_bytes(&strides_bytes);
        let offset_arr = runner.buffer_i32(0i32);
        let scale_buf = runner.buffer_f32_scalar(1.0f32);
        let offset_stride_buf = runner.buffer_i64(1i64);
        let n_head_buf = runner.buffer_i32(h as i32);
        let dummy = runner.buffer_zeros(4);
        let base_buf = runner.buffer_f32_scalar(base_val);
        let mt_h_stride = runner.buffer_u32(d as u32);
        let mt_seq_stride = runner.buffer_u32((h * d) as u32);
        let mt_grid_x = runner.buffer_u32(gx as u32);
        let mt_base = runner.buffer_f32_scalar(base_val);
        let bytes = (n_elems * 2 * 2) as f64;

        let ref_perf = rk.as_ref().and_then(|rk| {
            let out = runner.buffer_zeros(n_elems * 2);
            bench_gbps(
                runner,
                rk,
                &[
                    &inp,
                    &out,
                    &offset_arr,
                    &scale_buf,
                    &strides_buf,
                    &strides_buf,
                    &offset_stride_buf,
                    &n_head_buf,
                    &dummy,
                    &dummy,
                    &base_buf,
                ],
                [gx, gy, gz],
                [1, 1, 1],
                bytes,
            )
        });
        let mt_out = runner.buffer_zeros(n_elems * 2);
        let mt_perf = bench_gbps(
            runner,
            &mk,
            &[&inp, &mt_out, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
            [gx, gy, gz],
            [1, 1, 1],
            bytes,
        );
        let shape = format!("B{b}H{h}L{l}D{d} f16");
        vec![bench.result_sub(Some(self.subop), shape, ref_perf, mt_perf, equiv)]
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    fn run_attention(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        shapes: &[(usize, usize, usize)],
        tpg: usize,
        mlx_src: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let msl = match self.msl_reduction(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        const REF_FCS: &[(usize, bool)] =
            &[(20, false), (21, false), (22, false), (23, false), (24, false), (25, false)];
        let ref_name = match dt {
            DType::F32 => "sdpa_vector_float_128_128",
            DType::F16 => "sdpa_vector_float16_t_128_128",
            _ => return vec![],
        };
        let rk =
            mlx_src.and_then(|src| runner.compile_with_bool_constants(src, ref_name, REF_FCS).ok());
        let mut results = Vec::new();
        for &(h, n_kv, d) in shapes {
            let scale = 1.0_f32 / (d as f32).sqrt();
            // Correctness: cpu_sdpa on small H=2, N=64
            let ch = 2usize;
            let cn = 64usize;
            let cq: Vec<f32> = (0..ch * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let ck_: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
            let cv: Vec<f32> = (0..ch * cn * d).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let ref_out: Vec<f32> = {
                let mut out = vec![0.0f32; ch * d];
                for head in 0..ch {
                    let q_base = head * d;
                    let kv_base = head * cn * d;
                    let mut scores = vec![0.0f32; cn];
                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..cn {
                        let base = kv_base + t * d;
                        let qk: f32 =
                            (0..d).map(|e| cq[q_base + e] * ck_[base + e]).sum::<f32>() * scale;
                        scores[t] = qk;
                        max_score = max_score.max(qk);
                    }
                    let mut sum = 0.0f32;
                    let mut o = vec![0.0f32; d];
                    for t in 0..cn {
                        let w = (scores[t] - max_score).exp();
                        sum += w;
                        for e in 0..d {
                            o[e] += w * cv[kv_base + t * d + e];
                        }
                    }
                    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
                    for e in 0..d {
                        out[q_base + e] = o[e] * inv;
                    }
                }
                out
            };
            let (q_b, k_b, v_b, out_b, n_b, sc_b) = if dt == DType::F32 {
                let q_b = buffer_typed(runner, &cq, dt);
                let k_b = buffer_typed(runner, &ck_, dt);
                let v_b = buffer_typed(runner, &cv, dt);
                let out_b = zeros_typed(runner, ch * d, dt);
                let n_b = runner.buffer_u32(cn as u32);
                let sc_b = runner.buffer_f32_scalar(scale);
                (q_b, k_b, v_b, out_b, n_b, sc_b)
            } else {
                let f32_to_f16 = |v: f32| -> u16 {
                    let x = v.to_bits();
                    let sign = ((x >> 31) as u16) << 15;
                    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
                    let mant32 = x & 0x7F_FFFF;
                    if exp <= 0 {
                        return sign;
                    }
                    if exp >= 31 {
                        return sign | 0x7C00;
                    }
                    let mant16 = mant32 >> 13;
                    sign | ((exp as u16) << 10) | (mant16 as u16)
                };
                let q_f16: Vec<u16> = cq.iter().copied().map(f32_to_f16).collect();
                let k_f16: Vec<u16> = ck_.iter().copied().map(f32_to_f16).collect();
                let v_f16: Vec<u16> = cv.iter().copied().map(f32_to_f16).collect();
                let q_b = runner.buffer_f16(&q_f16);
                let k_b = runner.buffer_f16(&k_f16);
                let v_b = runner.buffer_f16(&v_f16);
                let out_b = runner.buffer_zeros(ch * d * 2);
                let n_b = runner.buffer_u32(cn as u32);
                let sc_b = runner.buffer_f32_scalar(scale);
                (q_b, k_b, v_b, out_b, n_b, sc_b)
            };
            runner.measure(
                &mk,
                &[&q_b, &k_b, &v_b, &out_b, &n_b, &sc_b],
                [ch, 1, 1],
                [tpg, 1, 1],
                0,
                1,
            );
            let mt_chk = if dt == DType::F32 {
                runner.read_f32_slice(&out_b, ch * d)
            } else {
                runner.read_f16_slice(&out_b, ch * d)
            };
            let equiv = check_equiv_with(&ref_out, &mt_chk, EquivTolerance::new(self.tol, 0.999));

            let vals: Vec<f32> =
                (0..h * n_kv * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
            let bytes = (h * n_kv * d * ctx.eb * 2 + h * d * ctx.eb * 2) as f64;
            let (q_buf, k_buf, v_buf, n_buf, sc_buf) = if dt == DType::F32 {
                let qb = buffer_typed(runner, &vals[..h * d], dt);
                let kb = buffer_typed(runner, &vals[..h * n_kv * d], dt);
                let vb = buffer_typed(runner, &vals[..h * n_kv * d], dt);
                let nb = runner.buffer_u32(n_kv as u32);
                let sb = runner.buffer_f32_scalar(scale);
                (qb, kb, vb, nb, sb)
            } else {
                let f32_to_f16 = |v: f32| -> u16 {
                    let x = v.to_bits();
                    let sign = ((x >> 31) as u16) << 15;
                    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
                    let mant16 = (x & 0x7F_FFFF) >> 13;
                    if exp <= 0 {
                        return sign;
                    }
                    if exp >= 31 {
                        return sign | 0x7C00;
                    }
                    sign | ((exp as u16) << 10) | (mant16 as u16)
                };
                let qb = runner
                    .buffer_f16(&vals[..h * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>());
                let kb = runner.buffer_f16(
                    &vals[..h * n_kv * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
                );
                let vb = runner.buffer_f16(
                    &vals[..h * n_kv * d].iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
                );
                let nb = runner.buffer_u32(n_kv as u32);
                let sb = runner.buffer_f32_scalar(scale);
                (qb, kb, vb, nb, sb)
            };
            let ref_perf = rk.as_ref().and_then(|rk| {
                let gqa = runner.buffer_i32(1i32);
                let n_i32 = runner.buffer_i32(n_kv as i32);
                let khs = runner.buffer_u64((n_kv * d) as u64);
                let kss = runner.buffer_u64(d as u64);
                let out = if dt == DType::F32 {
                    zeros_typed(runner, h * d, dt)
                } else {
                    runner.buffer_zeros(h * d * 2)
                };
                bench_gbps(
                    runner,
                    rk,
                    &[&q_buf, &k_buf, &v_buf, &out, &gqa, &n_i32, &khs, &kss, &khs, &kss, &sc_buf],
                    [h, 1, 1],
                    [1024, 1, 1],
                    bytes,
                )
            });
            let mt_perf = {
                let out = if dt == DType::F32 {
                    zeros_typed(runner, h * d, dt)
                } else {
                    runner.buffer_zeros(h * d * 2)
                };
                bench_gbps(
                    runner,
                    &mk,
                    &[&q_buf, &k_buf, &v_buf, &out, &n_buf, &sc_buf],
                    [h, 1, 1],
                    [tpg, 1, 1],
                    bytes,
                )
            };
            results.push(bench.result_sub(
                Some(self.subop),
                format!("H={h} N={n_kv} D={d} {}", ctx.label),
                ref_perf,
                mt_perf,
                Some(equiv),
            ));
        }
        results
    }

    // ── StridedCopy ───────────────────────────────────────────────────────────

    fn run_strided_copy(
        &self,
        runner: &GpuRunner,
        dt: DType,
        bench: &OpBench,
        m: usize,
        n: usize,
        pad: usize,
        mlx_src: &Option<&str>,
        mlx_pattern: &Option<&str>,
    ) -> Vec<OpResult> {
        let ctx = DtypeCtx::elementwise(dt);
        let msl = match self.msl_grid3d(dt) {
            Some(s) => s,
            None => return vec![],
        };
        let mk = match Self::compile_mt(runner, &msl, self.kernel_name) {
            Some(k) => k,
            None => return vec![],
        };
        let ref_kernel = Self::compile_mlx(runner, mlx_src, mlx_pattern, ctx.tn);

        // Correctness: 8×16 copy from 8×(16+4) source
        let cm = 8usize;
        let cn = 16usize;
        let cp = 4usize;
        let src_stride = cn + cp;
        let src_vals: Vec<f32> = (0..cm * src_stride)
            .map(|i| {
                let row = i / src_stride;
                let col = i % src_stride;
                if col < cn { (row * cn + col) as f32 + 1.0 } else { -999.0 }
            })
            .collect();
        let expected: Vec<f32> = (0..cm * cn).map(|i| i as f32 + 1.0).collect();
        let src_buf = buffer_typed(runner, &src_vals, dt);
        let src_shape_check = runner.buffer_bytes(
            &[cm as u32, cn as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let src_strides_check = runner.buffer_bytes(
            &[src_stride as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let cols_buf = runner.buffer_u32(cn as u32);
        let out_check = zeros_typed(runner, cm * cn, dt);
        let mt_chk = run_typed_once(
            runner,
            &mk,
            &[&src_buf, &src_shape_check, &src_strides_check, &out_check, &cols_buf],
            &out_check,
            cm * cn,
            [cm, cn, 1],
            [1, 1, 1],
            dt,
        );
        let equiv = check_equiv(&expected, &mt_chk, self.tol);

        // Throughput: full M×N copy from M×(N+PAD) source
        let full_src: Vec<f32> = (0..m * (n + pad)).map(|i| (i % 256) as f32 * 0.01).collect();
        let full_src_buf = buffer_typed(runner, &full_src, dt);
        let full_src_shape = runner.buffer_bytes(
            &[m as u32, n as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let full_src_strides = runner.buffer_bytes(
            &[(n + pad) as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let full_strides_i64 = runner.buffer_bytes(
            &[(n + pad) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let full_cols = runner.buffer_u32(n as u32);
        let bytes = (m * n * ctx.eb * 2) as f64;

        let ref_perf = ref_kernel.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, m * n, dt);
            bench_gbps(
                runner,
                rk,
                &[&full_src_buf, &out, &full_strides_i64],
                [n, m, 1],
                [1, 1, 1],
                bytes,
            )
        });
        let mt_perf = {
            let out = zeros_typed(runner, m * n, dt);
            bench_gbps(
                runner,
                &mk,
                &[&full_src_buf, &full_src_shape, &full_src_strides, &out, &full_cols],
                [m, n, 1],
                [1, 1, 1],
                bytes,
            )
        };
        vec![bench.result_sub(
            Some(self.subop),
            format!("M={m} N={n}+{pad} {}", ctx.label),
            ref_perf,
            mt_perf,
            Some(equiv),
        )]
    }
}

enum InterpMode {
    Elementwise(usize),
    Reduction(usize),
    Grid3D(usize, usize, usize),
}
