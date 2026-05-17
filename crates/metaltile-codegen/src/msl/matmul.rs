//! Tiled matrix multiply emission.
//!
//! Two paths:
//! - **Scalar path** (M1/M2): each thread computes RPT×CPT outputs via scalar inner loop
//!   using `threadgroup half` shared memory.
//! - **Simdgroup path** (M1+, dedicated HW on M3+): `simdgroup_multiply_accumulate`
//!   with double-buffered threadgroup memory for overlapped prefetch + compute.

use std::fmt::Write;

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, ValueId},
    shape::{Dim, Shape},
};

use crate::wl;

impl super::MslGenerator {
    /// Emit the full tiled matmul body (dispatches to scalar or simdgroup path).
    #[allow(non_snake_case)]
    pub(super) fn emit_tiled(
        &self,
        out: &mut String,
        pad: &str,
        kernel: &Kernel,
        dot_vid: Option<ValueId>,
    ) -> crate::error::Result<()> {
        let s = &self.config.tile_schedule;
        let (TM, TN, TK) = match dot_vid.and_then(|v| kernel.tile_annotations.get(&v)) {
            Some(&(tm, tn, tk)) => (tm, tn, tk),
            None if self.config.use_simd_matrix => (64, 64, 16),
            None => (s.tile_m, s.tile_n, s.tile_k),
        };
        let (THY, THX) =
            if self.config.use_simd_matrix { (8u32, 16u32) } else { (s.threads.0, s.threads.1) };
        let (RPT, CPT) = (s.rows_per_thread, s.cols_per_thread);
        let thrds = THY * THX;

        let tensors: Vec<_> = kernel.params.iter().filter(|p| p.shape.rank() == 2).collect();
        if tensors.len() < 3 {
            wl!(out, "{pad}// emit_tiled: need >= 3 2D tensor params (a, b, c)");
            return Ok(());
        }

        let a = &tensors[0].name;
        let b = &tensors[1].name;
        let c = &tensors[2].name;
        let m = dim_name(&tensors[0].shape, 0).unwrap_or("M");
        let k = dim_name(&tensors[0].shape, 1).unwrap_or("K");
        let n = dim_name(&tensors[1].shape, 1).unwrap_or("N");

        wl!(out);
        wl!(out, "{pad}// Tiled matmul: {a}[{m}×{k}] @ {b}[{k}×{n}] → {c}[{m}×{n}]");
        wl!(out, "{pad}// TM={TM} TN={TN} TK={TK} threads={THY}×{THX} RPT={RPT} CPT={CPT}");

        if self.config.use_simd_matrix {
            self.emit_tiled_simdgroup(out, pad, a, b, c, m, k, n, TM, TN, TK, THY, THX, thrds)?;
        } else {
            self.emit_tiled_scalar(
                out, pad, a, b, c, m, k, n, TM, TN, TK, THY, THX, RPT, CPT, thrds,
            )?;
        }
        wl!(out);
        Ok(())
    }

    /// Scalar GEMM path (M1/M2): each thread computes RPT×CPT outputs.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn emit_tiled_scalar(
        &self,
        out: &mut String,
        pad: &str,
        a: &str,
        b: &str,
        c: &str,
        m: &str,
        k: &str,
        n: &str,
        TM: u32,
        TN: u32,
        TK: u32,
        _THY: u32,
        THX: u32,
        RPT: u32,
        CPT: u32,
        thrds: u32,
    ) -> crate::error::Result<()> {
        wl!(out);
        wl!(out, "{pad}threadgroup half A_tile[{TM} * ({TK} + 1)];");
        wl!(out, "{pad}threadgroup half B_tile[{TK} * ({TN} + 1)];");
        wl!(out, "{pad}const uint row_start = tgid.y * {TM};");
        wl!(out, "{pad}const uint col_start = tgid.x * {TN};");
        wl!(out, "{pad}float acc[{RPT}][{CPT}];");
        wl!(
            out,
            "{pad}for (uint r = 0; r < {RPT}; r++) for (uint c = 0; c < {CPT}; c++) acc[r][c] = 0.0f;"
        );
        wl!(out);
        wl!(out, "{pad}for (uint kb = 0; kb < {k}; kb += {TK}) {{");
        wl!(out, "{pad}    for (uint i = tid.y*{THX}+tid.x; i < {TM}*({TK}+1); i += {thrds}) {{");
        wl!(out, "{pad}        uint r = i/({TK}+1), ko = kb + (i%({TK}+1));");
        wl!(
            out,
            "{pad}        A_tile[i] = (row_start+r < {m} && ko < {k}) ? {a}[(row_start+r)*{k}+ko] : 0.0h;"
        );
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}    for (uint i = tid.y*{THX}+tid.x; i < {TK}*({TN}+1); i += {thrds}) {{");
        wl!(out, "{pad}        uint ko = kb + (i/({TN}+1)), c2 = i%({TN}+1);");
        wl!(
            out,
            "{pad}        B_tile[i] = (ko < {k} && col_start+c2 < {n}) ? {b}[ko*{n}+col_start+c2] : 0.0h;"
        );
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);");
        wl!(out, "{pad}    for (uint r = 0; r < {RPT}; r++) {{");
        wl!(out, "{pad}        uint lr = tid.y*{RPT}+r;");
        wl!(out, "{pad}        for (uint c2 = 0; c2 < {CPT}; c2++) {{");
        wl!(out, "{pad}            uint lc = tid.x*{CPT}+c2;");
        wl!(out, "{pad}            float s = 0.0f;");
        wl!(out, "{pad}            for (uint kk = 0; kk < {TK}; kk++)");
        wl!(
            out,
            "{pad}                s += float(A_tile[lr*({TK}+1)+kk]) * float(B_tile[kk*({TN}+1)+lc]);"
        );
        wl!(out, "{pad}            acc[r][c2] += s;");
        wl!(out, "{pad}        }}");
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);");
        wl!(out, "{pad}}}");
        wl!(out, "{pad}for (uint r = 0; r < {RPT}; r++) {{");
        wl!(out, "{pad}    uint gr = row_start + tid.y*{RPT}+r;");
        wl!(out, "{pad}    if (gr >= {m}) continue;");
        wl!(out, "{pad}    for (uint c2 = 0; c2 < {CPT}; c2++) {{");
        wl!(out, "{pad}        uint gc = col_start + tid.x*{CPT}+c2;");
        wl!(out, "{pad}        if (gc >= {n}) continue;");
        wl!(out, "{pad}        {c}[gr*{n}+gc] = half(acc[r][c2]);");
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}}}");
        Ok(())
    }

    /// Simdgroup matrix multiply path (Apple7+/M1+).
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn emit_tiled_simdgroup(
        &self,
        out: &mut String,
        pad: &str,
        a: &str,
        b: &str,
        c: &str,
        m: &str,
        k: &str,
        n: &str,
        TM: u32,
        TN: u32,
        TK: u32,
        _THY: u32,
        THX: u32,
        _thrds: u32,
    ) -> crate::error::Result<()> {
        const SGM: u32 = 4;
        const SGN: u32 = 4;
        const SG_COLS: u32 = 2;

        let AB_SZ: u32 = TM * (TK + 1);
        let B_HALF_SZ: u32 = TK * (TN + 1);
        wl!(out);
        wl!(out, "{pad}threadgroup half A_tile[2 * {TM} * ({TK} + 1)];");
        wl!(out, "{pad}threadgroup half B_tile[2 * {TK} * ({TN} + 1)];");
        wl!(out, "{pad}const uint sg_x = simd_group % {SG_COLS};");
        wl!(out, "{pad}const uint sg_y = simd_group / {SG_COLS};");

        for fi in 0..SGM {
            for fj in 0..SGN {
                let idx = fi * SGN + fj;
                wl!(
                    out,
                    "{pad}simdgroup_float8x8 c{idx} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
                );
            }
        }

        const VEC: u32 = 8;
        let a_vld_shift = (TK / VEC).ilog2();
        let a_vld_mask = TK / VEC - 1;
        let b_vld_shift = (TN / VEC).ilog2();
        let b_vld_mask = TN / VEC - 1;

        wl!(out);
        wl!(out, "{pad}const uint ld_flat = tid.y*{THX}+tid.x;");
        wl!(out, "{pad}const uint a_rv = ld_flat >> {a_vld_shift}u;");
        wl!(out, "{pad}const uint a_cv = (ld_flat & {a_vld_mask}u) << 3u;");
        wl!(out, "{pad}const uint b_rv = ld_flat >> {b_vld_shift}u;");
        wl!(out, "{pad}const uint b_cv = (ld_flat & {b_vld_mask}u) << 3u;");

        wl!(out, "{pad}{{");
        wl!(out, "{pad}    const uint a_gm = tgid.y*{TM} + a_rv;");
        wl!(out, "{pad}    if (a_gm < {m} && a_cv + {VEC}u <= {k}) {{");
        wl!(
            out,
            "{pad}        *((threadgroup uint4*)(A_tile + a_rv*({TK}+1) + a_cv)) = *((const device uint4*)({a} + a_gm*{k} + a_cv));"
        );
        wl!(out, "{pad}    }} else {{");
        wl!(
            out,
            "{pad}        for (uint _i = 0; _i < {VEC}u; _i++) A_tile[a_rv*({TK}+1) + a_cv + _i] = (a_gm < {m} && a_cv+_i < {k}) ? {a}[a_gm*{k} + a_cv + _i] : 0.0h;"
        );
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}}}");
        wl!(out, "{pad}{{");
        wl!(out, "{pad}    const uint b_gc = tgid.x*{TN} + b_cv;");
        wl!(out, "{pad}    if (b_rv < {k} && b_gc + {VEC}u <= {n}) {{");
        wl!(
            out,
            "{pad}        *((threadgroup uint4*)(B_tile + b_rv*({TN}+1) + b_cv)) = *((const device uint4*)({b} + b_rv*{n} + b_gc));"
        );
        wl!(out, "{pad}    }} else {{");
        wl!(
            out,
            "{pad}        for (uint _i = 0; _i < {VEC}u; _i++) B_tile[b_rv*({TN}+1) + b_cv + _i] = (b_rv < {k} && tgid.x*{TN}+b_cv+_i < {n}) ? {b}[b_rv*{n} + tgid.x*{TN}+b_cv+_i] : 0.0h;"
        );
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}}}");
        wl!(out, "{pad}threadgroup_barrier(mem_flags::mem_threadgroup);");
        wl!(out);
        wl!(out, "{pad}uint cur_buf = 0;");
        wl!(out, "{pad}for (uint kb = 0; kb < {k}; kb += {TK}) {{");
        wl!(out, "{pad}    uint nxt_buf = cur_buf ^ 1;");
        wl!(out, "{pad}    uint ca = cur_buf * {AB_SZ};");
        wl!(out, "{pad}    uint cb = cur_buf * {B_HALF_SZ};");
        wl!(out, "{pad}    uint na = nxt_buf * {AB_SZ};");
        wl!(out, "{pad}    uint nb = nxt_buf * {B_HALF_SZ};");
        wl!(out, "{pad}    if (kb + {TK} < {k}) {{");
        wl!(out, "{pad}        {{");
        wl!(out, "{pad}            const uint a_gm = tgid.y*{TM} + a_rv;");
        wl!(out, "{pad}            const uint a_kc = kb + {TK} + a_cv;");
        wl!(out, "{pad}            if (a_gm < {m} && a_kc + {VEC}u <= {k}) {{");
        wl!(
            out,
            "{pad}                *((threadgroup uint4*)(A_tile + na + a_rv*({TK}+1) + a_cv)) = *((const device uint4*)({a} + a_gm*{k} + a_kc));"
        );
        wl!(out, "{pad}            }} else {{");
        wl!(
            out,
            "{pad}                for (uint _i = 0; _i < {VEC}u; _i++) A_tile[na + a_rv*({TK}+1) + a_cv + _i] = (a_gm < {m} && a_kc+_i < {k}) ? {a}[a_gm*{k} + a_kc + _i] : 0.0h;"
        );
        wl!(out, "{pad}            }}");
        wl!(out, "{pad}        }}");
        wl!(out, "{pad}        {{");
        wl!(out, "{pad}            const uint b_kr = kb + {TK} + b_rv;");
        wl!(out, "{pad}            const uint b_gc = tgid.x*{TN} + b_cv;");
        wl!(out, "{pad}            if (b_kr < {k} && b_gc + {VEC}u <= {n}) {{");
        wl!(
            out,
            "{pad}                *((threadgroup uint4*)(B_tile + nb + b_rv*({TN}+1) + b_cv)) = *((const device uint4*)({b} + b_kr*{n} + b_gc));"
        );
        wl!(out, "{pad}            }} else {{");
        wl!(
            out,
            "{pad}                for (uint _i = 0; _i < {VEC}u; _i++) B_tile[nb + b_rv*({TN}+1) + b_cv + _i] = (b_kr < {k} && tgid.x*{TN}+b_cv+_i < {n}) ? {b}[b_kr*{n} + tgid.x*{TN}+b_cv+_i] : 0.0h;"
        );
        wl!(out, "{pad}            }}");
        wl!(out, "{pad}        }}");
        wl!(out, "{pad}    }}");
        wl!(out);
        wl!(out, "{pad}    #pragma clang loop unroll(full)");
        wl!(out, "{pad}    for (uint kk = 0; kk < {TK}; kk += 8) {{");
        for fj in 0..SGN {
            wl!(out, "{pad}        simdgroup_half8x8 bv{fj};");
            wl!(
                out,
                "{pad}        simdgroup_load(bv{fj}, B_tile + cb + kk*({TN}+1) + (sg_x*{SGN}+{fj})*8, {TN}+1, ulong2(0,0), false);"
            );
        }
        for fi in 0..SGM {
            wl!(out, "{pad}        simdgroup_half8x8 av{fi};");
            wl!(
                out,
                "{pad}        simdgroup_load(av{fi}, A_tile + ca + (sg_y*{SGM}+{fi})*8*({TK}+1) + kk, {TK}+1, ulong2(0,0), false);"
            );
            for fj in 0..SGN {
                let idx = fi * SGN + fj;
                wl!(
                    out,
                    "{pad}        simdgroup_multiply_accumulate(c{idx}, av{fi}, bv{fj}, c{idx});"
                );
            }
        }
        wl!(out, "{pad}    }}");
        wl!(out, "{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);");
        wl!(out, "{pad}    cur_buf = nxt_buf;");
        wl!(out, "{pad}}}");
        wl!(out);
        wl!(out, "{pad}const uint row_base = tgid.y * {TM}, col_base = tgid.x * {TN};");
        wl!(out, "{pad}const bool n_aligned = (({n} & 1u) == 0u);");
        let sgm8 = SGM * 8;
        let sgn8 = SGN * 8;
        for fi in 0..SGM {
            for fj in 0..SGN {
                let idx = fi * SGN + fj;
                let fi8 = fi * 8;
                let fj8 = fj * 8;
                wl!(out, "{pad}{{");
                wl!(
                    out,
                    "{pad}    thread float2& c{idx}_e = (thread float2&)c{idx}.thread_elements();"
                );
                wl!(out, "{pad}    uint pair = simd_lane >> 1;");
                wl!(out, "{pad}    uint row_in_frag = (pair & 3u) + ((simd_lane >> 4) << 2);");
                wl!(
                    out,
                    "{pad}    uint col0 = ((simd_lane & 1u) << 1) + (((simd_lane >> 3) & 1u) << 2);"
                );
                wl!(out, "{pad}    uint gr = row_base + sg_y*{sgm8} + {fi8} + row_in_frag;");
                wl!(out, "{pad}    uint gc = col_base + sg_x*{sgn8} + {fj8} + col0;");
                wl!(out, "{pad}    if (gr < {m}) {{");
                wl!(out, "{pad}        if (n_aligned && gc + 1 < {n}) {{");
                wl!(
                    out,
                    "{pad}            *((device half2*)({c} + gr*{n} + gc)) = half2(half(c{idx}_e.x), half(c{idx}_e.y));"
                );
                wl!(out, "{pad}        }} else {{");
                wl!(out, "{pad}            if (gc < {n}) {c}[gr*{n}+gc] = half(c{idx}_e.x);");
                wl!(out, "{pad}            if (gc + 1 < {n}) {c}[gr*{n}+gc+1] = half(c{idx}_e.y);");
                wl!(out, "{pad}        }}");
                wl!(out, "{pad}    }}");
                wl!(out, "{pad}}}");
            }
        }
        Ok(())
    }
}

pub(super) fn dim_name(shape: &Shape, idx: usize) -> Option<&str> {
    match shape.dim(idx)? {
        Dim::ConstExpr(ce) => Some(ce.name()),
        _ => None,
    }
}

pub(super) fn dim_to_msl_str(dim: &Dim) -> String {
    match dim {
        Dim::Known(n) => n.to_string(),
        Dim::ConstExpr(ce) => ce.name().to_string(),
        Dim::Any => "1".to_string(),
    }
}

pub(super) fn fmt_float(v: f64, dtype: &DType) -> String {
    match dtype {
        DType::F16 => format!("{}h", v),
        DType::F32 => format!("{}f", v),
        _ => v.to_string(),
    }
}
