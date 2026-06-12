//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Correctness test for the CUTLASS grouped block-scaled NVFP4 MoE GEMM
//! (`CudaDevice::moe_grouped_cutlass_fp4`): runs a small grouped problem on a
//! real sm_120a/sm_121a device and compares the f16 output against an f32
//! oracle that dequantizes the SAME e2m1 + ue4m3 inputs and does the grouped
//! GEMM on the host. The fp4 inputs are bit-identical on both sides, so the
//! only deviation is the f16 output round-trip — the match is near bit-exact.
//!
//! Skips (no failure) when there is no CUDA device OR the runtime was built
//! without CUTLASS (`CUTLASS_DIR` unset → the FFI returns a "built without
//! CUTLASS" error), so CI shards without the toolkit stay green.
#![cfg(feature = "cuda")]

use metaltile_runtime::CudaDevice;

// ── fp4 element decode. e2m1: sign | exp(2) | mant(1). 0,.5,1,1.5,2,3,4,6 (±). ──
const FP4_LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn decode_fp4(code: u8) -> f32 {
    let v = FP4_LUT[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

// ── block scale-factor decode. ue4m3: exp(4, bias 7) | mant(3); e==0 → subnormal. ──
fn decode_sf(b: u8) -> f32 {
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if e == 0 { (m / 8.0) * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

// ── per-(row, 16-block) scale-factor swizzle slot inside a 128-row atom. ──
// The block-scaled config lays scales out in 512-byte atoms over 128 rows × 4
// k-blocks: a row's (r%32, (r/32)%4) selects the MN position, the k-block's
// (kb = blk/4, ks = blk%4) selects the atom and its k-lane. This is the layout
// the kernel reads via `tile_atom_to_shape_SFA/SFB`; the oracle fills + decodes
// the identical slot so the test is layout-faithful, not just value-faithful.
fn sf_slot(row: usize, blk16: usize, k_blocks16: usize) -> usize {
    let k_atoms = k_blocks16.div_ceil(4);
    let (row_block, row_in) = (row / 128, row % 128);
    let (r32, r4) = (row_in % 32, (row_in / 32) % 4);
    let (kb, ks) = (blk16 / 4, blk16 % 4);
    (row_block * k_atoms + kb) * 512 + r32 * 16 + r4 * 4 + ks
}

// f16 bits → f32 (test-only; avoids a `half` dev-dep for the readback).
// Branchless: subnormals are normalized by scaling the f32 result, so there is
// no decrement loop that could underflow in a debug build.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h as u32) & 0x3FF;
    let bits = if exp == 0 {
        // zero or subnormal: value = ±mant·2^-24 (mant·2^-10·2^-14).
        let mag = (mant as f32) * (1.0f32 / 16_777_216.0);
        return if sign != 0 { -mag } else { mag };
    } else if exp == 0x1F {
        sign | (0xFF << 23) | (mant << 13) // inf / nan
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// Tiny LCG so the test is deterministic without a rng dev-dep.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 { lo + self.next_u32() % (hi - lo + 1) }
}

#[test]
fn grouped_nvfp4_moe_gemm_matches_f32_oracle() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping NVFP4 grouped GEMM test");
        return;
    };
    let (maj, min) = dev.compute_capability();
    eprintln!("CUDA device compute capability: sm_{maj}{min}");

    // ── small grouped problem: 4 experts, M per group 16..64, N=256, K=128 ──
    const N: usize = 256;
    const K: usize = 128;
    const K2: usize = K / 2; // packed e2m1 bytes per row
    let kb16 = K.div_ceil(16); // #16-element K-blocks
    let rows: [usize; 4] = [16, 32, 48, 64];
    let expert_ids: [i32; 4] = [0, 1, 2, 3];
    let n_groups = rows.len();
    let n_exp = 4usize;
    let mt: usize = rows.iter().sum();

    let mut rng = Lcg(0x1234_5678_9abc_def0);
    // ue4m3 codes that decode to ~[0.25, 2] keep products inside fp4's [0, 6].
    let nib = |r: &mut Lcg| r.range(0, 15) as u8;
    let sfc = |r: &mut Lcg| r.range(0x28, 0x40) as u8;

    // ── A: packed e2m1 [mt, K/2] + per-group dense SFA blobs ──
    let mut a_bytes = vec![0u8; mt * K2];
    for b in &mut a_bytes {
        *b = nib(&mut rng);
    }
    // sfa_code[g][row*kb16 + blk] — the logical scale codes, for the oracle.
    let mut sfa_code: Vec<Vec<u8>> = Vec::with_capacity(n_groups);
    let mut sfa_off: Vec<i64> = Vec::with_capacity(n_groups);
    let mut sfa_bytes: Vec<u8> = Vec::new();
    for &m in &rows {
        sfa_off.push(sfa_bytes.len() as i64);
        let blob = m.div_ceil(128) * 512 * K.div_ceil(64);
        let start = sfa_bytes.len();
        sfa_bytes.resize(start + blob, 0);
        let mut codes = vec![0u8; m * kb16];
        for r in 0..m {
            for blk in 0..kb16 {
                let c = sfc(&mut rng);
                codes[r * kb16 + blk] = c;
                sfa_bytes[start + sf_slot(r, blk, kb16)] = c;
            }
        }
        sfa_code.push(codes);
    }

    // ── B: per-expert packed e2m1 [n_exp, N, K/2] (row-major n,k) + SFB ──
    let mut b_bytes = vec![0u8; n_exp * N * K2];
    for b in &mut b_bytes {
        *b = nib(&mut rng);
    }
    let sfb_exp = N.div_ceil(128) * 512 * K.div_ceil(64);
    let mut sfb_bytes = vec![0u8; n_exp * sfb_exp];
    let mut sfb_code: Vec<Vec<u8>> = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let mut codes = vec![0u8; N * kb16];
        for n in 0..N {
            for blk in 0..kb16 {
                let c = sfc(&mut rng);
                codes[n * kb16 + blk] = c;
                sfb_bytes[e * sfb_exp + sf_slot(n, blk, kb16)] = c;
            }
        }
        sfb_code.push(codes);
    }

    // ── f32 oracle: out[t,n] = Σ_k A[t,k]·sfa · B[eid][n,k]·sfb ──
    let mut oracle = vec![0.0f32; mt * N];
    let mut rowoff = 0usize;
    for (g, &m) in rows.iter().enumerate() {
        let e = expert_ids[g] as usize;
        for r in 0..m {
            let t = rowoff + r;
            for n in 0..N {
                let mut acc = 0.0f32;
                for k in 0..K {
                    let ab = a_bytes[t * K2 + k / 2];
                    let an = if k & 1 == 1 { (ab >> 4) & 0xF } else { ab & 0xF };
                    let av = decode_fp4(an) * decode_sf(sfa_code[g][r * kb16 + k / 16]);
                    let bb = b_bytes[(e * N + n) * K2 + k / 2];
                    let bn = if k & 1 == 1 { (bb >> 4) & 0xF } else { bb & 0xF };
                    let bv = decode_fp4(bn) * decode_sf(sfb_code[e][n * kb16 + k / 16]);
                    acc += av * bv;
                }
                oracle[t * N + n] = acc;
            }
        }
        rowoff += m;
    }

    // ── upload, run, read back ──
    let da = dev.upload(&a_bytes).expect("upload A");
    let dsfa = dev.upload(&sfa_bytes).expect("upload SFA");
    let db = dev.upload(&b_bytes).expect("upload B");
    let dsfb = dev.upload(&sfb_bytes).expect("upload SFB");
    let dd = dev.alloc(mt * N * 2).expect("alloc D (f16)");

    let group_rows: Vec<i32> = rows.iter().map(|&m| m as i32).collect();
    let res = dev.moe_grouped_cutlass_fp4(
        da.device_ptr(),
        dsfa.device_ptr(),
        db.device_ptr(),
        dsfb.device_ptr(),
        dd.device_ptr(),
        &group_rows,
        &expert_ids,
        &sfa_off,
        0, // alpha_vec = none → alpha 1
        N,
        K,
    );
    if let Err(e) = &res
        && e.to_string().contains("without CUTLASS")
    {
        eprintln!("runtime built without CUTLASS — skipping NVFP4 grouped GEMM test");
        return;
    }
    res.expect("moe_grouped_cutlass_fp4 dispatch");

    let mut out_bytes = vec![0u8; mt * N * 2];
    dev.download(&dd, &mut out_bytes).expect("download D");
    let got: Vec<f32> =
        out_bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_ne_bytes([c[0], c[1]]))).collect();

    // ── compare: cosine + max-abs ──
    let (mut max_abs, mut sgg, mut soo, mut sgo) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, o) in got.iter().zip(&oracle) {
        let (g, o) = (*g as f64, *o as f64);
        max_abs = max_abs.max((g - o).abs());
        sgg += g * g;
        soo += o * o;
        sgo += g * o;
    }
    let cos = sgo / (sgg.sqrt() * soo.sqrt() + 1e-30);
    eprintln!(
        "grouped NVFP4 MoE GEMM: mt={mt} N={N} K={K} groups={n_groups}  \
         maxAbsErr={max_abs:.4}  cos={cos:.6}"
    );

    // fp4 inputs are exact on both sides; only the f16 output round-trip drifts.
    // Output magnitudes reach K·6·6·sf, so the f16 ULP floor is ~O(0.1) — a tight
    // abs bound plus a near-1 cosine pins true correctness.
    assert!(cos > 0.9999, "NVFP4 grouped GEMM cosine too low: {cos:.6}");
    assert!(max_abs < 1.0, "NVFP4 grouped GEMM max|Δ| too high: {max_abs:.4}");
}
