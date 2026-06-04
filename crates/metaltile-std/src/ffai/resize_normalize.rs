//! Fused image resize (bilinear) + per-channel normalize + interleaved→NCHW.
//!
//! The single GPU op every VL preprocess needs: take an interleaved
//! `[src_h, src_w, 3]` source image (float `[0,1]`), bilinearly resample to
//! `target_h × target_w`, normalize each channel `(v − mean[c]) / std[c]`,
//! and write the planar NCHW `[3, target_h, target_w]` tensor the patch-embed
//! / conv stem consumes — all in one dispatch, replacing the scalar CPU
//! triple-loop in `ImagePreprocessing.resize` + `preprocess`.
//!
//! Bilinear matches `align_corners=false` (half-pixel centers), the HF /
//! transformers image-processor convention:
//!   `src = (out + 0.5)·scale − 0.5`, clamped to `[0, src_dim − 1]`.
//!
//! `src_w/src_h/target_w/target_h` are 1-element `u32` buffers (runtime
//! scalars) so ONE compiled kernel serves every (variable-resolution) image
//! size instead of specialising per shape. `mean`/`std` are `[3]` f32.
//!
//! Grid3D — one thread per output element `(ox, oy, c)`, no cooperation, so
//! no reduction TPG / freeze hazard.
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: grid = `[target_w, target_h, 3]` threadgroups, tpg `[1,1,1]`.
//!   - `input` count == `src_h · src_w · 3`; `out` count == `3 · target_h ·
//!     target_w`; `mean`/`std` length 3; the four dim buffers are 1-element
//!     and equal the grid extents (`target_w/target_h`) / source dims.

use metaltile::kernel;

#[kernel]
pub fn ffai_resize_normalize<T>(
    input: Tensor<T>,
    mean: Tensor<f32>,
    std: Tensor<f32>,
    out: Tensor<T>,
    src_w: Tensor<u32>,
    src_h: Tensor<u32>,
    target_w: Tensor<u32>,
    target_h: Tensor<u32>,
) {
    let ox = program_id::<0>();
    let oy = program_id::<1>();
    let c = program_id::<2>();

    let sw = load(src_w[0]);
    let sh = load(src_h[0]);
    let tw = load(target_w[0]);
    let th = load(target_h[0]);
    let sw_f = sw.cast::<f32>();
    let sh_f = sh.cast::<f32>();
    let sw_m1 = sw_f - 1.0f32;
    let sh_m1 = sh_f - 1.0f32;
    let scale_x = sw_f / tw.cast::<f32>();
    let scale_y = sh_f / th.cast::<f32>();

    // Half-pixel source coords, clamped into the valid range (the DSL has
    // no `clamp` — clamp = `select(v<lo,lo, select(v>hi,hi,v))`).
    let sx_raw = (ox.cast::<f32>() + 0.5f32) * scale_x - 0.5f32;
    let sx_lo = select(sx_raw < 0.0f32, 0.0f32, sx_raw);
    let src_x = select(sx_lo > sw_m1, sw_m1, sx_lo);
    let sy_raw = (oy.cast::<f32>() + 0.5f32) * scale_y - 0.5f32;
    let sy_lo = select(sy_raw < 0.0f32, 0.0f32, sy_raw);
    let src_y = select(sy_lo > sh_m1, sh_m1, sy_lo);

    let x0 = floor(src_x);
    let y0 = floor(src_y);
    let wx = src_x - x0;
    let wy = src_y - y0;
    let x0u = x0.cast::<u32>();
    let y0u = y0.cast::<u32>();
    let x1 = x0 + 1.0f32;
    let x1c = select(x1 > sw_m1, sw_m1, x1);
    let x1u = x1c.cast::<u32>();
    let y1 = y0 + 1.0f32;
    let y1c = select(y1 > sh_m1, sh_m1, y1);
    let y1u = y1c.cast::<u32>();

    // Interleaved source index: (y * src_w + x) * 3 + c.
    let p00 = load(input[(y0u * sw + x0u) * 3u32 + c]).cast::<f32>();
    let p01 = load(input[(y0u * sw + x1u) * 3u32 + c]).cast::<f32>();
    let p10 = load(input[(y1u * sw + x0u) * 3u32 + c]).cast::<f32>();
    let p11 = load(input[(y1u * sw + x1u) * 3u32 + c]).cast::<f32>();
    let top = p00 * (1.0f32 - wx) + p01 * wx;
    let bot = p10 * (1.0f32 - wx) + p11 * wx;
    let v = top * (1.0f32 - wy) + bot * wy;

    let m = load(mean[c]);
    let s = load(std[c]);
    let normed = (v - m) / s;

    // Planar NCHW index: (c * target_h + oy) * target_w + ox.
    store(out[(c * th + oy) * tw + ox], normed.cast::<T>());
}

/// Higher-quality **bicubic** variant — a 4×4 cubic-convolution sampler
/// (Keys, a = −0.75) instead of the 2×2 bilinear above, preserving edges far
/// better when down-scaling a large photo to a tower's native grid (where
/// bilinear aliases). Same fused interface; matches
/// `torch.interpolate(mode='bicubic', align_corners=False)`: half-pixel
/// centers, 4 taps `floor(src)−1…+2`, edge-clamp per tap, partition-of-unity
/// weights (no renormalize). Per-tap branch specialization is exact at
/// boundaries (cubic1(1)=cubic2(1)=0).
#[kernel]
pub fn ffai_resize_normalize_bicubic<T>(
    input: Tensor<T>,
    mean: Tensor<f32>,
    std: Tensor<f32>,
    out: Tensor<T>,
    src_w: Tensor<u32>,
    src_h: Tensor<u32>,
    target_w: Tensor<u32>,
    target_h: Tensor<u32>,
) {
    let ox = program_id::<0>();
    let oy = program_id::<1>();
    let c = program_id::<2>();

    let sw = load(src_w[0]);
    let sh = load(src_h[0]);
    let tw = load(target_w[0]);
    let th = load(target_h[0]);
    let sw_f = sw.cast::<f32>();
    let sh_f = sh.cast::<f32>();
    let sw_m1 = sw_f - 1.0f32;
    let sh_m1 = sh_f - 1.0f32;
    let scale_x = sw_f / tw.cast::<f32>();
    let scale_y = sh_f / th.cast::<f32>();

    // Half-pixel source center (UNclamped — each of the 4 taps clamps its
    // own coordinate below; the fractional offset must stay true near edges).
    let src_x = (ox.cast::<f32>() + 0.5f32) * scale_x - 0.5f32;
    let src_y = (oy.cast::<f32>() + 0.5f32) * scale_y - 0.5f32;
    let fx = floor(src_x);
    let fy = floor(src_y);
    let tx = src_x - fx;
    let ty = src_y - fy;

    // Keys cubic-convolution weights, a = -0.75. Taps sit at fx-1 … fx+2;
    // their |distance| from src_x is (tx+1), tx, (1-tx), (2-tx). The first
    // and last fall in 1 < |d| < 2 (the `cubic2` branch); the middle two in
    // |d| ≤ 1 (the `cubic1` branch). Specializing per tap avoids a runtime
    // branch and is exact at the boundaries (cubic1(1)=cubic2(1)=0).
    let a = -0.75f32;
    let a2 = a + 2.0f32;
    let a3 = a + 3.0f32;
    // cubic1(t) = (a+2)t³ − (a+3)t² + 1   (for the |d| ≤ 1 taps)
    let tx2 = tx * tx;
    let tx3 = tx2 * tx;
    let wx1 = a2 * tx3 - a3 * tx2 + 1.0f32;
    let om = 1.0f32 - tx;
    let om2 = om * om;
    let om3 = om2 * om;
    let wx2 = a2 * om3 - a3 * om2 + 1.0f32;
    // cubic2(t) = a·t³ − 5a·t² + 8a·t − 4a   (for the 1 < |d| < 2 taps)
    let p1 = tx + 1.0f32;
    let p1_2 = p1 * p1;
    let p1_3 = p1_2 * p1;
    let wx0 = a * p1_3 - 5.0f32 * a * p1_2 + 8.0f32 * a * p1 - 4.0f32 * a;
    let p2 = 2.0f32 - tx;
    let p2_2 = p2 * p2;
    let p2_3 = p2_2 * p2;
    let wx3 = a * p2_3 - 5.0f32 * a * p2_2 + 8.0f32 * a * p2 - 4.0f32 * a;
    // Same in y.
    let ty2 = ty * ty;
    let ty3 = ty2 * ty;
    let wy1 = a2 * ty3 - a3 * ty2 + 1.0f32;
    let omy = 1.0f32 - ty;
    let omy2 = omy * omy;
    let omy3 = omy2 * omy;
    let wy2 = a2 * omy3 - a3 * omy2 + 1.0f32;
    let q1 = ty + 1.0f32;
    let q1_2 = q1 * q1;
    let q1_3 = q1_2 * q1;
    let wy0 = a * q1_3 - 5.0f32 * a * q1_2 + 8.0f32 * a * q1 - 4.0f32 * a;
    let q2 = 2.0f32 - ty;
    let q2_2 = q2 * q2;
    let q2_3 = q2_2 * q2;
    let wy3 = a * q2_3 - 5.0f32 * a * q2_2 + 8.0f32 * a * q2 - 4.0f32 * a;

    // 4×4 weighted sum, each tap clamped into the image (edge replicate).
    let mut acc = 0.0f32;
    for j in range(0u32, 4u32, 1u32) {
        let yj = fy + j.cast::<f32>() - 1.0f32;
        let yj_lo = select(yj < 0.0f32, 0.0f32, yj);
        let yj_c = select(yj_lo > sh_m1, sh_m1, yj_lo);
        let yu = yj_c.cast::<u32>();
        let wy = select(j == 0u32, wy0, select(j == 1u32, wy1, select(j == 2u32, wy2, wy3)));
        let row = yu * sw;
        for i in range(0u32, 4u32, 1u32) {
            let xi = fx + i.cast::<f32>() - 1.0f32;
            let xi_lo = select(xi < 0.0f32, 0.0f32, xi);
            let xi_c = select(xi_lo > sw_m1, sw_m1, xi_lo);
            let xu = xi_c.cast::<u32>();
            let wx = select(i == 0u32, wx0, select(i == 1u32, wx1, select(i == 2u32, wx2, wx3)));
            let px = load(input[(row + xu) * 3u32 + c]).cast::<f32>();
            acc = acc + wy * wx * px;
        }
    }

    let m = load(mean[c]);
    let s = load(std[c]);
    let normed = (acc - m) / s;
    store(out[(c * th + oy) * tw + ox], normed.cast::<T>());
}

/// New-syntax correctness for `ffai_resize_normalize` vs a CPU bilinear
/// reference. Grid3D, grid `[target_w, target_h, 3]`, tpg `[1,1,1]`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{ffai_resize_normalize, ffai_resize_normalize_bicubic};
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ref(
        src: &[f32],
        sw: usize,
        sh: usize,
        tw: usize,
        th: usize,
        mean: &[f32],
        std: &[f32],
    ) -> Vec<f32> {
        let scale_x = sw as f32 / tw as f32;
        let scale_y = sh as f32 / th as f32;
        let mut out = vec![0.0f32; 3 * th * tw];
        for oy in 0..th {
            let sy = ((oy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, sh as f32 - 1.0);
            let y0 = sy.floor();
            let wy = sy - y0;
            let y0u = y0 as usize;
            let y1u = (y0 + 1.0).clamp(0.0, sh as f32 - 1.0) as usize;
            for ox in 0..tw {
                let sx = ((ox as f32 + 0.5) * scale_x - 0.5).clamp(0.0, sw as f32 - 1.0);
                let x0 = sx.floor();
                let wx = sx - x0;
                let x0u = x0 as usize;
                let x1u = (x0 + 1.0).clamp(0.0, sw as f32 - 1.0) as usize;
                for c in 0..3 {
                    let p00 = src[(y0u * sw + x0u) * 3 + c];
                    let p01 = src[(y0u * sw + x1u) * 3 + c];
                    let p10 = src[(y1u * sw + x0u) * 3 + c];
                    let p11 = src[(y1u * sw + x1u) * 3 + c];
                    let top = p00 * (1.0 - wx) + p01 * wx;
                    let bot = p10 * (1.0 - wx) + p11 * wx;
                    let v = top * (1.0 - wy) + bot * wy;
                    out[(c * th + oy) * tw + ox] = (v - mean[c]) / std[c];
                }
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_resize_normalize(dt: DType) -> TestSetup {
        // Up-size a small non-square source (exercises both scale dirs).
        let (sw, sh, tw, th) = (5usize, 4usize, 8usize, 6usize);
        let mean = [0.5f32, 0.45, 0.4];
        let std = [0.5f32, 0.5, 0.5];
        let src_f: Vec<f32> = (0..sh * sw * 3).map(|i| ((i % 17) as f32) / 17.0).collect();
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let exp = cpu_ref(&src, sw, sh, tw, th, &mean, &std);
        TestSetup::new(ffai_resize_normalize::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec(
                "mean",
                mean.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec(
                "std",
                std.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", 3 * th * tw, dt))
            .input(TestBuffer::from_vec("src_w", u32_bytes(sw as u32), DType::U32))
            .input(TestBuffer::from_vec("src_h", u32_bytes(sh as u32), DType::U32))
            .input(TestBuffer::from_vec("target_w", u32_bytes(tw as u32), DType::U32))
            .input(TestBuffer::from_vec("target_h", u32_bytes(th as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(tw as u32, th as u32, 3, [1, 1, 1])
    }

    // ── Bicubic variant ──
    // Keys cubic-convolution weight for one tap distance `d` (a = -0.75).
    fn cubic(d: f32, a: f32) -> f32 {
        let ad = d.abs();
        if ad <= 1.0 {
            (a + 2.0) * ad * ad * ad - (a + 3.0) * ad * ad + 1.0
        } else if ad < 2.0 {
            a * ad * ad * ad - 5.0 * a * ad * ad + 8.0 * a * ad - 4.0 * a
        } else {
            0.0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ref_bicubic(
        src: &[f32],
        sw: usize,
        sh: usize,
        tw: usize,
        th: usize,
        mean: &[f32],
        std: &[f32],
    ) -> Vec<f32> {
        let a = -0.75f32;
        let scale_x = sw as f32 / tw as f32;
        let scale_y = sh as f32 / th as f32;
        let clamp = |v: isize, hi: usize| -> usize { v.max(0).min(hi as isize) as usize };
        let mut out = vec![0.0f32; 3 * th * tw];
        for oy in 0..th {
            let sy = (oy as f32 + 0.5) * scale_y - 0.5;
            let fy = sy.floor();
            for ox in 0..tw {
                let sx = (ox as f32 + 0.5) * scale_x - 0.5;
                let fx = sx.floor();
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for j in 0..4 {
                        let yj = fy as isize + j - 1;
                        let wy = cubic(sy - (fy + (j - 1) as f32), a);
                        let yu = clamp(yj, sh - 1);
                        for i in 0..4 {
                            let xi = fx as isize + i - 1;
                            let wx = cubic(sx - (fx + (i - 1) as f32), a);
                            let xu = clamp(xi, sw - 1);
                            acc += wy * wx * src[(yu * sw + xu) * 3 + c];
                        }
                    }
                    out[(c * th + oy) * tw + ox] = (acc - mean[c]) / std[c];
                }
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-4, 5e-3, 3e-2])]
    fn test_resize_normalize_bicubic(dt: DType) -> TestSetup {
        // Down-size a non-square source (the case bicubic improves on).
        let (sw, sh, tw, th) = (10usize, 8usize, 5usize, 6usize);
        let mean = [0.5f32, 0.45, 0.4];
        let std = [0.5f32, 0.5, 0.5];
        let src_f: Vec<f32> = (0..sh * sw * 3).map(|i| ((i % 23) as f32) / 23.0).collect();
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let exp = cpu_ref_bicubic(&src, sw, sh, tw, th, &mean, &std);
        TestSetup::new(ffai_resize_normalize_bicubic::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec(
                "mean",
                mean.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec(
                "std",
                std.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", 3 * th * tw, dt))
            .input(TestBuffer::from_vec("src_w", u32_bytes(sw as u32), DType::U32))
            .input(TestBuffer::from_vec("src_h", u32_bytes(sh as u32), DType::U32))
            .input(TestBuffer::from_vec("target_w", u32_bytes(tw as u32), DType::U32))
            .input(TestBuffer::from_vec("target_h", u32_bytes(th as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(tw as u32, th as u32, 3, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-4, 5e-3, 3e-2])]
    fn test_resize_normalize_bicubic_up(dt: DType) -> TestSetup {
        let (sw, sh, tw, th) = (5usize, 4usize, 8usize, 7usize);
        let mean = [0.5f32, 0.45, 0.4];
        let std = [0.5f32, 0.5, 0.5];
        let src_f: Vec<f32> = (0..sh * sw * 3).map(|i| ((i % 17) as f32) / 17.0).collect();
        let src = unpack_f32(&pack_f32(&src_f, dt), dt);
        let exp = cpu_ref_bicubic(&src, sw, sh, tw, th, &mean, &std);
        TestSetup::new(ffai_resize_normalize_bicubic::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&src_f, dt), dt))
            .input(TestBuffer::from_vec(
                "mean",
                mean.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec(
                "std",
                std.iter().flat_map(|x| x.to_le_bytes()).collect(),
                DType::F32,
            ))
            .input(TestBuffer::zeros("out", 3 * th * tw, dt))
            .input(TestBuffer::from_vec("src_w", u32_bytes(sw as u32), DType::U32))
            .input(TestBuffer::from_vec("src_h", u32_bytes(sh as u32), DType::U32))
            .input(TestBuffer::from_vec("target_w", u32_bytes(tw as u32), DType::U32))
            .input(TestBuffer::from_vec("target_h", u32_bytes(th as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(tw as u32, th as u32, 3, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_resize_normalize` at a representative VL
/// preprocess shape (≈640×480 source → 448×448).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_resize_normalize, ffai_resize_normalize_bicubic};

    #[bench(name = "ffai/resize/resize_normalize", dtypes = [f32, f16, bf16])]
    fn bench_resize_normalize(dt: DType) -> BenchSetup {
        let (sw, sh, tw, th) = (640usize, 480usize, 448usize, 448usize);
        BenchSetup::new(ffai_resize_normalize::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", sh * sw * 3, dt))
            .buffer(BenchBuffer::random("mean", 3, DType::F32))
            .buffer(BenchBuffer::random("std", 3, DType::F32))
            .buffer(BenchBuffer::zeros("out", 3 * th * tw, dt).output())
            .buffer(BenchBuffer::from_vec("src_w", (sw as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::from_vec("src_h", (sh as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::from_vec(
                "target_w",
                (tw as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "target_h",
                (th as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .with_shape_label(format!(
                "{sw}x{sh}->{tw}x{th} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(tw as u32, th as u32, 3, [1, 1, 1])
            .bytes_moved(((sh * sw * 3 + 3 * th * tw) * dt.size_bytes()) as u64)
            // 4-tap bilinear (4 MAC = 8) + normalize (2) per output element.
            .flops((3 * th * tw) as u64 * 10)
    }

    #[bench(name = "ffai/resize/resize_normalize_bicubic", dtypes = [f32, f16, bf16])]
    fn bench_resize_normalize_bicubic(dt: DType) -> BenchSetup {
        let (sw, sh, tw, th) = (640usize, 480usize, 448usize, 448usize);
        BenchSetup::new(ffai_resize_normalize_bicubic::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", sh * sw * 3, dt))
            .buffer(BenchBuffer::random("mean", 3, DType::F32))
            .buffer(BenchBuffer::random("std", 3, DType::F32))
            .buffer(BenchBuffer::zeros("out", 3 * th * tw, dt).output())
            .buffer(BenchBuffer::from_vec("src_w", (sw as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::from_vec("src_h", (sh as u32).to_le_bytes().to_vec(), DType::U32))
            .buffer(BenchBuffer::from_vec(
                "target_w",
                (tw as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .buffer(BenchBuffer::from_vec(
                "target_h",
                (th as u32).to_le_bytes().to_vec(),
                DType::U32,
            ))
            .with_shape_label(format!(
                "{sw}x{sh}->{tw}x{th} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(tw as u32, th as u32, 3, [1, 1, 1])
            .bytes_moved(((sh * sw * 3 + 3 * th * tw) * dt.size_bytes()) as u64)
            // 4×4-tap bicubic (16 MAC = 32) + normalize (2) per output element.
            .flops((3 * th * tw) as u64 * 34)
    }
}
