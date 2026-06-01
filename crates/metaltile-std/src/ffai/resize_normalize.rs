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

#[kernel(
    bench(
        op="resize",
        subop="resize_normalize",
        class=GenericEmpty,
        tol=1e-4,
        kernel_mode=Grid3D,
    )
)]
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

/// New-syntax correctness for `ffai_resize_normalize` vs a CPU bilinear
/// reference. Grid3D, grid `[target_w, target_h, 3]`, tpg `[1,1,1]`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_resize_normalize;
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
}
