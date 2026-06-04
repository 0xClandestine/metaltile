//! Token-major ↔ head-major transpose for vision / audio attention.
//!
//! A vision (or audio) tower's attention stage-1 — per-head RMSNorm plus
//! `ffai_rope_2d` — emits Q/K/V in **token-major** layout
//! `[n_tokens, n_heads, head_dim]` (one contiguous head block per token).
//! But `ffai_sdpa_bidirectional` reads K/V **head-major**
//! `[n_heads, n_tokens, head_dim]` — its `kv_slab = kvh * kv_stride *
//! head_dim` indexing walks one head's full token run contiguously. This
//! kernel performs that physical reshape on the GPU so the whole attention
//! pipeline (norm → rope → transpose → SDPA) stays GPU-resident instead of
//! bouncing K/V back to the CPU for a re-pack.
//!
//! Pure element copy with an index remap:
//!
//!   in  [n_tokens, n_heads, head_dim]  T   — token-major  (stage-1 output)
//!   out [n_heads, n_tokens, head_dim]  T   — head-major    (SDPA K/V input)
//!
//!   in_idx  = (token * n_heads + head) * head_dim + d
//!   out_idx = (head * n_tokens + token) * head_dim + d
//!
//! Grid3D — one thread per output element, no cross-thread cooperation, so
//! there is no reduction TPG to get wrong (and therefore no machine-freeze
//! hazard the way the SDPA / RMSNorm reduction kernels have).
//!
//! Codegen-only. Correctness validated by `transpose_th_gpu_correctness`
//! and the inline `kernel_tests` oracle.
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: grid = `[n_tokens, n_heads, head_dim]` threadgroups,
//!     tpg = `[1, 1, 1]` (one thread per output element). NEVER a
//!     reduction TPG — this is a genuine one-thread-per-output kernel.
//!   - `n_tokens`, `n_heads`, `head_dim` passed as constexpr MUST equal
//!     the grid dimensions used to dispatch.
//!   - `input` and `out` element counts both == `n_tokens * n_heads *
//!     head_dim`; the buffers must not alias (distinct allocations).

use metaltile::kernel;

#[kernel]
pub fn ffai_transpose_th<T>(
    input: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_tokens: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] head_dim: u32,
) {
    let token = program_id::<0>();
    let head = program_id::<1>();
    let d = program_id::<2>();
    let in_idx = (token * n_heads + head) * head_dim + d;
    let out_idx = (head * n_tokens + token) * head_dim + d;
    store(out[out_idx], load(input[in_idx]));
}

/// New-syntax correctness for `ffai_transpose_th`. Grid3D, grid
/// `[n_tokens, n_heads, head_dim]`, tpg `[1,1,1]`. Oracle moves element
/// `(token, head, d)` of the token-major input to `(head, token, d)` of
/// the head-major output.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_transpose_th;
    use crate::utils::{pack_f32, unpack_f32};

    fn oracle(input: &[f32], n_tokens: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n_tokens * n_heads * head_dim];
        for token in 0..n_tokens {
            for head in 0..n_heads {
                for d in 0..head_dim {
                    let in_idx = (token * n_heads + head) * head_dim + d;
                    let out_idx = (head * n_tokens + token) * head_dim + d;
                    out[out_idx] = input[in_idx];
                }
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [0.0, 0.0, 0.0])]
    fn test_transpose_th(dt: DType) -> TestSetup {
        // GQA K/V shape: a handful of tokens, a couple of heads, head_dim 64.
        let (n_tokens, n_heads, head_dim) = (5usize, 3usize, 64usize);
        let input_f: Vec<f32> =
            (0..n_tokens * n_heads * head_dim).map(|i| ((i % 29) as f32 - 14.0) * 0.1).collect();
        // Round-trip through the dtype so the oracle matches the kernel's
        // (lossy for f16 / bf16) stored values exactly — a pure copy, so
        // tol = 0.
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let exp = oracle(&input, n_tokens, n_heads, head_dim);
        TestSetup::new(ffai_transpose_th::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::zeros("out", n_tokens * n_heads * head_dim, dt))
            .constexpr("n_tokens", n_tokens as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n_tokens as u32, n_heads as u32, head_dim as u32, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_transpose_th` at the SigLIP production
/// shape (576 patches, 16 heads, head_dim 64).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_transpose_th;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_transpose_th(dt: DType) -> BenchSetup {
        let (n_tokens, n_heads, head_dim) = (576usize, 16usize, 64usize);
        BenchSetup::new(ffai_transpose_th::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n_tokens * n_heads * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * n_heads * head_dim, dt).output())
            .constexpr("n_tokens", n_tokens as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("head_dim", head_dim as u32)
            .with_shape_label(format!(
                "tok{n_tokens} h{n_heads} d{head_dim} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(n_tokens as u32, n_heads as u32, head_dim as u32, [1, 1, 1])
            .bytes_moved((2 * n_tokens * n_heads * head_dim * dt.size_bytes()) as u64)
    }
}
