//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Random benchmark — #[kernel] DSL vs MLX metal/random.metal

use metaltile::kernel;

#[kernel]
pub fn mt_random_hash(out: Tensor<u32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let mut s = gid + 1u32;
    s = s ^ (s << 13u32);
    s = s ^ (s >> 17u32);
    s = s ^ (s << 5u32);
    store(out[gid], s);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn cpu_random_hash(n: usize) -> Vec<u32> {
        (0..n)
            .map(|gid| {
                let mut s = gid as u32 + 1;
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
            .collect()
    }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    #[test_kernel(name = "mlx/random/hash_oracle", dtypes = [u32], tol = 0.0)]
    fn test_random_hash_oracle(dt: DType) -> TestSetup {
        let n = 1024usize;
        let expected = cpu_random_hash(n);
        let _ = dt; // single non-generic kernel
        TestSetup::new(mt_random_hash::kernel_ir_for())
            .expect(TestBuffer::from_vec("out", pack_u32(&expected), DType::U32))
            .constexpr("n", n as u32)
            .grid_1d(n, 1024)
    }
}
