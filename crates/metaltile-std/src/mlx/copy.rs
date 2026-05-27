//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal

use metaltile::kernel;

#[kernel(
    bench(
        op="copy",
        subop="copy",
        class=Unary,
        input=Signed,
        tol=1e-6,
        mlx="v_copy{tn}{tn}",
        metal_file="copy.metal",
    )
)]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}
