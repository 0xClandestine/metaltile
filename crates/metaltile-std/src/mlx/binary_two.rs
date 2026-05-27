//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! binary_two benchmark — #[kernel] DSL fused two-output elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_binary_two<T>(a: Tensor<T>, b: Tensor<T>, mut c: Tensor<T>, mut d: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    let y = load(b[idx]);
    store(c[idx], x + y);
    store(d[idx], x * y);
}
