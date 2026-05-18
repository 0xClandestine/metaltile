//! `#[kernel]` rejects `start..=end` (inclusive range) in `for` loops —
//! Metal has no inclusive-loop primitive, so the macro emits a clear
//! diagnostic at parse time instead of letting the codegen panic later.

use metaltile::prelude::*;

#[kernel]
fn loop_with_inclusive_range(out: Tensor<f32>) {
    for _i in 0u32..=8u32 {
        store(out[0u32], 0.0f32);
    }
}

fn main() {}
