//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `#[kernel]` rejects `start..` (open-ended range) in `for` loops —
//! the macro needs a static upper bound to generate the IR `Loop` node.

use metaltile::prelude::*;

#[kernel]
fn loop_with_open_ended_range(out: Tensor<f32>) {
    for _i in 0u32.. {
        store(out[0u32], 0.0f32);
    }
}

fn main() {}
