//! `#[kernel]` rejects `while` loops — the body parser does not lower them
//! and would otherwise silently drop the loop, shipping a kernel that does
//! zero iterations. The macro fails loudly instead.

use metaltile::prelude::*;

#[kernel]
fn kernel_with_while(out: Tensor<f32>) {
    while program_id(0) < 8u32 {
        store(out[0u32], 0.0f32);
    }
}

fn main() {}
