//! `#[kernel]` rejects `return` — the body parser does not lower it and
//! would otherwise silently drop it, letting execution fall through. The
//! macro fails loudly instead.

use metaltile::prelude::*;

#[kernel]
fn kernel_with_return(out: Tensor<f32>) {
    let idx = program_id(0);
    if idx > 8u32 {
        return;
    }
    store(out[idx], 0.0f32);
}

fn main() {}
