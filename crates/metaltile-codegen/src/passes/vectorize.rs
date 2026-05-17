//! Vectorization pass: promote scalar ops to `half4`/`half8`/`float4`/`bfloat4` etc.
//!
//! Scans for consecutive scalar Load/Store ops with contiguous indices and
//! replaces them with vectorized VectorLoad/VectorStore ops.
//!
//! ## v2 changes (CODEGEN_OVERHAUL §4.4)
//!
//! - **BF16 support**: `DType::BF16` params are now vectorizable (`bfloat4` on Metal 3.1+).
//! - **Structural contiguity**: instead of relying on ValueId encoding heuristics,
//!   the pass examines the defining op of each index ValueId.  After ConstFold +
//!   CSE + LICM, consecutive loads at `base+0, base+1, …` show up as
//!   `BinOp(Add, invariant_vid, Const(k))` with incrementing *k*.
//! - **Width 8**: `MAX_VEC_LEN` is 8; the emitter decomposes `float8`/`half8` into
//!   `float2x4` when the native 8-wide vector isn't available.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    error::Result,
    ir::{BinOpKind, Block, BlockId, IndexExpr, Kernel, Op, Param, ValueId},
};

pub struct VectorizePass;

impl super::Pass for VectorizePass {
    fn name(&self) -> &str { "vectorize" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        let params = &kernel.params;
        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                vectorize_block(block, params);
            }
        }
        vectorize_block(&mut kernel.body, params);
        Ok(())
    }
}

/// Maximum vector width to try (2, 4, or 8).
const MAX_VEC_LEN: usize = 8;

#[allow(clippy::needless_range_loop)]
fn vectorize_block(block: &mut Block, params: &[Param]) {
    let n = block.ops.len();
    let mut skip: Vec<bool> = vec![false; n];

    // Phase 1: find contiguous Load sequences.
    for i in 0..n {
        if skip[i] {
            continue;
        }

        if let Op::Load { src, indices, .. } = &block.ops[i] {
            if indices.len() != 1 {
                continue;
            }
            let base_vid = match &indices[0] {
                IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                _ => continue,
            };

            let Some(param) = params.iter().find(|p| &p.name == src) else {
                continue;
            };
            if !is_vectorizable(param.dtype) {
                continue;
            }

            // Use structural analysis: find the (invariant_base, const_offset) for this index.
            let (inv_base, offset) = decompose_index(block, base_vid);

            // Collect a run of loads with same src, same invariant_base, and consecutive offsets.
            let mut run_indices: Vec<usize> = vec![i];
            for j in (i + 1)..n.min(i + MAX_VEC_LEN) {
                if skip[j] {
                    break;
                }
                match &block.ops[j] {
                    Op::Load { src: s2, indices: idx2, .. } if *s2 == *src && idx2.len() == 1 => {
                        let next_base = match &idx2[0] {
                            IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                            _ => break,
                        };
                        let (next_inv, next_off) = decompose_index(block, next_base);
                        if next_inv == inv_base && next_off == offset + (j - i) as i64 {
                            run_indices.push(j);
                        } else {
                            break;
                        }
                    },
                    _ => break,
                }
            }

            if run_indices.len() >= 2 {
                let vlen = run_indices.len() as u32;
                block.ops[i] =
                    Op::VectorLoad { src: src.clone(), byte_offset: base_vid, len: vlen };
                for &idx in run_indices[1..].iter().rev() {
                    skip[idx] = true;
                }
            }
        }
    }

    // Phase 2: find contiguous Store sequences.
    for i in 0..n {
        if skip[i] {
            continue;
        }

        if let Op::Store { dst, indices, .. } = &block.ops[i] {
            if indices.len() != 1 {
                continue;
            }
            let base_vid = match &indices[0] {
                IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                _ => continue,
            };

            let Some(param) = params.iter().find(|p| &p.name == dst) else {
                continue;
            };
            if !is_vectorizable(param.dtype) {
                continue;
            }

            let (inv_base, offset) = decompose_index(block, base_vid);

            let mut run_indices: Vec<usize> = vec![i];
            for j in (i + 1)..n.min(i + MAX_VEC_LEN) {
                if skip[j] {
                    break;
                }
                match &block.ops[j] {
                    Op::Store { dst: d2, indices: idx2, .. } if *d2 == *dst && idx2.len() == 1 => {
                        let next_base = match &idx2[0] {
                            IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                            _ => break,
                        };
                        let (next_inv, next_off) = decompose_index(block, next_base);
                        if next_inv == inv_base && next_off == offset + (j - i) as i64 {
                            run_indices.push(j);
                        } else {
                            break;
                        }
                    },
                    _ => break,
                }
            }

            if run_indices.len() >= 2 {
                let vlen = run_indices.len() as u32;
                let first_value = match &block.ops[i] {
                    Op::Store { value, .. } => *value,
                    _ => continue,
                };
                block.ops[i] = Op::VectorStore {
                    dst: dst.clone(),
                    byte_offset: base_vid,
                    len: vlen,
                    value: first_value,
                };
                for &idx in run_indices[1..].iter().rev() {
                    skip[idx] = true;
                }
            }
        }
    }

    // Phase 3: rebuild the block without skipped ops.
    let mut old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);

    // result_remap: skipped scalar load ValueId → replacement VectorLoad ValueId.
    let mut result_remap: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    let mut i = 0usize;
    while i < n {
        if skip[i] {
            i += 1;
            continue;
        }

        // If this is a VectorLoad (start of a vectorized run), register the result
        // so that subsequent code referencing the old scalar results gets the vector.
        if let Op::VectorLoad { len, .. } = &old_ops[i] {
            let vlen = *len as usize;
            let first_vid = old_results[i].unwrap_or(ValueId::new(0));
            for k in 1..vlen {
                // The i+k load results are now subsumed by the VectorLoad result.
                if let Some(Some(skipped_vid)) = old_results.get(i + k) {
                    result_remap.insert(*skipped_vid, first_vid);
                }
            }
        }

        new_ops.push(std::mem::replace(&mut old_ops[i], Op::Const { value: 0 }));
        new_results.push(old_results[i]);
        i += 1;
    }

    // Remap value references in surviving ops.
    for op in new_ops.iter_mut() {
        remap_values_in_op(op, &result_remap);
    }

    block.ops = new_ops;
    block.results = new_results;
}

/// Decompose a ValueId index into (invariant_base, const_offset).
///
/// If the index is defined by `BinOp(Add, base, Const(k))`, returns `(base, k)`.
/// Otherwise returns `(vid, 0)` — treating the index itself as the base with offset 0.
fn decompose_index(block: &Block, vid: ValueId) -> (ValueId, i64) {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid)) {
            if let Op::BinOp { op: BinOpKind::Add, lhs, rhs } = op {
                if let Some(k) = find_const_in_block(block, *rhs) {
                    return (*lhs, k);
                }
                if let Some(k) = find_const_in_block(block, *lhs) {
                    return (*rhs, k);
                }
            }
            break;
        }
    }
    (vid, 0)
}

/// Check if a ValueId is defined by an Op::Const in this block.
fn find_const_in_block(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::Const { value } = op
        {
            return Some(*value);
        }
    }
    None
}

/// Whether a dtype supports vectorization. BF16 is vectorizable on Metal 3.1+
/// (bfloat2, bfloat4 are valid MSL vector types).
fn is_vectorizable(dtype: DType) -> bool { matches!(dtype, DType::F16 | DType::F32 | DType::BF16) }

fn remap_values_in_op(op: &mut Op, remap: &BTreeMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new_vid) = remap.get(v) {
            *v = new_vid;
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            s(value);
        },
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, .. } => {
            s(value);
        },
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::VectorStore { value, .. } => {
            s(value);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops.iter_mut() {
                remap_values_in_op(sub, remap);
            },
        _ => {},
    }
}
