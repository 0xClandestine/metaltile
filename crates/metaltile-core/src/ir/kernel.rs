//! Kernel structure: [`KernelMode`], [`Block`], [`Kernel`].

use std::{collections::BTreeMap, fmt};

use rustc_hash::FxHashMap;

use super::{
    ids::{BlockId, ValueId},
    op::Op,
    param::{ConstExprDecl, Param},
};
use crate::dsl::shape::Shape;

// ---------------------------------------------------------------------------
// KernelMode
// ---------------------------------------------------------------------------

/// Thread-indexing mode — controls which Metal built-in attributes are emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelMode {
    /// `uint tid [[thread_position_in_grid]]`
    /// Used for flat elementwise kernels.
    #[default]
    Elementwise,
    /// `uint3 _tid3/tgid3/lsize3` with `.x`/`.y` aliases injected.
    /// Used for row-reduction kernels (softmax, rms_norm, layer_norm, …).
    Reduction,
    /// `uint3 gid [[thread_position_in_grid]]`
    /// Used for 3-axis grid kernels (rope).
    Grid3D,
    /// `uint2 tid [[thread_position_in_threadgroup]] + uint2 tgid`
    /// Used for tiled 2-D kernels (gemv, matmul).
    Tile2D,
    /// `uint3 tid [[threadgroup_position_in_grid]]` + `uint3 lid` +
    /// `uint simd_lane` + `uint simd_group`.
    /// Used for tiled simdgroup-matmul kernels (steel GEMM) and
    /// any 3-axis kernel that needs `tgid_z` (e.g. batched SDPA
    /// prefill).
    SimdGroup2D,
}

impl fmt::Display for KernelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KernelMode::Elementwise => "Elementwise",
            KernelMode::Reduction => "Reduction",
            KernelMode::Grid3D => "Grid3D",
            KernelMode::Tile2D => "Tile2D",
            KernelMode::SimdGroup2D => "SimdGroup",
        })
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// A basic block: a sequence of operations with a terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub id: BlockId,
    /// Operations in this block.
    pub ops: Vec<Op>,
    /// Parallel to `ops`: the SSA value ID produced by each op, or `None` for
    /// no-result ops (Store, Loop, Barrier, etc.).
    /// Invariant: `results.len() == ops.len()`.
    pub results: Vec<Option<ValueId>>,
    /// Name hints for values (for debugging and MSL variables).
    pub names: BTreeMap<ValueId, String>,
}

impl Block {
    pub fn new(id: BlockId) -> Self {
        Block { id, ops: Vec::new(), results: Vec::new(), names: BTreeMap::new() }
    }

    /// Push an op that produces a value.
    pub fn push_op(&mut self, op: Op, value_id: ValueId) {
        self.ops.push(op);
        self.results.push(Some(value_id));
    }

    /// Push an op that does not produce a value (Store, Loop, Barrier, etc.).
    pub fn push_op_no_result(&mut self, op: Op) {
        self.ops.push(op);
        self.results.push(None);
    }

    /// Give a name hint to a value for prettier MSL output.
    pub fn name_value(&mut self, id: ValueId, name: impl Into<String>) {
        self.names.insert(id, name.into());
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// A complete kernel in the IR.
#[derive(Debug, PartialEq)]
pub struct Kernel {
    /// Kernel name.
    pub name: String,
    /// Thread-indexing mode — controls which Metal built-in attributes are emitted.
    pub mode: KernelMode,
    /// Input/output parameters (tensors).
    pub params: Vec<Param>,
    /// Constexpr declarations.
    pub constexprs: Vec<ConstExprDecl>,
    /// Entry block of the kernel body.
    pub body: Block,
    /// All blocks in this kernel (including nested loop bodies, etc.).
    pub blocks: FxHashMap<BlockId, Block>,
    /// Return shapes — for each output tensor, the shape of the written region.
    pub return_shapes: Vec<Shape>,
    /// Tile schedule annotations set by SchedulePass.
    /// Keys are ValueId of Dot ops; values are (tile_m, tile_n, tile_k).
    pub tile_annotations: FxHashMap<ValueId, (u32, u32, u32)>,
    /// Per-kernel opt-in for the MFA-style f32→bf16 reinterpret cast.
    pub bfloat_reinterpret_cast: bool,
    /// Per-kernel opt-in for the indirect-dispatch Swift wrapper variant.
    pub wants_indirect_variant: bool,
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        Kernel {
            name: name.into(),
            mode: KernelMode::default(),
            params: Vec::new(),
            constexprs: Vec::new(),
            body: Block::new(BlockId::new(0)),
            // `kernel.blocks` holds NESTED blocks only — loop bodies,
            // if-then/else branches, etc.  The entry block lives at
            // `kernel.body` and is the canonical source of truth for
            // block 0.  Earlier versions also inserted a clone of the
            // entry block here under `BlockId(0)`; that copy was never
            // updated by passes mutating `kernel.body`, and any code
            // walking `kernel.blocks.values()` for analysis ended up
            // reading stale state.  Lookups now go through
            // `kernel.body` for the entry block and `kernel.blocks`
            // for everything else — see [`iter_blocks`] for a unified
            // walk.
            blocks: FxHashMap::default(),
            return_shapes: Vec::new(),
            tile_annotations: FxHashMap::default(),
            bfloat_reinterpret_cast: false,
            wants_indirect_variant: false,
        }
    }

    /// Add a block to the kernel, returning its ID.
    ///
    /// Panics if `block.id == self.body.id` — the entry block lives in
    /// `self.body`, not `self.blocks`.
    pub fn add_block(&mut self, block: Block) -> BlockId {
        debug_assert_ne!(
            block.id, self.body.id,
            "entry block lives in kernel.body, not kernel.blocks"
        );
        let id = block.id;
        self.blocks.insert(id, block);
        id
    }

    /// Iterate every block in the kernel — entry block first, then
    /// nested blocks in `kernel.blocks` insertion order.  Use this when
    /// an analysis needs to walk all SSA defs / uses across the kernel
    /// (liveness, use-counting, identifier scanning).  Walking only
    /// `self.blocks.values()` would skip the entry block; walking only
    /// `self.body` would skip nested loop/if bodies.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        std::iter::once(&self.body).chain(self.blocks.values())
    }

    /// Mutable variant of [`iter_blocks`].  Order matches [`iter_blocks`]:
    /// entry block first, then nested blocks.
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        std::iter::once(&mut self.body).chain(self.blocks.values_mut())
    }

    /// Get a block by ID.
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        if id == self.body.id {
            return Some(&self.body);
        }
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        if id == self.body.id {
            return Some(&mut self.body);
        }
        self.blocks.get_mut(&id)
    }
}

impl Clone for Kernel {
    fn clone(&self) -> Self {
        // `kernel.blocks` no longer holds the entry block — see the
        // `Kernel::new` comment for the data-structure rationale.  A
        // plain field-by-field clone is correct now; previously this
        // re-inserted the entry block under `body.id` to paper over
        // the stale-copy invariant.
        Kernel {
            name: self.name.clone(),
            mode: self.mode,
            params: self.params.clone(),
            constexprs: self.constexprs.clone(),
            body: self.body.clone(),
            blocks: self.blocks.clone(),
            return_shapes: self.return_shapes.clone(),
            tile_annotations: self.tile_annotations.clone(),
            bfloat_reinterpret_cast: self.bfloat_reinterpret_cast,
            wants_indirect_variant: self.wants_indirect_variant,
        }
    }
}

// ── Display / pretty-printing ────────────────────────────────────────────────

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode_str = match self.mode {
            KernelMode::Elementwise => "Elementwise",
            KernelMode::Reduction => "Reduction",
            KernelMode::Grid3D => "Grid3D",
            KernelMode::Tile2D => "Tile2D",
            KernelMode::SimdGroup2D => "SimdGroup2D",
        };
        let params_str: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let io = if p.is_output { "out:" } else { "" };
                format!("{io}{}:{:?}", p.name, p.dtype)
            })
            .collect();
        writeln!(f, "kernel {}  mode={mode_str}  params=[{}]", self.name, params_str.join(", "))?;

        // Entry block
        write!(f, "{}", self.body)?;

        // Nested blocks (sorted by ID)
        let mut block_ids: Vec<BlockId> = self.blocks.keys().copied().collect();
        block_ids.sort_unstable();
        for id in block_ids {
            if id == self.body.id {
                continue;
            }
            if let Some(block) = self.blocks.get(&id) {
                write!(f, "{}", block)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  block b{}:", self.id.as_u32())?;
        for (i, op) in self.ops.iter().enumerate() {
            let result_id = self.results.get(i).and_then(|r| *r);
            if let Some(vid) = result_id {
                write!(f, "    v{:<4} = ", vid.as_u32())?;
            } else {
                write!(f, "         ")?;
            }
            op.fmt_ir(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{Block, BlockId, Kernel, Op, ValueId};
    use crate::{
        dsl::{dtype::DType, shape::Shape},
        ir::{
            op::{BinOpKind, IndexExpr},
            param::{Param, ParamKind},
        },
    };

    #[test]
    fn entry_block_lives_in_body_not_blocks() {
        let mut kernel = Kernel::new("body");
        kernel.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        assert_eq!(kernel.body.ops.len(), 1);
        assert!(
            !kernel.blocks.contains_key(&kernel.body.id),
            "entry block must NOT be present in kernel.blocks"
        );
    }

    #[test]
    fn getters_treat_body_as_authoritative_entry_block() {
        let mut kernel = Kernel::new("body");
        kernel.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        assert_eq!(kernel.get_block(BlockId::new(0)).expect("entry block must exist").ops.len(), 1);

        let body = kernel.get_block_mut(BlockId::new(0)).expect("entry block must be mutable");
        body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        assert_eq!(kernel.body.ops.len(), 2);
        assert_eq!(
            kernel.get_block(BlockId::new(0)).expect("entry block must exist").ops.len(),
            2,
            "get_block(body.id) must reflect post-mutation state"
        );
    }

    #[test]
    fn iter_blocks_yields_body_then_nested() {
        let mut kernel = Kernel::new("iter");
        kernel.body.push_op(Op::Const { value: 10 }, ValueId::new(0));
        let mut nested = Block::new(BlockId::new(1));
        nested.push_op(Op::Const { value: 20 }, ValueId::new(1));
        kernel.add_block(nested);

        let ids: Vec<u32> = kernel.iter_blocks().map(|b| b.id.as_u32()).collect();
        assert_eq!(ids, vec![0, 1], "iter_blocks yields body first then nested");
    }

    #[test]
    fn display_format_shows_kernel_structure() {
        use super::KernelMode;

        let mut k = Kernel::new("mt_vadd");
        k.mode = KernelMode::Elementwise;
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        });
        k.params.push(Param {
            name: "b".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        });
        k.params.push(Param {
            name: "out".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: ParamKind::Tensor,
        });
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
            mask: None,
        });

        let output = format!("{k}");
        assert!(output.contains("kernel mt_vadd"), "should show kernel name: {output}");
        assert!(output.contains("mode=Elementwise"), "should show mode: {output}");
        assert!(output.contains("v0    = ProgramId(axis=0)"), "should show ProgramId: {output}");
        assert!(output.contains("BinOp(Add, v1, v2)"), "should show BinOp: {output}");
        assert!(output.contains("Store(out, v3, [v0])"), "should show Store: {output}");
    }
}
