//! Variable naming, index expression emission, and tile allocation helpers.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    ir::{Block, IndexExpr, Kernel, Op, ParamKind, ValueId},
    shape::Shape,
};

use super::{MslGenerator, matmul::dim_to_msl_str};

impl MslGenerator {
    /// Return the MSL type name for a dtype, respecting the `native_bfloat` config flag.
    pub(crate) fn msl_type_name(&self, dtype: DType) -> &'static str {
        if dtype == DType::BF16 && !self.config.native_bfloat {
            "bfloat16_t"
        } else {
            dtype.msl_name()
        }
    }

    /// Emit a cast expression: `bfloat(val)` for BF16, `static_cast<T>(val)` otherwise.
    pub(crate) fn emit_cast_expr(&self, dtype: DType, value: &str) -> String {
        if dtype == DType::BF16 {
            format!("{}({value})", self.msl_type_name(dtype))
        } else {
            format!("static_cast<{}>({value})", self.msl_type_name(dtype))
        }
    }

    /// Resolve a `ValueId` to its MSL variable name string.
    pub(super) fn vname(
        &self,
        vid: Option<ValueId>,
        block: &Block,
        extra_names: &BTreeMap<ValueId, String>,
    ) -> String {
        let vid = match vid {
            Some(v) => v,
            None => return "_".into(),
        };
        if let Some(name) = extra_names.get(&vid) {
            return name.clone();
        }
        if let Some(hint) = block.names.get(&vid) {
            return format!("v_{hint}");
        }
        format!("v{}", vid.as_u32())
    }

    /// Emit a flat index expression for a Load/Store, handling 1-D, multi-dim, and strided params.
    pub(super) fn emit_idx(
        &self,
        indices: &[IndexExpr],
        block: &Block,
        extra_names: &BTreeMap<ValueId, String>,
        kernel: &Kernel,
        src_or_dst: &str,
    ) -> String {
        let is_strided = kernel
            .params
            .iter()
            .any(|p| p.name == src_or_dst && matches!(p.kind, ParamKind::Strided));

        if is_strided {
            // Strided indexing: use shape/stride arrays for each dimension.
            let parts: Vec<String> = indices
                .iter()
                .enumerate()
                .map(|(dim, ix)| {
                    let ix_str = self.idx_expr_str(ix, block, extra_names);
                    let stride = format!("{}_strides[{dim}]", src_or_dst);
                    format!("({ix_str}) * {stride}")
                })
                .collect();
            parts.join(" + ")
        } else if indices.len() == 1 {
            self.idx_expr_str(&indices[0], block, extra_names)
        } else {
            // Multi-dim into flat: N is first stride, 1 is last stride.
            let param = kernel.params.iter().find(|p| p.name == src_or_dst);
            let shape = param.map(|p| &p.shape);
            let stride1 = shape.and_then(|s| s.dim(1)).map(dim_to_msl_str).unwrap_or("1".into());
            let mut offset = String::new();
            for (dim, ix) in indices.iter().enumerate() {
                let ix_str = self.idx_expr_str(ix, block, extra_names);
                if dim == 0 {
                    offset.push_str(&format!("({ix_str}) * {stride1}"));
                } else {
                    offset.push_str(&format!(" + ({ix_str})"));
                }
            }
            offset
        }
    }

    pub(super) fn idx_expr_str(
        &self,
        ix: &IndexExpr,
        block: &Block,
        extra_names: &BTreeMap<ValueId, String>,
    ) -> String {
        match ix {
            IndexExpr::Value(v) => self.vname(Some(*v), block, extra_names),
            IndexExpr::Const(n) => n.to_string(),
            IndexExpr::Range(v, _) => self.vname(Some(*v), block, extra_names),
        }
    }

    pub(super) fn shape_nelems_str(&self, shape: &Shape) -> String {
        shape.num_elements().map(|n| n.to_string()).unwrap_or_else(|| {
            let rank = shape.rank();
            (0..rank)
                .filter_map(|i| shape.dim(i))
                .map(dim_to_msl_str)
                .collect::<Vec<_>>()
                .join(" * ")
        })
    }

    pub(super) fn emit_tile_alloc(
        &self,
        dtype: &DType,
        shape: &Shape,
        name: &str,
        fill: f64,
    ) -> (String, Vec<String>) {
        let t = self.msl_type_name(*dtype);
        let n = self.shape_nelems_str(shape);
        let decl = format!("{t} {name}[{n}]");
        let init = if fill == 0.0 {
            vec![format!("for (uint _i = 0; _i < {n}; _i++) {name}[_i] = 0;")]
        } else {
            vec![format!("for (uint _i = 0; _i < {n}; _i++) {name}[_i] = {fill};")]
        };
        (decl, init)
    }
}

/// Extract the result `ValueId` encoded inside certain leaf ops (used by fused expression emission).
pub(super) fn op_to_vid(op: &Op) -> ValueId {
    match op {
        Op::ProgramId { .. } => ValueId::new(0),
        Op::Const { value } => ValueId::new(*value as u32),
        _ => ValueId::new(0),
    }
}
