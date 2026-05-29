//! Kernel-level parameter types: [`Param`], [`ParamKind`], [`TypedSlot`], [`ConstExprDecl`].

use crate::dsl::{constexpr::ConstExpr, dtype::DType, shape::Shape};

/// How a kernel parameter is bound and represented in MSL.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ParamKind {
    /// `device T*` — a flat tensor buffer (default).
    #[default]
    Tensor,
    /// `device T*` + `constant uint* name_shape` + `constant uint* name_strides`
    /// — a strided tensor that also passes its shape and stride arrays.
    Strided,
    /// `constant T& name` — a single scalar value (e.g., `eps`, `scale`, `n`).
    Scalar,
}

/// A kernel parameter: a tensor input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// Human-readable name.
    pub name: String,
    /// Data type of the tensor elements.
    pub dtype: DType,
    /// Shape of the tensor.
    pub shape: Shape,
    /// Whether this is read-write (output) or read-only (input).
    pub is_output: bool,
    /// How this parameter is bound in Metal.
    pub kind: ParamKind,
}

/// A typed slot: used for inline MSL outputs and other typed holes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedSlot {
    pub dtype: DType,
    pub shape: Shape,
}

/// A constexpr declaration in the kernel signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstExprDecl {
    pub name: ConstExpr,
    /// The scalar type of this constexpr (default `U32`).
    pub dtype: DType,
    /// Optional fixed value if known at definition time.
    pub value: Option<usize>,
}
