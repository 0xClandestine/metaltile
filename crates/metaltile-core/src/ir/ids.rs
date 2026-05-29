//! SSA identifier types: [`ValueId`], [`BlockId`], [`VarId`].

/// Unique identifier for a value in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ValueId(u32);

impl ValueId {
    pub const fn new(id: u32) -> Self { ValueId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "%{}", self.0) }
}

/// Unique identifier for a block in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlockId(u32);

impl BlockId {
    pub const fn new(id: u32) -> Self { BlockId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}

/// Unique identifier for a loop/block-level variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VarId(u32);

impl VarId {
    pub const fn new(id: u32) -> Self { VarId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}
