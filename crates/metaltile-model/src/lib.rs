//! MetalTile model crate — TOML-defined model architectures.
//!
//! `metaltile-model` provides a declarative system for defining model
//! forward passes via TOML files that specify kernel dispatch ordering,
//! tensor wiring, and buffer lifetimes. The compiler resolves these
//! definitions into `ExecutionPlan`s that can be dispatched on the GPU
//! via `metaltile-runtime`.
//!
//! ## Architecture
//!
//! ```text
//! model.toml ─► compiler.rs ─► ExecutionPlan ─► executor.rs ─► dispatch_chain
//!                    │                              │
//!              schema.rs  registry.rs          plan.rs  liveness.rs
//!              expr.rs
//! ```
//!
//! ## Quick start
//!
//! ```ignore
//! use metaltile_model::{compile, KernelRegistry, CompileParams};
//!
//! let toml_src = std::fs::read_to_string("models/llama_decode.toml")?;
//! let def: ModelDef = toml::from_str(&toml_src)?;
//! let reg = KernelRegistry::build();
//! let plan = compile(&def, &CompileParams { ... }, &reg)?;
//! // plan can now be dispatched: execute_plan(&ctx, &plan, &weights, &state, &resident)
//! ```
//!
//! ## Crate features
//!
//! - `msl-validate`: enables MSL generation for compile-time validation
//!   of kernel IR (requires `metaltile-codegen`).

pub mod compiler;
pub mod error;
pub mod executor;
pub mod expr;
pub mod liveness;
pub mod plan;
pub mod registry;
pub mod schema;

// Re-export the main types for convenience.
pub use compiler::compile;
pub use error::ModelError;
pub use executor::{execute_plan, WeightMap, StateMap};
pub use compiler::CompileParams;
pub use plan::{BufferSlot, ConstexprValue, DispatchNode, ExecutionPlan, SlotRef};
pub use registry::KernelRegistry;
pub use schema::{KernelNode, LayerDef, ModelDef, ModelMeta, TensorDecl};
