//! Toolchain plumbing: project discovery, runner ↔ CLI wire protocol,
//! and bench/test infrastructure.
//!
//! These modules are used by the `tile` CLI and the runner binary that the
//! CLI spawns. They are not part of the kernel IR or the DSL type system.

pub mod bench;
pub mod config;
pub mod protocol;
pub mod registry;
pub mod test;
