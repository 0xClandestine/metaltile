//! Dispatch orchestration: single-kernel and fused multi-pass.

pub(crate) mod buffer_plan;
#[cfg(target_os = "macos")]
pub(crate) mod chain_dispatch;
#[cfg(target_os = "macos")]
pub(crate) mod single_dispatch;
