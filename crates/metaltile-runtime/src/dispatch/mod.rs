//! Dispatch orchestration: single-kernel and fused multi-pass.

#[cfg(any(target_os = "macos", test))]
pub(crate) mod buffer_plan;
#[cfg(target_os = "macos")]
pub(crate) mod chain_dispatch;
#[cfg(target_os = "macos")]
pub(crate) mod single_dispatch;
// Pure geometry validation — no Metal types, host-testable on any platform.
pub(crate) mod validate;
