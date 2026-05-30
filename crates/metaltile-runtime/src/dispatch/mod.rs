//! Dispatch orchestration: single-kernel and fused multi-pass.

#[cfg(any(target_os = "macos", test))]
pub(crate) mod buffer_plan;
#[cfg(target_os = "macos")]
pub(crate) mod chain_dispatch;
#[cfg(target_os = "macos")]
pub(crate) mod single_dispatch;
// Pure geometry validation — no Metal types. Its only production callers live in
// the macOS-gated single/chain_dispatch, so gate it the same way as buffer_plan;
// the `test` arm keeps it (and its unit tests) compiling/host-testable on every
// platform, including the Linux CI clippy job (without it, the fn is dead code
// off-macOS and `-D warnings` fails).
#[cfg(any(target_os = "macos", test))]
pub(crate) mod validate;
