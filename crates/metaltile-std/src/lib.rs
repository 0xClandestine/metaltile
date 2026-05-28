pub mod ffai;
pub mod mlx;
pub mod utils;

// Link anchor: referenced by __tile_runner in metaltile-cli/src/runner_main.rs
// to pull this crate's single codegen unit into the binary so that all
// inventory::submit! kernel registrations are included. Zero runtime cost.
#[doc(hidden)]
pub static __STD_LINK_ANCHOR: () = ();
