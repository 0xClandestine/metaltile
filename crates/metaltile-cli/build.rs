//! Force all object files from every static library into `__tile_runner` so
//! that `inventory::submit!` kernel registrations (bench + test) defined in
//! any crate (e.g. `metaltile-std`) are included in the link even when no
//! symbol from that crate is directly referenced by the runner entry point.
//!
//! Rust's per-crate name mangling guarantees unique symbols, so loading all
//! objects never causes duplicate-symbol errors.

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-arg-bin=__tile_runner=-Wl,-all_load");
        }
        "linux" | "android" => {
            println!("cargo:rustc-link-arg-bin=__tile_runner=-Wl,--whole-archive");
            println!("cargo:rustc-link-arg-bin=__tile_runner=-Wl,--no-whole-archive");
        }
        _ => {}
    }
}
