//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Apple GPU family detection.
//!
//! Two orthogonal detection strategies:
//!
//! 1. **Hardware probe** ([`GpuFamily::detect`]) — queries the Metal
//!    runtime for the highest supported `MTLGPUFamily`.  Only works on
//!    macOS with a Metal device.  Returns [`GpuFamily::Unknown`] on
//!    other platforms or when no GPU is available.
//!
//! 2. **Name heuristic** ([`GpuFamily::from_device_name`]) — parses
//!    the human‑readable device name (e.g. `"Apple M4 Max"`).  Works
//!    cross‑platform with no Metal dependency.
//!
//! The two methods agree on real hardware; the name heuristic exists
//! so non‑macOS hosts (and the CLI snapshot format) can still
//! classify a device without linking Metal frameworks.

use std::fmt;

/// Apple GPU family, inferred from hardware or device name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuFamily {
    /// M1 / A14 — Apple GPU Family 7
    Apple7,
    /// M2 / A15 / A16 — Apple GPU Family 8
    Apple8,
    /// M3 / M4 / A17 / A18 — Apple GPU Family 9
    Apple9,
    /// M5 — Apple GPU Family 10
    Apple10,
    /// Unrecognised or no Metal device available.
    Unknown,
}

impl GpuFamily {
    // ── constructors ────────────────────────────────────────────────

    /// Probe the default Metal device for the highest supported GPU
    /// family.
    ///
    /// Apple families are cumulative — a chip that returns `true` for
    /// `Apple10` also returns `true` for `Apple9`/`8`/`7`.  We report
    /// the newest.  Returns [`GpuFamily::Unknown`] off‑macOS or when
    /// no Metal device is available.
    #[cfg(target_os = "macos")]
    pub fn detect() -> Self {
        use objc2_metal::{MTLDevice, MTLGPUFamily};
        let Some(dev) = objc2_metal::MTLCreateSystemDefaultDevice() else {
            return GpuFamily::Unknown;
        };
        for (family, level) in [
            (MTLGPUFamily::Apple10, GpuFamily::Apple10),
            (MTLGPUFamily::Apple9, GpuFamily::Apple9),
            (MTLGPUFamily::Apple8, GpuFamily::Apple8),
            (MTLGPUFamily::Apple7, GpuFamily::Apple7),
        ] {
            if dev.supportsFamily(family) {
                return level;
            }
        }
        GpuFamily::Unknown
    }

    /// Stub for non‑macOS platforms.
    #[cfg(not(target_os = "macos"))]
    pub fn detect() -> Self { GpuFamily::Unknown }

    /// Infer the GPU family from a Metal device name string
    /// (e.g. `"Apple M4 Max"`, `"Apple M1 Pro"`).
    ///
    /// M‑series is checked before A‑series since `"M1 Pro"` contains
    /// no A‑chip substring.  Newer chips are checked first so `"M4"`
    /// doesn't shadow the broader `M5` substring on future strings.
    pub fn from_device_name(name: &str) -> Self {
        if name.contains("M5") {
            GpuFamily::Apple10
        } else if name.contains("M4") || name.contains("M3") {
            GpuFamily::Apple9
        } else if name.contains("M2") {
            GpuFamily::Apple8
        } else if name.contains("M1") {
            GpuFamily::Apple7
        } else if name.contains("A18") || name.contains("A17") {
            GpuFamily::Apple9
        } else if name.contains("A16") || name.contains("A15") {
            GpuFamily::Apple8
        } else if name.contains("A14") {
            GpuFamily::Apple7
        } else {
            GpuFamily::Unknown
        }
    }

    // ── numeric family level ────────────────────────────────────────

    /// The numeric GPU family level: 7 = M1, 8 = M2, 9 = M3/M4,
    /// 10 = M5.  Returns `None` for [`GpuFamily::Unknown`].
    pub fn family_level(self) -> Option<u32> {
        match self {
            GpuFamily::Apple7 => Some(7),
            GpuFamily::Apple8 => Some(8),
            GpuFamily::Apple9 => Some(9),
            GpuFamily::Apple10 => Some(10),
            GpuFamily::Unknown => None,
        }
    }

    // ── capability queries ──────────────────────────────────────────

    /// True for Apple9+ (M3, M4, M5, A17, A18).
    pub const fn is_apple9_or_later(self) -> bool {
        matches!(self, GpuFamily::Apple9 | GpuFamily::Apple10)
    }

    /// Threadgroup memory in KB.  All Apple7‑10 GPUs have 32 KB.
    pub const fn threadgroup_mem_kb(self) -> u32 { 32 }

    /// Maximum threads per threadgroup.  All Apple7‑10 GPUs support 1024.
    pub const fn max_threads_per_threadgroup(self) -> u32 { 1024 }

    // ── display ─────────────────────────────────────────────────────

    /// Human‑readable label for display
    /// (e.g. `"Apple9 (M3+)"`).
    pub fn display_label(self) -> &'static str {
        match self {
            GpuFamily::Apple7 => "Apple7 (M1/A14)",
            GpuFamily::Apple8 => "Apple8 (M2/A15+)",
            GpuFamily::Apple9 => "Apple9 (M3+)",
            GpuFamily::Apple10 => "Apple10 (M5)",
            GpuFamily::Unknown => "unknown",
        }
    }

    /// Bare label used in snapshot metadata (e.g. `"Apple9"`).
    /// Returns `None` for [`GpuFamily::Unknown`].
    pub fn code(self) -> Option<&'static str> {
        match self {
            GpuFamily::Apple7 => Some("Apple7"),
            GpuFamily::Apple8 => Some("Apple8"),
            GpuFamily::Apple9 => Some("Apple9"),
            GpuFamily::Apple10 => Some("Apple10"),
            GpuFamily::Unknown => None,
        }
    }

    // ── static helpers ──────────────────────────────────────────────

    /// Known SLC (System Level Cache) size string per chip tier.
    ///
    /// Returns `"varies"` when the tier is not recognised.
    pub fn slc_label(device_name: &str) -> &'static str {
        if device_name.contains("Ultra") {
            "~96 MB"
        } else if device_name.contains("Max")
            && (device_name.contains("M5") || device_name.contains("M4"))
        {
            // M4/M5 Max share the ~64 MB SLC tier; revisit if M5
            // specs differ.
            "~64 MB"
        } else if device_name.contains("Max") {
            "~48 MB"
        } else {
            "varies"
        }
    }
}

impl fmt::Display for GpuFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.display_label()) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs_on_every_target() {
        let family = GpuFamily::detect();
        // On macOS with a real Apple GPU, returns ≥ Apple7.  On
        // virtualised CI runners (GitHub Actions) or non‑macOS,
        // returns Unknown.  Both are valid.
        if cfg!(target_os = "macos") && std::env::var_os("CI").is_none() {
            assert!(matches!(
                family,
                GpuFamily::Apple7 | GpuFamily::Apple8 | GpuFamily::Apple9 | GpuFamily::Apple10
            ));
        } else {
            assert_eq!(family, GpuFamily::Unknown);
        }
    }

    #[test]
    fn from_device_name_parses_m_series() {
        assert_eq!(GpuFamily::from_device_name("Apple M5 Ultra"), GpuFamily::Apple10);
        assert_eq!(GpuFamily::from_device_name("Apple M4 Max"), GpuFamily::Apple9);
        assert_eq!(GpuFamily::from_device_name("Apple M3"), GpuFamily::Apple9);
        assert_eq!(GpuFamily::from_device_name("Apple M2 Pro"), GpuFamily::Apple8);
        assert_eq!(GpuFamily::from_device_name("Apple M1"), GpuFamily::Apple7);
    }

    #[test]
    fn from_device_name_parses_a_series() {
        assert_eq!(GpuFamily::from_device_name("Apple A18 Pro"), GpuFamily::Apple9);
        assert_eq!(GpuFamily::from_device_name("Apple A16 Bionic"), GpuFamily::Apple8);
        assert_eq!(GpuFamily::from_device_name("Apple A14"), GpuFamily::Apple7);
    }

    #[test]
    fn from_device_name_unknown_for_junk() {
        assert_eq!(GpuFamily::from_device_name("NVIDIA GeForce RTX 4090"), GpuFamily::Unknown);
    }

    #[test]
    fn family_level_mapping() {
        assert_eq!(GpuFamily::Apple7.family_level(), Some(7));
        assert_eq!(GpuFamily::Apple8.family_level(), Some(8));
        assert_eq!(GpuFamily::Apple9.family_level(), Some(9));
        assert_eq!(GpuFamily::Apple10.family_level(), Some(10));
        assert_eq!(GpuFamily::Unknown.family_level(), None);
    }

    #[test]
    fn is_apple9_or_later() {
        assert!(!GpuFamily::Apple7.is_apple9_or_later());
        assert!(!GpuFamily::Apple8.is_apple9_or_later());
        assert!(GpuFamily::Apple9.is_apple9_or_later());
        assert!(GpuFamily::Apple10.is_apple9_or_later());
        assert!(!GpuFamily::Unknown.is_apple9_or_later());
    }

    #[test]
    fn code_and_display_label() {
        assert_eq!(GpuFamily::Apple9.code(), Some("Apple9"));
        assert_eq!(GpuFamily::Unknown.code(), None);
        assert_eq!(GpuFamily::Apple9.display_label(), "Apple9 (M3+)");
    }

    #[test]
    fn slc_label_per_tier() {
        assert_eq!(GpuFamily::slc_label("Apple M4 Ultra"), "~96 MB");
        assert_eq!(GpuFamily::slc_label("Apple M5 Max"), "~64 MB");
        assert_eq!(GpuFamily::slc_label("Apple M4 Max"), "~64 MB");
        assert_eq!(GpuFamily::slc_label("Apple M3 Max"), "~48 MB");
        assert_eq!(GpuFamily::slc_label("Apple M1 Pro"), "varies");
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", GpuFamily::Apple10), "Apple10 (M5)");
    }
}
