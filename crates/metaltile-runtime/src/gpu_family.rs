//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Apple GPU family detection from Metal device name strings.
//!
//! Provides a lightweight [`GpuFamily`] enum with a `from_device_name`
//! constructor that uses substring heuristics. This is intentionally
//! a pure-data type with no platform dependencies so all crates can
//! use it.

use std::fmt;

/// Apple GPU family, inferred from device name.
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
    /// Unrecognised device name.
    Unknown,
}

impl GpuFamily {
    /// Infer the GPU family from a Metal device name string
    /// (e.g. `"Apple M4 Max"`, `"Apple M1 Pro"`).
    pub fn from_device_name(name: &str) -> Self {
        // M-series checked before A-series since "M1 Pro" etc.
        // contain no A-chip substring. Newer chips checked first so
        // "M4" doesn't shadow the broader M5 substring on future strings.
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

    /// Human-readable label for display (e.g. `"Apple9 (M4)"`).
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
    pub fn code(self) -> Option<&'static str> {
        match self {
            GpuFamily::Apple7 => Some("Apple7"),
            GpuFamily::Apple8 => Some("Apple8"),
            GpuFamily::Apple9 => Some("Apple9"),
            GpuFamily::Apple10 => Some("Apple10"),
            GpuFamily::Unknown => None,
        }
    }

    /// True for Apple9+ (M3, M4, M5, A17, A18).
    pub const fn is_apple9_or_later(self) -> bool {
        matches!(self, GpuFamily::Apple9 | GpuFamily::Apple10)
    }

    /// Threadgroup memory in KB. All Apple7-9 GPUs have 32 KB.
    pub const fn threadgroup_mem_kb(self) -> u32 { 32 }

    /// Maximum threads per threadgroup. All Apple7-9 GPUs support 1024.
    pub const fn max_threads_per_threadgroup(self) -> u32 { 1024 }

    /// Known SLC (System Level Cache) size string per chip tier.
    /// Returns `"varies"` when the tier is not recognised.
    pub fn slc_label(device_name: &str) -> &'static str {
        if device_name.contains("Ultra") {
            "~96 MB"
        } else if device_name.contains("Max")
            && (device_name.contains("M5") || device_name.contains("M4"))
        {
            // M4/M5 Max share the ~64 MB SLC tier; revisit if M5 specs differ.
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
