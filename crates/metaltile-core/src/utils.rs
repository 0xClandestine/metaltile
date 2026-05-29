//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Utility types for metaltile-core.

/// A counter for generating unique IDs.
#[derive(Debug, Clone, Default)]
pub struct IdCounter {
    next: u32,
}

impl IdCounter {
    pub fn new() -> Self { IdCounter { next: 0 } }
}

impl Iterator for IdCounter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let id = self.next;
        self.next += 1;
        Some(id)
    }
}

/// Generate `len` random bytes using a fast non-cryptographic RNG.
pub fn random_bytes(len: usize) -> Vec<u8> {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(42);
    let mut state = seed as u64 ^ 0x9e3779b97f4a7c15;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.push(state as u8);
    }
    data
}
