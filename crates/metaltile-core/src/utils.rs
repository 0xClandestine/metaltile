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
