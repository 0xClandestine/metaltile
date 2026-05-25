//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StdError {
    #[error("runner error: {0}")]
    Runner(String),
    #[error("Metal error: {0}")]
    Metal(String),
}
