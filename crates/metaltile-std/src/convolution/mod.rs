//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Convolution kernels (1-D, 2-D, 3-D, depthwise, winograd, audio/speech).

pub mod audio_conv1d;
pub mod audio_conv1d_block_scaled;
pub mod consolidated;
pub mod conv1d_causal_step_silu_cast_many;
pub mod conv1d_dilated_transpose;
pub mod conv1d_transpose_depthwise;
pub mod conv2d;
pub mod conv2d_block_scaled;
pub mod conv2d_mma;
pub mod conv2d_mma_block_scaled;
pub mod conv3d;
pub mod conv3d_block_scaled;
pub mod conv3d_mma;
pub mod conv3d_mma_block_scaled;
pub mod depthwise_conv2d;
pub mod depthwise_conv2d_block_scaled;
pub mod depthwise_conv2d_nhwc;
pub mod fishspeech_conv1d_block_scaled;
pub mod steel_conv;
pub mod winograd_conv;

// `mlx_conv_stub.rs` is a stale placeholder from the old metaltile-bench
// crate; it references `crate::runner` from metaltile-cli and does not
// compile.  Kept for future-work notes but intentionally not declared here.
