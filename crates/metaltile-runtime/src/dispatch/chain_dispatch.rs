//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//!
//! Multi‑pass fused dispatch chain.
//!
//! Dispatching N kernel passes through a single Metal command buffer
//! eliminates intermediate host round‑trips: buffers that are outputs
//! of pass `i` and inputs of pass `j>i` are allocated as
//! `MTLStorageModePrivate` and shared across passes without ever
//! touching host RAM.
//!
//! For a 2‑pass SDPA decode this replaces two separate cmd‑buffer
//! commits + a ~MB‑sized host `memcpy` of `partial_o/m/l` with one
//! commit and zero host traffic between passes.

use std::{
    collections::{BTreeMap, HashSet},
    ptr::NonNull,
};

use metaltile_core::ir::ParamKind;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBarrierScope,
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandEncoder,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLSize,
};
use rustc_hash::FxHashMap;

use crate::{
    DispatchResult,
    DispatchSpec,
    device::{buffer_pool::BufRc, metal_device::MetalDevice},
    dispatch::buffer_plan::{ParamBufferPlan, build_param_buffer_plans, resolve_strided_metadata},
    error::MetalTileError,
};

// ---------------------------------------------------------------------------
// Chain dispatch
// ---------------------------------------------------------------------------

/// Orchestrates a fused multi‑pass Metal dispatch chain.
///
/// Created by [`Context::dispatch_chain`](super::Context::dispatch_chain).
/// Each pass is a [`DispatchSpec`] with its own kernel, buffers, and
/// grid parameters.
pub(crate) struct ChainDispatch<'a> {
    /// The Metal device adapter.
    dev: &'a MetalDevice,
    /// Pass specifications, in execution order.
    specs: &'a [DispatchSpec<'a>],
}

impl<'a> ChainDispatch<'a> {
    /// Prepare a chain dispatch.
    pub fn new(dev: &'a MetalDevice, specs: &'a [DispatchSpec<'a>]) -> Self {
        ChainDispatch { dev, specs }
    }

    /// Execute all passes on one command buffer and return per‑pass
    /// results.  GPU time is attributed to the first result (chained
    /// passes share one command buffer and cannot be split).
    pub fn execute(&self) -> Result<Vec<DispatchResult>, MetalTileError> {
        let (msl_sources, binding_plans) = self.generate_msl_and_plans()?;

        let later_inputs: Vec<HashSet<&str>> = (0..self.specs.len())
            .map(|i| {
                self.specs[i + 1..]
                    .iter()
                    .flat_map(|s| s.kernel.params.iter())
                    .filter(|p| !p.is_output)
                    .map(|p| p.name.as_str())
                    .collect()
            })
            .collect();

        let pipes = self.compile_pso_pipeline(&msl_sources)?;

        let cb = self.dev.command_buffer()?;
        let per_spec_bufs = self.encode_passes(&cb, &pipes, &binding_plans, &later_inputs)?;

        (*cb).commit();
        (*cb).waitUntilCompleted();

        let elapsed_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;

        self.read_results(&per_spec_bufs, &binding_plans, &later_inputs, elapsed_us)
    }

    // ── private helpers ─────────────────────────────────────────────

    fn generate_msl_and_plans(
        &self,
    ) -> Result<(Vec<String>, Vec<Vec<ParamBufferPlan>>), MetalTileError> {
        let mut msl_sources = Vec::with_capacity(self.specs.len());
        let mut binding_plans = Vec::with_capacity(self.specs.len());

        for spec in self.specs {
            let key = crate::cache::pso_cache::pso_cache_key(spec.kernel, spec.fn_consts);
            let msl = self.dev.get_msl(spec.kernel, key)?;
            msl_sources.push(msl);
            binding_plans.push(build_param_buffer_plans(spec.kernel, spec.buffers)?);
        }

        Ok((msl_sources, binding_plans))
    }

    fn compile_pso_pipeline(
        &self,
        msl_sources: &[String],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, MetalTileError> {
        let mut pipes = Vec::with_capacity(self.specs.len());

        for (spec, msl) in self.specs.iter().zip(msl_sources) {
            let key = crate::cache::pso_cache::pso_cache_key(spec.kernel, spec.fn_consts);
            let pso = self.dev.get_pso(key, msl, &spec.kernel.name, spec.fn_consts)?;
            pipes.push(pso);
        }

        Ok(pipes)
    }

    fn encode_passes(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        pipes: &[Retained<ProtocolObject<dyn MTLComputePipelineState>>],
        binding_plans: &[Vec<ParamBufferPlan>],
        later_inputs: &[HashSet<&str>],
    ) -> Result<Vec<Vec<BufRc>>, MetalTileError> {
        let mut alias_pool: FxHashMap<String, BufRc> = FxHashMap::default();
        let mut per_spec_bufs: Vec<Vec<BufRc>> = Vec::with_capacity(self.specs.len());

        for (i, spec) in self.specs.iter().enumerate() {
            let bufs = self.allocate_pass_buffers(
                spec,
                &binding_plans[i],
                &later_inputs[i],
                &mut alias_pool,
            )?;

            let enc = (*cb).computeCommandEncoder().ok_or(MetalTileError::NoDevice)?;

            enc.setComputePipelineState(&pipes[i]);
            for (idx, buf) in bufs.iter().enumerate() {
                // SAFETY: `buf` is a valid MTLBuffer; offset is 0.
                unsafe { enc.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, idx) };
            }

            let tensor_binding_count = bufs.len();
            self.bind_constexpr_scalars(&enc, spec, tensor_binding_count)?;

            let (g, t) = (spec.grid_groups, spec.threads_per_group);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: g[0], height: g[1], depth: g[2] },
                MTLSize { width: t[0], height: t[1], depth: t[2] },
            );

            if i + 1 < self.specs.len() {
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }

            (*enc).endEncoding();
            per_spec_bufs.push(bufs);
        }

        Ok(per_spec_bufs)
    }

    fn allocate_pass_buffers(
        &self,
        spec: &DispatchSpec<'_>,
        plans: &[ParamBufferPlan],
        later_inputs_this: &HashSet<&str>,
        alias_pool: &mut FxHashMap<String, BufRc>,
    ) -> Result<Vec<BufRc>, MetalTileError> {
        let mut bufs: Vec<BufRc> = Vec::with_capacity(spec.kernel.params.len() * 2);

        for (param, plan) in spec.kernel.params.iter().zip(plans) {
            // Input resolution: resident pre‑uploaded > aliased from
            // earlier pass.
            if !param.is_output {
                let pre = spec
                    .resident
                    .get(&param.name)
                    .map(|r| r.inner.clone())
                    .or_else(|| alias_pool.get(&param.name).cloned());

                if let Some(buf) = pre {
                    bufs.push(buf);
                    self.push_strided(&mut bufs, param, spec.buffers)?;
                    continue;
                }
            }

            // Outputs aliased to later specs → private storage.
            // Otherwise shared (host‑readable).
            let new_buf = if param.is_output && later_inputs_this.contains(param.name.as_str()) {
                let b = self.dev.acquire_private(plan.data_len)?;
                alias_pool.insert(param.name.clone(), b.clone());
                b
            } else {
                let bytes = spec.buffers.get(&param.name).map(Vec::as_slice);
                self.dev.acquire_shared(bytes, plan.data_len)?
            };

            bufs.push(new_buf);
            self.push_strided(&mut bufs, param, spec.buffers)?;
        }

        Ok(bufs)
    }

    /// Append shape + strides metadata buffers for a strided parameter.
    fn push_strided(
        &self,
        bufs: &mut Vec<BufRc>,
        param: &metaltile_core::ir::Param,
        src: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), MetalTileError> {
        if param.kind == ParamKind::Strided {
            let (shape, strides) = resolve_strided_metadata(param, src)?;
            bufs.push(self.dev.acquire_shared(Some(shape.as_ref()), shape.len())?);
            bufs.push(self.dev.acquire_shared(Some(strides.as_ref()), strides.len())?);
        }
        Ok(())
    }

    fn bind_constexpr_scalars(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        spec: &DispatchSpec<'_>,
        tensor_binding_count: usize,
    ) -> Result<(), MetalTileError> {
        for (j, decl) in spec.kernel.constexprs.iter().enumerate() {
            let key = decl.name.name();
            let elem = decl.dtype.size_bytes().max(4);
            let bytes = spec.buffers.get(key).map(Vec::as_slice).unwrap_or(&[]);

            // Pad to `elem` if the caller supplied fewer bytes
            // (legacy `alloc_shared` zeroed beyond `elem` too).
            // 16 bytes is the largest possible constexpr scalar.
            const MAX_CONSTEXPR_BYTES: usize = 16;
            let mut staged = [0u8; MAX_CONSTEXPR_BYTES];
            let n = bytes.len().min(elem).min(staged.len());
            staged[..n].copy_from_slice(&bytes[..n]);

            // SAFETY: `staged` is a stack array — its pointer is
            // always non‑null and valid for `elem` bytes.  Metal
            // copies the bytes during `setBytes_length_atIndex` so
            // the stack frame can unwind safely.
            unsafe {
                enc.setBytes_length_atIndex(
                    NonNull::new(staged.as_ptr() as *mut _)
                        .ok_or_else(|| MetalTileError::Buffer("setBytes null".into()))?,
                    elem,
                    tensor_binding_count + j,
                );
            }
        }
        Ok(())
    }

    fn read_results(
        &self,
        per_spec_bufs: &[Vec<BufRc>],
        binding_plans: &[Vec<ParamBufferPlan>],
        later_inputs: &[HashSet<&str>],
        elapsed_us: f64,
    ) -> Result<Vec<DispatchResult>, MetalTileError> {
        let mut results = Vec::with_capacity(self.specs.len());

        for (i, spec) in self.specs.iter().enumerate() {
            let mut outputs: BTreeMap<String, Vec<u8>> = BTreeMap::new();

            for (param, plan) in spec.kernel.params.iter().zip(&binding_plans[i]) {
                if !param.is_output || later_inputs[i].contains(param.name.as_str()) {
                    continue;
                }

                let Some(buf) = per_spec_bufs[i].get(plan.data_binding_index) else {
                    continue;
                };

                // SAFETY: `buf.contents()` is valid for the buffer's
                // lifetime.  `waitUntilCompleted` has been called.
                let ptr = buf.contents();
                let bytes =
                    unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, plan.data_len) }
                        .to_vec();
                outputs.insert(param.name.clone(), bytes);
            }

            let us = if i == 0 { elapsed_us } else { 0.0 };
            results.push(DispatchResult { elapsed_us: us, gflops: 0.0, outputs });
        }

        Ok(results)
    }
}
