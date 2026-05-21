//! `tile emit` — emit a `kernels.metallib` + manifest + Swift wrappers.
//!
//! Walks the explicit kernel registry (`register_kernels`) and produces
//! artifacts under `<out>/`:
//!
//!   Resources/kernels/<name>.metal   — MSL source per kernel
//!   Resources/kernels.metallib       — compiled Metal library
//!   Resources/manifest.json          — per-kernel metadata
//!   Generated/MetalTileKernels.swift — typed Swift dispatch wrappers
//!
//! Usage:
//!   tile emit --out <swift-package-dir> [--sdk macosx] [--no-compile]

use std::{fs, path::PathBuf};

use metaltile_codegen::{
    MslGenerator,
    emit::{compile_metallib, write_manifest, write_msl, write_swift_wrappers},
};
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};
use metaltile_std::{
    ffai::{
        arg_reduce::ffai_argmax,
        dequant_gather::{
            dequant_gather_int3,
            dequant_gather_int4,
            dequant_gather_int5,
            dequant_gather_int6,
            dequant_gather_int8,
        },
        dequant_gemv::{
            dequant_gemv_int3,
            dequant_gemv_int4,
            dequant_gemv_int5,
            dequant_gemv_int6,
            dequant_gemv_int8,
        },
        gated_delta::{mt_gated_delta_chunk, mt_gated_delta_step},
        gated_delta_prep::mt_gated_delta_prep_step,
        gather::ffai_gather,
        kv_cache::{
            bulk_dequant_kv_int4,
            bulk_dequant_kv_int8,
            kv_cache_update,
            quantize_kv_int4,
            quantize_kv_int8,
        },
        moe::mt_moe_gather_qmm_mma_int4_bm16,
        moe_mpp,
        moe_mpp_bm64,
        moe_mpp_bm8,
        rope_llama::ffai_rope_llama,
        sampling::softmax_categorical_sample,
        sdpa_decode::ffai_sdpa_decode,
        ssm::{conv1d_causal_step, ssm_step},
    },
    mlx::{
        binary::{mt_mul, vector_add},
        gemv::mt_gemv,
        quantized::mt_qmm_mma,
        quantized_mpp,
        rms_norm::{mt_gated_mixer_norm, mt_rms_norm},
        steel::attn::steel_attention_mma::mt_sdpa_prefill_mma,
        unary::{mt_cast_to_f32, mt_gelu, mt_relu, mt_sigmoid, mt_silu, mt_softplus},
    },
    probe::mpp_matmul_smoke,
};

use crate::{EmitArgs, CliError};

pub fn run(args: &EmitArgs) -> Result<(), CliError> {
    let out = PathBuf::from(&args.out);
    let resources_dir = out.join("Resources");
    let kernels_dir = resources_dir.join("kernels");
    let generated_dir = out.join("Generated");

    fs::create_dir_all(&kernels_dir)?;
    fs::create_dir_all(&generated_dir)?;

    let kernels = register_kernels();
    println!("tile emit: {} kernels", kernels.len());

    let generator = MslGenerator::default();
    let mut metal_paths = Vec::new();

    for kernel in &kernels {
        let path = write_msl(kernel, &kernels_dir, &generator)
            .map_err(|e| CliError::Other(format!("MSL for {}: {e}", kernel.name)))?;
        println!("  wrote {}", path.display());
        metal_paths.push(path);
    }

    write_manifest(&kernels, &resources_dir.join("manifest.json"))
        .map_err(|e| CliError::Other(e.to_string()))?;
    println!("  wrote {}", resources_dir.join("manifest.json").display());

    write_swift_wrappers(&kernels, &generated_dir.join("MetalTileKernels.swift"))
        .map_err(|e| CliError::Other(e.to_string()))?;
    println!("  wrote {}", generated_dir.join("MetalTileKernels.swift").display());

    if args.no_compile {
        println!("--no-compile: skipping metallib build");
    } else {
        let metallib = resources_dir.join("kernels.metallib");
        let air_dir = std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("target"))
            .join("tile-emit-air");
        compile_metallib(&metal_paths, &metallib, &args.sdk, &air_dir)
            .map_err(|e| CliError::Other(e.to_string()))?;
        println!("  wrote {}", metallib.display());
    }

    println!("tile emit: done");
    Ok(())
}

fn register_kernels() -> Vec<Kernel> {
    let mut kernels: Vec<Kernel> = Vec::new();
    let dtypes = [DType::F32, DType::F16, DType::BF16];

    for &dt in &dtypes {
        let mut k = vector_add::kernel_ir_for(dt);
        k.name = format!("vector_add_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_mul::kernel_ir_for(dt);
        k.name = format!("mt_mul_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_silu::kernel_ir_for(dt);
        k.name = format!("mt_silu_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_softplus::kernel_ir_for(dt);
        k.name = format!("mt_softplus_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = ffai_gather::kernel_ir_for(dt);
        k.name = format!("ffai_gather_{}", dt_suffix(dt));
        k.mode = KernelMode::Grid3D;
        kernels.push(k);

        let mut k = mt_gemv::kernel_ir_for(dt);
        k.name = format!("mt_gemv_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_rms_norm::kernel_ir_for(dt);
        k.name = format!("mt_rms_norm_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_gated_mixer_norm::kernel_ir_for(dt);
        k.name = format!("mt_gated_mixer_norm_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = ffai_rope_llama::kernel_ir_for(dt);
        k.name = format!("ffai_rope_llama_{}", dt_suffix(dt));
        k.mode = KernelMode::Grid3D;
        kernels.push(k);

        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.name = format!("ffai_sdpa_decode_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        for &suffix in &["d64", "d256", "d512"] {
            let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
            k.name = format!("ffai_sdpa_decode_{}_{}", suffix, dt_suffix(dt));
            k.mode = KernelMode::Reduction;
            kernels.push(k);
        }

        let mut k = kv_cache_update::kernel_ir_for(dt);
        k.name = format!("kv_cache_update_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = quantize_kv_int8::kernel_ir_for(dt);
        k.name = format!("quantize_kv_int8_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = bulk_dequant_kv_int8::kernel_ir_for(dt);
        k.name = format!("bulk_dequant_kv_int8_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = quantize_kv_int4::kernel_ir_for(dt);
        k.name = format!("quantize_kv_int4_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = bulk_dequant_kv_int4::kernel_ir_for(dt);
        k.name = format!("bulk_dequant_kv_int4_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = ssm_step::kernel_ir_for(dt);
        k.name = format!("ssm_step_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = conv1d_causal_step::kernel_ir_for(dt);
        k.name = format!("conv1d_causal_step_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = softmax_categorical_sample::kernel_ir_for(dt);
        k.name = format!("softmax_categorical_sample_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = ffai_argmax::kernel_ir_for(dt);
        k.name = format!("ffai_argmax_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_gelu::kernel_ir_for(dt);
        k.name = format!("mt_gelu_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_relu::kernel_ir_for(dt);
        k.name = format!("mt_relu_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_sigmoid::kernel_ir_for(dt);
        k.name = format!("mt_sigmoid_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_cast_to_f32::kernel_ir_for(dt);
        k.name = format!("mt_cast_to_f32_{}", dt_suffix(dt));
        kernels.push(k);

        let mut k = mt_gated_delta_step::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_step_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_gated_delta_chunk::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_chunk_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_sdpa_prefill_mma::kernel_ir_for(dt);
        k.name = format!("mt_sdpa_prefill_mma_{}", dt_suffix(dt));
        k.mode = KernelMode::SimdGroup2D;
        kernels.push(k);

        let mut k = mt_gated_delta_prep_step::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_prep_step_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        for bits in ["int4", "int8", "int3", "int5", "int6"] {
            let (gemv_ir, gather_ir): (fn(DType) -> Kernel, fn(DType) -> Kernel) = match bits {
                "int4" => (dequant_gemv_int4::kernel_ir_for, dequant_gather_int4::kernel_ir_for),
                "int8" => (dequant_gemv_int8::kernel_ir_for, dequant_gather_int8::kernel_ir_for),
                "int3" => (dequant_gemv_int3::kernel_ir_for, dequant_gather_int3::kernel_ir_for),
                "int5" => (dequant_gemv_int5::kernel_ir_for, dequant_gather_int5::kernel_ir_for),
                "int6" => (dequant_gemv_int6::kernel_ir_for, dequant_gather_int6::kernel_ir_for),
                _ => unreachable!(),
            };
            let mut k = gemv_ir(dt);
            k.name = format!("dequant_gemv_{}_{}", bits, dt_suffix(dt));
            k.mode = KernelMode::Reduction;
            kernels.push(k);

            let mut k = gather_ir(dt);
            k.name = format!("dequant_gather_{}_{}", bits, dt_suffix(dt));
            kernels.push(k);
        }
    }

    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = mt_qmm_mma::kernel_ir_for(dt);
        k.name = format!("mt_qmm_mma_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm16_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = moe_mpp::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm16_mpp_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = moe_mpp_bm64::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm64_mpp_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = moe_mpp_bm8::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm8_mpp_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    for &dt in &[DType::F32, DType::F16] {
        let mut k = quantized_mpp::kernel_ir_for(dt);
        k.name = format!("mt_qmm_mma_mpp_{}", dt_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    kernels.push(mpp_matmul_smoke::kernel_ir());

    kernels
}

fn dt_suffix(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I64 => "i64",
        DType::U64 => "u64",
        DType::I4 => "i4",
        DType::Bool => "bool",
    }
}
