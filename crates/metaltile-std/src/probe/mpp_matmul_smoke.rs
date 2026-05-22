//! Cooperative tile matmul smoke kernel.
//!
//! Single simdgroup, 16×32 fp16 → 16×16 fp32 matmul using the
//! `CoopTile*` primitive ops, which lower to `mpp::tensor_ops::matmul2d`
//! on Metal 4 (macOS 26+).
//!
//! Shapes:
//!   A = [M=16, K=32], row-major fp16
//!   B = [K=32, N=16], row-major fp16
//!   C = [M=16, N=16], row-major fp32
//!
//! Geometry: single threadgroup, single simdgroup (32 threads).

use metaltile_core::{
    dtype::DType,
    ir::{
        Block,
        BlockId,
        CoopTileAccMode,
        CoopTileScope,
        Kernel,
        KernelMode,
        Op,
        Param,
        ParamKind,
    },
    shape::{Dim, Shape},
};
use rustc_hash::FxHashMap;

/// Build the [`Kernel`] IR for `mt_mpp_matmul_smoke`.
///
/// Expresses the matmul via 6 primitive `CoopTile*` ops:
/// `CoopTileSetup → CoopTileZero → CoopTileLoadA → CoopTileLoadB → CoopTileRun → CoopTileStoreC`.
///
/// Dispatch geometry: 1 threadgroup × 32 threads = one simdgroup.
pub fn kernel_ir() -> Kernel {
    let mut k = Kernel::new("mt_mpp_matmul_smoke");
    k.mode = KernelMode::Elementwise;

    // Params: A [M=16, K=32] fp16, B [K=32, N=16] fp16, C [M=16, N=16] fp32.
    k.params.push(Param {
        name: "A".into(),
        dtype: DType::F16,
        shape: Shape::new([Dim::Known(16), Dim::Known(32)]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "B".into(),
        dtype: DType::F16,
        shape: Shape::new([Dim::Known(32), Dim::Known(16)]),
        is_output: false,
        kind: ParamKind::Tensor,
    });
    k.params.push(Param {
        name: "C".into(),
        dtype: DType::F32,
        shape: Shape::new([Dim::Known(16), Dim::Known(16)]),
        is_output: true,
        kind: ParamKind::Tensor,
    });
    k.return_shapes.push(Shape::new([Dim::Known(16), Dim::Known(16)]));

    let mut body = Block::new(BlockId::new(0));

    // Step 1: declare descriptor + cooperative tensor objects.
    // M=16, N=16, K=32, A=[M,K] not transposed, B=[K,N] not transposed.
    // act_dtype=F16 (A, B elements), acc_dtype=F32 (C accumulator/output).
    body.push_op_no_result(Op::CoopTileSetup {
        name: "gemm".into(),
        m: 16,
        n: 16,
        k: 32,
        ta: false,
        tb: false,
        tc: false,
        acc_mode: CoopTileAccMode::Overwrite,
        exec_scope: CoopTileScope::SimdGroup,
        act_dtype: DType::F16,
        acc_dtype: DType::F32,
        direct_inputs: false,
        a_is_tg: false,
        a_ei: 0,
        a_eo: 0,
        b_is_tg: false,
        b_ei: 0,
        b_eo: 0,
    });

    // Step 2: zero the C accumulator.
    body.push_op_no_result(Op::CoopTileZero { name: "gemm".into() });

    // Step 3: load A tile from device buffer.
    // A is [M=16, K=32] row-major → extents<int, K=32, M=16>.
    body.push_op_no_result(Op::CoopTileLoadA {
        name: "gemm".into(),
        ptr_name: "A".into(),
        ptr_offset: None,
        is_tg: false,
        dtype: DType::F16,
        ei: 32,
        eo: 16,
        direct: false,
    });

    // Step 4: load B tile from device buffer.
    // B is [K=32, N=16] row-major → extents<int, N=16, K=32>.
    body.push_op_no_result(Op::CoopTileLoadB {
        name: "gemm".into(),
        ptr_name: "B".into(),
        ptr_offset: None,
        is_tg: false,
        dtype: DType::F16,
        ei: 16,
        eo: 32,
        direct: false,
    });

    // Step 5: execute A·B → C.
    body.push_op_no_result(Op::CoopTileRun { name: "gemm".into(), direct: false });

    // Step 6: store C tile to device buffer.
    // C is [M=16, N=16] row-major → extents<int, N=16, M=16>.
    body.push_op_no_result(Op::CoopTileStoreC {
        name: "gemm".into(),
        ptr_name: "C".into(),
        ptr_offset: None,
        is_tg: false,
        dtype: DType::F32,
        ei: 16,
        eo: 16,
    });

    k.body = body.clone();
    let mut blocks = FxHashMap::default();
    blocks.insert(BlockId::new(0), body);
    k.blocks = blocks;

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_ir_constructs_and_has_three_params() {
        let k = kernel_ir();
        assert_eq!(k.name, "mt_mpp_matmul_smoke");
        assert_eq!(k.params.len(), 3);
        assert_eq!(k.params[0].name, "A");
        assert_eq!(k.params[1].name, "B");
        assert_eq!(k.params[2].name, "C");
        assert!(k.params[2].is_output);
        // 6 CoopTile* ops.
        assert_eq!(k.body.ops.len(), 6);
        assert!(matches!(&k.body.ops[0], Op::CoopTileSetup { .. }));
        assert!(matches!(&k.body.ops[5], Op::CoopTileStoreC { .. }));
    }

    #[test]
    fn codegen_emits_mpp_include() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        assert!(
            msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
            "MPP include missing from generated MSL:\n{msl}"
        );
        assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
        assert!(msl.contains("kernel void mt_mpp_matmul_smoke"));
    }

    /// Developer aid — `cargo test -p metaltile-std --lib -- dump_generated_msl --nocapture`
    #[test]
    fn dump_generated_msl() {
        use metaltile_codegen::msl::MslGenerator;
        let k = kernel_ir();
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
