// CUTLASS grouped MoE GEMM, extern "C" entry for the metaltile-runtime FFI.
// out[t,n] = sum_k A[t,k] * W[eid(t)][n,k]  (A RowMajor [mt,K], W[n,k] => B ColumnMajor [K,N], C RowMajor [mt,N]).
// Sorted tokens: group g owns a contiguous row range; W[expert_ids[g]] is its weight slab.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
using StrideI = typename LayoutA::Stride::Index;

// Returns 0 on success, nonzero on failure. group_rows/expert_ids are HOST arrays.
extern "C" int moe_grouped_gemm_cutlass(
    const void* A, const void* W, void* Cout,
    const int* group_rows, const int* expert_ids,
    int n_groups, int N, int K, void* stream_v)
{
    cudaStream_t stream = (cudaStream_t)stream_v;
    const ElementA* Ap = (const ElementA*)A;
    const ElementB* Wp = (const ElementB*)W;
    ElementOutput* Cp = (ElementOutput*)Cout;

    std::vector<cutlass::gemm::GemmCoord> ps(n_groups);
    std::vector<ElementA*> pA(n_groups);
    std::vector<ElementB*> pB(n_groups);
    std::vector<ElementOutput*> pC(n_groups);
    std::vector<StrideI> lda(n_groups), ldb(n_groups), ldc(n_groups);
    long rowoff = 0;
    for (int g = 0; g < n_groups; g++) {
        int m = group_rows[g];
        ps[g] = cutlass::gemm::GemmCoord(m, N, K);
        pA[g] = (ElementA*)(Ap + rowoff * K);
        pB[g] = (ElementB*)(Wp + (long)expert_ids[g] * N * K);
        pC[g] = (ElementOutput*)(Cp + rowoff * N);
        lda[g] = K; ldb[g] = K; ldc[g] = N;
        rowoff += m;
    }

    // device arrays
    cutlass::gemm::GemmCoord* ps_d; ElementA** pA_d; ElementB** pB_d; ElementOutput** pC_d;
    StrideI *lda_d,*ldb_d,*ldc_d;
    size_t gp = n_groups;
    cudaMallocAsync((void**)&ps_d, gp*sizeof(cutlass::gemm::GemmCoord), stream);
    cudaMallocAsync((void**)&pA_d, gp*sizeof(ElementA*), stream);
    cudaMallocAsync((void**)&pB_d, gp*sizeof(ElementB*), stream);
    cudaMallocAsync((void**)&pC_d, gp*sizeof(ElementOutput*), stream);
    cudaMallocAsync((void**)&lda_d, gp*sizeof(StrideI), stream);
    cudaMallocAsync((void**)&ldb_d, gp*sizeof(StrideI), stream);
    cudaMallocAsync((void**)&ldc_d, gp*sizeof(StrideI), stream);
    cudaMemcpyAsync(ps_d, ps.data(), gp*sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pA_d, pA.data(), gp*sizeof(ElementA*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pB_d, pB.data(), gp*sizeof(ElementB*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pC_d, pC.data(), gp*sizeof(ElementOutput*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lda_d, lda.data(), gp*sizeof(StrideI), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldb_d, ldb.data(), gp*sizeof(StrideI), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ldc_d, ldc.data(), gp*sizeof(StrideI), cudaMemcpyHostToDevice, stream);

    int tbc = Gemm::sufficient(ps.data(), n_groups);
    if (!tbc) return 10;
    typename Gemm::EpilogueOutputOp::Params epi(ElementAccumulator(1), ElementAccumulator(0));
    typename Gemm::Arguments args(ps_d, n_groups, tbc, epi,
        pA_d, pB_d, pC_d, pC_d, lda_d, ldb_d, ldc_d, ldc_d, ps.data());

    Gemm gemm;
    size_t ws = gemm.get_workspace_size(args);
    uint8_t* work = nullptr;
    if (ws) cudaMallocAsync((void**)&work, ws, stream);
    cutlass::Status st = gemm.initialize(args, work, stream);
    if (st != cutlass::Status::kSuccess) return 1;
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) return 2;

    cudaFreeAsync(ps_d, stream); cudaFreeAsync(pA_d, stream); cudaFreeAsync(pB_d, stream);
    cudaFreeAsync(pC_d, stream); cudaFreeAsync(lda_d, stream); cudaFreeAsync(ldb_d, stream);
    cudaFreeAsync(ldc_d, stream); if (work) cudaFreeAsync(work, stream);
    return 0;
}
