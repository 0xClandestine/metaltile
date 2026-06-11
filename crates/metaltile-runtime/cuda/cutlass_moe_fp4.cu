// CUTLASS grouped block-scaled NVFP4 MoE GEMM, extern "C" entry for the
// metaltile-runtime FFI (sm_120a/sm_121a only; AOT-built when CUTLASS_DIR set).
//
// out[t,n](f16) = sum_k A[t,k] * W[eid(t)][n,k]
//   A  = sorted-token activations, packed e2m1 [mt, K/2] bytes row-major
//   SFA= per-group ue4m3 scale blocks (canonical 512B-block swizzle, one
//        16-elem K-block per scale); group g's blob starts at SFA+sfa_off[g]
//        and is laid out for the GROUP-LOCAL row index (M_g rows pad to 128).
//   W  = contiguous packed e2m1 expert slab [n_exp, N, K/2] bytes (W[n,k]
//        row-major per expert == ColumnMajor [K,N] for the GEMM's B operand)
//   SFB= per-expert ue4m3 scale slab [n_exp, ceil(N/128)*512*ceil(K/64)] bytes
//   D  = f16 out [mt, N] (plain LinearCombination epilogue, alpha=1 beta=0 —
//        no SFD output fusion; the result feeds relu2 / scatter in f16)
//
// Sorted tokens: group g owns a contiguous row range of `group_rows[g]` rows;
// W[expert_ids[g]] is its weight slab. All per-group pointer/stride/layout
// arrays are built host-side here and shipped in ONE device blob (graph-safety
// device-side build is a follow-up; host-side first per the integration plan).

#include "cutlass/cutlass.h"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using ElementInput = cutlass::float_e2m1_t;

// A: activations, nvfp4 (e2m1 + ue4m3 block-16 SF), RowMajor [M,K]
using ElementA   = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

// B: per-expert weights, nvfp4, ColumnMajor [K,N] (== W[n,k] row-major)
using ElementB   = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D: f16 out, plain LinearCombination (no block-scaled output fusion)
using ElementD   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128,_128,_128>;
using ClusterShape       = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

int queried_sm_count() {
    static int sm_count = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    }();
    return sm_count;
}

} // namespace

// Returns 0 on success. group_rows/expert_ids/sfa_off are HOST arrays
// (n_groups each); sfa_off[g] = BYTE offset of group g's SF blob inside SFA.
// alpha_vec: optional DEVICE float[n_groups] of per-group output scales
// (act_global * expert_global, folding both operands' per-tensor globals back
// in); null = alpha 1.
extern "C" int moe_grouped_gemm_cutlass_fp4(
    const void* A, const void* SFA, const void* B, const void* SFB, void* D,
    const int* group_rows, const int* expert_ids, const long long* sfa_off,
    const void* alpha_vec, int n_groups, int N, int K, void* stream_v)
{
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (K % 32 != 0 || N % 32 != 0) return 20; // e2m1 TMA alignment (32 elems)
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    // ── per-group host arrays ───────────────────────────────────────────────
    std::vector<UnderlyingProblemShape> ps_h(n_groups);
    std::vector<const ElementInput*> pA_h(n_groups);
    std::vector<const ElementInput*> pB_h(n_groups);
    std::vector<const ElementSF*> pSFA_h(n_groups);
    std::vector<const ElementSF*> pSFB_h(n_groups);
    std::vector<ElementD*> pD_h(n_groups);
    std::vector<StrideA> dA_h(n_groups);
    std::vector<StrideB> dB_h(n_groups);
    std::vector<StrideD> dD_h(n_groups);
    std::vector<LayoutSFA> lSFA_h(n_groups);
    std::vector<LayoutSFB> lSFB_h(n_groups);
    long rowoff = 0;
    for (int g = 0; g < n_groups; ++g) {
        const int m = group_rows[g];
        const long eid = expert_ids[g];
        ps_h[g]   = {m, N, K};
        pA_h[g]   = (const ElementInput*)((const uint8_t*)A + rowoff * (K / 2));
        pSFA_h[g] = (const ElementSF*)((const uint8_t*)SFA + sfa_off[g]);
        pB_h[g]   = (const ElementInput*)((const uint8_t*)B + eid * w_slab_bytes);
        pSFB_h[g] = (const ElementSF*)((const uint8_t*)SFB + eid * sfb_exp_bytes);
        pD_h[g]   = (ElementD*)((uint8_t*)D + rowoff * (long)N * 2);
        dA_h[g]   = cutlass::make_cute_packed_stride(StrideA{}, {m, K, 1});
        dB_h[g]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        dD_h[g]   = cutlass::make_cute_packed_stride(StrideD{}, {m, N, 1});
        lSFA_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, N, K, 1));
        lSFB_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, N, K, 1));
        rowoff += m;
    }

    // ── ship everything in ONE device blob (16B-aligned sections) ──────────
    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    const size_t off_ps   = 0;
    const size_t off_pA   = off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    const size_t off_pB   = off_pA   + sec(n_groups * sizeof(void*));
    const size_t off_pSFA = off_pB   + sec(n_groups * sizeof(void*));
    const size_t off_pSFB = off_pSFA + sec(n_groups * sizeof(void*));
    const size_t off_pD   = off_pSFB + sec(n_groups * sizeof(void*));
    const size_t off_dA   = off_pD   + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(LayoutSFA));
    const size_t off_pAl  = off_lSFB + sec(n_groups * sizeof(LayoutSFB));
    const size_t blob_bytes = off_pAl + sec(n_groups * sizeof(float*));

    std::vector<uint8_t> staging(blob_bytes, 0);
    std::memcpy(staging.data() + off_ps,   ps_h.data(),   n_groups * sizeof(UnderlyingProblemShape));
    std::memcpy(staging.data() + off_pA,   pA_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pB,   pB_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pSFA, pSFA_h.data(), n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_pD,   pD_h.data(),   n_groups * sizeof(void*));
    std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(StrideA));
    std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(StrideB));
    std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(StrideD));
    std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(LayoutSFA));
    std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(LayoutSFB));
    if (alpha_vec) {
        // per-group alpha POINTER array (values stay device-side, addresses
        // are host-computable from the device base — no extra sync).
        std::vector<const float*> pAl_h(n_groups);
        for (int g = 0; g < n_groups; ++g) pAl_h[g] = (const float*)alpha_vec + g;
        std::memcpy(staging.data() + off_pAl, pAl_h.data(), n_groups * sizeof(float*));
    }

    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    int rc = 0;
#define MT_CUDA_CK(call) do { if ((call) != cudaSuccess) { rc = 3; goto cleanup; } } while (0)
    MT_CUDA_CK(cudaMallocAsync((void**)&blob, blob_bytes, stream));
    MT_CUDA_CK(cudaMemcpyAsync(blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice, stream));

    {
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = queried_sm_count();

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {n_groups, (UnderlyingProblemShape*)(blob + off_ps), ps_h.data()},
            {(const ElementA::DataType**)(blob + off_pA),  (StrideA*)(blob + off_dA),
             (const ElementB::DataType**)(blob + off_pB),  (StrideB*)(blob + off_dB),
             (const ElementSF**)(blob + off_pSFA), (LayoutSFA*)(blob + off_lSFA),
             (const ElementSF**)(blob + off_pSFB), (LayoutSFB*)(blob + off_lSFB)},
            {{}, // fusion args set below
             nullptr, (StrideC*)(blob + off_dD),                 // C unused (beta=0)
             (ElementD**)(blob + off_pD), (StrideD*)(blob + off_dD)},
            hw_info
        };
        if (alpha_vec) {
            args.epilogue.thread.alpha = 0.0f; // ignored when ptr_array set
            args.epilogue.thread.alpha_ptr_array = (const float* const*)(blob + off_pAl);
            args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 1};
        } else {
            args.epilogue.thread.alpha = 1.0f;
        }
        args.epilogue.thread.beta = 0.0f;

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) { rc = 10; goto cleanup; }
        size_t ws = Gemm::get_workspace_size(args);
        if (ws) MT_CUDA_CK(cudaMallocAsync((void**)&work, ws, stream));
        cutlass::Status st = gemm.initialize(args, work, stream);
        if (st != cutlass::Status::kSuccess) { rc = 1; goto cleanup; }
        st = gemm.run(stream);
        if (st != cutlass::Status::kSuccess) { rc = 2; goto cleanup; }
    }
#undef MT_CUDA_CK

cleanup:
    if (blob) cudaFreeAsync(blob, stream);
    if (work) cudaFreeAsync(work, stream);
    return rc;
}


// ───────────────────────── device-side descriptor build ─────────────────────
// prepare(): one-time per (n_groups,N,K[,D-base...]) — allocates a persistent
// device blob + workspace, fills every M-INDEPENDENT section host-side once
// (expert ptrs, strides, SF layouts at worst-case M extent, alpha ptr array),
// initializes the GEMM with host_problem_shapes=nullptr, and returns a handle.
// run(): per call — ONE small kernel derives ps/pA/pSFA/pD from the DEVICE
// offsets array (no download, no host build, no allocs), then gemm.run().
// Graph-safe: fixed launch geometry, fixed pointers, stream-ordered only.

namespace {

struct Fp4GroupedHandle {
    Gemm gemm;
    uint8_t* blob = nullptr;
    uint8_t* work = nullptr;
    int n_groups = 0, N = 0, K = 0;
    // section offsets (same layout as the one-shot path)
    size_t off_ps, off_pA, off_pB, off_pSFA, off_pSFB, off_pD, off_pAl;
};

} // namespace

// one thread per group: derive the M-dependent sections from device offsets.
// FILE SCOPE (not anon-namespace: nvcc's cudafe stub collides __global__
// symbols in anon namespaces with CUTLASS's own). The problem-shape entry is
// written as 3 contiguous ints (asserted == UnderlyingProblemShape below).
// sfa blocks are laid out densely: group g's blob starts at
// (sum over j<g of ceil(M_j/128)) * 512 * ceil(K/64) bytes.
__global__ void mt_fp4_fill_group_args(
    const unsigned* __restrict__ off,   // [n_groups+1] device row offsets
    const uint8_t* A, const uint8_t* SFA, uint8_t* D,
    int* ps3,                            // [n_groups*3] (M,N,K) triples
    const void** pA, const void** pSFA, void** pD,
    int n_groups, int N, int K)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const int m = (int)(off[g + 1] - off[g]);
    ps3[g * 3 + 0] = m; ps3[g * 3 + 1] = N; ps3[g * 3 + 2] = K;
    pA[g] = A + (size_t)off[g] * (K / 2);
    pD[g] = D + (size_t)off[g] * (size_t)N * 2;
    // dense prefix of ceil(M_j/128) — n_groups is small (<=130), linear scan
    size_t blk = 0;
    for (int j = 0; j < g; ++j) blk += (size_t)((off[j + 1] - off[j] + 127) / 128);
    pSFA[g] = SFA + blk * 512 * (size_t)((K + 63) / 64);
}

static_assert(sizeof(UnderlyingProblemShape) == 3 * sizeof(int),
    "GroupProblemShape underlying entry must be 3 contiguous ints");

extern "C" void* moe_grouped_gemm_cutlass_fp4_prepare(
    const void* B, const void* SFB, const void* alpha_vec,
    int n_groups, int N, int K, int max_m_total)
{
    if (K % 32 != 0 || N % 32 != 0) return nullptr;
    auto* h = new Fp4GroupedHandle();
    h->n_groups = n_groups; h->N = N; h->K = K;
    const size_t w_slab_bytes  = (size_t)N * (size_t)K / 2;
    const size_t sfb_exp_bytes = (size_t)((N + 127) / 128) * 512 * (size_t)((K + 63) / 64);

    auto sec = [](size_t bytes) { return (bytes + 15) & ~(size_t)15; };
    h->off_ps   = 0;
    h->off_pA   = h->off_ps   + sec(n_groups * sizeof(UnderlyingProblemShape));
    h->off_pB   = h->off_pA   + sec(n_groups * sizeof(void*));
    h->off_pSFA = h->off_pB   + sec(n_groups * sizeof(void*));
    h->off_pSFB = h->off_pSFA + sec(n_groups * sizeof(void*));
    h->off_pD   = h->off_pSFB + sec(n_groups * sizeof(void*));
    const size_t off_dA   = h->off_pD + sec(n_groups * sizeof(void*));
    const size_t off_dB   = off_dA   + sec(n_groups * sizeof(StrideA));
    const size_t off_dD   = off_dB   + sec(n_groups * sizeof(StrideB));
    const size_t off_lSFA = off_dD   + sec(n_groups * sizeof(StrideD));
    const size_t off_lSFB = off_lSFA + sec(n_groups * sizeof(LayoutSFA));
    h->off_pAl  = off_lSFB + sec(n_groups * sizeof(LayoutSFB));
    const size_t blob_bytes = h->off_pAl + sec(n_groups * sizeof(float*));

    // host-fill every M-independent section once. SF layouts use the
    // WORST-CASE M extent (max_m_total): per-128-row-block strides are
    // M-independent and the tile scheduler bounds reads by the device
    // problem shapes, so an over-sized extent is safe.
    std::vector<uint8_t> staging(blob_bytes, 0);
    {
        std::vector<const ElementInput*> pB_h(n_groups);
        std::vector<const ElementSF*> pSFB_h(n_groups);
        std::vector<StrideA> dA_h(n_groups);
        std::vector<StrideB> dB_h(n_groups);
        std::vector<StrideD> dD_h(n_groups);
        std::vector<LayoutSFA> lSFA_h(n_groups);
        std::vector<LayoutSFB> lSFB_h(n_groups);
        for (int g = 0; g < n_groups; ++g) {
            pB_h[g]   = (const ElementInput*)((const uint8_t*)B + (size_t)g * w_slab_bytes);
            pSFB_h[g] = (const ElementSF*)((const uint8_t*)SFB + (size_t)g * sfb_exp_bytes);
            dA_h[g]   = cutlass::make_cute_packed_stride(StrideA{}, {max_m_total, K, 1});
            dB_h[g]   = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            dD_h[g]   = cutlass::make_cute_packed_stride(StrideD{}, {max_m_total, N, 1});
            lSFA_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m_total, N, K, 1));
            lSFB_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(max_m_total, N, K, 1));
        }
        std::memcpy(staging.data() + h->off_pB,   pB_h.data(),   n_groups * sizeof(void*));
        std::memcpy(staging.data() + h->off_pSFB, pSFB_h.data(), n_groups * sizeof(void*));
        std::memcpy(staging.data() + off_dA,   dA_h.data(),   n_groups * sizeof(StrideA));
        std::memcpy(staging.data() + off_dB,   dB_h.data(),   n_groups * sizeof(StrideB));
        std::memcpy(staging.data() + off_dD,   dD_h.data(),   n_groups * sizeof(StrideD));
        std::memcpy(staging.data() + off_lSFA, lSFA_h.data(), n_groups * sizeof(LayoutSFA));
        std::memcpy(staging.data() + off_lSFB, lSFB_h.data(), n_groups * sizeof(LayoutSFB));
        if (alpha_vec) {
            std::vector<const float*> pAl_h(n_groups);
            for (int g = 0; g < n_groups; ++g) pAl_h[g] = (const float*)alpha_vec + g;
            std::memcpy(staging.data() + h->off_pAl, pAl_h.data(), n_groups * sizeof(float*));
        }
    }
    if (cudaMalloc((void**)&h->blob, blob_bytes) != cudaSuccess) { delete h; return nullptr; }
    if (cudaMemcpy(h->blob, staging.data(), blob_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = queried_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {n_groups, (UnderlyingProblemShape*)(h->blob + h->off_ps), nullptr},
        {(const ElementA::DataType**)(h->blob + h->off_pA),  (StrideA*)(h->blob + off_dA),
         (const ElementB::DataType**)(h->blob + h->off_pB),  (StrideB*)(h->blob + off_dB),
         (const ElementSF**)(h->blob + h->off_pSFA), (LayoutSFA*)(h->blob + off_lSFA),
         (const ElementSF**)(h->blob + h->off_pSFB), (LayoutSFB*)(h->blob + off_lSFB)},
        {{},
         nullptr, (StrideC*)(h->blob + off_dD),
         (ElementD**)(h->blob + h->off_pD), (StrideD*)(h->blob + off_dD)},
        hw_info
    };
    if (alpha_vec) {
        args.epilogue.thread.alpha = 0.0f;
        args.epilogue.thread.alpha_ptr_array = (const float* const*)(h->blob + h->off_pAl);
        args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 1};
    } else {
        args.epilogue.thread.alpha = 1.0f;
    }
    args.epilogue.thread.beta = 0.0f;

    if (h->gemm.can_implement(args) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    size_t ws = Gemm::get_workspace_size(args);
    if (ws && cudaMalloc((void**)&h->work, ws) != cudaSuccess) {
        cudaFree(h->blob); delete h; return nullptr;
    }
    if (h->gemm.initialize(args, h->work) != cutlass::Status::kSuccess) {
        cudaFree(h->blob); if (h->work) cudaFree(h->work); delete h; return nullptr;
    }
    return h;
}

// Per-call: fill M-dependent sections from DEVICE offsets, then run. A/SFA/D
// must be the SAME base pointers across calls if used under graph capture.
extern "C" int moe_grouped_gemm_cutlass_fp4_run(
    void* handle, const void* A, const void* SFA, void* D,
    const void* off_dev, void* stream_v)
{
    auto* h = (Fp4GroupedHandle*)handle;
    if (!h) return 1;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    int blocks = (h->n_groups + threads - 1) / threads;
    mt_fp4_fill_group_args<<<blocks, threads, 0, stream>>>(
        (const unsigned*)off_dev,
        (const uint8_t*)A, (const uint8_t*)SFA, (uint8_t*)D,
        (int*)(h->blob + h->off_ps),
        (const void**)(h->blob + h->off_pA),
        (const void**)(h->blob + h->off_pSFA),
        (void**)(h->blob + h->off_pD),
        h->n_groups, h->N, h->K);
    if (cudaGetLastError() != cudaSuccess) return 3;
    return h->gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 2;
}

extern "C" void moe_grouped_gemm_cutlass_fp4_release(void* handle)
{
    auto* h = (Fp4GroupedHandle*)handle;
    if (!h) return;
    if (h->blob) cudaFree(h->blob);
    if (h->work) cudaFree(h->work);
    delete h;
}

#else // !CUTLASS_ARCH_MMA_SM120_SUPPORTED && !CUTLASS_ARCH_MMA_SM121_SUPPORTED

extern "C" int moe_grouped_gemm_cutlass_fp4(
    const void*, const void*, const void*, const void*, void*,
    const int*, const int*, const long long*, const void*, int, int, int, void*)
{
    return 100; // built without sm_120a/sm_121a block-scaled mma support
}

extern "C" void* moe_grouped_gemm_cutlass_fp4_prepare(
    const void*, const void*, const void*, int, int, int, int) { return nullptr; }
extern "C" int moe_grouped_gemm_cutlass_fp4_run(
    void*, const void*, const void*, void*, const void*, void*) { return 100; }
extern "C" void moe_grouped_gemm_cutlass_fp4_release(void*) {}

#endif
