# MetalTile vs MLX Reference Kernel Coverage

## Per-Op Reference Status

Legend: ✅ exists in Metal source  ❌ NOT found  — no reference

| Op | MT Kernel | Dtype | MLX Reference | Status |
|---|---|---|---|---|
| unary | `mt_exp` | f32 | `v_Expfloat32float32` | ✅ |
|  | `mt_exp` | f16 | `v_Expfloat16float16` | ✅ |
|  | `mt_exp` | bf16 | `v_Expbfloat16bfloat16` | ✅ |
|  | `mt_log` | f32 | `v_Logfloat32float32` | ✅ |
|  | `mt_log` | f16 | `v_Logfloat16float16` | ✅ |
|  | `mt_log` | bf16 | `v_Logbfloat16bfloat16` | ✅ |
|  | `mt_sqrt` | f32 | `v_Sqrtfloat32float32` | ✅ |
|  | `mt_sqrt` | f16 | `v_Sqrtfloat16float16` | ✅ |
|  | `mt_sqrt` | bf16 | `v_Sqrtbfloat16bfloat16` | ✅ |
|  | `mt_rsqrt` | f32 | `v_Rsqrtfloat32float32` | ✅ |
|  | `mt_rsqrt` | f16 | `v_Rsqrtfloat16float16` | ✅ |
|  | `mt_rsqrt` | bf16 | `v_Rsqrtbfloat16bfloat16` | ✅ |
|  | `mt_abs` | f32 | `v_Absfloat32float32` | ✅ |
|  | `mt_abs` | f16 | `v_Absfloat16float16` | ✅ |
|  | `mt_abs` | bf16 | `v_Absbfloat16bfloat16` | ✅ |
|  | `mt_silu` | f32 | *MLX computes as x·sigmoid(x), no standalone unary kernel* | — |
|  | `mt_silu` | f16 | *MLX computes as x·sigmoid(x), no standalone unary kernel* | — |
|  | `mt_silu` | bf16 | *MLX computes as x·sigmoid(x), no standalone unary kernel* | — |
|  | `mt_gelu` | f32 | *MLX uses composite poly, no standalone unary kernel* | — |
|  | `mt_gelu` | f16 | *MLX uses composite poly, no standalone unary kernel* | — |
|  | `mt_gelu` | bf16 | *MLX uses composite poly, no standalone unary kernel* | — |
|  | `mt_relu` | f32 | *MLX uses vvn_Maximum with scalar 0, not a unary op* | — |
|  | `mt_relu` | f16 | *MLX uses vvn_Maximum with scalar 0, not a unary op* | — |
|  | `mt_relu` | bf16 | *MLX uses vvn_Maximum with scalar 0, not a unary op* | — |
|  | `mt_cos` | f32 | `v_Cosfloat32float32` | ✅ |
|  | `mt_cos` | f16 | `v_Cosfloat16float16` | ✅ |
|  | `mt_cos` | bf16 | `v_Cosbfloat16bfloat16` | ✅ |
|  | `mt_sin` | f32 | `v_Sinfloat32float32` | ✅ |
|  | `mt_sin` | f16 | `v_Sinfloat16float16` | ✅ |
|  | `mt_sin` | bf16 | `v_Sinbfloat16bfloat16` | ✅ |
|  | `mt_ceil` | f32 | `v_Ceilfloat32float32` | ✅ |
|  | `mt_ceil` | f16 | `v_Ceilfloat16float16` | ✅ |
|  | `mt_ceil` | bf16 | `v_Ceilbfloat16bfloat16` | ✅ |
|  | `mt_floor` | f32 | `v_Floorfloat32float32` | ✅ |
|  | `mt_floor` | f16 | `v_Floorfloat16float16` | ✅ |
|  | `mt_floor` | bf16 | `v_Floorbfloat16bfloat16` | ✅ |
|  | `mt_erf` | f32 | `v_Erffloat32float32` | ✅ |
|  | `mt_erf` | f16 | `v_Erffloat16float16` | ✅ |
|  | `mt_erf` | bf16 | `v_Erfbfloat16bfloat16` | ✅ |
|  | `mt_exp2` | f32 | *not in instantiate_unary_float; MLX uses exp(x·ln2)* | — |
|  | `mt_exp2` | f16 | *not in instantiate_unary_float; MLX uses exp(x·ln2)* | — |
|  | `mt_exp2` | bf16 | *not in instantiate_unary_float; MLX uses exp(x·ln2)* | — |
|  | `mt_log2` | f32 | `v_Log2float32float32` | ✅ |
|  | `mt_log2` | f16 | `v_Log2float16float16` | ✅ |
|  | `mt_log2` | bf16 | `v_Log2bfloat16bfloat16` | ✅ |
|  | `mt_sign` | f32 | `v_Signfloat32float32` | ✅ |
|  | `mt_sign` | f16 | `v_Signfloat16float16` | ✅ |
|  | `mt_sign` | bf16 | `v_Signbfloat16bfloat16` | ✅ |
|  | `mt_round` | f32 | `v_Roundfloat32float32` | ✅ |
|  | `mt_round` | f16 | `v_Roundfloat16float16` | ✅ |
|  | `mt_round` | bf16 | `v_Roundbfloat16bfloat16` | ✅ |
|  | `mt_neg` | f32 | `v_Negativefloat32float32` | ✅ |
|  | `mt_neg` | f16 | `v_Negativefloat16float16` | ✅ |
|  | `mt_neg` | bf16 | `v_Negativebfloat16bfloat16` | ✅ |
|  | `mt_recip` | f32 | *not in unary.metal; MLX uses binary divide kernel* | — |
|  | `mt_recip` | f16 | *not in unary.metal; MLX uses binary divide kernel* | — |
|  | `mt_recip` | bf16 | *not in unary.metal; MLX uses binary divide kernel* | — |
|  | `mt_square` | f32 | `v_Squarefloat32float32` | ✅ |
|  | `mt_square` | f16 | `v_Squarefloat16float16` | ✅ |
|  | `mt_square` | bf16 | `v_Squarebfloat16bfloat16` | ✅ |
|  | `mt_sigmoid` | f32 | `v_Sigmoidfloat32float32` | ✅ |
|  | `mt_sigmoid` | f16 | `v_Sigmoidfloat16float16` | ✅ |
|  | `mt_sigmoid` | bf16 | `v_Sigmoidbfloat16bfloat16` | ✅ |
|  | `mt_log1p` | f32 | `v_Log1pfloat32float32` | ✅ |
|  | `mt_log1p` | f16 | `v_Log1pfloat16float16` | ✅ |
|  | `mt_log1p` | bf16 | `v_Log1pbfloat16bfloat16` | ✅ |
| binary | `vector_add` | f32 | `vvn_Addfloat32` | ✅ |
|  | `vector_add` | f16 | `vvn_Addfloat16` | ✅ |
|  | `vector_add` | bf16 | `vvn_Addbfloat16` | ✅ |
|  | `mt_mul` | f32 | `vvn_Multiplyfloat32` | ✅ |
|  | `mt_mul` | f16 | `vvn_Multiplyfloat16` | ✅ |
|  | `mt_mul` | bf16 | `vvn_Multiplybfloat16` | ✅ |
|  | `mt_sub` | f32 | `vvn_Subtractfloat32` | ✅ |
|  | `mt_sub` | f16 | `vvn_Subtractfloat16` | ✅ |
|  | `mt_sub` | bf16 | `vvn_Subtractbfloat16` | ✅ |
|  | `mt_div` | f32 | `vvn_Dividefloat32` | ✅ |
|  | `mt_div` | f16 | `vvn_Dividefloat16` | ✅ |
|  | `mt_div` | bf16 | `vvn_Dividebfloat16` | ✅ |
|  | `mt_max_elem` | f32 | `vvn_Maximumfloat32` | ✅ |
|  | `mt_max_elem` | f16 | `vvn_Maximumfloat16` | ✅ |
|  | `mt_max_elem` | bf16 | `vvn_Maximumbfloat16` | ✅ |
|  | `mt_min_elem` | f32 | `vvn_Minimumfloat32` | ✅ |
|  | `mt_min_elem` | f16 | `vvn_Minimumfloat16` | ✅ |
|  | `mt_min_elem` | bf16 | `vvn_Minimumbfloat16` | ✅ |
|  | `mt_pow` | f32 | `vvn_Powerfloat32` | ✅ |
|  | `mt_pow` | f16 | `vvn_Powerfloat16` | ✅ |
|  | `mt_pow` | bf16 | `vvn_Powerbfloat16` | ✅ |
|  | `mt_logaddexp` | f32 | `vvn_LogAddExpfloat32` | ✅ |
|  | `mt_logaddexp` | f16 | `vvn_LogAddExpfloat16` | ✅ |
|  | `mt_logaddexp` | bf16 | `vvn_LogAddExpbfloat16` | ✅ |
| binary_two | `mt_binary_two` | f32 | *no MLX equivalent — MT benchmarks 2-output fused pass that MLX doesn't expose* | — |
|  | `mt_binary_two` | f16 | *no MLX equivalent — MT benchmarks 2-output fused pass that MLX doesn't expose* | — |
|  | `mt_binary_two` | bf16 | *no MLX equivalent — MT benchmarks 2-output fused pass that MLX doesn't expose* | — |
| copy | `mt_copy` | f32 | `v_copyfloat32float32` | ✅ |
|  | `mt_copy` | f16 | `v_copyfloat16float16` | ✅ |
|  | `mt_copy` | bf16 | `v_copybfloat16bfloat16` | ✅ |
| arange | `mt_arange` | f32 | `arangefloat32` | ✅ |
|  | `mt_arange` | f16 | `arangefloat16` | ✅ |
|  | `mt_arange` | bf16 | `arangebfloat16` | ✅ |
| ternary | `mt_select` | f32 | `v_Selectfloat32` | ✅ |
|  | `mt_select` | f16 | `v_Selectfloat16` | ✅ |
|  | `mt_select` | bf16 | `v_Selectbfloat16` | ✅ |
| softmax | `mt_softmax` | f32 | `looped_softmax_float32` | ✅ |
|  | `mt_softmax` | f16 | `looped_softmax_float16` | ✅ |
|  | `mt_softmax` | bf16 | `looped_softmax_bfloat16` | ✅ |
| rms_norm | `mt_rms_norm` | f32 | `rmsfloat32` | ✅ |
|  | `mt_rms_norm` | f16 | `rmsfloat16` | ✅ |
|  | `mt_rms_norm` | bf16 | `rmsbfloat16` | ✅ |
| layer_norm | `mt_layer_norm` | f32 | `layer_norm_loopedfloat32` | ✅ |
|  | `mt_layer_norm` | f16 | `layer_norm_loopedfloat16` | ✅ |
|  | `mt_layer_norm` | bf16 | `layer_norm_loopedbfloat16` | ✅ |
| logsumexp | `mt_logsumexp` | f32 | `looped_logsumexp_float32` | ✅ |
|  | `mt_logsumexp` | f16 | `looped_logsumexp_float16` | ✅ |
|  | `mt_logsumexp` | bf16 | `looped_logsumexp_bfloat16` | ✅ |
| reduce | `mt_all_reduce` | f32 | `all_reduce_sumfloat32` | ✅ |
|  | `mt_all_reduce` | f16 | `all_reduce_sumfloat16` | ✅ |
|  | `mt_all_reduce` | bf16 | `all_reduce_sumbfloat16` | ✅ |
|  | `mt_all_reduce_max` | f32 | `all_reduce_maxfloat32` | ✅ |
|  | `mt_all_reduce_max` | f16 | `all_reduce_maxfloat16` | ✅ |
|  | `mt_all_reduce_max` | bf16 | `all_reduce_maxbfloat16` | ✅ |
|  | `mt_all_reduce_min` | f32 | `all_reduce_minfloat32` | ✅ |
|  | `mt_all_reduce_min` | f16 | `all_reduce_minfloat16` | ✅ |
|  | `mt_all_reduce_min` | bf16 | `all_reduce_minbfloat16` | ✅ |
|  | `mt_row_reduce` | f32 | `row_reduce_simple_sumfloat32` | ✅ |
|  | `mt_row_reduce` | f16 | `row_reduce_simple_sumfloat16` | ✅ |
|  | `mt_row_reduce` | bf16 | `row_reduce_simple_sumbfloat16` | ✅ |
|  | `mt_row_reduce_max` | f32 | `row_reduce_simple_maxfloat32` | ✅ |
|  | `mt_row_reduce_max` | f16 | `row_reduce_simple_maxfloat16` | ✅ |
|  | `mt_row_reduce_max` | bf16 | `row_reduce_simple_maxbfloat16` | ✅ |
|  | `mt_row_reduce_min` | f32 | `row_reduce_simple_minfloat32` | ✅ |
|  | `mt_row_reduce_min` | f16 | `row_reduce_simple_minfloat16` | ✅ |
|  | `mt_row_reduce_min` | bf16 | `row_reduce_simple_minbfloat16` | ✅ |
| gemv | `mt_gemv` | f32 | `gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0` | ✅ |
|  | `mt_gemv` | f16 | `gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0` | ✅ |
|  | `mt_gemv` | bf16 | `gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0` | ✅ |
| gemv_masked | `mt_gemv_masked` | f32 | *no nomask/nomask variant in instantiate_gemv_base;              all MLX variants require explicit mask buffers* | — |
|  | `mt_gemv_masked` | f16 | *no nomask/nomask variant in instantiate_gemv_base;              all MLX variants require explicit mask buffers* | — |
|  | `mt_gemv_masked` | bf16 | *no nomask/nomask variant in instantiate_gemv_base;              all MLX variants require explicit mask buffers* | — |
| rope | `mt_rope_f16` | f16 | `rope_float16` | ✅ |
|  | `mt_rope_f32` | f32 | *f32 rope not yet in MT bench; MLX rope_float32 exists* | — |
|  | `mt_rope_bf16` | bf16 | *bf16 rope not yet in MT bench; MLX rope_bfloat16 exists* | — |
| scaled_dot_product_attention | `mt_sdpa` | f32 | `sdpa_vector_float_128_128` | ✅ |
|  | `mt_sdpa` | f16 | `sdpa_vector_float16_t_128_128` | ✅ |
|  | `—` | bf16 | *bf16 SDPA not yet implemented in MT bench* | — |
| scan | `mt_scan_f32` | f32 | `contig_scan_inclusive_sum_float32_float32` | ✅ |
|  | `mt_scan_f16` | f16 | *f16/bf16 scan not yet in MT bench;                  MLX contig_scan_inclusive_sum_float16_float16 exists* | — |
|  | `mt_scan_bf16` | bf16 | *f16/bf16 scan not yet in MT bench;                  MLX contig_scan_inclusive_sum_bfloat16_bfloat16 exists* | — |
| arg_reduce | `mt_argmax_f32` | f32 | `argmax_float32` | ✅ |
|  | `mt_argmax_f16` | f16 | *f16/bf16 argmax not yet in MT bench; MLX argmax_float16 exists* | — |
|  | `mt_argmax_bf16` | bf16 | *f16/bf16 argmax not yet in MT bench; MLX argmax_bfloat16 exists* | — |
| sort | `mt_sort_f32` | f32 | `c_block_sort_float32_float32_bn256_tn4` | ✅ |
|  | `mt_sort_f16` | f16 | *f16/bf16 sort not yet in MT bench;                  MLX c_block_sort_float16_float16_* exists* | — |
|  | `mt_sort_bf16` | bf16 | *f16/bf16 sort not yet in MT bench;                  MLX c_block_sort_bfloat16_bfloat16_* exists* | — |
| random | `mt_random_hash` |  | `rbitsc` | ✅ |
| fp_quantized | `mt_fp4_quant_dequant` |  | `nvfp4_quantize_dequantize_float_gs_16_b_4` | ✅ |
| quantized | `mt_qmv_f32` |  | `affine_qmv_fast_float16_t_gs_64_b_4_batch_0` | ✅ |
| strided | `mt_strided_copy` | f32 | `copy_g_nd2float32float32` | ❌ |
|  | `mt_strided_copy` | f16 | `copy_g_nd2float16float16` | ❌ |
|  | `mt_strided_copy` | bf16 | `copy_g_nd2bfloat16bfloat16` | ❌ |
| steel/gemm/steel_gemm_fused | `mt_matmul` |  | `steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2` | ✅ |
| steel/gemm/steel_gemm_gather | `mt_matmul` |  | `steel_gather_mm_rhs_nn_float16_float16_bm64_bn64_bk16_wm2_wn2` | ❌ |
| steel/gemm/steel_gemm_masked | `mt_matmul` |  | `steel_gemm_block_outmask_nomask_opmask_nomask_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_MN_taligned_K_taligned` | ❌ |
| steel/gemm/steel_gemm_segmented | `mt_matmul` |  | `steel_segmented_mm_nn_float16_float16_bm64_bn64_bk16_wm2_wn2` | ✅ |
| steel/gemm/steel_gemm_splitk | `—` |  | *split-K GEMM requires two-kernel pipeline (compute + accumulate pass);              DSL has no support for cross-kernel scratch buffers* | — |
| steel/attn/steel_attention | `—` |  | *prefill flash-attention requires online softmax with per-tile rescaling              and tiled Q/K/V staging; not yet in DSL* | — |
| steel/conv/steel_conv | `—` |  | *2D implicit GEMM convolution not yet in MT bench* | — |
| steel/conv/steel_conv_3d | `—` |  | *3D implicit GEMM convolution not yet in MT bench* | — |
| steel/conv/steel_conv_general | `—` |  | *general convolution not yet in MT bench* | — |
| conv | `—` |  | *2D/depthwise convolution not yet implemented in MT* | — |
| fft | `—` |  | *FFT not yet implemented in MT* | — |
| fence | `—` |  | *GPU memory barrier — handled by Metal command encoder; no MT kernel needed* | — |

## Metal File Coverage

How many of each Metal file's instantiated kernels are used as references.

| Metal File | Total kernels | Benchmarked | % | Unbenchmarked examples |
|---|---|---|---|---|
| `arange.metal` | 12 | 3 | ⚠️ 25% | `arange`, `arangeint16`, `arangeint32`, … (+6 more) |
| `arg_reduce.metal` | 25 | 1 | ⚠️ 4% | `arg_reduce_general`, `argmax_bfloat16`, `argmax_bool_`, … (+21 more) |
| `binary.metal` | 4133 | 24 | ❌ 0% | `binary_g`, `binary_g_nd1`, `binary_g_nd2`, … (+4106 more) |
| `binary_two.metal` | 236 | 0 | ❌ 0% | `binary_g`, `binary_g_nd1`, `binary_g_nd2`, … (+233 more) |
| `conv.metal` | 43 | 0 | ❌ 0% | `depthwise_conv_1d`, `depthwise_conv_1d_bfloat16`, `depthwise_conv_1d_bfloat16_large`, … (+40 more) |
| `copy.metal` | 2512 | 3 | ❌ 0% | `copy_g`, `copy_g_nd1`, `copy_g_nd2`, … (+2506 more) |
| `fence.metal` | 3 | 0 | ❌ 0% | `fence_update`, `fence_wait`, `input_coherent` |
| `fft.metal` | 79 | 0 | ❌ 0% | `bluestein_fft`, `bluestein_fft_mem_1024_float2_float`, `bluestein_fft_mem_1024_float2_float2`, … (+76 more) |
| `fp_quantized.metal` | 296 | 1 | ❌ 0% | `fp_dequantize`, `fp_gather_qmm_n`, `fp_gather_qmm_rhs`, … (+292 more) |
| `fp_quantized_nax.metal` | 77 | 0 | ❌ 0% | `fp_gather_qmm_n_nax`, `fp_gather_qmm_rhs_nax`, `fp_gather_qmm_t_nax`, … (+74 more) |
| `gemv.metal` | 224 | 3 | ⚠️ 1% | `gemv_bfloat16_bm1_bn1_sm8_sn4_tm1_tn4_nc0_axpby0`, `gemv_bfloat16_bm1_bn1_sm8_sn4_tm1_tn4_nc0_axpby1`, `gemv_bfloat16_bm1_bn1_sm8_sn4_tm1_tn4_nc1_axpby0`, … (+218 more) |
| `gemv_masked.metal` | 528 | 0 | ❌ 0% | `gemv_outmask_bfloat16_opmask_bfloat16_bfloat16_bm2_bn1_sm2_sn16_tm1_tn4_nc0`, `gemv_outmask_bfloat16_opmask_bfloat16_bfloat16_bm2_bn1_sm2_sn16_tm1_tn4_nc1`, `gemv_outmask_bfloat16_opmask_bfloat16_bfloat16_bm2_bn1_sm2_sn16_tm4_tn4_nc0`, … (+525 more) |
| `layer_norm.metal` | 16 | 3 | ⚠️ 18% | `layer_norm_looped`, `layer_norm_single_row`, `layer_normbfloat16`, … (+10 more) |
| `logsumexp.metal` | 8 | 3 | ⚠️ 37% | `block_logsumexp_bfloat16`, `block_logsumexp_float16`, `block_logsumexp_float32`, … (+2 more) |
| `quantized.metal` | 1636 | 1 | ❌ 0% | `affine_dequantize`, `affine_dequantize_bfloat16_t_gs_128_b_2`, `affine_dequantize_bfloat16_t_gs_128_b_3`, … (+1632 more) |
| `quantized_nax.metal` | 545 | 0 | ❌ 0% | `affine_gather_qmm_n_nax`, `affine_gather_qmm_rhs_nax`, `affine_gather_qmm_rhs_nax_nn_bfloat16_t_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2`, … (+542 more) |
| `random.metal` | 2 | 1 | ⚠️ 50% | `rbits` |
| `reduce.metal` | 2177 | 18 | ❌ 0% | `all_reduce`, `all_reduce_andbool_`, `all_reduce_andint16`, … (+2156 more) |
| `rms_norm.metal` | 16 | 3 | ⚠️ 18% | `rms_looped`, `rms_loopedbfloat16`, `rms_loopedfloat16`, … (+10 more) |
| `rope.metal` | 22 | 1 | ⚠️ 4% | `rope`, `rope_bfloat16`, `rope_float32`, … (+18 more) |
| `scaled_dot_product_attention.metal` | 39 | 2 | ⚠️ 5% | `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_1_bfloat16_t_128_128`, … (+34 more) |
| `scan.metal` | 458 | 1 | ❌ 0% | `contig_scan_exclusive_logaddexp_bfloat16_bfloat16`, `contig_scan_exclusive_logaddexp_complex64_complex64`, `contig_scan_exclusive_logaddexp_float16_float16`, … (+454 more) |
| `softmax.metal` | 12 | 3 | ⚠️ 25% | `block_softmax_bfloat16`, `block_softmax_float16`, `block_softmax_float32`, … (+6 more) |
| `sort.metal` | 265 | 1 | ❌ 0% | `c_block_sort_bfloat16_bfloat16_bn128_tn4`, `c_block_sort_bfloat16_bfloat16_bn256_tn4`, `c_block_sort_bfloat16_bfloat16_bn32_tn4`, … (+261 more) |
| `steel/attn/steel_attention.metal` | 18 | 0 | ❌ 0% | `steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1_maskbfloat16`, `steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1_maskbool_`, `steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16`, … (+15 more) |
| `steel/attn/steel_attention_nax.metal` | 24 | 0 | ❌ 0% | `steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1_maskbfloat16`, `steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1_maskbool_`, `steel_attention_bfloat16_bq64_bk32_bd64_wm4_wn1_maskbfloat16`, … (+21 more) |
| `steel/conv/steel_conv.metal` | 109 | 0 | ❌ 0% | `implicit_gemm_conv_2d`, `implicit_gemm_conv_2d_bfloat16_bm32_bn32_bk16_wm2_wn2_channel_1_filter_l`, `implicit_gemm_conv_2d_bfloat16_bm32_bn32_bk16_wm2_wn2_channel_2_filter_l`, … (+106 more) |
| `steel/conv/steel_conv_3d.metal` | 36 | 0 | ❌ 0% | `implicit_gemm_conv_3d_bfloat16_bm32_bn32_bk16_wm2_wn2_filter_l`, `implicit_gemm_conv_3d_bfloat16_bm32_bn32_bk16_wm2_wn2_filter_s`, `implicit_gemm_conv_3d_bfloat16_bm32_bn64_bk16_wm2_wn2_filter_l`, … (+33 more) |
| `steel/conv/steel_conv_general.metal` | 19 | 0 | ❌ 0% | `implicit_gemm_conv_2d_general`, `implicit_gemm_conv_2d_general_bfloat16_bm32_bn32_bk16_wm2_wn2`, `implicit_gemm_conv_2d_general_bfloat16_bm32_bn64_bk16_wm2_wn2`, … (+16 more) |
| `steel/gemm/steel_gemm_fused.metal` | 96 | 1 | ⚠️ 1% | `steel_gemm_fused_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2`, `steel_gemm_fused_nn_bfloat16_bfloat16_bm32_bn64_bk16_wm1_wn2`, `steel_gemm_fused_nn_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2`, … (+92 more) |
| `steel/gemm/steel_gemm_fused_nax.metal` | 72 | 0 | ❌ 0% | `steel_gemm_fused_nax_nn_bfloat16_bfloat16_bm128_bn128_bk256_wm4_wn4`, `steel_gemm_fused_nax_nn_bfloat16_bfloat16_bm128_bn128_bk512_wm4_wn4`, `steel_gemm_fused_nax_nn_bfloat16_bfloat16_bm128_bn128_bk64_wm4_wn4`, … (+69 more) |
| `steel/gemm/steel_gemm_gather.metal` | 66 | 0 | ❌ 0% | `steel_gather_mm_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2`, `steel_gather_mm_nn_bfloat16_bfloat16_bm32_bn64_bk16_wm1_wn2`, `steel_gather_mm_nn_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2`, … (+63 more) |
| `steel/gemm/steel_gemm_gather_nax.metal` | 12 | 0 | ❌ 0% | `steel_gather_mm_rhs_nax_nn_bfloat16_bfloat16_bm16_bn128_bk128_wm1_wn4`, `steel_gather_mm_rhs_nax_nn_bfloat16_bfloat16_bm32_bn128_bk128_wm1_wn4`, `steel_gather_mm_rhs_nax_nn_bfloat16_bfloat16_bm64_bn128_bk128_wm2_wn4`, … (+9 more) |
| `steel/gemm/steel_gemm_masked.metal` | 768 | 0 | ❌ 0% | `steel_gemm_block_outmask_bfloat16_opmask_bfloat16_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_naligned`, `steel_gemm_block_outmask_bfloat16_opmask_bfloat16_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_taligned`, `steel_gemm_block_outmask_bfloat16_opmask_bfloat16_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_naligned`, … (+765 more) |
| `steel/gemm/steel_gemm_segmented.metal` | 60 | 1 | ⚠️ 1% | `steel_segmented_mm_nn_bfloat16_bfloat16_bm32_bn32_bk16_wm2_wn2`, `steel_segmented_mm_nn_bfloat16_bfloat16_bm32_bn64_bk16_wm1_wn2`, `steel_segmented_mm_nn_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2`, … (+56 more) |
| `steel/gemm/steel_gemm_splitk.metal` | 266 | 0 | ❌ 0% | `gemm_splitk_accum`, `gemm_splitk_accum_axpby`, `steel_gemm_splitk_accum_bfloat16_float32`, … (+263 more) |
| `steel/gemm/steel_gemm_splitk_nax.metal` | 24 | 0 | ❌ 0% | `steel_gemm_splitk_nax_nn_bfloat16_float32_bm128_bn128_bk512_wm4_wn4`, `steel_gemm_splitk_nax_nn_bfloat16_float32_bm64_bn64_bk256_wm2_wn2`, `steel_gemm_splitk_nax_nn_float16_float32_bm128_bn128_bk512_wm4_wn4`, … (+21 more) |
| `ternary.metal` | 218 | 3 | ⚠️ 1% | `g1_Selectbfloat16`, `g1_Selectbool_`, `g1_Selectcomplex64`, … (+212 more) |
| `unary.metal` | 880 | 51 | ⚠️ 5% | `gn1_Absbfloat16bfloat16`, `gn1_Absbool_bool_`, `gn1_Abscomplex64complex64`, … (+826 more) |

**Total**: 128/16032 instantiated kernels benchmarked (0%)
