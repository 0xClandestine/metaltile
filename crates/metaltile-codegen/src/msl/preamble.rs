//! MSL preamble emission: BF16 compatibility struct and activation helpers.
//!
//! These are emitted once at the top of the generated MSL, before the kernel
//! function, based on the `KernelFeatures` analysis.

use std::fmt::Write;

use super::features::KernelFeatures;
use crate::wl;

impl super::MslGenerator {
    /// Emit the BF16 compatibility struct for pre-Metal-3.1 targets.
    pub(super) fn emit_bf16_preamble(&self, out: &mut String) {
        wl!(out);
        wl!(out, "// BF16 compatibility struct for pre-Metal-3.1 targets");
        wl!(out, "struct bfloat16_t {{");
        wl!(out, "    uint16_t bits;");
        wl!(out, "    bfloat16_t() = default;");
        wl!(out, "    bfloat16_t(float v) {{");
        wl!(out, "        uint32_t x = as_type<uint32_t>(v);");
        wl!(out, "        bits = uint16_t((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const device {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const threadgroup {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "}};");
    }

    /// Emit activation helper template functions.
    pub(super) fn emit_activation_helpers(&self, feat: &KernelFeatures, out: &mut String) {
        if feat.needs_silu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_silu(T x) {{ return x / (T(1) + exp(-x)); }}");
        }
        if feat.needs_gelu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_gelu(T x) {{");
            wl!(out, "    const T k = T(0.7978845608f);");
            wl!(out, "    return T(0.5f) * x * (T(1) + tanh(k * (x + T(0.044715f) * x*x*x)));");
            wl!(out, "}}");
        }
        if feat.needs_relu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_relu(T x) {{ return max(T(0), x); }}");
        }
        if feat.needs_sigmoid {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_sigmoid(T x) {{ return T(1) / (T(1) + exp(-x)); }}");
        }
        if feat.needs_erf {
            wl!(out);
            // Polynomial approximation matching MLX erf.h (max error < 1 ulp)
            wl!(out, "inline float mt_erf_impl(float a) {{");
            wl!(out, "    float r, s, t, u;");
            wl!(out, "    t = metal::abs(a);");
            wl!(out, "    s = a * a;");
            wl!(out, "    if (t > 0.927734375f) {{");
            wl!(out, "        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);");
            wl!(out, "        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);");
            wl!(out, "        r = metal::fma(r, s, u);");
            wl!(out, "        r = metal::fma(r, t, -1.06777877e-1f);");
            wl!(out, "        r = metal::fma(r, t, -6.34846687e-1f);");
            wl!(out, "        r = metal::fma(r, t, -1.28717512e-1f);");
            wl!(out, "        r = metal::fma(r, t, -t);");
            wl!(out, "        r = -(exp(r) - 1.0f);");
            wl!(out, "        r = metal::copysign(r, a);");
            wl!(out, "    }} else {{");
            wl!(out, "        r = -5.96761703e-4f;");
            wl!(out, "        r = metal::fma(r, s,  4.99119423e-3f);");
            wl!(out, "        r = metal::fma(r, s, -2.67681349e-2f);");
            wl!(out, "        r = metal::fma(r, s,  1.12819925e-1f);");
            wl!(out, "        r = metal::fma(r, s, -3.76125336e-1f);");
            wl!(out, "        r = metal::fma(r, s,  1.28379166e-1f);");
            wl!(out, "        r = metal::fma(r, a, a);");
            wl!(out, "    }}");
            wl!(out, "    return r;");
            wl!(out, "}}");
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_erf_impl(T x) {{ return T(mt_erf_impl(float(x))); }}");
        }
    }
}
