//! PSO-reflection probe.
//!
//! Compiles a small fleet of representative kernels at varying
//! `expected_tpg` values and dumps each PSO's reflection
//! (`maxTotalThreadsPerThreadgroup`, `staticThreadgroupMemoryLength`,
//! `threadExecutionWidth`). Goal: show that reflection values are
//! *real and discriminating* — i.e., they change with kernel
//! complexity and TPG, so they're useful features for the autotune
//! cache.
//!
//! Usage:
//!     cargo run -p metaltile-std --example pso_reflection_probe --release
//!
//! macOS-only.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("pso_reflection_probe requires macOS — PSO reflection is Metal-specific.");
}

#[cfg(target_os = "macos")]
fn main() {
    use metaltile_std::runner::GpuRunner;

    // Three kernels of escalating register/memory pressure. The
    // simplest is a load-store. The middle adds a few multiplies. The
    // heaviest does a per-thread polynomial — enough ALU to plausibly
    // push register usage above the small-kernel floor.
    let kernels: &[(&str, &str)] = &[
        (
            "add_one",
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void add_one(device float* in_  [[buffer(0)]],
                                device float* out_ [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
                out_[gid] = in_[gid] + 1.0;
            }
            "#,
        ),
        (
            "saxpy",
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void saxpy(device float* x [[buffer(0)]],
                              device float* y [[buffer(1)]],
                              device float* out_ [[buffer(2)]],
                              constant float& a [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
                out_[gid] = a * x[gid] + y[gid];
            }
            "#,
        ),
        (
            "poly8",
            // Horner-form degree-8 polynomial. Long dependency chain,
            // many live values — exercises the register allocator.
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void poly8(device float* in_  [[buffer(0)]],
                              device float* out_ [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
                float x = in_[gid];
                float r = 0.5;
                r = r * x + 0.25;
                r = r * x + 0.125;
                r = r * x + 0.0625;
                r = r * x + 0.03125;
                r = r * x + 0.015625;
                r = r * x + 0.0078125;
                r = r * x + 0.00390625;
                r = r * x + 0.001953125;
                out_[gid] = r;
            }
            "#,
        ),
        (
            "reduce_tg",
            // Tree-reduction over a 1024-wide threadgroup using
            // threadgroup memory. `static_tgm` should jump to 4096
            // (1024 × sizeof(float)).
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void reduce_tg(device float* in_   [[buffer(0)]],
                                  device float* out_  [[buffer(1)]],
                                  uint  gid [[thread_position_in_grid]],
                                  uint  lid [[thread_position_in_threadgroup]],
                                  uint  tgid [[threadgroup_position_in_grid]]) {
                threadgroup float scratch[1024];
                scratch[lid] = in_[gid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = 512; s > 0; s >>= 1) {
                    if (lid < s) scratch[lid] += scratch[lid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lid == 0) out_[tgid] = scratch[0];
            }
            "#,
        ),
        (
            "tile8x8",
            // Each thread accumulates an 8×8 register tile. That's 64
            // floats per thread = 256 bytes of register state per
            // thread, which on a 32-thread simdgroup is 8 KB — close
            // to the per-simdgroup register file budget on Apple GPUs.
            // Expect `max_tpg` to drop below 1024.
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void tile8x8(device float* a   [[buffer(0)]],
                                device float* b   [[buffer(1)]],
                                device float* out_ [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
                float acc[8][8];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[i][j] = 0.0;
                #pragma unroll
                for (int k = 0; k < 16; ++k) {
                    float av[8], bv[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) av[i] = a[gid * 128 + k * 8 + i];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) bv[j] = b[gid * 128 + k * 8 + j];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[i][j] += av[i] * bv[j];
                }
                float s = 0.0;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        s += acc[i][j];
                out_[gid] = s;
            }
            "#,
        ),
    ];

    let runner = GpuRunner::new().expect("Metal device");
    println!("device: {}", runner.device_name);
    println!();
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "kernel", "max_tpg", "static_tgm", "exec_width", "src_lines",
    );
    println!("{}", "-".repeat(58));

    for (name, msl) in kernels {
        let compiled = match runner.compile(msl, name) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{name}: compile failed: {e}");
                continue;
            },
        };
        let r = runner.pso_reflection(&compiled);
        // Source lines are a rough proxy for "kernel complexity" so
        // the reader can correlate reflection deltas with code size.
        let src_lines = msl.lines().count();
        println!(
            "{:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            name,
            r.max_total_threads_per_threadgroup,
            r.static_threadgroup_memory_length,
            r.thread_execution_width,
            src_lines,
        );
    }
}
