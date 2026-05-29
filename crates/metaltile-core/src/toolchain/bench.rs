//! Benchmark types: [`BenchBuffer`], [`RefKernel`], [`BenchSetup`],
//! [`KernelBench`], [`KernelBenchEntry`], and supporting primitives.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{dsl::dtype::DType, ir::Kernel};

pub(super) fn random_bytes(len: usize) -> Vec<u8> {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(42);
    let mut state = seed as u64 ^ 0x9e3779b97f4a7c15;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.push(state as u8);
    }
    data
}

// ---------------------------------------------------------------------------
// Primitive value types
// ---------------------------------------------------------------------------

/// Throughput in gigabytes per second.
///
/// Prevents accidental swapping of throughput and latency values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Gbps(pub f64);

impl fmt::Display for Gbps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:.1} GB/s", self.0) }
}

/// Latency in microseconds.
///
/// Prevents accidental swapping of latency and throughput values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Microseconds(pub f64);

impl fmt::Display for Microseconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:.1} µs", self.0) }
}

/// A compile-time constant value forwarded to a kernel at dispatch time.
///
/// These correspond to `#[constexpr]` parameters in the kernel signature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// Pointer-sized unsigned integer.
    Usize(usize),
}

impl ConstValue {
    /// Return the value as a `u32` if it is representable, or an error.
    pub fn as_u32(&self) -> crate::Result<u32> {
        match *self {
            ConstValue::U32(v) => Ok(v),
            ConstValue::I32(v) => u32::try_from(v)
                .map_err(|_| crate::Error::Internal(format!("ConstValue {v} out of u32 range"))),
            ConstValue::Usize(v) => u32::try_from(v)
                .map_err(|_| crate::Error::Internal(format!("ConstValue {v} out of u32 range"))),
            _ => Err(crate::Error::Internal(format!(
                "ConstValue {self:?} is not representable as u32"
            ))),
        }
    }
}

impl From<u32> for ConstValue {
    fn from(v: u32) -> Self { ConstValue::U32(v) }
}
impl From<i32> for ConstValue {
    fn from(v: i32) -> Self { ConstValue::I32(v) }
}
impl From<f32> for ConstValue {
    fn from(v: f32) -> Self { ConstValue::F32(v) }
}
impl From<u64> for ConstValue {
    fn from(v: u64) -> Self { ConstValue::U64(v) }
}
impl From<i64> for ConstValue {
    fn from(v: i64) -> Self { ConstValue::I64(v) }
}
impl From<usize> for ConstValue {
    fn from(v: usize) -> Self { ConstValue::Usize(v) }
}

/// Dispatch dimensions for a kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    /// Threadgroups per grid axis (x, y, z).
    pub grid: [u32; 3],
    /// Threads per threadgroup axis (x, y, z).
    pub tpg: [u32; 3],
}

impl Grid {
    /// Create a 1D grid from total elements and threads-per-group.
    pub fn new_1d(n: usize, tpg: u32) -> Self {
        let grid_x = (n as u32).div_ceil(tpg);
        Grid { grid: [grid_x, 1, 1], tpg: [tpg, 1, 1] }
    }

    /// Create a 2D grid.
    pub fn new_2d(x: u32, y: u32, tpg: [u32; 2]) -> Self {
        Grid { grid: [x, y, 1], tpg: [tpg[0], tpg[1], 1] }
    }

    /// Create a 3D grid.
    pub fn new_3d(x: u32, y: u32, z: u32, tpg: [u32; 3]) -> Self { Grid { grid: [x, y, z], tpg } }
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}x{}x{}) / ({}x{}x{})",
            self.grid[0], self.grid[1], self.grid[2], self.tpg[0], self.tpg[1], self.tpg[2]
        )
    }
}

// ---------------------------------------------------------------------------
// BenchBuffer
// ---------------------------------------------------------------------------

/// How a GPU buffer should be initialised before running a benchmark.
#[derive(Debug, Clone)]
enum BufferInit {
    Random,
    Zeros,
    FromVec(Vec<u8>),
}

/// Describes a GPU buffer for a benchmark run.
///
/// Fields are private — construction goes through named constructors.
#[derive(Debug, Clone)]
pub struct BenchBuffer {
    name: String,
    len: usize,
    dtype: DType,
    is_output: bool,
    init: BufferInit,
}

impl BenchBuffer {
    /// Create a buffer initialised with random data.
    pub fn random(name: &str, len: usize, dtype: DType) -> Self {
        BenchBuffer {
            name: name.to_string(),
            len,
            dtype,
            is_output: false,
            init: BufferInit::Random,
        }
    }

    /// Create a buffer initialised with zeros.
    pub fn zeros(name: &str, len: usize, dtype: DType) -> Self {
        BenchBuffer {
            name: name.to_string(),
            len,
            dtype,
            is_output: false,
            init: BufferInit::Zeros,
        }
    }

    /// Create a buffer from concrete byte data.
    pub fn from_vec(name: &str, data: Vec<u8>, dtype: DType) -> Self {
        let len = data.len() / dtype.size_bytes();
        BenchBuffer {
            name: name.to_string(),
            len,
            dtype,
            is_output: false,
            init: BufferInit::FromVec(data),
        }
    }

    /// Mark this buffer as an output slot.
    pub fn output(mut self) -> Self {
        self.is_output = true;
        self
    }

    /// Buffer name (matches the kernel parameter name).
    pub fn name(&self) -> &str { &self.name }

    /// Number of elements.
    pub fn len(&self) -> usize { self.len }

    /// Whether the buffer has zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Element data type.
    pub fn dtype(&self) -> DType { self.dtype }

    /// Whether this buffer is an output.
    pub fn is_output(&self) -> bool { self.is_output }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> u64 { (self.len * self.dtype.size_bytes()) as u64 }

    /// Allocate and fill the initial byte content for this buffer.
    pub fn initial_bytes(&self) -> Vec<u8> {
        let n = self.size_bytes() as usize;
        match &self.init {
            BufferInit::Random => random_bytes(n),
            BufferInit::Zeros => vec![0u8; n],
            BufferInit::FromVec(v) => v.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// RefKernel
// ---------------------------------------------------------------------------

/// A reference Metal kernel to benchmark against a MetalTile kernel.
#[derive(Debug, Clone)]
pub struct RefKernel {
    /// The Metal kernel function name to dispatch, e.g. `"vvn_expfloat32"`.
    pub fn_name: String,
    /// Path to the `.metal` source file, relative to `reference_metal_path`.
    pub metal_file: String,
    /// Buffers bound positionally (`[[buffer(0)]]`, `[[buffer(1)]]`, etc.).
    pub buffers: Vec<BenchBuffer>,
    /// Dispatch grid for the reference kernel.
    pub grid: Grid,
}

// ---------------------------------------------------------------------------
// BenchSetup
// ---------------------------------------------------------------------------

/// Complete benchmark configuration for a single kernel/dtype combination.
///
/// Built via a consuming builder pattern. Call `build()` to finalise —
/// it returns an error if no grid was set.
#[derive(Debug, Clone)]
pub struct BenchSetup {
    kernel: Kernel,
    buffers: Vec<BenchBuffer>,
    constexprs: Vec<(String, ConstValue)>,
    grid: Option<Grid>,
    bytes_moved: Option<u64>,
    ref_kernel: Option<RefKernel>,
}

impl BenchSetup {
    /// Create a new `BenchSetup` builder for the given kernel IR.
    pub fn new(kernel: Kernel) -> Self {
        BenchSetup {
            kernel,
            buffers: Vec::new(),
            constexprs: Vec::new(),
            grid: None,
            bytes_moved: None,
            ref_kernel: None,
        }
    }

    /// Add a GPU buffer.
    pub fn buffer(mut self, b: BenchBuffer) -> Self {
        self.buffers.push(b);
        self
    }

    /// Add a compile-time constant.
    pub fn constexpr(mut self, name: &str, v: impl Into<ConstValue>) -> Self {
        self.constexprs.push((name.to_string(), v.into()));
        self
    }

    /// Set a 1D grid.
    pub fn grid_1d(mut self, n: usize, tpg: u32) -> Self {
        self.grid = Some(Grid::new_1d(n, tpg));
        self
    }

    /// Set a 2D grid.
    pub fn grid_2d(mut self, x: u32, y: u32, tpg: [u32; 2]) -> Self {
        self.grid = Some(Grid::new_2d(x, y, tpg));
        self
    }

    /// Set a 3D grid.
    pub fn grid_3d(mut self, x: u32, y: u32, z: u32, tpg: [u32; 3]) -> Self {
        self.grid = Some(Grid::new_3d(x, y, z, tpg));
        self
    }

    /// Override the bytes-moved figure for bandwidth computation.
    pub fn bytes_moved(mut self, bytes: u64) -> Self {
        self.bytes_moved = Some(bytes);
        self
    }

    /// Attach a reference Metal kernel.
    pub fn with_reference(mut self, ref_kernel: RefKernel) -> Self {
        self.ref_kernel = Some(ref_kernel);
        self
    }

    /// Finalise the builder. Returns an error if no grid was set.
    pub fn build(self) -> crate::Result<BenchSetup> {
        if self.grid.is_none() {
            return Err(crate::Error::Internal(
                "BenchSetup missing grid — call grid_1d(), grid_2d(), or grid_3d()".into(),
            ));
        }
        Ok(self)
    }

    /// The kernel IR for this benchmark.
    pub fn kernel(&self) -> &Kernel { &self.kernel }

    /// The buffers to allocate.
    pub fn buffers(&self) -> &[BenchBuffer] { &self.buffers }

    /// Dispatch grid. Panics if `build()` was not called.
    pub fn grid(&self) -> &Grid {
        self.grid.as_ref().expect("BenchSetup grid accessed before build()")
    }

    /// Constexpr values for the kernel.
    pub fn constexprs(&self) -> &[(String, ConstValue)] { &self.constexprs }

    /// Total bytes moved for bandwidth calculation.
    pub fn compute_bytes_moved(&self) -> u64 {
        self.bytes_moved.unwrap_or_else(|| self.buffers.iter().map(|b| b.size_bytes()).sum())
    }

    /// Byte size of a named buffer, or 0 if not found.
    pub fn buffer_bytes(&self, name: &str) -> u64 {
        self.buffers.iter().find(|b| b.name == name).map(|b| b.size_bytes()).unwrap_or(0)
    }

    /// Optional reference Metal kernel.
    pub fn ref_kernel(&self) -> Option<&RefKernel> { self.ref_kernel.as_ref() }
}

// ---------------------------------------------------------------------------
// KernelBench trait + inventory
// ---------------------------------------------------------------------------

/// Trait for benchmark definitions.
///
/// Prefer the `#[bench]` macro over implementing this directly.
pub trait KernelBench: Send + Sync {
    /// Unique benchmark name (e.g. `"unary/exp"`).
    fn name(&self) -> &str;

    /// Data types to benchmark.
    fn dtypes(&self) -> &[DType];

    /// Build the `BenchSetup` for a specific dtype.
    fn setup(&self, dt: DType) -> BenchSetup;

    /// Optional reference Metal kernel for live comparison.
    fn reference_kernel(&self) -> Option<RefKernel> { None }

    /// Bytes moved for bandwidth calculation (default: sum of all buffer sizes).
    fn bytes_moved(&self, setup: &BenchSetup) -> u64 { setup.compute_bytes_moved() }
}

/// Inventory wrapper for a [`KernelBench`] implementation.
pub struct KernelBenchEntry {
    pub(crate) inner: &'static dyn KernelBench,
}

impl KernelBenchEntry {
    /// Wrap a `KernelBench` impl for inventory submission.
    pub const fn new(inner: &'static dyn KernelBench) -> Self { KernelBenchEntry { inner } }
}

impl AsRef<dyn KernelBench + 'static> for KernelBenchEntry {
    fn as_ref(&self) -> &(dyn KernelBench + 'static) { self.inner }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;

    #[test]
    fn bench_buffer_named_constructors() {
        let r = BenchBuffer::random("x", 100, DType::F32);
        assert_eq!(r.name(), "x");
        assert_eq!(r.len(), 100);
        assert_eq!(r.dtype(), DType::F32);
        assert!(!r.is_output());

        let z = BenchBuffer::zeros("y", 200, DType::F16).output();
        assert_eq!(z.name(), "y");
        assert!(z.is_output());
    }

    #[test]
    fn bench_buffer_size_bytes() {
        let b = BenchBuffer::random("x", 1024, DType::F32);
        assert_eq!(b.size_bytes(), 4096);
    }

    #[test]
    fn bench_setup_consuming_builder() {
        let setup = BenchSetup::new(Kernel::new("k"))
            .buffer(BenchBuffer::random("in", 64, DType::F32))
            .buffer(BenchBuffer::zeros("out", 64, DType::F32).output())
            .constexpr("n", 64u32)
            .grid_1d(64, 16)
            .build()
            .unwrap();

        assert_eq!(setup.buffers().len(), 2);
        assert_eq!(setup.constexprs().len(), 1);
        assert_eq!(setup.grid().grid[0], 4);
        assert_eq!(setup.grid().tpg[0], 16);
    }

    #[test]
    fn bench_setup_build_requires_grid() {
        let err = BenchSetup::new(Kernel::new("k"))
            .buffer(BenchBuffer::random("in", 64, DType::F32))
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("missing grid"));
    }

    #[test]
    fn grid_1d_computes_correctly() {
        let g = Grid::new_1d(1000, 256);
        assert_eq!(g.grid[0], 4);
        assert_eq!(g.grid[1], 1);
        assert_eq!(g.grid[2], 1);
        assert_eq!(g.tpg[0], 256);
    }

    #[test]
    fn grid_display() {
        let g = Grid::new_2d(8, 4, [16, 8]);
        let s = format!("{g}");
        assert!(s.contains("8") && s.contains("4") && s.contains("16"));
    }
}
