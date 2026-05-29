//! Test setup types: [`TestBuffer`], [`TestSetup`], [`KernelTest`], [`KernelTestEntry`].

use super::bench::{ConstValue, Grid, random_bytes};
use crate::{dsl::dtype::DType, ir::Kernel};

// ---------------------------------------------------------------------------
// TestBuffer
// ---------------------------------------------------------------------------

/// Describes a CPU-side buffer used as input or expected output in correctness tests.
#[derive(Debug, Clone)]
pub struct TestBuffer {
    name: String,
    data: Vec<u8>,
    dtype: DType,
}

impl TestBuffer {
    /// Create a test buffer filled with random data.
    pub fn random(name: &str, len: usize, dtype: DType) -> Self {
        let data = random_bytes(len * dtype.size_bytes());
        TestBuffer { name: name.to_string(), data, dtype }
    }

    /// Create a test buffer filled with the given data.
    pub fn from_vec(name: &str, data: Vec<u8>, dtype: DType) -> Self {
        TestBuffer { name: name.to_string(), data, dtype }
    }

    /// Map each element (interpreted as f32) through a CPU function.
    ///
    /// # Panics
    ///
    /// Panics if the buffer's element size is not 4 bytes (f32/i32/u32 only).
    pub fn map_f32<F>(&self, f: F) -> Self
    where F: Fn(f32) -> f32 {
        assert_eq!(
            self.dtype.size_bytes(),
            4,
            "map_f32 requires 4-byte dtype, but buffer '{}' has {:?}",
            self.name,
            self.dtype
        );
        let out_bytes: Vec<u8> = self
            .data
            .chunks_exact(4)
            .flat_map(|chunk| {
                let val = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                f(val).to_ne_bytes()
            })
            .collect();
        TestBuffer { name: self.name.clone(), data: out_bytes, dtype: self.dtype }
    }

    /// Map bytes through a raw function (for non-f32 dtypes).
    pub fn map_raw<F>(&self, f: F) -> Self
    where F: Fn(&[u8]) -> Vec<u8> {
        TestBuffer { name: self.name.clone(), data: f(&self.data), dtype: self.dtype }
    }

    /// Rename this buffer.
    pub fn rename(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Buffer name.
    pub fn name(&self) -> &str { &self.name }

    /// Reference to the byte data.
    pub fn data(&self) -> &[u8] { &self.data }

    /// Element data type.
    pub fn dtype(&self) -> DType { self.dtype }

    /// Number of elements.
    pub fn len(&self) -> usize { self.data.len() / self.dtype.size_bytes() }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

// ---------------------------------------------------------------------------
// TestSetup
// ---------------------------------------------------------------------------

/// Complete test configuration for verifying kernel correctness.
///
/// Built via a consuming builder pattern. Call `build()` to finalise —
/// it returns an error if no grid was set.
#[derive(Debug, Clone)]
pub struct TestSetup {
    kernel: Kernel,
    inputs: Vec<TestBuffer>,
    expected: Vec<TestBuffer>,
    constexprs: Vec<(String, ConstValue)>,
    grid: Option<Grid>,
    ref_setup: Option<Box<TestSetup>>,
}

impl TestSetup {
    /// Create a new `TestSetup` builder for the given kernel IR.
    pub fn new(kernel: Kernel) -> Self {
        TestSetup {
            kernel,
            inputs: Vec::new(),
            expected: Vec::new(),
            constexprs: Vec::new(),
            grid: None,
            ref_setup: None,
        }
    }

    /// Add an input buffer (sent to the GPU).
    pub fn input(mut self, b: TestBuffer) -> Self {
        self.inputs.push(b);
        self
    }

    /// Add an expected output buffer (compared against GPU result).
    pub fn expect(mut self, b: TestBuffer) -> Self {
        self.expected.push(b);
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

    /// Set a GPU-vs-GPU reference setup (no CPU oracle needed).
    pub fn compare_against(mut self, ref_setup: TestSetup) -> Self {
        self.ref_setup = Some(Box::new(ref_setup));
        self
    }

    /// Finalise the builder. Returns an error if no grid was set.
    pub fn build(self) -> crate::Result<TestSetup> {
        if self.grid.is_none() {
            return Err(crate::Error::Internal(
                "TestSetup missing grid — call grid_1d(), grid_2d(), or grid_3d()".into(),
            ));
        }
        Ok(self)
    }

    /// The kernel IR for this test.
    pub fn kernel(&self) -> &Kernel { &self.kernel }

    /// Input buffers.
    pub fn inputs(&self) -> &[TestBuffer] { &self.inputs }

    /// Expected output buffers.
    pub fn expected(&self) -> &[TestBuffer] { &self.expected }

    /// Dispatch grid. Panics if `build()` was not called.
    pub fn grid(&self) -> &Grid {
        self.grid.as_ref().expect("TestSetup grid accessed before build()")
    }

    /// Constexpr values.
    pub fn constexprs(&self) -> &[(String, ConstValue)] { &self.constexprs }

    /// Optional GPU-vs-GPU reference setup.
    pub fn ref_setup(&self) -> Option<&TestSetup> { self.ref_setup.as_deref() }
}

// ---------------------------------------------------------------------------
// KernelTest trait + inventory
// ---------------------------------------------------------------------------

/// Trait for correctness test definitions.
///
/// Prefer the `#[test_kernel]` macro over implementing this directly.
pub trait KernelTest: Send + Sync {
    /// Unique test name (e.g. `"unary/exp"`).
    fn name(&self) -> &str;

    /// Data types to test.
    fn dtypes(&self) -> &[DType];

    /// Build the `TestSetup` for a specific dtype.
    fn setup(&self, dt: DType) -> TestSetup;

    /// Tolerance for element-wise comparison (default: `1e-4`).
    fn tolerance(&self, _dt: DType) -> f64 { 1e-4 }
}

/// Inventory wrapper for a [`KernelTest`] implementation.
pub struct KernelTestEntry {
    pub(crate) inner: &'static dyn KernelTest,
}

impl KernelTestEntry {
    /// Wrap a `KernelTest` impl for inventory submission.
    pub const fn new(inner: &'static dyn KernelTest) -> Self { KernelTestEntry { inner } }
}

impl AsRef<dyn KernelTest + 'static> for KernelTestEntry {
    fn as_ref(&self) -> &(dyn KernelTest + 'static) { self.inner }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;

    #[test]
    fn test_setup_build_requires_grid() {
        let err = TestSetup::new(Kernel::new("k"))
            .input(TestBuffer::random("x", 64, DType::F32))
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("missing grid"));
    }

    #[test]
    fn test_buffer_map_f32() {
        let input = TestBuffer::random("input", 100, DType::F32);
        assert_eq!(input.len(), 100);
        let expected = input.map_f32(|x| x * 2.0).rename("out");
        assert_eq!(expected.name(), "out");
        assert_eq!(expected.len(), 100);
    }
}
