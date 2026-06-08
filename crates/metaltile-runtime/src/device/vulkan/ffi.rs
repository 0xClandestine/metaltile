//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Hand-rolled FFI to the Vulkan compute pipeline and shaderc — only the
//! symbols the Phase-1 elementwise compute path uses
//! (`VULKAN_BACKEND_SPEC.md §3-§6.1`).
//!
//! Avoiding `ash` here keeps Phase 1 dependency-light and the FFI surface
//! identical in shape to the CUDA/HIP backends. `ash`/`vulkano` adoption is
//! tracked for Phase 6 alongside the broader API surface (descriptor
//! lifetimes, fences, queues).
//!
//! ## Subset
//!
//! We implement only the path needed for: create instance → enumerate
//! physical device → create logical device + compute queue → create buffers
//! + memory + descriptor set → create compute pipeline from SPIR-V →
//! `vkCmdDispatch` → readback. Everything else (windowing, graphics,
//! sparse memory, ray-tracing) is intentionally out of scope.

#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ── Handles ────────────────────────────────────────────────────────────
pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandPool = u64;
pub type VkCommandBuffer = *mut c_void;
pub type VkBuffer_ = u64;
pub type VkDeviceMemory = u64;
pub type VkDescriptorPool = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkDescriptorSet = u64;
pub type VkPipelineLayout = u64;
pub type VkPipeline_ = u64;
pub type VkPipelineCache = u64;
pub type VkShaderModule = u64;
pub type VkFence = u64;

pub type VkResult = c_int;
pub type VkFlags = u32;
pub type VkBool32 = u32;
pub type VkDeviceSize = u64;

pub const VK_SUCCESS: VkResult = 0;
pub const VK_NULL_HANDLE: u64 = 0;
pub const VK_TRUE: VkBool32 = 1;
pub const VK_FALSE: VkBool32 = 0;

// Structure types we use.
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 35;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35 + 1;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
// Vulkan 1.1 / 1.2 feature struct sType values — used to chain feature
// enables through `VkDeviceCreateInfo.pNext`. Promoted-from-KHR features
// (shader_float16_int8, 16/8bit_storage, bfloat16, scalar block layout)
// are all reached through this single chain on a Vulkan 1.2 driver.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: u32 = 1000059000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: u32 = 49;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: u32 = 51;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: u32 = 53;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: u32 = 1000141000;
/// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` — gates the
/// `VK_KHR_cooperative_matrix` device feature (coopMatMulAdd fragment
/// ops). Chained into device creation only when the gated coopmat GEMM
/// path is active.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: u32 = 1000506000;
/// `VK_KHR_cooperative_matrix` device-extension name (NUL-terminated).
pub const VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME: &[u8] = b"VK_KHR_cooperative_matrix\0";
/// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` — Vulkan 1.3
/// core (promoted from VK_EXT_subgroup_size_control). Pinned to 32 in
/// the compute pipeline so `subgroupAdd` etc. reduce within a 32-lane
/// SIMD group (matches the metaltile kernels' Apple `simdgroup`
/// assumption). Without this AMD's wave32-default-but-driver-chosen
/// behaviour silently summed across the whole workgroup on some
/// kernels.
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: u32 =
    1000225001;

// Queue flags.
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;

// Buffer usage.
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;

pub const VK_SHARING_MODE_EXCLUSIVE: c_int = 0;

// Memory property flags.
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;

// Descriptor types.
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: c_int = 7;

// Shader stage.
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;

// Pipeline bind point.
pub const VK_PIPELINE_BIND_POINT_COMPUTE: c_int = 1;

// Command buffer level.
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: c_int = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;

// Command pool flags.
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;

// Pipeline stage flags / access flags (for memory barrier).
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x00001000;
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000100;
pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;
pub const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x00000800;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x00001000;

pub const VK_WHOLE_SIZE: u64 = !0u64;

// ── Structs ────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Default)]
pub struct VkApplicationInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct VkInstanceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pApplicationInfo: *const VkApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}

#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queueFlags: VkFlags,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: [u32; 3],
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const c_void,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharingMode: c_int,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memoryTypeBits: u32,
}

#[repr(C)]
pub struct VkMemoryType {
    pub propertyFlags: VkFlags,
    pub heapIndex: u32,
}

#[repr(C)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkFlags,
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [VkMemoryType; 32],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [VkMemoryHeap; 16],
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub allocationSize: VkDeviceSize,
    pub memoryTypeIndex: u32,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub codeSize: usize,
    pub pCode: *const u32,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptorType: c_int,
    pub descriptorCount: u32,
    pub stageFlags: VkFlags,
    pub pImmutableSamplers: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub bindingCount: u32,
    pub pBindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkPushConstantRange {
    pub stageFlags: VkFlags,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const VkPushConstantRange,
}

#[repr(C)]
pub struct VkSpecializationInfo {
    pub mapEntryCount: u32,
    pub pMapEntries: *const c_void,
    pub dataSize: usize,
    pub pData: *const c_void,
}

/// `VkPhysicalDeviceVulkan13Features` — Vulkan 1.3 promoted feature
/// chain. We need `subgroupSizeControl` so we can pin the subgroup
/// size to 32 at pipeline creation.
#[repr(C)]
pub struct VkPhysicalDeviceVulkan13Features {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub robustImageAccess: VkBool32,
    pub inlineUniformBlock: VkBool32,
    pub descriptorBindingInlineUniformBlockUpdateAfterBind: VkBool32,
    pub pipelineCreationCacheControl: VkBool32,
    pub privateData: VkBool32,
    pub shaderDemoteToHelperInvocation: VkBool32,
    pub shaderTerminateInvocation: VkBool32,
    pub subgroupSizeControl: VkBool32,
    pub computeFullSubgroups: VkBool32,
    pub synchronization2: VkBool32,
    pub textureCompressionASTC_HDR: VkBool32,
    pub shaderZeroInitializeWorkgroupMemory: VkBool32,
    pub dynamicRendering: VkBool32,
    pub shaderIntegerDotProduct: VkBool32,
    pub maintenance4: VkBool32,
}

#[repr(C)]
pub struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub requiredSubgroupSize: u32,
}

/// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`. Chained into
/// `vkCreateDevice` to enable `coopMatMulAdd`. `cooperativeMatrix` is the
/// only bit we set; `cooperativeMatrixRobustBufferAccess` stays off.
#[repr(C)]
pub struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub cooperativeMatrix: VkBool32,
    pub cooperativeMatrixRobustBufferAccess: VkBool32,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkFlags,
    pub module: VkShaderModule,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const VkSpecializationInfo,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub basePipelineHandle: VkPipeline_,
    pub basePipelineIndex: i32,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub typ: c_int,
    pub descriptorCount: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub maxSets: u32,
    pub poolSizeCount: u32,
    pub pPoolSizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub descriptorPool: VkDescriptorPool,
    pub descriptorSetCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer_,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub sType: u32,
    pub pNext: *const c_void,
    pub dstSet: VkDescriptorSet,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
    pub descriptorType: c_int,
    pub pImageInfo: *const c_void,
    pub pBufferInfo: *const VkDescriptorBufferInfo,
    pub pTexelBufferView: *const c_void,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub commandPool: VkCommandPool,
    pub level: c_int,
    pub commandBufferCount: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pInheritanceInfo: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const c_void,
    pub pWaitDstStageMask: *const u32,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const VkCommandBuffer,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const c_void,
}

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
}

/// `VkPhysicalDeviceVulkan11Features` — used here only for the 16-bit
/// storage bits. Layout must match `vulkan_core.h` exactly (sType + pNext
/// + every `VkBool32` in declared order, no padding).
#[repr(C)]
pub struct VkPhysicalDeviceVulkan11Features {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub storageBuffer16BitAccess: VkBool32,
    pub uniformAndStorageBuffer16BitAccess: VkBool32,
    pub storagePushConstant16: VkBool32,
    pub storageInputOutput16: VkBool32,
    pub multiview: VkBool32,
    pub multiviewGeometryShader: VkBool32,
    pub multiviewTessellationShader: VkBool32,
    pub variablePointersStorageBuffer: VkBool32,
    pub variablePointers: VkBool32,
    pub protectedMemory: VkBool32,
    pub samplerYcbcrConversion: VkBool32,
    pub shaderDrawParameters: VkBool32,
}

/// `VkPhysicalDeviceVulkan12Features` — Phase 3 uses `shaderFloat16`,
/// `shaderInt8`, `storageBuffer8BitAccess`, `scalarBlockLayout`. The full
/// 49-field layout must match `vulkan_core.h` byte-for-byte (we send the
/// struct into the driver, no per-field offset table).
#[repr(C)]
pub struct VkPhysicalDeviceVulkan12Features {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub samplerMirrorClampToEdge: VkBool32,
    pub drawIndirectCount: VkBool32,
    pub storageBuffer8BitAccess: VkBool32,
    pub uniformAndStorageBuffer8BitAccess: VkBool32,
    pub storagePushConstant8: VkBool32,
    pub shaderBufferInt64Atomics: VkBool32,
    pub shaderSharedInt64Atomics: VkBool32,
    pub shaderFloat16: VkBool32,
    pub shaderInt8: VkBool32,
    pub descriptorIndexing: VkBool32,
    pub shaderInputAttachmentArrayDynamicIndexing: VkBool32,
    pub shaderUniformTexelBufferArrayDynamicIndexing: VkBool32,
    pub shaderStorageTexelBufferArrayDynamicIndexing: VkBool32,
    pub shaderUniformBufferArrayNonUniformIndexing: VkBool32,
    pub shaderSampledImageArrayNonUniformIndexing: VkBool32,
    pub shaderStorageBufferArrayNonUniformIndexing: VkBool32,
    pub shaderStorageImageArrayNonUniformIndexing: VkBool32,
    pub shaderInputAttachmentArrayNonUniformIndexing: VkBool32,
    pub shaderUniformTexelBufferArrayNonUniformIndexing: VkBool32,
    pub shaderStorageTexelBufferArrayNonUniformIndexing: VkBool32,
    pub descriptorBindingUniformBufferUpdateAfterBind: VkBool32,
    pub descriptorBindingSampledImageUpdateAfterBind: VkBool32,
    pub descriptorBindingStorageImageUpdateAfterBind: VkBool32,
    pub descriptorBindingStorageBufferUpdateAfterBind: VkBool32,
    pub descriptorBindingUniformTexelBufferUpdateAfterBind: VkBool32,
    pub descriptorBindingStorageTexelBufferUpdateAfterBind: VkBool32,
    pub descriptorBindingUpdateUnusedWhilePending: VkBool32,
    pub descriptorBindingPartiallyBound: VkBool32,
    pub descriptorBindingVariableDescriptorCount: VkBool32,
    pub runtimeDescriptorArray: VkBool32,
    pub samplerFilterMinmax: VkBool32,
    pub scalarBlockLayout: VkBool32,
    pub imagelessFramebuffer: VkBool32,
    pub uniformBufferStandardLayout: VkBool32,
    pub shaderSubgroupExtendedTypes: VkBool32,
    pub separateDepthStencilLayouts: VkBool32,
    pub hostQueryReset: VkBool32,
    pub timelineSemaphore: VkBool32,
    pub bufferDeviceAddress: VkBool32,
    pub bufferDeviceAddressCaptureReplay: VkBool32,
    pub bufferDeviceAddressMultiDevice: VkBool32,
    pub vulkanMemoryModel: VkBool32,
    pub vulkanMemoryModelDeviceScope: VkBool32,
    pub vulkanMemoryModelAvailabilityVisibilityChains: VkBool32,
    pub shaderOutputViewportIndex: VkBool32,
    pub shaderOutputLayer: VkBool32,
    pub subgroupBroadcastDynamicId: VkBool32,
}

#[repr(C)]
pub struct VkMemoryBarrier {
    pub sType: u32,
    pub pNext: *const c_void,
    pub srcAccessMask: VkFlags,
    pub dstAccessMask: VkFlags,
}

#[repr(C)]
pub struct VkBufferCopy {
    pub srcOffset: VkDeviceSize,
    pub dstOffset: VkDeviceSize,
    pub size: VkDeviceSize,
}

// ── Vulkan FFI ─────────────────────────────────────────────────────────
// Windows: linker resolves against `vulkan-1.lib` -> loader picks
// `vulkan-1.dll` at runtime. Linux: `libvulkan.so`.
#[cfg(windows)]
#[link(name = "vulkan-1")]
unsafe extern "C" {
    pub fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut VkInstance,
    ) -> VkResult;
    pub fn vkDestroyInstance(instance: VkInstance, pAllocator: *const c_void);
    pub fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut VkPhysicalDevice,
    ) -> VkResult;
    pub fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: VkPhysicalDevice,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );
    pub fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: VkPhysicalDevice,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );
    pub fn vkCreateDevice(
        physicalDevice: VkPhysicalDevice,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut VkDevice,
    ) -> VkResult;
    pub fn vkDestroyDevice(device: VkDevice, pAllocator: *const c_void);
    pub fn vkGetDeviceQueue(
        device: VkDevice,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut VkQueue,
    );
    pub fn vkQueueSubmit(
        queue: VkQueue,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;
    pub fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
    pub fn vkDeviceWaitIdle(device: VkDevice) -> VkResult;

    pub fn vkCreateBuffer(
        device: VkDevice,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut VkBuffer_,
    ) -> VkResult;
    pub fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer_, pAllocator: *const c_void);
    pub fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer_,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    pub fn vkAllocateMemory(
        device: VkDevice,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut VkDeviceMemory,
    ) -> VkResult;
    pub fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, pAllocator: *const c_void);
    pub fn vkBindBufferMemory(
        device: VkDevice,
        buffer: VkBuffer_,
        memory: VkDeviceMemory,
        memoryOffset: VkDeviceSize,
    ) -> VkResult;
    pub fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        ppData: *mut *mut c_void,
    ) -> VkResult;
    pub fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);

    pub fn vkCreateShaderModule(
        device: VkDevice,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut VkShaderModule,
    ) -> VkResult;
    pub fn vkDestroyShaderModule(
        device: VkDevice,
        shaderModule: VkShaderModule,
        pAllocator: *const c_void,
    );

    pub fn vkCreateDescriptorSetLayout(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut VkDescriptorSetLayout,
    ) -> VkResult;
    pub fn vkDestroyDescriptorSetLayout(
        device: VkDevice,
        descriptorSetLayout: VkDescriptorSetLayout,
        pAllocator: *const c_void,
    );

    pub fn vkCreatePipelineLayout(
        device: VkDevice,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut VkPipelineLayout,
    ) -> VkResult;
    pub fn vkDestroyPipelineLayout(
        device: VkDevice,
        pipelineLayout: VkPipelineLayout,
        pAllocator: *const c_void,
    );

    pub fn vkCreateComputePipelines(
        device: VkDevice,
        pipelineCache: VkPipelineCache,
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut VkPipeline_,
    ) -> VkResult;
    pub fn vkDestroyPipeline(
        device: VkDevice,
        pipeline: VkPipeline_,
        pAllocator: *const c_void,
    );

    pub fn vkCreateDescriptorPool(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut VkDescriptorPool,
    ) -> VkResult;
    pub fn vkDestroyDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        pAllocator: *const c_void,
    );
    pub fn vkResetDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        flags: VkFlags,
    ) -> VkResult;
    pub fn vkAllocateDescriptorSets(
        device: VkDevice,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut VkDescriptorSet,
    ) -> VkResult;
    pub fn vkUpdateDescriptorSets(
        device: VkDevice,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );

    pub fn vkCreateCommandPool(
        device: VkDevice,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut VkCommandPool,
    ) -> VkResult;
    pub fn vkDestroyCommandPool(
        device: VkDevice,
        commandPool: VkCommandPool,
        pAllocator: *const c_void,
    );
    pub fn vkAllocateCommandBuffers(
        device: VkDevice,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut VkCommandBuffer,
    ) -> VkResult;
    pub fn vkBeginCommandBuffer(
        commandBuffer: VkCommandBuffer,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> VkResult;
    pub fn vkEndCommandBuffer(commandBuffer: VkCommandBuffer) -> VkResult;
    pub fn vkResetCommandBuffer(commandBuffer: VkCommandBuffer, flags: VkFlags) -> VkResult;
    pub fn vkFreeCommandBuffers(
        device: VkDevice,
        commandPool: VkCommandPool,
        commandBufferCount: u32,
        pCommandBuffers: *const VkCommandBuffer,
    );
    pub fn vkCmdBindPipeline(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: c_int,
        pipeline: VkPipeline_,
    );
    pub fn vkCmdBindDescriptorSets(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: c_int,
        layout: VkPipelineLayout,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const VkDescriptorSet,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );
    pub fn vkCmdPushConstants(
        commandBuffer: VkCommandBuffer,
        layout: VkPipelineLayout,
        stageFlags: VkFlags,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );
    pub fn vkCmdDispatch(
        commandBuffer: VkCommandBuffer,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );
    pub fn vkCmdCopyBuffer(
        commandBuffer: VkCommandBuffer,
        srcBuffer: VkBuffer_,
        dstBuffer: VkBuffer_,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );
    pub fn vkCmdPipelineBarrier(
        commandBuffer: VkCommandBuffer,
        srcStageMask: VkFlags,
        dstStageMask: VkFlags,
        dependencyFlags: VkFlags,
        memoryBarrierCount: u32,
        pMemoryBarriers: *const VkMemoryBarrier,
        bufferMemoryBarrierCount: u32,
        pBufferMemoryBarriers: *const c_void,
        imageMemoryBarrierCount: u32,
        pImageMemoryBarriers: *const c_void,
    );

    pub fn vkCreateFence(
        device: VkDevice,
        pCreateInfo: *const VkFenceCreateInfo,
        pAllocator: *const c_void,
        pFence: *mut VkFence,
    ) -> VkResult;
    pub fn vkDestroyFence(device: VkDevice, fence: VkFence, pAllocator: *const c_void);
    pub fn vkWaitForFences(
        device: VkDevice,
        fenceCount: u32,
        pFences: *const VkFence,
        waitAll: VkBool32,
        timeout: u64,
    ) -> VkResult;
}

#[cfg(not(windows))]
#[link(name = "vulkan")]
unsafe extern "C" {
    // Same signatures as above; duplicated under a different link name
    // because rustc's `#[link(name = ...)]` requires a single name per
    // extern block.
    pub fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut VkInstance,
    ) -> VkResult;
    pub fn vkDestroyInstance(instance: VkInstance, pAllocator: *const c_void);
    pub fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut VkPhysicalDevice,
    ) -> VkResult;
    pub fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: VkPhysicalDevice,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );
    pub fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: VkPhysicalDevice,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );
    pub fn vkCreateDevice(
        physicalDevice: VkPhysicalDevice,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut VkDevice,
    ) -> VkResult;
    pub fn vkDestroyDevice(device: VkDevice, pAllocator: *const c_void);
    pub fn vkGetDeviceQueue(
        device: VkDevice,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut VkQueue,
    );
    pub fn vkQueueSubmit(
        queue: VkQueue,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;
    pub fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
    pub fn vkDeviceWaitIdle(device: VkDevice) -> VkResult;

    pub fn vkCreateBuffer(
        device: VkDevice,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut VkBuffer_,
    ) -> VkResult;
    pub fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer_, pAllocator: *const c_void);
    pub fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer_,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    pub fn vkAllocateMemory(
        device: VkDevice,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut VkDeviceMemory,
    ) -> VkResult;
    pub fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, pAllocator: *const c_void);
    pub fn vkBindBufferMemory(
        device: VkDevice,
        buffer: VkBuffer_,
        memory: VkDeviceMemory,
        memoryOffset: VkDeviceSize,
    ) -> VkResult;
    pub fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        ppData: *mut *mut c_void,
    ) -> VkResult;
    pub fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);

    pub fn vkCreateShaderModule(
        device: VkDevice,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut VkShaderModule,
    ) -> VkResult;
    pub fn vkDestroyShaderModule(
        device: VkDevice,
        shaderModule: VkShaderModule,
        pAllocator: *const c_void,
    );

    pub fn vkCreateDescriptorSetLayout(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut VkDescriptorSetLayout,
    ) -> VkResult;
    pub fn vkDestroyDescriptorSetLayout(
        device: VkDevice,
        descriptorSetLayout: VkDescriptorSetLayout,
        pAllocator: *const c_void,
    );

    pub fn vkCreatePipelineLayout(
        device: VkDevice,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut VkPipelineLayout,
    ) -> VkResult;
    pub fn vkDestroyPipelineLayout(
        device: VkDevice,
        pipelineLayout: VkPipelineLayout,
        pAllocator: *const c_void,
    );

    pub fn vkCreateComputePipelines(
        device: VkDevice,
        pipelineCache: VkPipelineCache,
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut VkPipeline_,
    ) -> VkResult;
    pub fn vkDestroyPipeline(
        device: VkDevice,
        pipeline: VkPipeline_,
        pAllocator: *const c_void,
    );

    pub fn vkCreateDescriptorPool(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut VkDescriptorPool,
    ) -> VkResult;
    pub fn vkDestroyDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        pAllocator: *const c_void,
    );
    pub fn vkResetDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        flags: VkFlags,
    ) -> VkResult;
    pub fn vkAllocateDescriptorSets(
        device: VkDevice,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut VkDescriptorSet,
    ) -> VkResult;
    pub fn vkUpdateDescriptorSets(
        device: VkDevice,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );

    pub fn vkCreateCommandPool(
        device: VkDevice,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut VkCommandPool,
    ) -> VkResult;
    pub fn vkDestroyCommandPool(
        device: VkDevice,
        commandPool: VkCommandPool,
        pAllocator: *const c_void,
    );
    pub fn vkAllocateCommandBuffers(
        device: VkDevice,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut VkCommandBuffer,
    ) -> VkResult;
    pub fn vkBeginCommandBuffer(
        commandBuffer: VkCommandBuffer,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> VkResult;
    pub fn vkEndCommandBuffer(commandBuffer: VkCommandBuffer) -> VkResult;
    pub fn vkResetCommandBuffer(commandBuffer: VkCommandBuffer, flags: VkFlags) -> VkResult;
    pub fn vkFreeCommandBuffers(
        device: VkDevice,
        commandPool: VkCommandPool,
        commandBufferCount: u32,
        pCommandBuffers: *const VkCommandBuffer,
    );
    pub fn vkCmdBindPipeline(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: c_int,
        pipeline: VkPipeline_,
    );
    pub fn vkCmdBindDescriptorSets(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: c_int,
        layout: VkPipelineLayout,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const VkDescriptorSet,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );
    pub fn vkCmdPushConstants(
        commandBuffer: VkCommandBuffer,
        layout: VkPipelineLayout,
        stageFlags: VkFlags,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );
    pub fn vkCmdDispatch(
        commandBuffer: VkCommandBuffer,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );
    pub fn vkCmdCopyBuffer(
        commandBuffer: VkCommandBuffer,
        srcBuffer: VkBuffer_,
        dstBuffer: VkBuffer_,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );
    pub fn vkCmdPipelineBarrier(
        commandBuffer: VkCommandBuffer,
        srcStageMask: VkFlags,
        dstStageMask: VkFlags,
        dependencyFlags: VkFlags,
        memoryBarrierCount: u32,
        pMemoryBarriers: *const VkMemoryBarrier,
        bufferMemoryBarrierCount: u32,
        pBufferMemoryBarriers: *const c_void,
        imageMemoryBarrierCount: u32,
        pImageMemoryBarriers: *const c_void,
    );

    pub fn vkCreateFence(
        device: VkDevice,
        pCreateInfo: *const VkFenceCreateInfo,
        pAllocator: *const c_void,
        pFence: *mut VkFence,
    ) -> VkResult;
    pub fn vkDestroyFence(device: VkDevice, fence: VkFence, pAllocator: *const c_void);
    pub fn vkWaitForFences(
        device: VkDevice,
        fenceCount: u32,
        pFences: *const VkFence,
        waitAll: VkBool32,
        timeout: u64,
    ) -> VkResult;
}

// ── shaderc FFI ────────────────────────────────────────────────────────
// shaderc's C API (`shaderc_compile_into_spv`) gives us GLSL → SPIR-V at
// runtime. Linked statically on Windows via `shaderc_combined.lib`; the
// shared `libshaderc_shared` on Linux.
pub type shaderc_compiler_t = *mut c_void;
pub type shaderc_compile_options_t = *mut c_void;
pub type shaderc_compilation_result_t = *mut c_void;

pub type shaderc_shader_kind = c_int;
// shaderc_shader_kind enum (from shaderc/env.h): vertex=0, fragment=1,
// COMPUTE=2, geometry=3, tess_control=4, tess_eval=5. Getting this wrong
// silently picks a different stage — the GLSL `layout(local_size_x = N) in`
// then fails to validate because vertex/fragment don't accept it.
pub const shaderc_compute_shader: shaderc_shader_kind = 2;

pub type shaderc_compilation_status = c_int;
pub const shaderc_compilation_status_success: shaderc_compilation_status = 0;

pub type shaderc_target_env = c_int;
pub const shaderc_target_env_vulkan: shaderc_target_env = 0;

pub type shaderc_env_version = c_int;
// Maps to Vulkan 1.2 (0x402000). We use 1.2 features (subgroup ops with
// the size-control extension) but avoid 1.3-only conveniences for now.
pub const shaderc_env_version_vulkan_1_2: shaderc_env_version = (1 << 22) | (2 << 12);

#[cfg(windows)]
#[link(name = "shaderc_combined", kind = "static")]
unsafe extern "C" {
    pub fn shaderc_compiler_initialize() -> shaderc_compiler_t;
    pub fn shaderc_compiler_release(compiler: shaderc_compiler_t);
    pub fn shaderc_compile_options_initialize() -> shaderc_compile_options_t;
    pub fn shaderc_compile_options_release(options: shaderc_compile_options_t);
    pub fn shaderc_compile_options_set_target_env(
        options: shaderc_compile_options_t,
        target: shaderc_target_env,
        version: u32,
    );
    pub fn shaderc_compile_options_set_optimization_level(
        options: shaderc_compile_options_t,
        level: c_int,
    );
    pub fn shaderc_compile_into_spv(
        compiler: shaderc_compiler_t,
        source_text: *const c_char,
        source_text_size: usize,
        shader_kind: shaderc_shader_kind,
        input_file_name: *const c_char,
        entry_point_name: *const c_char,
        additional_options: shaderc_compile_options_t,
    ) -> shaderc_compilation_result_t;
    pub fn shaderc_result_release(result: shaderc_compilation_result_t);
    pub fn shaderc_result_get_compilation_status(
        result: shaderc_compilation_result_t,
    ) -> shaderc_compilation_status;
    pub fn shaderc_result_get_length(result: shaderc_compilation_result_t) -> usize;
    pub fn shaderc_result_get_bytes(result: shaderc_compilation_result_t) -> *const u8;
    pub fn shaderc_result_get_error_message(
        result: shaderc_compilation_result_t,
    ) -> *const c_char;
}

#[cfg(not(windows))]
#[link(name = "shaderc_shared")]
unsafe extern "C" {
    pub fn shaderc_compiler_initialize() -> shaderc_compiler_t;
    pub fn shaderc_compiler_release(compiler: shaderc_compiler_t);
    pub fn shaderc_compile_options_initialize() -> shaderc_compile_options_t;
    pub fn shaderc_compile_options_release(options: shaderc_compile_options_t);
    pub fn shaderc_compile_options_set_target_env(
        options: shaderc_compile_options_t,
        target: shaderc_target_env,
        version: u32,
    );
    pub fn shaderc_compile_options_set_optimization_level(
        options: shaderc_compile_options_t,
        level: c_int,
    );
    pub fn shaderc_compile_into_spv(
        compiler: shaderc_compiler_t,
        source_text: *const c_char,
        source_text_size: usize,
        shader_kind: shaderc_shader_kind,
        input_file_name: *const c_char,
        entry_point_name: *const c_char,
        additional_options: shaderc_compile_options_t,
    ) -> shaderc_compilation_result_t;
    pub fn shaderc_result_release(result: shaderc_compilation_result_t);
    pub fn shaderc_result_get_compilation_status(
        result: shaderc_compilation_result_t,
    ) -> shaderc_compilation_status;
    pub fn shaderc_result_get_length(result: shaderc_compilation_result_t) -> usize;
    pub fn shaderc_result_get_bytes(result: shaderc_compilation_result_t) -> *const u8;
    pub fn shaderc_result_get_error_message(
        result: shaderc_compilation_result_t,
    ) -> *const c_char;
}

// Silence unused param warnings on c_uint imports if any.
#[allow(dead_code)]
fn _force_use_uint(_x: c_uint) {}
