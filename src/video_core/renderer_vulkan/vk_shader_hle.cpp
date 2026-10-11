// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "core/memory.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;
static constexpr u64 COPY_KERNEL_HASH = 0x50784801;

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);
        buffer_cache.BumpMemEpoch();

        // Apply found buffer base.
        const auto vk_copies = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : vk_copies) {
            copy.srcOffset = copy.srcOffset - src_offset_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_offset_min + dst_buf_offset;
        }

        // Execute buffer copies.
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        runtime.CopyBuffer(src_buf, dst_buf, vk_copies);
        batch_start = batch_end;
    }

    return true;
}

// Kernel copying dst[t] = src[t % M] for t < min(N, cap): a buffer copy when M >= the count
static bool ExecuteCopyKernelHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    constexpr u32 CapUserData = 6;
    constexpr u64 ThreadsPerGroup = 64;
    if (!rasterizer.IsCopyKernelHleEnabled()) {
        return false;
    }

    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();
    auto* memory = Core::Memory::Instance();
    const auto reject = [](const char* reason) {
        LOG_DEBUG(Render_Vulkan, "Copy kernel HLE not applied: {}", reason);
        return false;
    };

    boost::container::small_vector<const Shader::BufferResource*, 4> guest;
    for (const auto& b : info.buffers) {
        if (!b.IsSpecial()) {
            guest.push_back(&b);
        }
    }

    if (guest.size() != 3) {
        return reject("buffer count");
    }

    if (guest[0]->is_written || guest[1]->is_written || !guest[2]->is_written) {
        return reject("written flags");
    }

    const auto params = guest[0]->GetSharp(info);
    const auto src = guest[1]->GetSharp(info);
    const auto dst = guest[2]->GetSharp(info);
    const auto plain32 = [](const Shader::BufferResource& desc, const AmdGpu::Buffer& b) {
        if (!desc.is_formatted) {
            return true;
        }
        return b.GetStride() == sizeof(u32) && b.swizzle_enable == 0 && b.add_tid_enable == 0 &&
               AmdGpu::DataFormat(b.data_format) == AmdGpu::DataFormat::Format32 &&
               b.GetNumberConversion() == AmdGpu::NumberConversion::None &&
               b.DstSelect().r == AmdGpu::CompSwizzle::Red;
    };

    if (guest[0]->is_formatted || !plain32(*guest[1], src) || !plain32(*guest[2], dst)) {
        return reject("formats");
    }

    struct Parameters {
        u32 count;     // N
        u32 src_count; // M
    } parameters{};

    const bool addresses_valid = params.GetSize() >= sizeof(parameters) &&
                                 memory->IsAllocatedAddress(params.base_address) &&
                                 memory->IsAllocatedAddress(src.base_address) &&
                                 memory->IsAllocatedAddress(dst.base_address);

    if (!addresses_valid) {
        return reject("addresses");
    }

    if (!buffer_cache.IsCpuCopyValid(params.base_address, sizeof(parameters))) {
        return reject("parameters pending on the GPU");
    }

    memory->CopySparseMemory(params.base_address, reinterpret_cast<u8*>(&parameters),
                             sizeof(parameters));
    const u64 cap = cs_program.user_data[CapUserData];
    const u64 threads = u64{cs_program.dim_x} * ThreadsPerGroup;
    const u64 count = std::min<u64>({parameters.count, cap, threads, dst.GetSize() / sizeof(u32)});
    if (count == 0) {
        return true;
    }

    const u64 size = count * sizeof(u32);
    if (parameters.src_count < count || src.GetSize() < size) {
        return reject("not a plain copy");
    }
    if (src.base_address == dst.base_address) {
        return true; // copies the buffer onto itself
    }
    if (src.base_address < dst.base_address + size && dst.base_address < src.base_address + size) {
        return reject("overlapping source and destination");
    }

    buffer_cache.BumpMemEpoch();
    const auto [src_buf, src_offset] = buffer_cache.ObtainBuffer(src.base_address, size, false);
    const auto [dst_buf, dst_offset] = buffer_cache.ObtainBuffer(dst.base_address, size, true);
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = size,
    };

    runtime.CopyBuffer(src_buf, dst_buf, std::span{&copy, 1});
    rasterizer.GetTextureCache().InvalidateMemoryFromGPU(dst.base_address, size);
    buffer_cache.CloseGpuWrites();
    buffer_cache.PredictCopy(src.base_address, dst.base_address, size);
    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    case COPY_KERNEL_HASH:
        return ExecuteCopyKernelHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
