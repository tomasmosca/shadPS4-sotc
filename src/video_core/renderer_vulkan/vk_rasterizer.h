// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);

    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    void DeferAfterGpu(Common::UniqueFunction<void>&& func);
    void FlushForLabels();

    void BumpMemEpoch() {
        buffer_cache.BumpMemEpoch();
    }

    [[nodiscard]] bool IsCopyKernelHleEnabled() const noexcept {
        return copy_kernel_hle;
    }

    void OnCpWait() {
        FlushForLabels();
        BumpMemEpoch();
    }
    bool WriteLodStats(VAddr address, u32 num_bytes);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    void RefreshImageDescriptorLayouts();

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);

    void ResetBindings(bool is_compute);

    bool CanReuseLastDraw(bool is_indexed, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

    struct ImageBindCacheEntry {
        std::array<u64, 4> tsharp{};
        u32 flags{};
        u64 generation = ~0ULL;
        VAddr range_addr{};
        u64 range_size{};
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        VideoCore::TextureCache::ImageDesc desc;
    };
    static constexpr size_t ImageBindCacheSize = 2048;
    struct DrawSignature {
        u64 vs_addr{};
        u64 ps_addr{};
        std::array<u32, 16> vs_user_data{};
        std::array<u32, 16> ps_user_data{};
        u16 vertex_sgpr_offset{};
        u16 instance_sgpr_offset{};
        bool is_indexed{};

        bool operator==(const DrawSignature&) const = default;
    };

    struct LastDraw {
        u64 regs_version = ~0ULL;
        u64 mem_epoch{};
        u64 bind_serial{};
        u64 tick{};
        bool had_writes{true};
    };

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    const bool reuse_draw_state;
    const bool copy_kernel_hle;
    const u32 flush_every_draws;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;

    static constexpr u32 MaxImageDescriptors =
        Shader::NUM_IMAGES + Shader::NUM_IMAGE_ARRAY_DESCRIPTORS;
    boost::container::static_vector<vk::DescriptorImageInfo, MaxImageDescriptors> image_infos;

    struct ImageDescriptorRef {
        u32 info_index{};
        const VideoCore::Image::BackingImage* backing{};
        u32 subres_idx{};
    };
    boost::container::static_vector<ImageDescriptorRef, MaxImageDescriptors> image_descriptor_refs;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, MaxImageDescriptors> bound_images;
    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;

        bool operator==(const BoundBuffer&) const = default;
    };
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;
    boost::container::small_vector<BoundBuffer, 8> ff_reads;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, MaxImageDescriptors> image_bindings;
    bool attachment_feedback_loop{};
    bool needs_barrier{};

    u64 bind_serial{};
    bool bind_had_writes{};
    std::vector<ImageBindCacheEntry> image_bind_cache;

    DrawSignature last_draw_signature{};
    LastDraw last_draw;
    const GraphicsPipeline* last_draw_pipeline{};
    RenderState last_draw_state{};
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> last_draw_buffers;
    boost::container::small_vector<BoundBuffer, 8> last_draw_ff_reads;

    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> last_access_buffers;
    u64 last_access_epoch = ~0ULL;
    bool last_access_compute{};

    u32 calls_since_flush{};
    u64 label_tick{};
    std::chrono::steady_clock::time_point label_since{};

    std::array<u32, 256> lod_stats_uses{};
    const bool lod_stats_enabled;
};

} // namespace Vulkan
