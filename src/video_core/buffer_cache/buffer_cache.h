// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is mapped for GPU access.
    [[nodiscard]] bool IsRegionGpuMapped(VAddr addr, size_t size) const;

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// True when no byte of the range holds pending GPU data (byte precise).
    [[nodiscard]] bool IsCpuCopyValid(VAddr addr, u64 size) const {
        return !gpu_modified_ranges.Intersects(addr, size);
    }

    /// IsCpuCopyValid for the many small reads of the SRT walkers, cached per page.
    [[nodiscard]] bool IsCpuCopyValidForWalker(VAddr addr, u64 size);

    /// Bumped by anything that may change memory the GPU or the SRT walkers read.
    [[nodiscard]] u64 MemEpoch() const noexcept {
        return mem_epoch;
    }

    void BumpMemEpoch() noexcept {
        ++mem_epoch;
    }

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Called once a draw or dispatch is recorded: its GPU writes belong to the current batch.
    void CloseGpuWrites();

    /// Writes the result of a GPU copy now when its source bytes are known on the CPU.
    bool PredictCopy(VAddr src, VAddr dst, u64 size);

    /// Keeps the arena memory of a range read by DMA shaders resident.
    void PinForDma(VAddr device_addr, u64 size);

    /// Over the device memory budget, frees the arena memory unused for a while.
    /// Called on submit, before SubmitPendingArenaBinds.
    void TrimArenaMemory();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

    /// Copies what the submitted batch wrote inside recently read back windows to the eager ring.
    void RecordEagerCopies();

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    /// Bytes of a GPU-written range held in eager_ring, or in eager_cpu_ring (PredictCopy).
    struct EagerEntry {
        VAddr end{};
        u64 ring_offset{};
        u64 tick{}; ///< batch that copied it
        bool cpu{};
    };

    /// Space of the ring used by a copy.
    struct EagerAlloc {
        u64 ring_begin{};
        u64 ring_end{};
        VAddr addr{};
        u64 size{};
    };

    struct HotWindow {
        u64 size{};
        u64 last_frame{};
    };

    /// An arena range synchronized for GPU reads, valid while its region has no new CPU write.
    struct SyncedRange {
        const Buffer* arena{};
        VAddr start{};
        VAddr end{};
        u64 generation{};

        bool Covers(VAddr addr, u64 size, u64 current_generation) const {
            return arena && addr >= start && addr + size <= end && generation == current_generation;
        }
    };

    /// A stream buffer upload, reused by an identical one in the same batch and memory epoch.
    struct StreamDedupEntry {
        VAddr addr{};
        u64 size{};
        u64 tick{};
        u64 epoch{};
        u64 wrap{};
        u64 offset{};
    };

    /// 16-byte blocks of a page that hold GPU-written bytes.
    static constexpr u64 WalkerBlockSize = 16;
    struct WalkerPageBlocks {
        u64 page = ~0ULL;
        u64 version = ~0ULL;
        std::bitset<BYTES_PER_PAGE / WalkerBlockSize> dirty{};
    };

    using ShadowPage = std::array<u8, BYTES_PER_PAGE>;
    static constexpr u32 StreamDedupBits = 12;

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    /// Retires the arena memory unused for min_idle_frames, oldest first, until bytes are retired.
    u64 EvictIdleArenaMemory(u64 bytes, u64 min_idle_frames);

    /// Unbinds and frees the retired arena memory whose last readers are done.
    void ReleaseRetiredArenaMemory();

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    // Eager readbacks: GPU writes in recently read back windows are copied to a host ring
    void EnsureEagerRing();
    u64 AllocEager(VAddr addr, u64 size);
    void EraseEager(VAddr addr, u64 size);
    void EraseEagerRing(const EagerAlloc& alloc);
    bool EagerCovers(VAddr addr, u64 size, u64& max_tick) const;
    void CopyFromEager(VAddr addr, u64 size, u8* out);
    bool ReadEagerComplete(VAddr addr, u64 size, u8* out);
    void WaitEager(u64 tick);
    void LearnEagerWindow(VAddr addr, u64 size);

    // Snapshots of pages becoming GPU written: downloads write back only the words the GPU changed
    void SnapshotWrittenRange(VAddr device_addr, u64 size);
    void ReleaseWriteShadows(VAddr addr, u64 size);

    void AddGpuModified(VAddr addr, u64 size);
    void SubtractGpuModified(VAddr addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;
    const bool eager_readbacks;
    const bool readbacks_ahead;
    const bool stream_upload_dedup;
    const bool snapshot_writes;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    FlatRangeSet gpu_modified_ranges;
    u64 gpu_ranges_version{}; ///< bumped by every change of gpu_modified_ranges
    u64 mem_epoch{};
    std::array<WalkerPageBlocks, 64> walker_page_blocks{};
    std::array<StreamDedupEntry, 1u << StreamDedupBits> stream_dedup{};
    std::array<SyncedRange, 8> synced_ranges{};
    u32 synced_next{};
    std::unordered_map<u64, std::unique_ptr<ShadowPage>> write_shadows; ///< by page
    std::vector<std::unique_ptr<ShadowPage>> free_shadows;

    std::map<VAddr, EagerEntry> eager_entries; ///< non-overlapping, by start address
    std::deque<EagerAlloc> eager_allocs;       ///< oldest first
    std::unique_ptr<Buffer> eager_ring;
    std::unique_ptr<Buffer> eager_marker; ///< tick of the last batch whose eager copies are done
    std::unique_ptr<u8[]> eager_cpu_ring; ///< same offsets as eager_ring
    u64 eager_ring_head{};
    bool eager_frozen{};                                    ///< a readback is using eager entries
    RangeSet eager_hot;                                     ///< windows read back recently
    std::unordered_map<VAddr, HotWindow> eager_hot_windows; ///< by window start
    u64 eager_hot_sweep_frame{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    /// GPU writes of the draw being bound and of the batch being recorded (tick_writes_tick)
    std::vector<std::pair<VAddr, u64>> open_writes;
    RangeSet tick_writes;
    u64 tick_writes_tick = ~0ULL;
    Vulkan::Semaphore memory_semaphore;

    /// Device memory made resident by one EnsureResident call
    struct ArenaAllocation {
        vk::DeviceMemory memory;
        u64 size{};
        u64 last_frame{};
        boost::container::small_vector<Interval, 2> blocks;
        bool dma{}; ///< read by DMA shaders, whose reads do not mark it as used
    };
    std::list<ArenaAllocation> arena_allocations;
    u64 arena_trim_frame{};

    /// Evicted arena memory, still bound for the commands recorded up to tick
    struct RetiredArenaMemory {
        vk::DeviceMemory memory;
        u64 tick{};
        boost::container::small_vector<Interval, 2> blocks;
    };
    std::vector<RetiredArenaMemory> retired_arena_memory;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        ArenaAllocation* allocation;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start), allocation};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
