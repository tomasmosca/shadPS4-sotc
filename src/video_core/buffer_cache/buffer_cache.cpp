// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "common/arch.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/ir/passes/srt_walker_cache.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef ARCH_X86_64
#include <immintrin.h>
#endif
#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

/// Pages of the write snapshots and of the SRT walker caches (BYTES_PER_PAGE)
static constexpr u64 PageBits = std::countr_zero(BYTES_PER_PAGE);
static constexpr u64 EagerRingSize = 64_MB;
static constexpr u64 EagerAlignment = 64;
static constexpr u64 MaxEagerBatchSize = EagerRingSize / 4;
static constexpr u64 MaxPredictedCopySize = 64_KB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      eager_readbacks{EmulatorSettings.IsEagerReadbacks()},
      readbacks_ahead{EmulatorSettings.IsReadbacksAhead()},
      stream_upload_dedup{EmulatorSettings.IsStreamUploadDedup()},
      snapshot_writes{EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    const auto* fault_buffer = fault_manager->GetFaultBuffer();
    runtime.FillBuffer(fault_buffer, 0u, fault_buffer->SizeBytes(), 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const auto flush_request = [this, device_addr, size, is_write] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

/// Copies to cpu the words of gpu that differ from shadow. n is a multiple of the word size.
static void MergeChangedWords(u8* cpu, const u8* gpu, const u8* shadow, u64 n) {
    constexpr u64 Word = sizeof(u32);
    u64 i = 0;
#ifdef ARCH_X86_64
    constexpr u64 Vector = sizeof(__m256i);
    constexpr int AllWordsEqual = (1 << (Vector / Word)) - 1;
    for (; i + Vector <= n; i += Vector) {
        const __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gpu + i));
        const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shadow + i));
        const int equal = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(g, s)));
        if (equal == AllWordsEqual) {
            continue;
        }

        for (u64 k = 0; k < Vector / Word; ++k) {
            if (!(equal & (1 << k))) {
                std::memcpy(cpu + i + k * Word, gpu + i + k * Word, Word);
            }
        }
    }
#endif
    for (; i < n; i += Word) {
        if (std::memcmp(gpu + i, shadow + i, Word) != 0) {
            std::memcpy(cpu + i, gpu + i, Word);
        }
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    const VAddr arena_base = arena->cpu_addr;
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 address, u64 range_size) {
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = start - arena_base,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                total_size_bytes += (new_size + align - 1) & ~(align - 1ULL);
            };
            gpu_modified_ranges.ForEachInRange(address, range_size, add_download);
            SubtractGpuModified(address, range_size);
        });
    if (total_size_bytes == 0) {
        ReleaseWriteShadows(device_addr, size);
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    // Copies whose bytes all have a valid eager copy are taken from the ring
    boost::container::small_vector<vk::BufferCopy, 16> eager_copies;
    u64 eager_tick = 0;

    if (!eager_entries.empty()) {
        auto out = copies.begin();
        for (const auto& copy : copies) {
            u64 tick = 0;
            if (EagerCovers(arena_base + copy.srcOffset, copy.size, tick)) {
                eager_copies.push_back(copy);
                eager_tick = std::max(eager_tick, tick);
            } else {
                *out++ = copy;
            }
        }

        copies.erase(out, copies.end());
        if (!eager_copies.empty()) {
            eager_frozen = true;
        }
    }

    // Bytes the recorded batch does not write are copied ahead of it, unless sparse binds pend
    const auto written_by_batch = [&] {
        const bool tick_valid = tick_writes_tick == scheduler.CurrentTick();
        for (const auto& copy : copies) {
            const VAddr lo = arena_base + copy.srcOffset;
            for (const auto& [write_addr, write_size] : open_writes) {
                if (lo < write_addr + write_size && write_addr < lo + copy.size) {
                    return true;
                }
            }

            if (tick_valid && tick_writes.Intersects(lo, copy.size)) {
                return true;
            }
        }
        return false;
    };

    if (!copies.empty()) {
        if (readbacks_ahead && pending_binds.empty() && !written_by_batch()) {
            const auto cmdbuf = scheduler.BeginAhead();
            cmdbuf.copyBuffer(arena->Handle(), download.buffer->Handle(), copies);
            scheduler.SubmitAheadAndWait();
        } else {
            runtime.CopyBuffer(arena, download.buffer, copies);
            scheduler.Finish();
        }
    }

    download.buffer->Invalidate(download.offset, download.size);
    if (!eager_copies.empty()) {
        WaitEager(eager_tick);
        boost::container::small_vector<vk::BufferCopy, 4> lost;

        for (const auto& copy : eager_copies) {
            const VAddr lo = arena_base + copy.srcOffset;
            u8* dst = download.mapped + (copy.dstOffset - download.offset);
            u64 unused_tick = 0;

            if (EagerCovers(lo, copy.size, unused_tick)) {
                CopyFromEager(lo, copy.size, dst);
            } else {
                lost.push_back(copy);
            }
        }

        if (!lost.empty()) {
            static std::once_flag logged;
            std::call_once(logged, [&] {
                LOG_ERROR(Render_Vulkan,
                          "Readback: {} copies lost their eager entries, read from the GPU",
                          lost.size());
            });

            runtime.CopyBuffer(arena, download.buffer, lost);
            scheduler.Finish();
            for (const auto& copy : lost) {
                download.buffer->Invalidate(copy.dstOffset, copy.size);
            }
        }

        copies.insert(copies.end(), eager_copies.begin(), eager_copies.end());
        eager_frozen = false;
    }
    for (const auto& copy : copies) {
        const VAddr dst = arena_base + copy.srcOffset;
        const u8* gpu_data = download.mapped + (copy.dstOffset - download.offset);
        const bool word_aligned = (dst | copy.size) % sizeof(u32) == 0;
        bool have_shadows = word_aligned && !write_shadows.empty();
        for (u64 page = dst >> PageBits; have_shadows && page <= (dst + copy.size - 1) >> PageBits;
             ++page) {
            have_shadows = write_shadows.contains(page);
        }

        if (!have_shadows) {
            memory->TryWriteBacking(std::bit_cast<u8*>(dst), gpu_data, copy.size);
            continue;
        }

        // Only the words the GPU changed: when the CPU changed one too, the GPU result wins
        const auto& address_space = memory->GetAddressSpace();
        for (u64 offset = 0; offset < copy.size;) {
            const VAddr addr = dst + offset;
            const u64 page_offset = addr & (BYTES_PER_PAGE - 1);
            const u64 length = std::min(copy.size - offset, BYTES_PER_PAGE - page_offset);
            const u8* shadow = write_shadows.at(addr >> PageBits)->data() + page_offset;

            if (u8* host = address_space.BackingAddress(addr)) {
                MergeChangedWords(host, gpu_data + offset, shadow, length);
            }
            offset += length;
        }
    }

    // The whole window is CPU owned again: a range kept without snapshots would overwrite CPU data
    ReleaseWriteShadows(device_addr, size);
    SubtractGpuModified(device_addr, size);
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
    EraseEager(device_addr, size);
    LearnEagerWindow(device_addr, size);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        StreamDedupEntry* slot{};
        u64 tick{};
        u64 epoch{};

        if (stream_upload_dedup) {
            tick = scheduler.CurrentTick();
            epoch = mem_epoch;
            constexpr u64 FibonacciHash = 0x9E3779B97F4A7C15ULL;
            const u64 key = device_addr ^ (u64{size} << 48);
            slot = &stream_dedup[(key * FibonacciHash) >> (64 - StreamDedupBits)];

            if (slot->addr == device_addr && slot->size == size && slot->tick == tick &&
                slot->epoch == epoch && slot->wrap == stream_buffer.WrapCount()) {
                return {&stream_buffer, slot->offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        if (slot) {
            *slot = {device_addr, size, tick, epoch, stream_buffer.WrapCount(), offset};
        }
        return {&stream_buffer, offset};
    }
    // Read-only ranges synchronized since the last CPU write in their region are used as is
    const bool cacheable =
        !is_written && !is_texel_buffer &&
        (device_addr >> HIGHER_PAGE_BITS) == ((device_addr + size - 1) >> HIGHER_PAGE_BITS);

    if (cacheable) {
        const u64 generation = memory_tracker->CpuGeneration(device_addr);
        if (generation != 0) {
            for (const auto& range : synced_ranges) {
                if (range.Covers(device_addr, size, generation)) {
                    return {range.arena, range.arena->Offset(device_addr)};
                }
            }
        }
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);

    if (cacheable) {
        // Synchronize the whole resident blocks (within the region) and remember them.
        const VAddr region_start = Common::AlignDown(device_addr, HIGHER_PAGE_SIZE);
        const VAddr start =
            std::max<VAddr>({first_block << block_shift, region_start, arena->cpu_addr});
        const VAddr end =
            std::min<VAddr>({(last_block + 1) << block_shift, region_start + HIGHER_PAGE_SIZE,
                             arena->cpu_addr + arena->size_bytes});

        const u64 generation = memory_tracker->CpuGeneration(device_addr);
        SynchronizeMemory(arena, start, static_cast<u32>(end - start), false, false);

        if (generation != 0) {
            synced_ranges[synced_next++ % synced_ranges.size()] = {arena, start, end, generation};
        }
        return {arena, arena->Offset(device_addr)};
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_texel_buffer && !is_written) {
        SynchronizeMemoryFromImage(arena, device_addr, size);
    }

    // A range already GPU owned has its snapshots (buffers the GPU rewrites every frame)
    const bool gpu_owned = is_written && gpu_modified_ranges.Contains(device_addr, size);
    if (is_written && !gpu_owned) {
        if (snapshot_writes) {
            SnapshotWrittenRange(device_addr, size);
        }
        AddGpuModified(device_addr, size);

        constexpr u64 MaxPagesToDrop = 16;
        const u64 first_page = device_addr >> PageBits;
        const u64 last_page = (device_addr + size - 1) >> PageBits;
        if (last_page - first_page < MaxPagesToDrop) {
            for (u64 page = first_page; page <= last_page; ++page) {
                Shader::g_walker_page_cache.Drop(page);
            }
        } else {
            Shader::g_walker_page_cache.DropAll();
        }
    }
    if (is_written) {
        open_writes.emplace_back(device_addr, size);
        EraseEager(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

bool BufferCache::IsCpuCopyValidForWalker(VAddr addr, u64 size) {
    const u64 page = addr >> PageBits;

    if (size == 0 || ((addr + size - 1) >> PageBits) != page) {
        return IsCpuCopyValid(addr, size);
    }

    auto& blocks = walker_page_blocks[page % walker_page_blocks.size()];
    const VAddr page_base = page << PageBits;
    if (blocks.page != page || blocks.version != gpu_ranges_version) {
        blocks.page = page;
        blocks.version = gpu_ranges_version;
        blocks.dirty.reset();
        gpu_modified_ranges.ForEachInRange(page_base, BYTES_PER_PAGE, [&](VAddr lo, VAddr hi) {
            for (u64 b = (lo - page_base) / WalkerBlockSize;
                 b <= (hi - 1 - page_base) / WalkerBlockSize; ++b) {
                blocks.dirty.set(b);
            }
        });
    }

    const u64 first = (addr - page_base) / WalkerBlockSize;
    const u64 last = (addr + size - 1 - page_base) / WalkerBlockSize;
    for (u64 b = first; b <= last; ++b) {
        if (blocks.dirty.test(b)) {
            return IsCpuCopyValid(addr, size); // a GPU-written block: the exact check
        }
    }

    return true;
}

void BufferCache::CloseGpuWrites() {
    if (open_writes.empty()) {
        return;
    }

    const u64 tick = scheduler.CurrentTick();
    if (tick_writes_tick != tick) {
        tick_writes.Clear();
        tick_writes_tick = tick;
    }

    for (const auto& [addr, size] : open_writes) {
        tick_writes.Add(addr, size);
    }
    open_writes.clear();
}

void BufferCache::ReleaseWriteShadows(VAddr addr, u64 size) {
    constexpr size_t MaxFreeShadows = 16384;
    for (u64 page = addr >> PageBits; page <= (addr + size - 1) >> PageBits; ++page) {
        if (const auto it = write_shadows.find(page); it != write_shadows.end()) {
            if (free_shadows.size() < MaxFreeShadows) {
                free_shadows.push_back(std::move(it->second));
            }
            write_shadows.erase(it);
        }
    }
}

void BufferCache::AddGpuModified(VAddr addr, u64 size) {
    gpu_modified_ranges.Add(addr, size);
    ++gpu_ranges_version;
}

void BufferCache::SubtractGpuModified(VAddr addr, u64 size) {
    gpu_modified_ranges.Subtract(addr, size);
    ++gpu_ranges_version;
}

void BufferCache::SnapshotWrittenRange(VAddr device_addr, u64 size) {
    if (size == 0) {
        return;
    }

    const u64 first_page = device_addr >> PageBits;
    const u64 last_page = (device_addr + size - 1) >> PageBits;
    for (u64 page = first_page; page <= last_page; ++page) {
        auto& shadow = write_shadows[page];
        if (shadow) {
            continue;
        }

        if (free_shadows.empty()) {
            shadow = std::make_unique<ShadowPage>();
        } else {
            shadow = std::move(free_shadows.back());
            free_shadows.pop_back();
        }
        memory->CopySparseMemory(page << PageBits, shadow->data(), shadow->size());
    }
}

/// Whether the batch numbered tick wrote marker (wrapping 32-bit tick).
static bool MarkerReached(u32 marker, u64 tick) {
    return static_cast<s32>(marker - static_cast<u32>(tick)) >= 0;
}

void BufferCache::EnsureEagerRing() {
    if (eager_ring) {
        return;
    }
    eager_ring = std::make_unique<Buffer>(instance, 0, EagerRingSize, MemoryType::HostCached,
                                          "Eager readback ring");
    eager_marker = std::make_unique<Buffer>(instance, 0, MarkerBufferSize, MemoryType::HostCached,
                                            "Eager readback marker");
    std::memset(eager_marker->mapped_data.data(), 0, MarkerBufferSize);
    eager_cpu_ring = std::make_unique_for_overwrite<u8[]>(EagerRingSize);
}

u64 BufferCache::AllocEager(VAddr addr, u64 size) {
    EnsureEagerRing();
    const u64 n = Common::AlignUp(size, EagerAlignment);

    if (eager_ring_head + n > EagerRingSize) {
        // The tail left unused belongs to the previous round: forget what is there.
        while (!eager_allocs.empty() && eager_allocs.front().ring_begin >= eager_ring_head) {
            EraseEagerRing(eager_allocs.front());
            eager_allocs.pop_front();
        }
        eager_ring_head = 0;
    }

    const u64 offset = eager_ring_head;
    eager_ring_head += n;

    while (!eager_allocs.empty() && eager_allocs.front().ring_begin < offset + n &&
           eager_allocs.front().ring_end > offset) {
        EraseEagerRing(eager_allocs.front());
        eager_allocs.pop_front();
    }

    eager_allocs.push_back({offset, offset + n, addr, size});
    return offset;
}

bool BufferCache::ReadEagerComplete(VAddr addr, u64 size, u8* out) {
    u64 tick = 0;
    if (!EagerCovers(addr, size, tick)) {
        return false;
    }

    const u32 marker = *reinterpret_cast<const volatile u32*>(eager_marker->mapped_data.data());
    if (tick != 0 && !MarkerReached(marker, tick)) {
        return false;
    }

    CopyFromEager(addr, size, out);
    return true;
}

void BufferCache::CopyFromEager(VAddr addr, u64 size, u8* out) {
    const VAddr end = addr + size;
    auto it = std::prev(eager_entries.upper_bound(addr));

    for (VAddr a = addr; a < end; ++it) {
        const VAddr piece_end = std::min(end, it->second.end);
        const u64 ring_offset = it->second.ring_offset + (a - it->first);
        const u8* ring_data = eager_cpu_ring.get();

        if (!it->second.cpu) {
            eager_ring->Invalidate(ring_offset, piece_end - a);
            ring_data = eager_ring->mapped_data.data();
        }

        std::memcpy(out + (a - addr), ring_data + ring_offset, piece_end - a);
        a = piece_end;
    }
}

// Without the memory lock: a guest thread unmapping memory may hold it while waiting for us
static bool WriteBackingNoLock(Core::MemoryManager* memory, VAddr addr, const u8* data, u64 size) {
    constexpr u64 PageSize = Core::AddressSpace::BackingPageSize;
    const auto& address_space = memory->GetAddressSpace();

    while (size != 0) {
        // A partial write is harmless: the caller keeps the copy in the eager ring instead
        u8* const backing = address_space.BackingAddress(addr);
        if (!backing) {
            return false;
        }

        const u64 length = std::min<u64>(PageSize - (addr & (PageSize - 1)), size);
        std::memcpy(backing, data, length);
        addr += length;
        data += length;
        size -= length;
    }
    return true;
}

bool BufferCache::PredictCopy(VAddr src, VAddr dst, u64 size) {
    if (!eager_readbacks || size == 0 || size > MaxPredictedCopySize) {
        return false;
    }

    std::array<u8, MaxPredictedCopySize> data;
    if (IsCpuCopyValid(src, size)) {
        memory->CopySparseMemory(src, data.data(), size);
    } else if (!ReadEagerComplete(src, size, data.data())) {
        return false;
    }

    // The GPU copy writes these same bytes, so dst has no pending GPU data unless they overlap
    const bool overlapping = src < dst + size && dst < src + size;
    if (!overlapping && WriteBackingNoLock(memory, dst, data.data(), size)) {
        if (gpu_modified_ranges.Intersects(dst, size)) {
            SubtractGpuModified(dst, size);
        }
        EraseEager(dst, size);
        return true;
    }

    const u64 offset = AllocEager(dst, size);
    std::memcpy(eager_cpu_ring.get() + offset, data.data(), size);

    EraseEager(dst, size);
    eager_entries.emplace(dst, EagerEntry{dst + size, offset, 0, true});
    return true;
}

void BufferCache::LearnEagerWindow(VAddr addr, u64 size) {
    if (!eager_readbacks || size == 0) {
        return;
    }

    constexpr u64 MaxAgeFrames = 240;
    constexpr u64 SweepIntervalFrames = 64;
    const u64 frame = g_present_frame.load(std::memory_order_relaxed);
    auto [it, inserted] = eager_hot_windows.try_emplace(addr, HotWindow{size, frame});

    it->second.last_frame = frame;
    if (inserted) {
        eager_hot.Add(addr, size);
    }

    if (frame - eager_hot_sweep_frame < SweepIntervalFrames) {
        return;
    }

    eager_hot_sweep_frame = frame;
    for (auto window = eager_hot_windows.begin(); window != eager_hot_windows.end();) {
        if (frame - window->second.last_frame > MaxAgeFrames) {
            eager_hot.Subtract(window->first, window->second.size);
            window = eager_hot_windows.erase(window);
        } else {
            ++window;
        }
    }
}

void BufferCache::EraseEager(VAddr addr, u64 size) {
    if (eager_entries.empty()) {
        return;
    }

    const VAddr end = addr + size;
    auto it = eager_entries.upper_bound(addr);
    if (it != eager_entries.begin() && std::prev(it)->second.end > addr) {
        --it;
    }

    while (it != eager_entries.end() && it->first < end) {
        const VAddr start = it->first;
        const EagerEntry entry = it->second;
        it = eager_entries.erase(it);
        if (start < addr) {
            eager_entries.emplace(start,
                                  EagerEntry{addr, entry.ring_offset, entry.tick, entry.cpu});
        }
        if (entry.end > end) {
            eager_entries.emplace(end, EagerEntry{entry.end, entry.ring_offset + (end - start),
                                                  entry.tick, entry.cpu});
        }
    }
}

void BufferCache::EraseEagerRing(const EagerAlloc& alloc) {
    // Drop what is left of the entry copied into this part of the ring.
    const VAddr end = alloc.addr + alloc.size;
    auto it = eager_entries.upper_bound(alloc.addr);
    if (it != eager_entries.begin() && std::prev(it)->second.end > alloc.addr) {
        --it;
    }

    while (it != eager_entries.end() && it->first < end) {
        if (it->second.ring_offset >= alloc.ring_begin && it->second.ring_offset < alloc.ring_end) {
            it = eager_entries.erase(it);
        } else {
            ++it;
        }
    }
}

bool BufferCache::EagerCovers(VAddr addr, u64 size, u64& max_tick) const {
    const VAddr end = addr + size;
    auto it = eager_entries.upper_bound(addr);
    if (it == eager_entries.begin()) {
        return false;
    }

    --it;
    VAddr covered = addr;
    while (covered < end) {
        if (it == eager_entries.end() || it->first > covered || it->second.end <= covered) {
            return false;
        }
        max_tick = std::max(max_tick, it->second.tick);
        covered = it->second.end;
        ++it;
    }
    return true;
}

void BufferCache::WaitEager(u64 tick) {
    if (tick == 0) {
        return;
    }
    const auto reached = [tick](u32 marker) { return MarkerReached(marker, tick); };
    scheduler.WaitSubmitted(tick);

    if (!PollMarker(*eager_marker, reached)) {
        scheduler.Wait(tick);
    }
}

void BufferCache::RecordEagerCopies() {
    if (!eager_readbacks || eager_frozen) {
        return;
    }

    // Called while submitting: the batch tick was just taken.
    const u64 tick = scheduler.CurrentTick() - 1;
    if (tick_writes_tick != tick || eager_hot_windows.empty()) {
        return;
    }

    struct Range {
        const Buffer* arena{};
        VAddr addr{};
        u64 size{};
    };

    boost::container::small_vector<Range, 32> ranges;
    u64 total = 0;
    tick_writes.ForEach([&](VAddr a, VAddr b) {
        eager_hot.ForEachInRange(a, b - a, [&](VAddr lo, VAddr hi) {
            const u64 first_block = lo >> block_shift;
            const u64 last_block = (hi - 1) >> block_shift;
            ranges.push_back({GetArena(first_block, last_block), lo, hi - lo});
            total += Common::AlignUp(hi - lo, EagerAlignment);
        });
    });

    if (ranges.empty()) {
        return;
    }

    if (total > MaxEagerBatchSize) {
        return;
    }

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &pre_barrier,
    });

    // One copy command per arena (normally one) with all its regions.
    boost::container::small_vector<std::pair<const Buffer*, std::vector<vk::BufferCopy>>, 2>
        regions;
    for (const auto& range : ranges) {
        const u64 offset = AllocEager(range.addr, range.size);
        size_t group = 0;

        while (group < regions.size() && regions[group].first != range.arena) {
            ++group;
        }

        if (group == regions.size()) {
            regions.emplace_back(range.arena, std::vector<vk::BufferCopy>{});
        }

        regions[group].second.push_back(vk::BufferCopy{
            .srcOffset = range.addr - range.arena->cpu_addr,
            .dstOffset = offset,
            .size = range.size,
        });

        EraseEager(range.addr, range.size);
        eager_entries.emplace(range.addr, EagerEntry{range.addr + range.size, offset, tick, false});
    }

    for (const auto& [arena, copies] : regions) {
        cmdbuf.copyBuffer(arena->Handle(), eager_ring->Handle(), copies);
    }

    const vk::MemoryBarrier2 copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &copy_barrier,
    });

    cmdbuf.fillBuffer(eager_marker->Handle(), 0, 4, static_cast<u32>(tick));
    // Later commands may overwrite what was copied, and the host reads the ring and the marker.
    const std::array<vk::MemoryBarrier2, 2> post_barriers = {
        vk::MemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask =
                vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        },
        vk::MemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        },
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pMemoryBarriers = post_barriers.data(),
    });
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionGpuMapped(VAddr addr, size_t size) const {
    return liverpool->IsMapped(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    // DMA shaders read any resident memory: upload what the CPU modified since the last sync
    std::vector<VAddr> bases;
    bool overflow = false;
    {
        std::scoped_lock lk{g_dma_dirty_regions.mutex};
        auto& q = g_dma_dirty_regions;
        bases.assign(q.bases.begin(), q.bases.begin() + q.count);
        overflow = q.overflow;
        q.count = 0;
        q.overflow = false;
    }

    for (const VAddr base : bases) {
        if (auto* region = memory_tracker->RegionAt(base)) {
            region->TakeDmaDirty();
        }
    }

    if (!overflow) {
        std::sort(bases.begin(), bases.end());
        for (const VAddr base : bases) {
            const u64 first_block = base >> block_shift;
            const u64 end_block = (base + HIGHER_PAGE_SIZE) >> block_shift;
            auto it = std::ranges::upper_bound(resident_ranges, first_block, {}, &Backing::end);

            for (; it != resident_ranges.end() && it->start < end_block; ++it) {
                const u64 start = std::max(it->start, first_block);
                const u64 end = std::min(it->end, end_block);
                const u64 page = start >> (ARENA_PAGE_BITS - block_shift);
                SynchronizeMemory(address_space[page], start << block_shift,
                                  static_cast<u32>((end - start) << block_shift), false, false);
            }
        }
        return;
    }
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");
    synced_ranges = {};

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    const u64 frame = g_present_frame.load(std::memory_order_relaxed);
    resident_ranges.ForEachInRange(first_block, last_block + 1, [frame](const Backing& backing) {
        backing.allocation->last_frame = frame;
    });

    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    // Newly resident memory still has to be uploaded (DMA shaders can read all of it).
    for (const auto& range : bind_ranges) {
        memory_tracker->NoteDmaDirty(range.start << block_shift, (range.end - range.start)
                                                                     << block_shift);
    }

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };

    // Out of device memory: free unused buffer memory and images, and retry
    constexpr u32 MaxAttempts = 3;
    constexpr u64 MinBytesToFree = 512_MB;
    constexpr u64 MinIdleFrames = 60;
    auto allocation = instance.GetDevice().allocateMemory(alloc_info);
    for (u32 attempt = 0;
         allocation.result == vk::Result::eErrorOutOfDeviceMemory && attempt < MaxAttempts;
         ++attempt) {
        const u64 wanted = std::max<u64>(alloc_info.allocationSize * 4, MinBytesToFree);
        const u64 freed_buffers = EvictIdleArenaMemory(wanted, MinIdleFrames);
        const u64 freed_images = freed_buffers < wanted
                                     ? texture_cache.EmergencyGarbageCollect(wanted - freed_buffers)
                                     : 0;
        LOG_ERROR(Render_Vulkan,
                  "Out of device memory making {}KB of buffers resident: freed {}MB of unused "
                  "buffers and {}MB of unused images, retrying",
                  alloc_info.allocationSize >> 10, freed_buffers >> 20, freed_images >> 20);
        // Freed memory is released once the GPU is done with it
        scheduler.Finish();

        if (freed_buffers != 0) {
            ReleaseRetiredArenaMemory();
            scheduler.Finish();
        }

        scheduler.PopPendingOperations();
        allocation = instance.GetDevice().allocateMemory(alloc_info);
    }

    const auto device_memory = Vulkan::Check(allocation);
    auto& arena_allocation = arena_allocations.emplace_back(ArenaAllocation{
        .memory = device_memory,
        .size = alloc_info.allocationSize,
        .last_frame = frame,
    });

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset;
        backing.allocation = &arena_allocation;
        resident_ranges.Add(backing);
        arena_allocation.blocks.push_back(range);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

u64 BufferCache::EvictIdleArenaMemory(u64 bytes, u64 min_idle_frames) {
    const u64 frame = g_present_frame.load(std::memory_order_relaxed);
    const auto mark_used = [frame](const Backing& backing) {
        backing.allocation->last_frame = frame;
    };
    // The synced range cache serves its ranges without EnsureResident
    for (const auto& range : synced_ranges) {
        if (range.arena) {
            resident_ranges.ForEachInRange(range.start >> block_shift,
                                           ((range.end - 1) >> block_shift) + 1, mark_used);
        }
    }

    std::vector<ArenaAllocation*> idle;
    for (auto& allocation : arena_allocations) {
        if (allocation.dma || frame - allocation.last_frame < min_idle_frames) {
            continue;
        }
        // GPU written data not read back yet only exists in the arena
        const bool holds_gpu_data = std::ranges::any_of(allocation.blocks, [&](const Interval& b) {
            const VAddr addr = b.start << block_shift;
            const u64 size = (b.end - b.start) << block_shift;
            return !IsCpuCopyValid(addr, size) || IsRegionGpuModified(addr, size);
        });
        if (!holds_gpu_data) {
            idle.push_back(&allocation);
        }
    }
    std::ranges::sort(idle, {}, [](const ArenaAllocation* a) { return a->last_frame; });

    u64 freed = 0;
    for (ArenaAllocation* allocation : idle) {
        if (freed >= bytes) {
            break;
        }

        // Recorded commands may still read it: it stays bound until they are done
        for (const Interval& b : allocation->blocks) {
            const VAddr addr = b.start << block_shift;
            const u64 size = (b.end - b.start) << block_shift;
            runtime.FillBuffer(bda_pagetable_buffer.get(), b.start * sizeof(vk::DeviceAddress),
                               (b.end - b.start) * sizeof(vk::DeviceAddress), 0);
            memory_tracker->MarkRegionAsCpuModified(addr, size);
        }
        retired_arena_memory.push_back({
            .memory = allocation->memory,
            .tick = scheduler.CurrentTick(),
            .blocks = allocation->blocks,
        });

        allocation->memory = nullptr;
        freed += allocation->size;
    }

    if (freed != 0) {
        resident_ranges.EraseIf([](const Backing& backing) { return !backing.allocation->memory; });
        std::erase_if(arena_allocations, [](const ArenaAllocation& a) { return !a.memory; });
    }
    return freed;
}

void BufferCache::ReleaseRetiredArenaMemory() {
    std::erase_if(retired_arena_memory, [this](const RetiredArenaMemory& retired) {
        if (!scheduler.IsFree(retired.tick)) {
            return false;
        }
        // Blocks made resident again are bound to their new memory already
        for (const Interval& b : retired.blocks) {
            resident_ranges.ForEachGap(b.start, b.end, [&](u64 start, u64 end) {
                const Buffer* arena = address_space[start >> blocks_per_arena_page_shift];
                BindsForArena(arena)->binds.push_back(vk::SparseMemoryBind{
                    .resourceOffset = (start << block_shift) - arena->cpu_addr,
                    .size = (end - start) << block_shift,
                });
            });
        }
        // The next submit waits for the unbind
        scheduler.DeferOperation([device = instance.GetDevice(), memory = retired.memory] {
            device.freeMemory(memory);
        });
        return true;
    });
}

void BufferCache::PinForDma(VAddr device_addr, u64 size) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    resident_ranges.ForEachInRange(first_block, last_block + 1,
                                   [](const Backing& backing) { backing.allocation->dma = true; });
}

void BufferCache::TrimArenaMemory() {
    constexpr u64 CheckIntervalFrames = 60;
    constexpr u64 MinIdleFrames = 600;
    constexpr u64 MaxBytesPerCheck = 512_MB;
    if (!retired_arena_memory.empty()) {
        ReleaseRetiredArenaMemory();
    }
    const u64 frame = g_present_frame.load(std::memory_order_relaxed);
    if (frame - arena_trim_frame < CheckIntervalFrames || !instance.CanReportMemoryUsage()) {
        return;
    }

    arena_trim_frame = frame;
    const u64 usage = instance.GetDeviceMemoryUsage();
    const u64 budget = instance.GetTotalMemoryBudget();
    if (usage < budget) {
        return;
    }

    const u64 target = budget - budget / 16;
    EvictIdleArenaMemory(std::min(usage - target, MaxBytesPerCheck), MinIdleFrames);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    if (!memory_tracker->IsSynchronized(device_addr, size, is_written)) {
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        size_t total_size_bytes{};
        memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
        });
        if (!copies.empty()) {
            const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
            for (auto& copy : copies) {
                memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset,
                                         copy.size);
                copy.srcOffset += staging.offset;
                copy.dstOffset -= arena->cpu_addr;
            }
            staging.Flush();
            runtime.CopyBuffer(staging.buffer, arena, copies);
        }
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            EraseEager(device_addr, size);
            open_writes.emplace_back(device_addr, size);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u64 written_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        written_size = std::max<u64>(written_size, mip_info.offset + mip_info.size);
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);

    EraseEager(device_addr, written_size);
    open_writes.emplace_back(device_addr, written_size);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();
    info.AddWait(signal_sema, signal_tick);

    // Runs in submission order, on the recording thread when there is one
    scheduler.RunOnQueue([this, binds = std::move(pending_binds), signal_tick, signal_sema] {
        std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
        buffer_binds.reserve(binds.size());
        for (const auto& bind : binds) {
            buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
                .buffer = bind.arena->Handle(),
                .bindCount = static_cast<u32>(bind.binds.size()),
                .pBinds = bind.binds.data(),
            });
        }
        const vk::TimelineSemaphoreSubmitInfo timeline_si = {
            .signalSemaphoreValueCount = 1u,
            .pSignalSemaphoreValues = &signal_tick,
        };
        const vk::BindSparseInfo sparse_info = {
            .pNext = &timeline_si,
            .bufferBindCount = static_cast<u32>(buffer_binds.size()),
            .pBufferBinds = buffer_binds.data(),
            .signalSemaphoreCount = 1u,
            .pSignalSemaphores = &signal_sema,
        };
        auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    });
    pending_binds.clear();
}

} // namespace VideoCore
