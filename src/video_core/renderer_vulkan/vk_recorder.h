// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/// Replays the commands recorded on fake command buffers on its own thread, in order.
class CommandRecorder {
public:
    CommandRecorder();
    ~CommandRecorder();

    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;

    /// Hooks the vkCmd functions of the dispatcher, once its device functions are loaded.
    static void InstallThunks();

    [[nodiscard]] static bool IsFake(VkCommandBuffer cmdbuf) noexcept;

    /// A handle for a command buffer the recording thread allocates later (see Bind).
    [[nodiscard]] vk::CommandBuffer NewFakeHandle();

    /// Runs func on the recording thread, after everything queued before.
    template <typename F>
    void Run(F&& func) {
        using T = std::decay_t<F>;
        static_assert(alignof(T) <= BlockAlign, "Closure over-aligned for the stream");
        void* payload = Allocate(sizeof(T), BlockKind::Command, &Invoke<T>);
        new (payload) T(std::forward<F>(func));
        Publish();
    }

    /// A block of size bytes for the arguments of invoke(payload), queued by Commit.
    [[nodiscard]] void* AllocateCommand(size_t size, void (*invoke)(void* payload)) {
        return Allocate(size, BlockKind::Command, invoke);
    }

    /// Queues the block of the last AllocateCommand.
    void Commit() {
        Publish();
    }

    /// Copies an argument array into the stream (the caller may free the original).
    template <typename T>
    [[nodiscard]] T* Copy(const T* src, size_t count = 1) {
        if (src == nullptr || count == 0) {
            return nullptr;
        }
        static_assert(std::is_trivially_copyable_v<T>);
        void* dst = Allocate(sizeof(T) * count, BlockKind::Data, nullptr);
        std::memcpy(dst, src, sizeof(T) * count);
        return static_cast<T*>(dst);
    }

    /// Makes room in the current chunk for the next blocks of one command.
    void Reserve(size_t bytes);

    /// Publishes everything queued and wakes the recording thread (at submits).
    void Kick();

    /// Makes the calling thread the producer (shutdown records from another thread).
    void ClaimProducer();

    /// Connects a fake handle to its real command buffer (recording thread).
    void Bind(VkCommandBuffer fake, VkCommandBuffer real) noexcept;

    /// The real command buffer of a fake handle (recording thread).
    [[nodiscard]] VkCommandBuffer Real(VkCommandBuffer fake) const noexcept;

    /// Frees the slots of count fake handles once their command buffers are submitted.
    void Retire(u32 count) noexcept {
        handles_retired.fetch_add(count, std::memory_order_release);
    }

    static constexpr size_t BlockAlign = 16;
    static constexpr u32 NumFakeHandles = 1U << 16;

private:
    enum class BlockKind : u32 { Data, Command, End };

    struct BlockHeader {
        /// Block size including the header, rounded up to BlockAlign for the next block
        u32 size;
        BlockKind kind;
        void (*invoke)(void* payload);
    };
    static_assert(sizeof(BlockHeader) == BlockAlign);

    struct Chunk {
        static constexpr size_t Size = 256_KB;
        std::atomic<u32> committed{};
        std::atomic<Chunk*> next{nullptr};
        alignas(64) u8 data[Size];
    };

    template <typename T>
    static void Invoke(void* payload) {
        T& func = *static_cast<T*>(payload);
        func();
        func.~T();
    }

    void* Allocate(size_t size, BlockKind kind, void (*invoke)(void*));
    void Publish();
    void PublishAll();
    void NextChunk();
    Chunk* AcquireChunk();
    void RecycleChunk(Chunk* chunk);
    void WakeConsumer();
    void WakeIfSleeping();
    void CheckProducer();
    void ConsumerLoop(std::stop_token stoken, Chunk* first_chunk);

    /// Up to 64 MiB of queued commands before the producer waits for the recording thread
    static constexpr size_t MaxChunksInFlight = 256;
    static constexpr u32 PublishBatch = 32;

    // Producer state
    std::vector<std::unique_ptr<Chunk>> chunks; ///< Owns every chunk, linked or free
    Chunk* write_chunk{};
    u32 write_pos{};
    u64 handles_issued{};
    u32 unpublished{};
    std::thread::id producer;
    u32 producer_switches{};

    // Recording thread state
    std::array<VkCommandBuffer, NumFakeHandles> real_handles{};

    // Shared state
    std::atomic<u64> handles_retired{};
    std::atomic<u32> chunks_in_flight{};
    std::atomic<bool> consumer_sleeping{};
    std::mutex wake_mutex;
    std::condition_variable wake_cv;
    bool wake_flag{};
    std::mutex free_mutex;
    std::vector<Chunk*> free_chunks;
    std::jthread thread;
};

} // namespace Vulkan
