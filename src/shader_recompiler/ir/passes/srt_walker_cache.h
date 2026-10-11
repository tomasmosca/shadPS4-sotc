// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include "common/types.h"

namespace Shader {

/// Untracked pages the walkers read, for the JIT: entry 16 * i holds the page and the backing.
struct WalkerPageCache {
    static constexpr u64 NumEntries = 16;
    struct Entry {
        std::atomic<u64> page{~0ULL};
        const u8* backing{};
    };
    std::array<Entry, NumEntries> entries{};

    void Set(u64 page_index, const u8* backing) {
        Entry& e = entries[page_index & (NumEntries - 1)];

        e.page.store(~0ULL, std::memory_order_relaxed);
        e.backing = backing;
        e.page.store(page_index, std::memory_order_release);
    }

    /// Forgets the page if it is cached.
    void Drop(u64 page_index) {
        u64 expected = page_index;
        entries[page_index & (NumEntries - 1)].page.compare_exchange_strong(
            expected, ~0ULL, std::memory_order_acq_rel);
    }

    void DropAll() {
        for (auto& e : entries) {
            e.page.store(~0ULL, std::memory_order_release);
        }
    }
};
static_assert(sizeof(WalkerPageCache::Entry) == 16);
inline WalkerPageCache g_walker_page_cache;

} // namespace Shader
