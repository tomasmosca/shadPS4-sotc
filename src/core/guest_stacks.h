// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_map.hpp>

#include "common/alignment.h"
#include "common/types.h"

// Guest stacks GPU tracking must not protect: Windows writes exception frames on the faulting
// stack and ends the process when that page is protected
namespace Core::GuestStacks {

using PageMap = boost::icl::interval_map<VAddr, u32>;

struct Registry {
    struct Stack {
        u64 size{};
        const void* owner{}; ///< the thread or fiber using it
    };
    std::shared_mutex mutex;
    PageMap pages;                 ///< page-aligned ranges -> number of stacks on them
    std::map<VAddr, Stack> stacks; ///< by start
    std::atomic<u64> generation{}; ///< bumped by every Register
};

inline Registry& Get() {
    static Registry registry;
    return registry;
}

constexpr VAddr PageSize = 4_KB;

inline PageMap::interval_type StackPages(VAddr start, u64 size) {
    return PageMap::interval_type::right_open(Common::AlignDown(start, PageSize),
                                              Common::AlignUp(start + size, PageSize));
}

/// A stack registered again (reused by a new thread or fiber) changes owner.
inline void Register(VAddr start, u64 size, const void* owner) {
    if (start == 0 || size == 0) {
        return;
    }
    auto& reg = Get();
    std::unique_lock lk{reg.mutex};
    const auto [it, inserted] = reg.stacks.try_emplace(start, Registry::Stack{size, owner});
    if (!inserted) {
        reg.pages -= std::make_pair(StackPages(start, it->second.size), u32{1});
        it->second = {size, owner};
    }
    reg.pages += std::make_pair(StackPages(start, size), u32{1});
    reg.generation.fetch_add(1, std::memory_order_release);
}

/// Changes when a stack is registered, which may race with protecting its pages.
inline u64 Generation() {
    return Get().generation.load(std::memory_order_acquire);
}

/// Does nothing when another owner registered the stack since.
inline void Unregister(VAddr start, const void* owner) {
    auto& reg = Get();
    std::unique_lock lk{reg.mutex};
    const auto it = reg.stacks.find(start);
    if (it == reg.stacks.end() || it->second.owner != owner) {
        return;
    }
    reg.pages -= std::make_pair(StackPages(start, it->second.size), u32{1});
    reg.stacks.erase(it);
}

/// Calls func(begin, end, is_stack) per piece, without the registry lock (func may take others).
template <typename Func>
inline void ForEachPiece(VAddr addr, u64 size, Func&& func) {
    auto& reg = Get();
    const VAddr end = addr + size;
    struct Piece {
        VAddr begin{};
        VAddr end{};
    };
    boost::container::small_vector<Piece, 8> pieces;
    {
        std::shared_lock lk{reg.mutex};
        const auto [first, last] =
            reg.pages.equal_range(PageMap::interval_type::right_open(addr, end));
        for (auto it = first; it != last; ++it) {
            pieces.push_back({std::max(it->first.lower(), addr), std::min(it->first.upper(), end)});
        }
    }
    VAddr cursor = addr;
    for (const Piece& piece : pieces) {
        if (cursor < piece.begin) {
            func(cursor, piece.begin, false);
        }
        func(piece.begin, piece.end, true);
        cursor = piece.end;
    }
    if (cursor < end) {
        func(cursor, end, false);
    }
}

} // namespace Core::GuestStacks
