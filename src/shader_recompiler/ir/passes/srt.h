// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            const void* /*load_helpers*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

/// How the walkers' guest loads reach the GPU memory tracking (set by the rasterizer).
struct WalkerMemoryCallbacks {
    bool (*read_protected)(u64 addr, u64 size){};
    bool (*cpu_valid)(u64 addr, u64 size){};
    void (*download)(u64 addr, u64 size){};
};
void SetWalkerMemoryCallbacks(const WalkerMemoryCallbacks& callbacks);

/// Downloads pending GPU data of a guest range read outside the walker code.
void SyncGuestRead(u64 addr, u64 size);

const void* GetWalkerLoadHelpers();

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
