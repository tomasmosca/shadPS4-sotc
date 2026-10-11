// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace Shader {

void Info::RefreshSharpTables() {
    constexpr VAddr AddressMask = 0xFFFFFFFFFFFFULL;
    auto* memory = Core::Memory::Instance();

    for (const SharpTable& table : sharp_tables) {
        u32* dst = flattened_ud_buf.data() + table.flatbuf_off;
        const u64 size = u64{table.NumDwords()} * sizeof(u32);
        const VAddr ptr =
            (VAddr{flattened_ud_buf[table.ptr_lo]} | VAddr{flattened_ud_buf[table.ptr_hi]} << 32) &
            AddressMask;
        const VAddr src = ptr + u64{table.offset_dw} * sizeof(u32);

        if (ptr == 0) {
            // Null T#s: the array binds null images, like the unflattened sharp did
            std::memset(dst, 0, size);
        } else {
            SyncGuestRead(src, size);
            memory->CopySparseMemory(src, reinterpret_cast<u8*>(dst), size);
        }
    }
}

} // namespace Shader
