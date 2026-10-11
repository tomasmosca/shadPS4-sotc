<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Notes on the Shadow of the Colossus fork

This file is for developers. It explains how the fork keeps guest memory, GPU memory and its caches coherent. It also lists the known limitations.

## Coherence

The CPU and the GPU both write guest memory. Each cache below has one rule that tells when its data is no longer valid.

| Mechanism | Where | Valid while |
|---|---|---|
| Memory epoch | `BufferCache::MemEpoch` | Nothing changed memory that the GPU or the SRT walker reads. Fences, `WaitRegMem`, memory semaphore waits, the first resume of each submit and GPU copies increment it. Buffer upload dedup and draw state reuse compare it. |
| CPU write generation | `RegionManager`, one per 16 MB region | The CPU did not write the region. The synced range cache compares it. |
| GPU modified ranges | `gpu_modified_ranges` and the GPU bits of the memory tracker | These are the bytes that the GPU wrote and the CPU did not read back yet. A CPU access to them starts a readback. |
| Write snapshots | `write_shadows` | A readback copies only the words that the GPU changed. Words that the CPU changed after the snapshot stay. |
| Eager ring | `eager_entries` | At the end of a batch, the GPU copies its writes in recently read windows to a host ring. A readback uses the ring only if the ring covers all the bytes. |
| Walker page cache | `g_walker_page_cache` | The page has no GPU writes, and nobody unmapped the memory. |
| Synced ranges | `synced_ranges` | The range is read only and its region has the same CPU write generation. |
| Arena residency | `arena_allocations` | Buffer memory stays resident while a draw or a dispatch uses it. Over the memory budget, the emulator releases memory that nothing used for 600 frames. It never releases memory with pending GPU writes, or memory that DMA shaders read through a fault. |
| Guest stacks | `Core::GuestStacks` | The page tracker does not protect the pages of a registered thread or fiber stack. Windows ends the process if a fault happens on a protected stack page. |

## Known limitations

- Fiber stacks of this game share pages with other heap data. The tracker does not protect these pages, so it does not see CPU writes to the other data in them.
- While the page tracker protects a range, a new stack can get a short protection before the tracker opens it again.
- A shader can write an end of pipe label address after the emulator defers the label. The deferred write then replaces the shader result. This did not happen in any measured session.
- If a guest thread exits from a signal handler, its exception handler stack is not released.
- The FMask size of a dynamic texture array comes from the first bind.
- `force_degamma` does not handle texture arrays that mix sRGB and UNORM formats.
- A DMA shader can read a buffer page that the emulator released. That read returns zero once. Then the page becomes resident again and stays resident.
- After the texture garbage collector frees images, the Vulkan memory allocator keeps some of their memory blocks.
- On Linux and macOS, the emulator can reuse a thread stack before the native thread ends. This comes from upstream. The fork fixes it on Windows only.
