<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# shadPS4 for Shadow of the Colossus

This is an experimental fork of [shadPS4](https://github.com/shadps4-emu/shadPS4), the PlayStation 4 emulator. It makes Shadow of the Colossus (CUSA08809) playable on a mid-range Windows PC.

The fork starts from upstream commit `94e2177` (September 29, 2026). It changes the GPU emulation only. For anything else, read the [upstream README](https://github.com/shadps4-emu/shadPS4#readme).

## Status

On an Intel Core i5-9400F with an NVIDIA RTX 2070 (8 GB), the game runs at about 30 fps most of the time. Before the speed changes, the game ran at about 8 fps.

Tested setup:

- Windows 10 and an NVIDIA GPU. Other systems and GPUs are not tested.
- Game version 1.00.
- Play sessions of more than 3 hours without a crash.

The first time you enter an area, the emulator compiles its shaders and the game stutters. The next visits use the shader cache.

## How to use

1. Download the latest release, or build the fork the same way as [upstream shadPS4](documents/building-windows.md).
2. Start the game the same way as with shadPS4.
3. Create the file `user/custom_configs/CUSA08809.json` with these settings:

```json
{
  "GPU": {
    "readbacks_mode": 2,
    "readback_linear_images_enabled": false,
    "direct_memory_access_enabled": true
  },
  "Vulkan": {
    "pipeline_cache_enabled": true
  },
  "General": {
    "redzone_patches": true
  }
}
```

The game needs `readbacks_mode` 2 (precise). Without it, the menu is black and the game has no textures.

## What changes

Speed:

- A second thread records the Vulkan commands, so the GPU thread does less work.
- GPU results that the CPU reads back are copied ahead of time. The copy keeps the bytes that the CPU changed in the meantime.
- The emulator runs a copy shader of the game as a plain buffer copy.
- The emulator reuses identical buffer uploads, draw state and shader variants.
- The shader resource walker caches the guest memory pages that it reads.

Graphics fixes:

- Barriers for shared memory in large compute workgroups. This fixes the sun and the exposure on NVIDIA.
- Labels at the end of a frame are written when the GPU finishes the work, not when the emulator reads the command. This reduces flickering on distant geometry.
- Fixes for 64-lane shader code on GPUs with 32-lane subgroups.
- Fixes for light rays, instance IDs and legacy multiplication.
- Texture LOD statistics, arrays of textures and early depth tests.

Stability:

- Buffer memory that was not used for some time is released when video memory is low. Before this, long sessions ended with an out of memory crash.
- The exception handler runs on its own stack. The game runs its jobs on small stacks.
- The emulator tracks the guest thread and fiber stacks, and it does not protect their pages.
- Safer fault handling for direct memory access shaders (upstream pull request #4738).

[documents/sotc-fork-notes.md](documents/sotc-fork-notes.md) explains how the parts keep memory coherent and lists the known limitations.

## Settings

The fork adds these settings to the `GPU` section of `config.json` and of the game config. The default values are the values that this game needs. Change them only to find the cause of a problem.

| Setting | Default | What it does |
|---|---|---|
| `record_thread` | `true` | Records Vulkan commands on a second thread. |
| `flush_every_draws` | `64` | Sends the commands to the GPU after this number of draws. 0 sends them only when needed. |
| `shader_loop_cap` | `0` | Stops shader loops after this number of iterations. 0 means no limit. Use 65536 if the GPU driver resets during play. |
| `copy_kernel_hle` | `true` | Runs the copy shader of the game as a buffer copy. |
| `eager_readbacks` | `true` | Copies GPU results to the CPU when a batch ends. |
| `readbacks_ahead` | `true` | Reads GPU results back outside the current batch when possible. |
| `stream_upload_dedup` | `true` | Reuses an identical buffer upload in the same batch. |
| `specialization_memo` | `true` | Reuses the shader variant of the last draw when its inputs did not change. |
| `reuse_draw_state` | `true` | Skips the binding work when a draw repeats the previous one. |
| `eop_labels_after_gpu` | `true` | Writes end of pipe labels when the GPU finishes the work. |
| `eop_after_earlier_compute` | `true` | Runs compute work that the game sent earlier before an end of pipe label. |
| `early_fragment_tests_from_z_order` | `true` | Enables early depth tests when the game asks for them. |
| `lod_stats_from_bindings` | `true` | Answers texture LOD statistics requests, so textures load at full detail. |
| `dynamic_tsharp_array_size` | `16` | Size of the texture arrays for dynamic indexing. 0 disables them. |
| `wave64_uniform_branches` | `true` | Treats branches on lowered 64-lane values as uniform in one-wave groups. |
| `wave64_missing_lane_identity` | `true` | Fills the lanes that a 32-lane GPU does not have with neutral values. |
| `lds_barriers_large_groups` | `true` | Adds shared memory barriers to workgroups larger than one wave. |
| `lds_barrier_uniform_readlane` | `true` | Adds barriers around lane reads that go through shared memory. |
| `nan_aware_min_max` | `true` | Uses the GCN rules for NaN in min and max. |
| `relative_instance_id` | `true` | Starts the instance ID at 0 in every draw, like the PS4 GPU. |

`shader_loop_cap` and the settings from `dynamic_tsharp_array_size` to `relative_instance_id` change the shaders. When you change one of them, the emulator compiles the shaders again.

## Credits

- The [shadPS4](https://github.com/shadps4-emu/shadPS4) team, for the emulator.
- [Pink-shadPS4](https://github.com/luizgustavs/Pink-shadPS4), implemented various of the graphical fixes.

## License

This fork uses the same license as shadPS4: [GPL-2.0-or-later](LICENSE).
