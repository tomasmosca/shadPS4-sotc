// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <optional>

namespace Libraries::AudioOut {

struct PortOut;

constexpr std::chrono::microseconds QueuedAudioTarget{30000};

class PortBackend {
public:
    virtual ~PortBackend() = default;

    /// Called with size equal to port buffer size. Without a queue report, it is guaranteed
    /// to be called in intervals of at least port buffer time. With one, it is called when
    /// less than QueuedAudioTarget is queued.
    virtual void Output(void* ptr) = 0;

    virtual void SetVolume(const std::array<int, 8>& ch_volumes) = 0;

    virtual std::optional<std::chrono::microseconds> QueuedTime() const {
        return std::nullopt;
    }
};

class AudioOutBackend {
public:
    AudioOutBackend() = default;
    virtual ~AudioOutBackend() = default;

    virtual std::unique_ptr<PortBackend> Open(PortOut& port) = 0;
};

class SDLAudioOut final : public AudioOutBackend {
public:
    std::unique_ptr<PortBackend> Open(PortOut& port) override;
};

class OpenALAudioOut final : public AudioOutBackend {
public:
    std::unique_ptr<PortBackend> Open(PortOut& port) override;
};

} // namespace Libraries::AudioOut
