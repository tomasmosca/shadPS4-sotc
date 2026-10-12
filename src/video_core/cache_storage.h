// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
    DriverPipelineCache,
    ProgramOrder,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    bool SaveFrom(BlobType type, const std::string& name, std::function<std::vector<u8>()> produce);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

    u32 AllocateShaderBinaryIndex(u64 pgm_hash, u32 min_index);

    std::vector<u64> StoredPrograms();
    std::vector<u32> StoredPermutations(u64 pgm_hash);

    void SetCloseHandler(std::function<void()> handler);
    void WaitForSaves();

private:
    void ScanShaderBinaryIndices();

    std::mutex index_mutex{};
    std::unordered_map<u64, u32> next_binary_index{}; ///< next free index of each program
    std::unordered_map<u64, std::vector<u32>> stored_permutations{};
    std::function<void()> close_handler{};
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
