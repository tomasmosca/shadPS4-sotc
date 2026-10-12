// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>

#include "common/hash.h"
#include "common/serdes.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 14u;
static constexpr u32 ShaderMetaVersion = 9u;
static constexpr u32 PipelineKeyVersion = 3u;
} // namespace Serialization

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);
    return !ar.Failed();
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);
    if (ar.Failed()) {
        return false;
    }

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;

    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);
    if (ar.Failed()) {
        return false;
    }

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    size_t perm_idx{};
    Program* program = LoadPermutation(ar, perm_idx);
    if (!program) {
        return false;
    }

    infos[stage] = &program->info;
    modules[stage] = program->modules[perm_idx].module;
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }

    return true;
}

Program* PipelineCache::LoadPermutation(Serialization::Archive& ar, size_t& perm_idx) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    if (!LoadShaderMeta(ar, program->info, spec, perm_idx)) {
        return nullptr;
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx),
                                       spv);
    if (spv.empty()) {
        return nullptr;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        // Empty slots (stored indices leave gaps) never match.
        const auto& it =
            std::ranges::find_if(it_pgm.value()->modules, [&](const Program::Module& m) {
                return m.module && m.spec == spec;
            });
        if (it != it_pgm.value()->modules.end()) {
            // A matching permutation is valid only at its original index. A different index means
            // the store holds entries from more than one cache generation, so this pipeline is
            // left to compile at runtime.
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, program->info.hw_stage, program->info.pgm_hash, idx);
                return nullptr;
            }
            module = it->module;
        } else {
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    return it_pgm.value().get();
}

std::unique_ptr<Program> PipelineCache::LoadProgram(u64 pgm_hash) {
    auto& storage = Storage::DataBase::Instance();
    std::unique_ptr<Program> program;

    for (const u32 stored_idx : storage.StoredPermutations(pgm_hash)) {
        std::vector<u8> meta_blob;
        storage.Load(Storage::BlobType::ShaderMeta,
                     fmt::format("{:#018x}", HashCombine(pgm_hash, u64{stored_idx})), meta_blob);
        if (meta_blob.empty()) {
            continue;
        }

        Serialization::Archive ar{std::move(meta_blob)};
        auto loaded = std::make_unique<Program>();
        Shader::StageSpecialization spec{};
        spec.info = &loaded->info;
        size_t perm_idx{};
        if (!LoadShaderMeta(ar, loaded->info, spec, perm_idx)) {
            continue;
        }

        std::vector<u32> spv{};
        storage.Load(Storage::BlobType::ShaderBinary,
                     fmt::format("{:#018x}_{}", pgm_hash, perm_idx), spv);
        if (spv.empty()) {
            continue;
        }

        if (!program) {
            program = std::move(loaded);
        } else if (std::ranges::any_of(program->modules, [&](const Program::Module& m) {
                       return m.module && m.spec == spec;
                   })) {
            // Entries from more than one cache generation: the first one wins
            continue;
        }
        spec.info = &program->info;
        program->InsertPermut(CompileSPV(spv, instance.GetDevice()), std::move(spec), perm_idx);
    }
    return program;
}

std::unique_ptr<Program> PipelineCache::TakeStoredProgram(u64 pgm_hash) {
    {
        std::unique_lock lk{preload_mutex};
        constexpr auto MaxWait = std::chrono::milliseconds{10};

        if (!preload_cv.wait_for(lk, MaxWait, [&] { return loading_program != pgm_hash; })) {
            loading_program_taken = true;
        } else if (const auto it = preloaded_programs.find(pgm_hash);
                   it != preloaded_programs.end()) {
            auto program = std::move(it.value());
            preloaded_programs.erase(it);

            return program;
        } else if (!claimed_programs.insert(pgm_hash).second) {
            return nullptr;
        }
    }
    return LoadProgram(pgm_hash);
}

void PipelineCache::PreloadPrograms(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:ShaderPreload");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);

    auto programs = Storage::DataBase::Instance().StoredPrograms();
    const auto rank = [this](u64 pgm_hash) {
        const auto it = program_order.find(pgm_hash);
        return it != program_order.end() ? it->second : std::numeric_limits<u32>::max();
    };

    std::ranges::stable_sort(programs, {}, rank);
    for (const u64 pgm_hash : programs) {
        if (stoken.stop_requested()) {
            return;
        }

        {
            std::scoped_lock lk{preload_mutex};
            if (!claimed_programs.insert(pgm_hash).second) {
                continue;
            }
            loading_program = pgm_hash;
        }

        auto program = LoadProgram(pgm_hash);
        {
            std::scoped_lock lk{preload_mutex};
            loading_program = 0;

            if (program && loading_program_taken) {
                for (const auto& module : program->modules) {
                    if (module.module) {
                        instance.GetDevice().destroyShaderModule(module.module);
                    }
                }
            } else if (program) {
                preloaded_programs.emplace(pgm_hash, std::move(program));
            }
            loading_program_taken = false;
        }
        preload_cv.notify_all();
    }
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    CreateDriverCache();
    LoadProgramOrder();
    if (!preload_pipelines) {
        Storage::DataBase::Instance().FinishPreload();
        if (!EmulatorSettings.IsPipelineCacheArchived()) {
            preload_thread =
                std::jthread{[this](std::stop_token stoken) { PreloadPrograms(stoken); }};
        }
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

void PipelineCache::CreateDriverCache() {
    std::vector<u8> data;
    Storage::DataBase::Instance().Load(Storage::BlobType::DriverPipelineCache, "pipelines", data);
    vk::PipelineCacheHeaderVersionOne header{};

    if (data.size() >= sizeof(header)) {
        std::memcpy(&header, data.data(), sizeof(header));
    }

    const auto uuid = instance.GetPipelineCacheUUID();
    if (!data.empty() &&
        (data.size() < sizeof(header) || header.headerSize < sizeof(header) ||
         header.headerSize > data.size() ||
         header.headerVersion != vk::PipelineCacheHeaderVersion::eOne ||
         header.vendorID != instance.GetVendorID() || header.deviceID != instance.GetDeviceID() ||
         std::memcmp(header.pipelineCacheUUID.data(), uuid.data(), VK_UUID_SIZE) != 0)) {
        LOG_WARNING(Render_Vulkan,
                    "Ignoring the saved driver pipeline cache: not from this device");
        data.clear();
    }

    auto [result, cache] = instance.GetDevice().createPipelineCacheUnique({
        .initialDataSize = data.size(),
        .pInitialData = data.data(),
    });

    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Ignoring the saved driver pipeline cache: {}",
                    vk::to_string(result));
        return;
    }

    LOG_INFO(Render_Vulkan, "Driver pipeline cache: {} KB loaded", data.size() >> 10);
    pipeline_cache = std::move(cache);
    last_driver_cache_save = std::chrono::steady_clock::now();
}

void PipelineCache::SaveDriverCache() {
    Storage::DataBase::Instance().SaveFrom(
        Storage::BlobType::DriverPipelineCache, "pipelines",
        [device = instance.GetDevice(), cache = *pipeline_cache] {
            auto [result, data] = device.getPipelineCacheData(cache);
            return result == vk::Result::eSuccess ? std::move(data) : std::vector<u8>{};
        });
}

void PipelineCache::OnPipelineCreated() {
    constexpr u32 MinNewPipelines = 64;
    constexpr auto MinInterval = std::chrono::minutes{2};
    const auto now = std::chrono::steady_clock::now();

    if (++pipelines_since_save >= MinNewPipelines && now - last_driver_cache_save >= MinInterval) {
        pipelines_since_save = 0;
        last_driver_cache_save = now;
        SaveDriverCache();
        SaveProgramOrder();
    }
}

void PipelineCache::LoadProgramOrder() {
    std::vector<u32> data;
    Storage::DataBase::Instance().Load(Storage::BlobType::ProgramOrder, "programs", data);

    for (size_t i = 0; i + 3 <= data.size(); i += 3) {
        program_order.emplace(u64{data[i]} | (u64{data[i + 1]} << 32), data[i + 2]);
    }
}

void PipelineCache::SaveProgramOrder() {
    std::vector<u64> used;
    {
        std::scoped_lock lk{preload_mutex};
        used = used_programs;
    }

    Storage::DataBase::Instance().SaveFrom(
        Storage::BlobType::ProgramOrder, "programs", [this, used = std::move(used)] {
            std::vector<u32> data;
            data.reserve((program_order.size() + used.size()) * 3);

            const auto add = [&](u64 pgm_hash, u32 rank) {
                data.push_back(static_cast<u32>(pgm_hash));
                data.push_back(static_cast<u32>(pgm_hash >> 32));
                data.push_back(rank);
            };

            tsl::robin_map<u64, u32> ranks;
            for (u32 rank = 0; rank < used.size(); ++rank) {
                ranks.try_emplace(used[rank], rank);
            }

            for (const auto& [pgm_hash, rank] : program_order) {
                const auto it = ranks.find(pgm_hash);
                add(pgm_hash, it != ranks.end() ? std::min(rank, it->second) : rank);
            }

            for (const auto& [pgm_hash, rank] : ranks) {
                if (!program_order.contains(pgm_hash)) {
                    add(pgm_hash, rank);
                }
            }

            const auto bytes = reinterpret_cast<const u8*>(data.data());
            return std::vector<u8>(bytes, bytes + data.size() * sizeof(u32));
        });
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
    fetch.Write(one_vgprs);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);
    fetch.Read(one_vgprs);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    walker_func = nullptr;
    if (walker_func_size > ar.Remaining()) {
        ar.Fail();
    } else if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }
    if (!walker_func) {
        walker_func_size = 0;

        if (flattened_bufsize_dw > NUM_USER_DATA_REGS) {
            ar.Fail();
        }
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (!fetch_shader_data.Empty()) {
        spec.Write(sizeof(fetch_shader_data));
        fetch_shader_data.Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
