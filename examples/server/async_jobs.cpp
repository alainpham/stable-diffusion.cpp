// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <utility>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion-internal.h"

namespace fs = std::filesystem;

static int64_t random_image_seed() {
    std::random_device random_device;
    std::seed_seq seed_data{random_device(), random_device(), random_device(), random_device()};
    std::mt19937_64 generator(seed_data);
    std::uniform_int_distribution<int64_t> distribution(0, std::numeric_limits<int64_t>::max() - 8);
    return distribution(generator);
}

static std::string history_image_stem(std::chrono::system_clock::time_point timestamp) {
    const std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    std::ostringstream output;
    output << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
    return output.str();
}

static bool save_history_image(const sd_image_t& image,
                               const json& settings,
                               std::string& filename,
                               std::string& error_message) {
    static std::mutex history_mutex;
    std::lock_guard<std::mutex> lock(history_mutex);
    const fs::path output_dir = "sdoutput";
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        error_message = "unable to create sdoutput: " + error.message();
        return false;
    }

    auto timestamp = std::chrono::system_clock::now();
    fs::path image_path;
    fs::path settings_path;
    for (int attempt = 0; attempt < 86400; ++attempt) {
        const std::string stem = history_image_stem(timestamp);
        image_path            = output_dir / (stem + ".png");
        settings_path         = output_dir / (stem + ".json");
        const bool image_exists = fs::exists(image_path, error);
        if (error) {
            error_message = "unable to inspect sdoutput: " + error.message();
            return false;
        }
        const bool settings_exist = fs::exists(settings_path, error);
        if (error) {
            error_message = "unable to inspect sdoutput: " + error.message();
            return false;
        }
        if (!image_exists && !settings_exist) {
            break;
        }
        timestamp += std::chrono::seconds(1);
        image_path.clear();
    }
    if (image_path.empty()) {
        error_message = "unable to choose a unique filename in sdoutput";
        return false;
    }

    if (!write_image_to_file(image_path.string(), image.data, image.width, image.height, image.channel)) {
        fs::remove(image_path, error);
        error_message = "unable to save generated image to sdoutput";
        return false;
    }

    std::ofstream settings_file(settings_path, std::ios::binary | std::ios::trunc);
    auto remove_partial_files = [&]() {
        settings_file.close();
        std::error_code cleanup_error;
        fs::remove(image_path, cleanup_error);
        cleanup_error.clear();
        fs::remove(settings_path, cleanup_error);
    };
    if (!settings_file) {
        remove_partial_files();
        error_message = "unable to save image settings to sdoutput";
        return false;
    }
    try {
        settings_file << settings.dump(2);
        settings_file.flush();
    } catch (const json::exception& e) {
        remove_partial_files();
        error_message = std::string("unable to serialize image settings: ") + e.what();
        return false;
    }
    settings_file.close();
    if (!settings_file) {
        remove_partial_files();
        error_message = "unable to write image settings to sdoutput";
        return false;
    }

    filename = image_path.filename().string();
    return true;
}

static void preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    (void)is_noisy;
    if (frame_count <= 0 || frames == nullptr || frames[0].data == nullptr) {
        return;
    }

    AsyncGenerationJob* job      = static_cast<AsyncGenerationJob*>(data);
    const sd_image_t& frame      = frames[0];
    const sd_preview_info_t info = sd_get_preview_info();

    auto image_bytes = encode_image_to_vector(EncodedImageFormat::PNG,
                                              frame.data,
                                              frame.width,
                                              frame.height,
                                              frame.channel,
                                              "",
                                              80);
    if (image_bytes.empty()) {
        return;
    }

    std::string b64 = base64_encode(image_bytes);
    std::lock_guard<std::mutex> lock(job->preview_mutex);
    job->preview_b64         = std::move(b64);
    job->preview_step        = step < 0 ? -step : step;
    job->preview_pass        = info.sample_pass;
    job->preview_total_steps = info.total_steps;
}

static void clear_preview_callback() {
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
}

static void set_preview_callback(AsyncGenerationJob& job) {
    preview_t mode = str_to_preview(job.kind == AsyncJobKind::ImgGen
                                        ? job.img_gen.preview_mode.c_str()
                                        : job.vid_gen.preview_mode.c_str());
    if (mode == PREVIEW_COUNT || mode == PREVIEW_NONE) {
        return;
    }
    int interval = job.kind == AsyncJobKind::ImgGen
                       ? job.img_gen.preview_interval
                       : job.vid_gen.preview_interval;
    if (interval <= 0) {
        interval = 1;
    }
    sd_set_preview_callback(preview_callback, mode, interval, true, false, &job);
}

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        default:
            return "img_gen";
    }
}

const char* async_job_status_name(AsyncJobStatus status) {
    switch (status) {
        case AsyncJobStatus::Queued:
            return "queued";
        case AsyncJobStatus::Generating:
            return "generating";
        case AsyncJobStatus::Completed:
            return "completed";
        case AsyncJobStatus::Failed:
            return "failed";
        case AsyncJobStatus::Cancelled:
            return "cancelled";
        default:
            return "failed";
    }
}

void purge_expired_jobs(AsyncJobManager& manager) {
    const int64_t now = unix_timestamp_now();

    for (auto it = manager.expired_jobs.begin(); it != manager.expired_jobs.end();) {
        if (it->second <= now) {
            it = manager.expired_jobs.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = manager.jobs.begin(); it != manager.jobs.end();) {
        const auto& job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }

        int64_t ttl_seconds = job->status == AsyncJobStatus::Completed
                                  ? manager.completed_ttl_seconds
                                  : manager.failed_ttl_seconds;
        if (now - job->completed_at >= ttl_seconds) {
            manager.expired_jobs[job->id] = now + std::max<int64_t>(ttl_seconds, 60);
            it                            = manager.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

size_t count_pending_jobs(const AsyncJobManager& manager) {
    size_t pending = 0;
    for (const auto& entry : manager.jobs) {
        if (entry.second->status == AsyncJobStatus::Queued ||
            entry.second->status == AsyncJobStatus::Generating) {
            ++pending;
        }
    }
    return pending;
}

std::string make_async_job_id(AsyncJobManager& manager) {
    std::ostringstream oss;
    oss << "job_" << std::hex << unix_timestamp_now() << "_" << std::setw(8)
        << std::setfill('0') << manager.next_id++;
    return oss.str();
}

bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job) {
    auto new_end = std::remove(manager.queue.begin(), manager.queue.end(), job.id);
    if (new_end == manager.queue.end()) {
        return false;
    }

    manager.queue.erase(new_end, manager.queue.end());
    job.status       = AsyncJobStatus::Cancelled;
    job.completed_at = unix_timestamp_now();
    job.result_images_b64.clear();
    job.result_image_filenames.clear();
    job.completed_image_filenames.clear();
    job.image_save_error.clear();
    job.result_media_b64.clear();
    job.result_media_mime_type.clear();
    job.result_frame_count = 0;
    job.result_fps         = 0;
    job.error_code         = "cancelled";
    job.error_message      = "job cancelled by client";
    return true;
}

json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job) {
    json result;
    result["id"]             = job.id;
    result["kind"]           = async_job_kind_name(job.kind);
    result["status"]         = async_job_status_name(job.status);
    result["created"]        = job.created_at;
    result["started"]        = job.started_at == 0 ? json(nullptr) : json(job.started_at);
    result["completed"]      = job.completed_at == 0 ? json(nullptr) : json(job.completed_at);
    result["queue_position"] = 0;
    if (job.kind == AsyncJobKind::ImgGen) {
        result["completed_images"] = job.completed_image_filenames;
    }

    if (job.status == AsyncJobStatus::Queued) {
        size_t position = 1;
        for (const auto& queued_id : manager.queue) {
            if (queued_id == job.id) {
                result["queue_position"] = position;
                break;
            }
            ++position;
        }
    }

    if (job.status == AsyncJobStatus::Completed) {
        if (job.kind == AsyncJobKind::VidGen) {
            result["result"] = {
                {"output_format", job.vid_gen.output_format},
                {"mime_type", job.result_media_mime_type},
                {"fps", job.result_fps},
                {"frame_count", job.result_frame_count},
                {"b64_json", job.result_media_b64},
            };
        } else {
            json images = json::array();
            for (size_t i = 0; i < job.result_images_b64.size(); ++i) {
                const std::string filename = i < job.result_image_filenames.size()
                                                 ? job.result_image_filenames[i]
                                                 : std::string();
                images.push_back({{"index", i}, {"b64_json", job.result_images_b64[i]}, {"filename", filename}});
            }
            result["result"] = {
                {"output_format", job.img_gen.output_format},
                {"images", images},
                {"save_error", job.image_save_error},
            };
        }
        result["error"] = nullptr;
    } else if (job.status == AsyncJobStatus::Failed ||
               job.status == AsyncJobStatus::Cancelled) {
        result["result"] = nullptr;
        result["error"]  = {
             {"code",
             job.error_code.empty()
                  ? (job.status == AsyncJobStatus::Cancelled ? "cancelled" : "generation_failed")
                  : job.error_code},
             {"message", job.error_message},
        };
    } else {
        result["result"] = nullptr;
        result["error"]  = nullptr;
    }

    if (job.status == AsyncJobStatus::Generating) {
        std::lock_guard<std::mutex> lock(job.preview_mutex);
        if (!job.preview_b64.empty()) {
            result["preview"] = {
                {"pass", job.preview_pass},
                {"step", job.preview_step},
                {"total_steps", job.preview_total_steps},
                {"b64_json", job.preview_b64},
            };
        }
    }

    return result;
}

struct ImageOutputContext {
    ServerRuntime& runtime;
    AsyncGenerationJob& job;
    int64_t initial_seed;
    EncodedImageFormat encoded_format;
    std::vector<std::string>& output_images;
    std::vector<std::string>& output_image_filenames;
    std::string& image_save_error;
};

static void append_image_error(std::string& errors, const std::string& error) {
    if (!errors.empty()) {
        errors += "; ";
    }
    errors += error;
}

static void collect_generated_image(const sd_image_t* image,
                                    size_t output_index,
                                    size_t batch_index,
                                    void* user_data) {
    (void)output_index;
    if (image == nullptr || image->data == nullptr || user_data == nullptr) {
        return;
    }

    auto& context = *static_cast<ImageOutputContext*>(user_data);
    const int64_t image_seed = context.initial_seed + static_cast<int64_t>(batch_index);
    const std::string metadata = context.job.img_gen.gen_params.embed_image_metadata
                                     ? get_image_params(*context.runtime.ctx_params,
                                                        context.job.img_gen.gen_params,
                                                        image_seed)
                                     : "";
    auto image_bytes = encode_image_to_vector(context.encoded_format,
                                              image->data,
                                              image->width,
                                              image->height,
                                              image->channel,
                                              metadata,
                                              context.job.img_gen.output_compression);
    if (image_bytes.empty()) {
        append_image_error(context.image_save_error, "unable to encode generated image");
        return;
    }
    context.output_images.push_back(base64_encode(image_bytes));

    json image_settings = context.job.ui_settings_snapshot;
    if (!image_settings.is_object()) {
        image_settings = json::object();
    }
    if (!image_settings.contains("form") || !image_settings["form"].is_object()) {
        image_settings["form"] = json::object();
    }
    image_settings["form"]["seed"] = image_seed;

    std::string saved_filename;
    std::string save_error;
    if (save_history_image(*image, image_settings, saved_filename, save_error)) {
        context.output_image_filenames.push_back(saved_filename);
        if (context.runtime.async_job_manager != nullptr) {
            std::lock_guard<std::mutex> lock(context.runtime.async_job_manager->mutex);
            context.job.completed_image_filenames.push_back(std::move(saved_filename));
        }
    } else {
        context.output_image_filenames.emplace_back();
        append_image_error(context.image_save_error, save_error);
        LOG_ERROR("%s", save_error.c_str());
    }
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         std::vector<std::string>& output_image_filenames,
                         std::string& image_save_error,
                         std::string& error_message) {
    sd_img_gen_params_t params = job.img_gen.to_sd_img_gen_params_t();
    if (params.seed < 0) {
        params.seed = random_image_seed();
    }

    EncodedImageFormat encoded_format = EncodedImageFormat::PNG;
    if (job.img_gen.output_format == "jpeg") {
        encoded_format = EncodedImageFormat::JPEG;
    } else if (job.img_gen.output_format == "webp") {
        encoded_format = EncodedImageFormat::WEBP;
    }
    ImageOutputContext output_context{runtime,
                                      job,
                                      params.seed,
                                      encoded_format,
                                      output_images,
                                      output_image_filenames,
                                      image_save_error};

    bool generated = false;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        set_preview_callback(job);
        int num_results = 0;
        generated = sd_generate_image_with_callback(runtime.sd_ctx,
                                                    &params,
                                                    &num_results,
                                                    collect_generated_image,
                                                    &output_context);
        clear_preview_callback();
    }
    if (!generated || output_images.empty()) {
        error_message = "generate_image returned empty encoded outputs";
        return false;
    }

    return true;
}

bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         std::string& error_message) {
    sd_vid_gen_params_t params = job.vid_gen.to_sd_vid_gen_params_t();
    std::string str_params     = job.vid_gen.gen_params.embed_image_metadata
                                     ? get_image_params(*runtime.ctx_params, job.vid_gen.gen_params, job.vid_gen.gen_params.seed, VID_GEN)
                                     : "";

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        set_preview_callback(job);
        sd_image_t* raw_results = nullptr;
        if (!generate_video(runtime.sd_ctx, &params, &raw_results, &num_results, &generated_audio, &output_fps)) {
            raw_results = nullptr;
        }
        clear_preview_callback();
        results.adopt(raw_results, num_results);
    }

    num_results = results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        error_message = "generate_video returned no results";
        return false;
    }

    std::vector<uint8_t> video_bytes = create_video_from_sd_images_to_vector(job.vid_gen.output_format,
                                                                             results.data(),
                                                                             num_results,
                                                                             output_fps,
                                                                             job.vid_gen.output_compression,
                                                                             generated_audio,
                                                                             str_params);
    free_sd_audio(generated_audio);
    if (video_bytes.empty()) {
        error_message = "failed to encode generated video container";
        return false;
    }

    output_media_b64       = base64_encode(video_bytes);
    output_media_mime_type = video_mime_type(job.vid_gen.output_format);
    output_frame_count     = num_results;
    return true;
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            manager.cv.wait(lock, [&]() { return manager.stop || !manager.queue.empty(); });

            if (manager.stop && manager.queue.empty()) {
                break;
            }

            purge_expired_jobs(manager);
            if (manager.queue.empty()) {
                continue;
            }

            const std::string job_id = manager.queue.front();
            manager.queue.pop_front();

            auto it = manager.jobs.find(job_id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job             = it->second;
            job->status     = AsyncJobStatus::Generating;
            job->started_at = unix_timestamp_now();
        }

        std::vector<std::string> output_images;
        std::vector<std::string> output_image_filenames;
        std::string image_save_error;
        std::string output_media_b64;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        std::string error_message;
        bool ok = false;

        if (job->kind == AsyncJobKind::ImgGen) {
            ok = execute_img_gen_job(runtime,
                                     *job,
                                     output_images,
                                     output_image_filenames,
                                     image_save_error,
                                     error_message);
        } else if (job->kind == AsyncJobKind::VidGen) {
            ok = execute_vid_gen_job(runtime,
                                     *job,
                                     output_media_b64,
                                     output_media_mime_type,
                                     output_frame_count,
                                     output_fps,
                                     error_message);
        } else {
            error_message = "unsupported job kind";
        }

        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto it = manager.jobs.find(job->id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job->completed_at = unix_timestamp_now();
            if (ok) {
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_image_filenames = std::move(output_image_filenames);
                job->image_save_error       = std::move(image_save_error);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
            } else {
                job->status        = AsyncJobStatus::Failed;
                job->error_code    = "generation_failed";
                job->error_message = error_message.empty() ? "unknown generation error" : error_message;
                job->result_images_b64.clear();
                job->result_image_filenames.clear();
                job->image_save_error.clear();
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
            }

            purge_expired_jobs(manager);
        }
    }
}
