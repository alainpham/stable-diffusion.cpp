#include "routes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <utility>

#include "async_jobs.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

namespace fs = std::filesystem;

static constexpr uint32_t k_max_upscale_dimension = 8192;
static constexpr std::size_t k_max_webui_settings_size = 1024 * 1024;

static bool valid_upscale_dimensions(const sd_image_t& image, int factor, int repeats) {
    if (image.width == 0 || image.height == 0 || factor < 1 || repeats < 1 || repeats > 4) {
        return false;
    }
    uint32_t width  = image.width;
    uint32_t height = image.height;
    for (int i = 0; i < repeats; ++i) {
        if (width > k_max_upscale_dimension / factor || height > k_max_upscale_dimension / factor) {
            return false;
        }
        width *= factor;
        height *= factor;
    }
    return true;
}

static bool parse_cache_mode(const std::string& mode_str, sd_cache_mode_t& mode_out) {
    if (mode_str == "disabled") {
        mode_out = SD_CACHE_DISABLED;
        return true;
    }
    if (mode_str == "easycache") {
        mode_out = SD_CACHE_EASYCACHE;
        return true;
    }
    if (mode_str == "ucache") {
        mode_out = SD_CACHE_UCACHE;
        return true;
    }
    if (mode_str == "dbcache") {
        mode_out = SD_CACHE_DBCACHE;
        return true;
    }
    if (mode_str == "taylorseer") {
        mode_out = SD_CACHE_TAYLORSEER;
        return true;
    }
    if (mode_str == "cache-dit") {
        mode_out = SD_CACHE_CACHE_DIT;
        return true;
    }
    if (mode_str == "spectrum") {
        mode_out = SD_CACHE_SPECTRUM;
        return true;
    }
    return false;
}

static json finite_number_or_null(float value) {
    return std::isfinite(value) ? json(value) : json(nullptr);
}

static const char* capability_scheduler_name(enum scheduler_t scheduler) {
    return scheduler < SCHEDULER_COUNT ? sd_scheduler_name(scheduler) : "default";
}

static const char* capability_sample_method_name(enum sample_method_t sample_method) {
    return sample_method < SAMPLE_METHOD_COUNT ? sd_sample_method_name(sample_method) : "default";
}

static json make_vae_tiling_json(const sd_tiling_params_t& params) {
    return {
        {"enabled", params.enabled},
        {"temporal_tiling", params.temporal_tiling},
        {"tile_size_w", params.tile_size_w},
        {"tile_size_h", params.tile_size_h},
        {"target_overlap", params.target_overlap},
        {"rel_size_w", params.rel_size_w},
        {"rel_size_h", params.rel_size_h},
        {"extra_tiling_args", params.extra_tiling_args ? params.extra_tiling_args : ""},
    };
}

static fs::path resolve_display_model_path(const ServerRuntime& runtime) {
    const auto& ctx = *runtime.ctx_params;
    if (!ctx.model_path.empty()) {
        return fs::path(ctx.model_path);
    }
    if (!ctx.diffusion_model_path.empty()) {
        return fs::path(ctx.diffusion_model_path);
    }
    return {};
}

static json make_sample_params_json(const sd_sample_params_t& sample_params,
                                    const std::vector<int>& skip_layers,
                                    const std::vector<float>* custom_sigmas = nullptr) {
    const auto& guidance = sample_params.guidance;
    json custom_sigmas_json = json::array();
    if (custom_sigmas != nullptr) {
        for (float sigma : *custom_sigmas) {
            custom_sigmas_json.push_back(sigma);
        }
    }
    return {
        {"scheduler", capability_scheduler_name(sample_params.scheduler)},
        {"sample_method", capability_sample_method_name(sample_params.sample_method)},
        {"sample_steps", sample_params.sample_steps},
        {"custom_sigmas", custom_sigmas_json},
        {"eta", finite_number_or_null(sample_params.eta)},
        {"shifted_timestep", sample_params.shifted_timestep},
        {"flow_shift", finite_number_or_null(sample_params.flow_shift)},
        {"guidance",
         {
             {"txt_cfg", guidance.txt_cfg},
             {"img_cfg", finite_number_or_null(guidance.img_cfg)},
             {"distilled_guidance", guidance.distilled_guidance},
             {"slg",
              {
                  {"layers", skip_layers},
                  {"layer_start", guidance.slg.layer_start},
                  {"layer_end", guidance.slg.layer_end},
                  {"scale", guidance.slg.scale},
              }},
         }},
    };
}

static json make_hires_json(const SDGenerationParams& defaults) {
    return {
        {"enabled", defaults.hires_enabled},
        {"upscaler", defaults.hires_upscaler},
        {"scale", defaults.hires_scale},
        {"target_width", defaults.hires_width},
        {"target_height", defaults.hires_height},
        {"steps", defaults.hires_steps},
        {"denoising_strength", defaults.hires_denoising_strength},
        {"custom_sigmas", defaults.hires_custom_sigmas},
        {"upscale_tile_size", defaults.hires_upscale_tile_size},
    };
}

static json make_img_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"batch_count", defaults.batch_count},
        {"qwen_image_layers", defaults.qwen_image_layers},
        {"ref_image_args", defaults.ref_image_args},
        {"image_preprocess", defaults.image_preprocess},
        {"increase_ref_index", defaults.increase_ref_index},
        {"control_strength", defaults.control_strength},
        {"ip_adapter_strength", defaults.ip_adapter_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers, &defaults.custom_sigmas)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_vid_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"video_frames", defaults.video_frames},
        {"image_preprocess", defaults.image_preprocess},
        {"fps", defaults.fps},
        {"moe_boundary", defaults.moe_boundary},
        {"vace_strength", defaults.vace_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers, &defaults.custom_sigmas)},
        {"high_noise_sample_params", make_sample_params_json(defaults.high_noise_sample_params, defaults.high_noise_skip_layers)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_img_gen_features_json() {
    return {
        {"init_image", true},
        {"mask_image", true},
        {"control_image", true},
        {"ip_adapter_image", true},
        {"ref_images", true},
        {"lora", true},
        {"vae_tiling", true},
        {"hires", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
        {"preview", true},
    };
}

static json make_vid_gen_features_json() {
    return {
        {"init_image", true},
        {"end_image", true},
        {"control_frames", true},
        {"high_noise_sample_params", true},
        {"lora", true},
        {"vae_tiling", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
        {"preview", true},
    };
}

static json make_preview_modes_json() {
    json modes = json::array();
    for (int i = 0; i < PREVIEW_COUNT; ++i) {
        modes.push_back(sd_preview_name((preview_t)i));
    }
    return modes;
}

static json make_capabilities_json(ServerRuntime& runtime) {
    refresh_lora_cache(runtime);
    refresh_upscaler_cache(runtime);

    AsyncJobManager& manager  = *runtime.async_job_manager;
    const auto& defaults      = *runtime.default_gen_params;
    const fs::path model_path = resolve_display_model_path(runtime);
    const bool supports_img   = runtime_supports_generation_mode(runtime, IMG_GEN);
    const bool supports_vid   = runtime_supports_generation_mode(runtime, VID_GEN);
    json samplers             = json::array();
    json schedulers           = json::array();
    json image_output_formats = supported_img_output_formats();
    json video_output_formats = supported_vid_output_formats();
    json available_loras      = json::array();
    json available_upscalers  = json::array();
    json supported_modes      = json::array();

    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        samplers.push_back(sd_sample_method_name((sample_method_t)i));
    }

    for (int i = 0; i < SCHEDULER_COUNT; ++i) {
        schedulers.push_back(sd_scheduler_name((scheduler_t)i));
        if (i == DISCRETE_SCHEDULER) {
            schedulers.push_back("normal");
        }
    }

    {
        std::lock_guard<std::mutex> lock(*runtime.lora_mutex);
        for (const auto& entry : *runtime.lora_cache) {
            available_loras.push_back({
                {"name", entry.name},
                {"path", entry.path},
            });
        }
    }

    available_upscalers.push_back({
        {"name", "None"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Lanczos"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Nearest"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest-exact)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (antialiased)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic antialiased)"},
        {"model", false},
        {"image_upscale", false},
    });
    bool have_upscaler_models = false;
    {
        std::lock_guard<std::mutex> lock(*runtime.upscaler_mutex);
        for (const auto& entry : *runtime.upscaler_cache) {
            available_upscalers.push_back({
                {"name", entry.name},
                {"model", true},
                {"image_upscale", entry.image_upscale_factor > 0},
            });
            have_upscaler_models = have_upscaler_models || entry.image_upscale_factor > 0;
        }
    }

    if (supports_img) {
        supported_modes.push_back("img_gen");
    }
    if (supports_vid) {
        supported_modes.push_back("vid_gen");
    }

    std::string default_img_output_format = "png";
    std::string default_vid_output_format = "avi";
    if (!image_output_formats.empty()) {
        default_img_output_format = image_output_formats[0].get<std::string>();
    }
    if (!video_output_formats.empty()) {
        default_vid_output_format = video_output_formats[0].get<std::string>();
    }

    json defaults_by_mode       = json::object();
    json output_formats_by_mode = json::object();
    json features_by_mode       = json::object();
    if (supports_img) {
        defaults_by_mode["img_gen"]       = make_img_gen_defaults_json(defaults, default_img_output_format);
        output_formats_by_mode["img_gen"] = image_output_formats;
        features_by_mode["img_gen"]       = make_img_gen_features_json();
    }
    if (supports_vid) {
        defaults_by_mode["vid_gen"]       = make_vid_gen_defaults_json(defaults, default_vid_output_format);
        output_formats_by_mode["vid_gen"] = video_output_formats;
        features_by_mode["vid_gen"]       = make_vid_gen_features_json();
    }

    json top_level_defaults       = json::object();
    json top_level_output_formats = json::array();
    json top_level_features       = {
              {"cancel_queued", true},
              {"cancel_generating", false},
    };
    std::string current_mode = "";
    if (supports_img) {
        current_mode             = "img_gen";
        top_level_defaults       = defaults_by_mode["img_gen"];
        top_level_output_formats = output_formats_by_mode["img_gen"];
        top_level_features       = features_by_mode["img_gen"];
    } else if (supports_vid) {
        current_mode             = "vid_gen";
        top_level_defaults       = defaults_by_mode["vid_gen"];
        top_level_output_formats = output_formats_by_mode["vid_gen"];
        top_level_features       = features_by_mode["vid_gen"];
    }

    json result;
    result["model"] = {
        {"name", model_path.filename().u8string()},
        {"stem", model_path.stem().u8string()},
        {"path", model_path.u8string()},
    };
    result["current_mode"]     = current_mode;
    result["supported_modes"]  = supported_modes;
    result["defaults"]         = top_level_defaults;
    result["defaults_by_mode"] = defaults_by_mode;
    result["limits"]           = {
                  {"min_width", 64},
                  {"max_width", 4096},
                  {"min_height", 64},
                  {"max_height", 4096},
                  {"max_batch_count", 8},
                  {"max_queue_size", manager.max_pending_jobs},
                  {"max_upscale_width", k_max_upscale_dimension},
                  {"max_upscale_height", k_max_upscale_dimension},
    };
    result["samplers"]               = samplers;
    result["schedulers"]             = schedulers;
    result["output_formats"]         = top_level_output_formats;
    result["output_formats_by_mode"] = output_formats_by_mode;
    result["features"]               = top_level_features;
    result["features_by_mode"]       = features_by_mode;
    result["loras"]                  = available_loras;
    result["upscalers"]              = available_upscalers;
    result["upscale"]                = have_upscaler_models;
    result["preview_modes"]          = make_preview_modes_json();
    return result;
}

static void remove_ui_image_fields(json& form) {
    for (const char* key : {"init_image", "end_image", "ref_images", "control_frames", "mask_image", "control_image"}) {
        form.erase(key);
    }
}

static json make_ui_settings_snapshot(const json& body, const char* generation_mode) {
    json supplied = body.value("ui_settings", json::object());
    json form = supplied.is_object() && supplied.contains("form") && supplied["form"].is_object()
                    ? supplied["form"]
                    : json::object();
    remove_ui_image_fields(form);
    return {{"form", std::move(form)}, {"generation_mode", generation_mode}};
}

static bool valid_webui_settings(const json& settings) {
    return settings.is_object() && settings.contains("form") && settings["form"].is_object() &&
           settings.contains("generation_mode") && settings["generation_mode"].is_string() &&
           (settings["generation_mode"] == "image" || settings["generation_mode"] == "video");
}

static bool valid_history_stem(const std::string& stem) {
    if (stem.size() != 19) {
        return false;
    }
    for (size_t i = 0; i < stem.size(); ++i) {
        const bool separator = i == 4 || i == 7 || i == 10 || i == 13 || i == 16;
        if (separator ? stem[i] != '-' : (stem[i] < '0' || stem[i] > '9')) {
            return false;
        }
    }
    return true;
}

static bool parse_img_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  ImgGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    json generation_body = body;
    generation_body.erase("ui_settings");
    if (!request.gen_params.from_json_str(generation_body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    std::string output_format = body.value("output_format", "png");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, true, error_message)) {
        return false;
    }
    request.preview_mode     = body.value("preview", std::string("none"));
    request.preview_interval = body.value("preview_interval", 1);
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

static bool parse_vid_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  VidGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    json generation_body = body;
    generation_body.erase("ui_settings");
    if (!request.gen_params.from_json_str(generation_body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    std::string output_format = body.value("output_format", "webm");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, error_message)) {
        return false;
    }
    request.preview_mode     = body.value("preview", std::string("none"));
    request.preview_interval = body.value("preview_interval", 1);
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    static std::mutex webui_settings_mutex;
    const fs::path webui_settings_path = "sdcpp-webui-settings.json";
    const fs::path webui_presets_path = "sdcpp-webui-presets.json";
    const fs::path history_dir = "sdoutput";

    svr.Get("/sdcpp/v1/capabilities", [runtime](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(make_capabilities_json(*runtime).dump(), "application/json");
    });

    svr.Get("/sdcpp/v1/settings", [webui_settings_path](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(webui_settings_mutex);
        std::ifstream file(webui_settings_path, std::ios::binary);
        if (!file) {
            std::error_code error;
            if (!fs::exists(webui_settings_path, error) && !error) {
                res.status = 200;
                res.set_content(R"({"form":{},"generation_mode":"image"})", "application/json");
            } else {
                res.status = 500;
                res.set_content(R"({"error":"unable to read saved UI settings"})", "application/json");
            }
            return;
        }

        try {
            json settings = json::parse(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            if (!settings.is_object() || !settings.contains("form") || !settings["form"].is_object() ||
                !settings.contains("generation_mode") || !settings["generation_mode"].is_string() ||
                (settings["generation_mode"] != "image" && settings["generation_mode"] != "video")) {
                res.status = 500;
                res.set_content(R"({"error":"saved UI settings are invalid"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(settings.dump(), "application/json");
        } catch (const json::exception&) {
            res.status = 500;
            res.set_content(R"({"error":"saved UI settings are invalid"})", "application/json");
        }
    });

    svr.Post("/sdcpp/v1/settings", [webui_settings_path](const httplib::Request& req, httplib::Response& res) {
        if (req.body.size() > k_max_webui_settings_size) {
            res.status = 413;
            res.set_content(R"({"error":"settings payload is too large"})", "application/json");
            return;
        }
        try {
            json body = json::parse(req.body);
            if (!body.is_object() || !body.contains("form") || !body["form"].is_object() ||
                !body.contains("generation_mode") || !body["generation_mode"].is_string()) {
                res.status = 400;
                res.set_content(R"({"error":"settings must contain a form object and generation_mode"})", "application/json");
                return;
            }

            const std::string generation_mode = body["generation_mode"].get<std::string>();
            if (generation_mode != "image" && generation_mode != "video") {
                res.status = 400;
                res.set_content(R"({"error":"generation_mode must be image or video"})", "application/json");
                return;
            }

            json form = body["form"];
            for (const char* key : {"init_image", "end_image", "ref_images", "control_frames", "mask_image",
                                    "control_image"}) {
                form.erase(key);
            }
            json settings = {{"form", std::move(form)}, {"generation_mode", generation_mode}};

            std::lock_guard<std::mutex> lock(webui_settings_mutex);
            std::ofstream file(webui_settings_path, std::ios::binary | std::ios::trunc);
            if (!file) {
                res.status = 500;
                res.set_content(R"({"error":"unable to save UI settings"})", "application/json");
                return;
            }
            file << settings.dump(2);
            file.flush();
            if (!file) {
                res.status = 500;
                res.set_content(R"({"error":"unable to save UI settings"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(R"({"saved":true})", "application/json");
        } catch (const json::exception&) {
            res.status = 400;
            res.set_content(R"({"error":"invalid settings JSON"})", "application/json");
        }
    });

    svr.Get("/sdcpp/v1/settings/presets", [webui_presets_path](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(webui_settings_mutex);
        std::ifstream file(webui_presets_path, std::ios::binary);
        json stored_presets = json::object();
        if (!file) {
            std::error_code error;
            if (fs::exists(webui_presets_path, error) || error) {
                res.status = 500;
                res.set_content(R"({"error":"unable to read saved settings presets"})", "application/json");
                return;
            }
        } else {
            try {
                stored_presets = json::parse(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            } catch (const json::exception&) {
                res.status = 500;
                res.set_content(R"({"error":"saved settings presets are invalid"})", "application/json");
                return;
            }
        }
        if (!stored_presets.is_object()) {
            res.status = 500;
            res.set_content(R"({"error":"saved settings presets are invalid"})", "application/json");
            return;
        }

        json presets = json::array();
        for (auto it = stored_presets.begin(); it != stored_presets.end(); ++it) {
            if (valid_webui_settings(it.value())) {
                presets.push_back({{"name", it.key()}, {"settings", it.value()}});
            }
        }
        res.status = 200;
        res.set_content(json({{"presets", presets}}).dump(), "application/json");
    });

    svr.Post("/sdcpp/v1/settings/presets", [webui_presets_path](const httplib::Request& req, httplib::Response& res) {
        if (req.body.size() > k_max_webui_settings_size) {
            res.status = 413;
            res.set_content(R"({"error":"settings preset is too large"})", "application/json");
            return;
        }
        try {
            json body = json::parse(req.body);
            if (!body.is_object() || !body.contains("name") || !body["name"].is_string() ||
                !body.contains("settings") || !valid_webui_settings(body["settings"])) {
                res.status = 400;
                res.set_content(R"({"error":"preset must contain a name and valid settings"})", "application/json");
                return;
            }
            const std::string name = body["name"].get<std::string>();
            if (name.empty() || name.size() > 80 || name.find_first_not_of(" \t\r\n") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error":"preset name must contain 1 to 80 non-space characters"})", "application/json");
                return;
            }

            json settings = body["settings"];
            remove_ui_image_fields(settings["form"]);

            std::lock_guard<std::mutex> lock(webui_settings_mutex);
            json presets = json::object();
            std::ifstream existing_file(webui_presets_path, std::ios::binary);
            if (existing_file) {
                presets = json::parse(std::istreambuf_iterator<char>(existing_file), std::istreambuf_iterator<char>());
                if (!presets.is_object()) {
                    res.status = 500;
                    res.set_content(R"({"error":"saved settings presets are invalid"})", "application/json");
                    return;
                }
            } else {
                std::error_code error;
                if (fs::exists(webui_presets_path, error) || error) {
                    res.status = 500;
                    res.set_content(R"({"error":"unable to read saved settings presets"})", "application/json");
                    return;
                }
            }
            presets[name] = std::move(settings);

            std::ofstream file(webui_presets_path, std::ios::binary | std::ios::trunc);
            if (!file) {
                res.status = 500;
                res.set_content(R"({"error":"unable to save settings preset"})", "application/json");
                return;
            }
            file << presets.dump(2);
            file.flush();
            if (!file) {
                res.status = 500;
                res.set_content(R"({"error":"unable to save settings preset"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(json({{"saved", true}, {"name", name}}).dump(), "application/json");
        } catch (const json::exception&) {
            res.status = 400;
            res.set_content(R"({"error":"invalid settings preset JSON"})", "application/json");
        }
    });

    svr.Get("/sdcpp/v1/history", [history_dir](const httplib::Request&, httplib::Response& res) {
        std::error_code error;
        if (!fs::exists(history_dir, error)) {
            if (error) {
                res.status = 500;
                res.set_content(R"({"error":"unable to read image history"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(R"({"images":[]})", "application/json");
            return;
        }

        std::vector<std::string> filenames;
        fs::directory_iterator it(history_dir, error);
        const fs::directory_iterator end;
        for (; !error && it != end; it.increment(error)) {
            if (!it->is_regular_file(error) || error) {
                continue;
            }
            const fs::path path = it->path();
            const std::string stem = path.stem().string();
            if (path.extension() != ".png" || !valid_history_stem(stem)) {
                continue;
            }
            fs::path settings_path = path;
            settings_path.replace_extension(".json");
            if (fs::is_regular_file(settings_path, error) && !error) {
                filenames.push_back(path.filename().string());
            }
        }
        if (error) {
            res.status = 500;
            res.set_content(R"({"error":"unable to read image history"})", "application/json");
            return;
        }
        std::sort(filenames.begin(), filenames.end(), std::greater<std::string>());
        json images = json::array();
        for (const auto& filename : filenames) {
            images.push_back({{"filename", filename}});
        }
        res.status = 200;
        res.set_content(json({{"images", images}}).dump(), "application/json");
    });

    svr.Get(R"(/sdcpp/v1/history/([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})/settings)",
            [history_dir](const httplib::Request& req, httplib::Response& res) {
                const std::string stem = req.matches[1].str();
                if (!valid_history_stem(stem)) {
                    res.status = 404;
                    res.set_content(R"({"error":"history entry not found"})", "application/json");
                    return;
                }
                std::ifstream file(history_dir / (stem + ".json"), std::ios::binary);
                if (!file) {
                    res.status = 404;
                    res.set_content(R"({"error":"history settings not found"})", "application/json");
                    return;
                }
                try {
                    json settings = json::parse(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
                    if (!valid_webui_settings(settings)) {
                        res.status = 500;
                        res.set_content(R"({"error":"history settings are invalid"})", "application/json");
                        return;
                    }
                    res.status = 200;
                    res.set_content(settings.dump(), "application/json");
                } catch (const json::exception&) {
                    res.status = 500;
                    res.set_content(R"({"error":"history settings are invalid"})", "application/json");
                }
            });

    svr.Get(R"(/sdcpp/v1/history/([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.png))",
            [history_dir](const httplib::Request& req, httplib::Response& res) {
                const std::string filename = req.matches[1].str();
                const std::string stem = fs::path(filename).stem().string();
                if (!valid_history_stem(stem)) {
                    res.status = 404;
                    res.set_content("history image not found", "text/plain");
                    return;
                }
                const fs::path image_path = history_dir / filename;
                if (req.has_param("thumbnail")) {
                    int width = 0;
                    int height = 0;
                    int channels = 0;
                    FreeUniquePtr<uint8_t> image(load_image_from_file(image_path.string().c_str(), width, height, channels, 256, 256, 3));
                    if (!image) {
                        res.status = 404;
                        res.set_content("history image not found", "text/plain");
                        return;
                    }
                    const auto bytes = encode_image_to_vector(EncodedImageFormat::JPEG, image.get(), width, height, channels, "", 82);
                    if (bytes.empty()) {
                        res.status = 500;
                        res.set_content("unable to create image thumbnail", "text/plain");
                        return;
                    }
                    res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/jpeg");
                    return;
                }

                std::ifstream file(image_path, std::ios::binary);
                if (!file) {
                    res.status = 404;
                    res.set_content("history image not found", "text/plain");
                    return;
                }
                std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
                res.set_content(bytes, "image/png");
            });

    svr.Delete(R"(/sdcpp/v1/history/([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.png))",
               [history_dir](const httplib::Request& req, httplib::Response& res) {
                   const std::string filename = req.matches[1].str();
                   const std::string stem = fs::path(filename).stem().string();
                   if (!valid_history_stem(stem)) {
                       res.status = 404;
                       res.set_content(R"({"error":"history entry not found"})", "application/json");
                       return;
                   }

                   const fs::path image_path = history_dir / filename;
                   std::error_code error;
                   if (!fs::is_regular_file(image_path, error)) {
                       res.status = error ? 500 : 404;
                       res.set_content(error ? R"({"error":"unable to inspect history image"})"
                                             : R"({"error":"history image not found"})",
                                       "application/json");
                       return;
                   }
                   if (!fs::remove(image_path, error) || error) {
                       res.status = 500;
                       res.set_content(R"({"error":"unable to delete history image"})", "application/json");
                       return;
                   }

                   fs::path settings_path = image_path;
                   settings_path.replace_extension(".json");
                   fs::remove(settings_path, error);
                   if (error) {
                       res.status = 500;
                       res.set_content(R"({"error":"image deleted but settings sidecar could not be removed"})",
                                       "application/json");
                       return;
                   }
                   res.status = 200;
                   res.set_content(R"({"deleted":true})", "application/json");
               });

    svr.Post("/sdcpp/v1/upscale", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            json body = json::parse(req.body);
            if (!body.is_object()) {
                res.status = 400;
                res.set_content(R"({"error":"body must be an object"})", "application/json");
                return;
            }
            for (const char* key : {"repeats", "tile_size", "output_compression"}) {
                if (!body.contains(key)) {
                    continue;
                }
                const auto& value = body[key];
                const bool valid  = value.is_number_unsigned()
                                        ? value.get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<int>::max())
                                        : value.is_number_integer() && value.get<int64_t>() >= std::numeric_limits<int>::min() &&
                                             value.get<int64_t>() <= std::numeric_limits<int>::max();
                if (!valid) {
                    res.status = 400;
                    res.set_content(json({{"error", std::string(key) + " must be a 32-bit integer"}}).dump(), "application/json");
                    return;
                }
            }
            ImgGenJobRequest output_options;
            std::string error_message;
            if (!assign_output_options(output_options,
                                       body.value("output_format", std::string("png")),
                                       body.value("output_compression", 100),
                                       true,
                                       error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            const int tile_size      = std::max(32, body.value("tile_size", runtime->default_gen_params->upscale_tile_size));
            const int repeats        = std::clamp(body.value("repeats", 1), 1, 4);
            const std::string wanted = body.value("upscaler", std::string());

            const std::string encoded = body.value("image", std::string());
            if (encoded.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"image is required"})", "application/json");
                return;
            }
            SDImageOwner input;
            if (!decode_base64_image(encoded, 3, 0, 0, input) || input.get().data == nullptr) {
                res.status = 400;
                res.set_content(R"({"error":"image could not be read"})", "application/json");
                return;
            }

            refresh_upscaler_cache(*runtime);
            int model_scale = 0;
            std::string model_path;
            std::string used_name;
            {
                std::lock_guard<std::mutex> lock(*runtime->upscaler_mutex);
                for (const auto& entry : *runtime->upscaler_cache) {
                    if (entry.image_upscale_factor > 0 && (wanted.empty() || entry.name == wanted)) {
                        model_path  = entry.fullpath;
                        used_name   = entry.name;
                        model_scale = entry.image_upscale_factor;
                        break;
                    }
                }
            }
            if (model_path.empty()) {
                res.status = 400;
                res.set_content(json({{"error", wanted.empty()
                                                    ? std::string("no RGB ESRGAN upscaler models are available; "
                                                                  "start the server with --hires-upscalers-dir")
                                                    : "no compatible image upscaler called " + wanted}})
                                    .dump(),
                                "application/json");
                return;
            }

            if (!valid_upscale_dimensions(input.get(), model_scale, repeats)) {
                res.status = 400;
                res.set_content(R"({"error":"upscaled dimensions must not exceed 8192 x 8192"})", "application/json");
                return;
            }

            // One GPU: an upscale must not run while a generation is using it.
            std::lock_guard<std::mutex> ctx_lock(*runtime->sd_ctx_mutex);
            UpscalerCtxPtr upscaler_ctx(new_upscaler_ctx(model_path.c_str(),
                                                         runtime->ctx_params->diffusion_conv_direct,
                                                         runtime->ctx_params->n_threads,
                                                         tile_size,
                                                         runtime->ctx_params->backend.c_str(),
                                                         runtime->ctx_params->params_backend.c_str()));
            if (upscaler_ctx == nullptr) {
                res.status = 500;
                res.set_content(R"({"error":"the upscaler model could not be loaded"})", "application/json");
                return;
            }
            const int factor = get_upscale_factor(upscaler_ctx.get());
            // The model file may have changed since its metadata was cached.
            if (!valid_upscale_dimensions(input.get(), factor, repeats)) {
                res.status = 400;
                res.set_content(R"({"error":"upscaled dimensions must not exceed 8192 x 8192"})", "application/json");
                return;
            }

            SDImageOwner current(input.release());
            for (int i = 0; i < repeats; ++i) {
                sd_image_t* out_images = nullptr;
                int out_count          = 0;
                if (!upscale(upscaler_ctx.get(), current.get(), (uint32_t)factor, &out_images, &out_count) ||
                    out_count <= 0 || out_images[0].data == nullptr) {
                    free_sd_images(out_images, out_count);
                    res.status = 500;
                    res.set_content(R"({"error":"upscale failed"})", "application/json");
                    return;
                }
                sd_image_t produced = out_images[0];
                out_images[0]       = {0, 0, 0, nullptr};
                free_sd_images(out_images, out_count);
                current.reset(produced);
            }

            const std::string& format = output_options.output_format;
            const int compression     = output_options.output_compression;
            const sd_image_t result   = current.get();
            auto image_bytes          = encode_image_to_vector(format == "jpeg"   ? EncodedImageFormat::JPEG
                                                               : format == "webp" ? EncodedImageFormat::WEBP
                                                                                  : EncodedImageFormat::PNG,
                                                      result.data,
                                                      result.width,
                                                      result.height,
                                                      result.channel,
                                                      "",
                                                      compression);
            if (image_bytes.empty()) {
                res.status = 500;
                res.set_content(R"({"error":"the result could not be encoded"})", "application/json");
                return;
            }

            json out;
            out["upscaler"]      = used_name;
            out["scale"]         = factor;
            out["repeats"]       = repeats;
            out["width"]         = result.width;
            out["height"]        = result.height;
            out["output_format"] = format;
            json images          = json::array();
            images.push_back({{"index", 0}, {"b64_json", base64_encode(image_bytes)}});
            out["images"] = std::move(images);
            res.set_content(out.dump(), "application/json");
            res.status = 200;
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid request"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", std::string("server_error: ") + e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/sdcpp/v1/img_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, IMG_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(IMG_GEN)}}).dump(), "application/json");
                return;
            }

            json body = json::parse(req.body);
            ImgGenJobRequest request;
            std::string error_message;
            if (!parse_img_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::ImgGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now();
            job->img_gen                            = std::move(request);
            job->ui_settings_snapshot               = make_ui_settings_snapshot(body, "image");

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/sdcpp/v1/vid_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, VID_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(VID_GEN)}}).dump(), "application/json");
                return;
            }

            json body = json::parse(req.body);
            VidGenJobRequest request;
            std::string error_message;
            if (!parse_vid_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::VidGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now();
            job->vid_gen                            = std::move(request);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    svr.Get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, *it->second).dump(), "application/json");
    });

    svr.Post(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/cancel)", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        auto& job = *it->second;
        if (job.status == AsyncJobStatus::Queued) {
            if (!cancel_queued_job(manager, job)) {
                res.status = 409;
                res.set_content(R"({"error":"job queue state changed before cancellation"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(make_async_job_json(manager, job).dump(), "application/json");
            return;
        }

        if (job.status == AsyncJobStatus::Generating) {
            res.status = 409;
            res.set_content(R"({"error":"job is currently generating and cannot be interrupted yet"})", "application/json");
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, job).dump(), "application/json");
    });
}
