#ifndef __SD_STABLE_DIFFUSION_INTERNAL_H__
#define __SD_STABLE_DIFFUSION_INTERNAL_H__

#include <functional>

#include "core/media.h"
#include "json.hpp"
#include "stable-diffusion.h"

namespace sd::pipeline {
    using ImageOutputCallback = std::function<void(const sd::Image&, size_t, size_t, const nlohmann::json&)>;
}

SD_API sd_status_t sd_generate_image_with_callback(sd_ctx_t* ctx,
                                                   const sd_img_gen_opt_t* params,
                                                   const sd::pipeline::ImageOutputCallback& callback);

#endif  // __SD_STABLE_DIFFUSION_INTERNAL_H__
