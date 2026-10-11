#pragma once

#include "stable-diffusion.h"

using sd_image_output_callback_t = void (*)(const sd_image_t* image,
                                           size_t output_index,
                                           size_t batch_index,
                                           void* user_data);

SD_API bool sd_generate_image_with_callback(sd_ctx_t* sd_ctx,
                                            const sd_img_gen_params_t* params,
                                            int* num_images_out,
                                            sd_image_output_callback_t callback,
                                            void* user_data);
