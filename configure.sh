#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

backend="CPU"
sd_cuda=OFF
sd_metal=OFF
sd_vulkan=OFF

has_vulkan_build_tools() {
    if [[ -n "${VULKAN_SDK:-}" && -x "${VULKAN_SDK}/bin/glslc" ]]; then
        return 0
    fi
    command -v glslc >/dev/null 2>&1 || return 1
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan; then
        return 0
    fi
    [[ -r /usr/include/vulkan/vulkan.h || -r /usr/local/include/vulkan/vulkan.h ]]
}

if [[ "$(uname -s)" == Darwin ]] && command -v xcrun >/dev/null 2>&1 && xcrun --find metal >/dev/null 2>&1; then
    backend="Metal"
    sd_metal=ON
elif command -v "${CUDACXX:-nvcc}" >/dev/null 2>&1; then
    backend="CUDA"
    sd_cuda=ON
elif has_vulkan_build_tools; then
    backend="Vulkan"
    sd_vulkan=ON
fi

printf 'Configuring for %s\n' "$backend"
cmake -S . -B build \
    "-DSD_CUDA=${sd_cuda}" \
    "-DSD_METAL=${sd_metal}" \
    "-DSD_VULKAN=${sd_vulkan}" \
    "-DGGML_CUDA=${sd_cuda}" \
    "-DGGML_METAL=${sd_metal}" \
    "-DGGML_VULKAN=${sd_vulkan}"
