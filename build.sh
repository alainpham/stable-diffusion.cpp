#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

build_jobs=""
if command -v lscpu >/dev/null 2>&1; then
    build_jobs="$(lscpu --parse=CORE,SOCKET 2>/dev/null | awk -F, '
        !/^#/ && NF >= 2 { cores[$1 "," $2] = 1 }
        END {
            for (core in cores) count++
            if (count > 0) print count
        }
    ')"
fi

if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    build_jobs="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
fi

printf 'Building with %s jobs\n' "$build_jobs"
cmake --build "build" --config Release -j "$build_jobs"
