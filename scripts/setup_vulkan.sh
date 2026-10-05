#!/bin/bash
# Make SAPIEN/ManiSkill GPU camera rendering work on a cluster node. Source this file, then call:
#   setup_vulkan <env_id>
# with the conda env already active. It exports VK_ICD_FILENAMES / VK_DRIVER_FILES (and LD_LIBRARY_PATH
# when needed) and returns non-zero if no working setup is found.
#
# Tried in order, keeping the first one that passes a 2-env render check:
#   1. VK_ICD_FILENAMES already set by the user
#   2. the node's NVIDIA ICD, with every other ICD (lavapipe, nouveau, intel, ...) ignored. Nodes with many
#      ICDs can make SAPIEN pick a non-NVIDIA device, and CUDA interop with it then hangs.
#   3. NVIDIA user-space driver libraries matching the node's kernel driver, downloaded once into
#      NVIDIA_DRIVER_CACHE (default ~/.cache/nvidia-vulkan). Covers nodes without the NVIDIA graphics libs.

NVIDIA_DRIVER_CACHE="${NVIDIA_DRIVER_CACHE:-${HOME}/.cache/nvidia-vulkan}"

_use_icd() {
    export VK_ICD_FILENAMES="$1"   # older Vulkan loaders
    export VK_DRIVER_FILES="$1"    # newer Vulkan loaders
    echo "[vulkan] using ICD $1 ($(grep -o '"library_path"[^,}]*' "$1"))"
}

# Render check: create a tiny env with cameras. The kill-after is needed because a hang inside the driver
# can ignore SIGTERM.
_render_ok() {
    timeout -k 10 180 python - "$1" <<'EOF'
import sys, time
import gymnasium as gym
import mani_skill.envs  # noqa: F401
t0 = time.time()
env = gym.make(sys.argv[1], num_envs=2, obs_mode="rgb+segmentation", sim_backend="physx_cuda")
env.reset(seed=0)
env.close()
print(f"[vulkan] render check passed in {time.time() - t0:.1f}s")
EOF
}

# Returns 0 if the ICD's library can be loaded (absolute path exists, or bare name is in the linker cache).
_icd_lib_present() {
    local lib
    lib=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['ICD']['library_path'])" "$1" 2>/dev/null) || return 1
    if [[ "${lib}" == /* ]]; then
        [[ -f "${lib}" ]]
    else
        { ldconfig -p || /sbin/ldconfig -p; } 2>/dev/null | grep -q "${lib}" || python -c "import ctypes,sys; ctypes.CDLL(sys.argv[1])" "${lib}" 2>/dev/null
    fi
}

# Download and extract the NVIDIA driver's user-space libs for this node's driver version (no root needed).
_install_user_driver() {
    local ver dir run url
    ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
    [[ -n "${ver}" ]] || { echo "[vulkan] could not read the driver version from nvidia-smi" >&2; return 1; }
    dir="${NVIDIA_DRIVER_CACHE}/${ver}"
    mkdir -p "${NVIDIA_DRIVER_CACHE}"
    (
        flock 8
        if [[ -f "${dir}/nvidia_icd.json" ]]; then exit 0; fi
        echo "[vulkan] installing NVIDIA ${ver} user-space libs into ${dir}"
        run="${NVIDIA_DRIVER_CACHE}/NVIDIA-Linux-x86_64-${ver}.run"
        for url in "https://us.download.nvidia.com/tesla/${ver}/NVIDIA-Linux-x86_64-${ver}.run" \
                   "https://us.download.nvidia.com/XFree86/Linux-x86_64/${ver}/NVIDIA-Linux-x86_64-${ver}.run"; do
            curl -fL --retry 3 -o "${run}.part" "${url}" && mv "${run}.part" "${run}" && break
        done
        [[ -f "${run}" ]] || { echo "[vulkan] could not download driver ${ver} from download.nvidia.com" >&2; exit 1; }
        rm -rf "${dir}"
        sh "${run}" --extract-only --target "${dir}" >/dev/null || exit 1
        rm -f "${run}"
        # keep CUDA and NVML coming from the system install
        rm -f "${dir}"/libcuda.so* "${dir}"/libnvidia-ml.so*
        # soname symlinks; works without root, but ldconfig is often only in /sbin
        { ldconfig -n "${dir}" || /sbin/ldconfig -n "${dir}"; } 2>/dev/null
        cat > "${dir}/nvidia_icd.json" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${dir}/libGLX_nvidia.so.${ver}",
        "api_version" : "1.3"
    }
}
EOF
    ) 8>"${NVIDIA_DRIVER_CACHE}/install.lock" || return 1
    export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    _use_icd "${dir}/nvidia_icd.json"
}

setup_vulkan() {
    local env_id="$1" icd
    echo "[vulkan] ICDs on $(hostname -s): $(ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | grep json | tr '\n' ' ')"

    if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
        _use_icd "${VK_ICD_FILENAMES}"
        _render_ok "${env_id}" && return 0
        echo "[vulkan] user-provided VK_ICD_FILENAMES failed the render check" >&2
        return 1
    fi

    for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
        [[ -f "${icd}" ]] || continue
        if _icd_lib_present "${icd}"; then
            _use_icd "${icd}"
            _render_ok "${env_id}" && return 0
            echo "[vulkan] system NVIDIA ICD failed the render check" >&2
        else
            echo "[vulkan] ${icd} points to a library that is not installed on this node"
        fi
        break
    done

    echo "[vulkan] falling back to user-space NVIDIA driver libs"
    _install_user_driver || return 1
    _render_ok "${env_id}" && return 0
    echo "[vulkan] user-space NVIDIA driver also failed the render check" >&2
    return 1
}
