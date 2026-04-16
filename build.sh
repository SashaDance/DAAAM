#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-daaam:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
HOST_ROS_DISTRO="${ROS_DISTRO:-}"
ROS_DISTRO="${DAAAM_ROS_DISTRO:-jazzy}"
COLCON_PARALLEL_WORKERS="${COLCON_PARALLEL_WORKERS:-1}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"

if [[ -n "${HOST_ROS_DISTRO}" && -z "${DAAAM_ROS_DISTRO:-}" && "${HOST_ROS_DISTRO}" != "jazzy" ]]; then
  printf 'Ignoring host ROS_DISTRO=%s for Docker build; using DAAAM_ROS_DISTRO=jazzy.\n' "${HOST_ROS_DISTRO}" >&2
  printf 'Set DAAAM_ROS_DISTRO explicitly if you really want a different ROS distro.\n' >&2
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  --build-arg COLCON_PARALLEL_WORKERS="${COLCON_PARALLEL_WORKERS}" \
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
  --tag "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"
