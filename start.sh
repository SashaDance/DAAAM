#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-daaam:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-daaam}"
DATA_DIR="${DATA_DIR:-}"

DOCKER_ARGS=(
  --rm
  --interactive
  --tty
  --gpus all
  --ipc host
  --network host
  --name "${CONTAINER_NAME}"
  --workdir /opt/daaam_ws/src/daaam
  --volume "${SCRIPT_DIR}:/opt/daaam_ws/src/daaam:rw"
  --env NVIDIA_VISIBLE_DEVICES=all
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
)

for env_name in DISPLAY ROS_DOMAIN_ID RMW_IMPLEMENTATION OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY; do
  if [[ -n "${!env_name:-}" ]]; then
    DOCKER_ARGS+=(--env "${env_name}=${!env_name}")
  fi
done

if [[ -d /tmp/.X11-unix ]]; then
  DOCKER_ARGS+=(--volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

if [[ -f "${HOME}/.Xauthority" ]]; then
  DOCKER_ARGS+=(--env XAUTHORITY=/root/.Xauthority)
  DOCKER_ARGS+=(--volume "${HOME}/.Xauthority:/root/.Xauthority:ro")
fi

if [[ -d "${HOME}/.cache/huggingface" ]]; then
  DOCKER_ARGS+=(--volume "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw")
fi

if [[ -d "${HOME}/.cache/torch" ]]; then
  DOCKER_ARGS+=(--volume "${HOME}/.cache/torch:/root/.cache/torch:rw")
fi

if [[ -n "${DATA_DIR}" ]]; then
  if [[ ! -d "${DATA_DIR}" ]]; then
    printf 'DATA_DIR does not exist: %s\n' "${DATA_DIR}" >&2
    exit 1
  fi

  DOCKER_ARGS+=(--volume "${DATA_DIR}:/data:rw")
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" \
  bash -lc 'source /opt/ros/${ROS_DISTRO}/setup.bash && source /opt/daaam_ws/install/setup.bash && exec bash -i'
