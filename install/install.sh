#!/usr/bin/env bash
# DAAAM workspace installer
# Usage:
#   cd ~/ros2_ws/src
#   git clone git@github.com:MIT-SPARK/DAAAM.git daaam
#   bash daaam/install/install.sh
#
# Assumes: ROS2 Iron+ sourced
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS_ROOT="$(cd "$WS_SRC/.." && pwd)"

echo "=== DAAAM workspace installer ==="
echo "  workspace : $WS_ROOT"
echo "  src       : $WS_SRC"
echo ""

# --- 0.5 Install ROS2 Prereqs ---
echo "--- [0.5/6] Installing prerequisite Python packages ---"
pip install pyem catkin_pkg lark catkin_pkg pyyaml

# --- 1. System dependencies ---
echo "--- [1/6] System dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
	python3-vcstool python3-tk libgoogle-glog-dev \
	nlohmann-json3-dev glpk-utils libglpk-dev ros-dev-tools

# --- 2. Clone workspace repos ---
echo "--- [2/6] Importing repos (vcs) ---"
cd "$WS_SRC"
vcs import . < "$SCRIPT_DIR/packages.yaml" --workers 1 --skip-existing

# --- 3. Rosdep ---
echo "--- [3/6] Rosdep ---"
rosdep update --rosdistro="${ROS_DISTRO:-jazzy}" || true
rosdep install --from-paths "$WS_SRC" --ignore-src -r -y

# --- 4. small_gicp ---
echo "--- [4/6] small_gicp ---"
COLCON_DEFAULTS="$WS_ROOT/colcon_defaults.yaml"
COLCON_PARALLEL_WORKERS="${COLCON_PARALLEL_WORKERS:-1}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
SMALL_GICP_BUILD_DIR="${SMALL_GICP_BUILD_DIR:-/tmp/daaam-small_gicp-build}"
if [ ! -f "$COLCON_DEFAULTS" ]; then
	cat > "$COLCON_DEFAULTS" <<'YAML'
---
build:
  symlink-install: true
  parallel-workers: 1
  packages-skip: [small_gicp, khronos_msgs, khronos_ros, khronos_eval, hydra_multi_ros, spark_fast_lio, ouroboros_ros, ouroboros_msgs]
  cmake-args:
    - --no-warn-unused-cli
    - -DCMAKE_BUILD_TYPE=RelWithDebInfo
    - -DCONFIG_UTILS_ENABLE_ROS=OFF
    - -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    - -DGTSAM_USE_SYSTEM_EIGEN=ON
YAML
	echo "  wrote $COLCON_DEFAULTS"
fi
cmake -S "$WS_SRC/small_gicp" -B "$SMALL_GICP_BUILD_DIR" \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build "$SMALL_GICP_BUILD_DIR" -j "$CMAKE_BUILD_PARALLEL_LEVEL"
sudo cmake --install "$SMALL_GICP_BUILD_DIR"
rm -rf "$SMALL_GICP_BUILD_DIR"

# --- 5. Colcon build ---
echo "--- [5/6] Colcon build ---"
cd "$WS_ROOT"
echo "  package workers : $COLCON_PARALLEL_WORKERS"
echo "  compiler jobs   : $CMAKE_BUILD_PARALLEL_LEVEL"
CMAKE_BUILD_PARALLEL_LEVEL="$CMAKE_BUILD_PARALLEL_LEVEL" \
	colcon build --continue-on-error --parallel-workers "$COLCON_PARALLEL_WORKERS" \
	--packages-skip small_gicp khronos khronos_msgs khronos_ros khronos_eval hydra_multi_ros spark_fast_lio ouroboros_ros ouroboros_msgs
source "$WS_ROOT/install/setup.bash"
CMAKE_BUILD_PARALLEL_LEVEL="$CMAKE_BUILD_PARALLEL_LEVEL" \
	colcon build --continue-on-error --parallel-workers "$COLCON_PARALLEL_WORKERS" --packages-select khronos

# --- 6. Python package ---
echo "--- [6/6] pip install ---"
pip install -r "$WS_SRC/daaam/requirements.txt"
pip install -e "$WS_SRC/spark_dsg"
pip install -e "$WS_SRC/daaam"

echo ""
echo "=== Done. source $WS_ROOT/install/setup.bash before running. ==="
