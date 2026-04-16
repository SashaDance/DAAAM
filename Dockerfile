ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG ROS_DISTRO=jazzy
ARG WS_ROOT=/opt/daaam_ws
ARG COLCON_PARALLEL_WORKERS=1
ARG CMAKE_BUILD_PARALLEL_LEVEL=1

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=${ROS_DISTRO}
ENV WS_ROOT=${WS_ROOT}
ENV COLCON_PARALLEL_WORKERS=${COLCON_PARALLEL_WORKERS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
ENV CMAKE_PREFIX_PATH=/usr/local
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1
ENV COLCON_DEFAULTS_FILE=${WS_ROOT}/colcon_defaults.yaml
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    gnupg2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    locales \
    lsb-release \
    ninja-build \
    openssh-client \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    software-properties-common \
    sudo && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

RUN . /etc/os-release && \
    case "${ROS_DISTRO}:${UBUNTU_CODENAME}" in \
      jazzy:noble|humble:jammy) ;; \
      *) \
        printf 'Unsupported ROS_DISTRO=%s on Ubuntu %s.\n' "${ROS_DISTRO}" "${UBUNTU_CODENAME}" >&2 && \
        printf 'Use the default Jazzy build on Ubuntu 24.04, or set CUDA_IMAGE to an Ubuntu 22.04 image before using DAAAM_ROS_DISTRO=humble.\n' >&2 && \
        exit 1 ;; \
    esac

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && printf '%s' \"$UBUNTU_CODENAME\") main" \
    > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-tk \
    libgoogle-glog-dev \
    nlohmann-json3-dev \
    glpk-utils \
    libglpk-dev \
    ros-${ROS_DISTRO}-ros-base \
    ros-dev-tools && \
    (rosdep init || true) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${WS_ROOT}/src
COPY install/packages.yaml ${WS_ROOT}/src/daaam/install/packages.yaml

RUN python3 -m pip install pyem catkin_pkg lark pyyaml && \
    sed -E 's#git@github.com:#https://github.com/#g' ${WS_ROOT}/src/daaam/install/packages.yaml > /tmp/packages.docker.yaml && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    vcs import ${WS_ROOT}/src < /tmp/packages.docker.yaml --workers 1 --skip-existing && \
    (rosdep update --rosdistro=${ROS_DISTRO} || true) && \
    apt-get update && \
    rosdep install --from-paths ${WS_ROOT}/src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

RUN cat > ${COLCON_DEFAULTS_FILE} <<'YAML'
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

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd ${WS_ROOT} && \
    cmake -S ${WS_ROOT}/src/small_gicp -B /tmp/small_gicp-build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/small_gicp-build -j ${CMAKE_BUILD_PARALLEL_LEVEL} && \
    cmake --install /tmp/small_gicp-build && \
    rm -rf /tmp/small_gicp-build

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd ${WS_ROOT} && \
    colcon build --continue-on-error --parallel-workers ${COLCON_PARALLEL_WORKERS} \
      --packages-skip small_gicp khronos khronos_msgs khronos_ros khronos_eval hydra_multi_ros spark_fast_lio ouroboros_ros ouroboros_msgs

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source ${WS_ROOT}/install/setup.bash && \
    cd ${WS_ROOT} && \
    colcon build --continue-on-error --parallel-workers ${COLCON_PARALLEL_WORKERS} --packages-select khronos

COPY pyproject.toml README.md LICENSE requirements.txt ${WS_ROOT}/src/daaam/
COPY src ${WS_ROOT}/src/daaam/src
COPY config ${WS_ROOT}/src/daaam/config
COPY scripts ${WS_ROOT}/src/daaam/scripts
COPY install ${WS_ROOT}/src/daaam/install

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source ${WS_ROOT}/install/setup.bash && \
    python3 -m pip install -r ${WS_ROOT}/src/daaam/requirements.txt && \
    python3 -m pip install "pybind11<3" scikit-build-core && \
    python3 -m pip install --no-build-isolation ${WS_ROOT}/src/spark_dsg && \
    python3 -m pip install -e ${WS_ROOT}/src/daaam

RUN printf 'source /opt/ros/%s/setup.bash\nsource %s/install/setup.bash\n' "${ROS_DISTRO}" "${WS_ROOT}" > /etc/profile.d/daaam.sh

WORKDIR ${WS_ROOT}/src/daaam

CMD ["bash"]
