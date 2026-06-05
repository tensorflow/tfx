# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Base image for the final image. Required.
ARG BASE_IMAGE
# Apache beam version to get Beam binaries.
ARG BEAM_VERSION

FROM ${BASE_IMAGE} AS base-with-gcc13
RUN if [ -x "/opt/conda/bin/conda" ]; then \
      /opt/conda/bin/conda install -y --override-channels -c conda-forge \
      gcc_linux-64=13 gxx_linux-64=13 binutils_linux-64=2.40 ld_impl_linux-64=2.40 ; \
    else \
      apt-get update && apt-get install -y gcc g++ build-essential binutils ; \
    fi

ENV CC=gcc
ENV CXX=g++

ENV BAZEL_COMPILER=gcc

FROM base-with-gcc13 AS wheel-builder
ARG BASE_IMAGE
ARG BEAM_VERSION
# Whether to use C++ wheels (tfdv, tfx_bsl) from the host cache (/tmp/wheels).
# If true, cached wheels are used if available (this saves time when debugging the docker build issues).
# If false, wheels are built from source even if they exist in the cache.
ARG USE_CPP_WHEELS_FROM_TEMP=false
# Whether to clear the persistent build caches (/tmp/wheels and bazel cache).
ARG CLEAN_CPP_TEMP_CACHE=false

# Specify what version of dependent libraries will be used. See dependencies.py.
ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}

ENV USE_BAZEL_VERSION=7.7.0

RUN apt-get update && apt-get install -y curl git openjdk-17-jdk-headless cmake && \
    (find /opt/conda/bin -name "python3-config" | head -n 1 | xargs -I {} ln -sf {} /usr/bin/python-config) && \
    (find /opt/conda/bin -name "python3-config" | head -n 1 | xargs -I {} ln -sf {} /opt/conda/bin/python-config)
RUN mkdir -p /usr/local/lib/bazel/bin && \
    curl -fLO https://github.com/bazelbuild/bazel/releases/download/${USE_BAZEL_VERSION}/bazel-${USE_BAZEL_VERSION}-linux-x86_64 && \
    mv bazel-${USE_BAZEL_VERSION}-linux-x86_64 /usr/local/lib/bazel/bin/bazel && \
    chmod +x /usr/local/lib/bazel/bin/bazel && \
    ln -sf /usr/local/lib/bazel/bin/bazel /usr/local/bin/bazel
ENV PATH="/usr/local/lib/bazel/bin:${PATH}"


COPY . /tfx/src/
WORKDIR /tfx/src/

# 1. C++ Wheels (tfdv, tfx_bsl, tfmd, mlmd) - Normal production build path
RUN if [ "$USE_CPP_WHEELS_FROM_TEMP" = "false" ]; then \
      echo "Rebuild of C++ wheels (tfdv, tfx_bsl, tfmd, mlmd)..." && \
      cp tfx/tools/docker/build_tfdv_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_tfx_bsl_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_tfmd_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_mlmd_wheels.sh /tmp/ && \
      cp tfx/tools/docker/*.patch /tmp/ && \
      mkdir -p /tfx/src/dist_wheels && \
      bash /tmp/build_tfmd_wheels.sh /tfx/src/dist_wheels && \
      bash /tmp/build_mlmd_wheels.sh /tfx/src/dist_wheels && \
      bash /tmp/build_tfdv_wheels.sh /tfx/src/dist_wheels && \
      bash /tmp/build_tfx_bsl_wheels.sh /tfx/src/dist_wheels ; \
    fi

# 2. C++ Wheels (tfdv, tfx_bsl, tfmd, mlmd) - Cached Path to avoid any CPP rebuilds
RUN --mount=type=cache,target=/tmp/wheels --mount=type=cache,target=/root/.cache/bazel \
    if [ "$USE_CPP_WHEELS_FROM_TEMP" = "true" ]; then \
      echo "Re-use cached build of C++ wheels (tfdv, tfx_bsl, tfmd, mlmd)..." && \
      cp tfx/tools/docker/build_tfdv_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_tfx_bsl_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_tfmd_wheels.sh /tmp/ && \
      cp tfx/tools/docker/build_mlmd_wheels.sh /tmp/ && \
      cp tfx/tools/docker/*.patch /tmp/ && \
      if [ ! -f /tmp/wheels/tensorflow_metadata-*.whl ]; then bash /tmp/build_tfmd_wheels.sh /tmp/wheels; fi && \
      if [ ! -f /tmp/wheels/ml_metadata-*.whl ]; then bash /tmp/build_mlmd_wheels.sh /tmp/wheels; fi && \
      if [ ! -f /tmp/wheels/tensorflow_data_validation-*.whl ]; then bash /tmp/build_tfdv_wheels.sh /tmp/wheels; fi && \
      if [ ! -f /tmp/wheels/tfx_bsl-*.whl ]; then bash /tmp/build_tfx_bsl_wheels.sh /tmp/wheels; fi && \
      mkdir -p /tfx/src/dist_wheels && \
      cp /tmp/wheels/tensorflow_metadata-*.whl /tfx/src/dist_wheels/ && \
      cp /tmp/wheels/ml_metadata-*.whl /tfx/src/dist_wheels/ && \
      cp /tmp/wheels/tensorflow_data_validation-*.whl /tfx/src/dist_wheels/ && \
      cp /tmp/wheels/tfx_bsl-*.whl /tfx/src/dist_wheels/ ; \
    fi

# TODO(b/175089240): clean up conditional checks on whether ml-pipelines-sdk is
# built after TFX versions <= 0.25 are no longer eligible for cherry-picks.
RUN echo "Building TFX and SDK wheels from source..." && \
    mkdir -p /tfx/src/dist_wheels && \
    if [ -e "package_build" ]; then \
      bash -x package_build/initialize.sh && \
      CFLAGS=$(python-config --cflags) python package_build/ml-pipelines-sdk/setup.py bdist_wheel && \
      CFLAGS=$(python-config --cflags) python package_build/tfx/setup.py bdist_wheel && \
      cp package_build/ml-pipelines-sdk/dist/*.whl /tfx/src/dist_wheels/ && \
      cp package_build/tfx/dist/*.whl /tfx/src/dist_wheels/ ; \
    else \
      CFLAGS=$(python-config --cflags) python setup.py bdist_wheel && \
      cp dist/*.whl /tfx/src/dist_wheels/ ; \
    fi

# --- Stage 2: beam-worker ---
FROM apache/beam_python3.10_sdk:${BEAM_VERSION} AS beam-worker

# --- Stage 3: install (Final) ---
FROM base-with-gcc13 AS install
ARG BASE_IMAGE
ARG ADDITIONAL_PACKAGES
ENV TF_CPP_MIN_LOG_LEVEL=3
# Disables Legacy "Virtual" Devices
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}
ENV TF_USE_LEGACY_KERAS=1

# 1. Apply OS security updates and install required system libraries
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libsnappy-dev \
    libcurl4-openssl-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy from image matching installed version of 'apache-beam'.
COPY --from=beam-worker /opt/apache/beam /opt/apache/beam
# Dataflow SDK worker image looks for pip at a different location by default
# without following pip environment variable.
# TODO(zhitaoli): Remove hack on pip environment after
# https://github.com/apache/beam/pull/13850 is widely available.
ENV pip=/opt/conda/bin/pip
ENTRYPOINT ["/opt/apache/beam/boot"]

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

COPY --from=wheel-builder /tfx/src /tfx/src

# 2. Consolidated Installation and Security Remediation in a Single RUN layer
# Combining upgrade, install, purge, and cache clean into one multi-command execution
# ensures intermediate vulnerable packages and conda caches never persist in layer tarballs.
RUN python -m pip install --upgrade pip setuptools==78.1.1 wheel \
    -c /tfx/src/tfx/tools/docker/requirements.txt \
    -c /tfx/src/tfx/tools/docker/build_constraints.txt && \
    if [ "${TFX_DEPENDENCY_SELECTOR}" = "NIGHTLY" ]; then \
	  python -m pip install --upgrade --upgrade-strategy=eager --no-cache-dir \
            --extra-index-url https://pypi-nightly.tensorflow.org/simple \
	    -c /tfx/src/tfx/tools/docker/requirements.txt \
	    -c /tfx/src/tfx/tools/docker/build_constraints.txt \
	    /tfx/src/dist_wheels/*.whl \
	    /tfx/src/tfx/tools/docker/wheels/*.whl \
	    "$(find /tfx/src/dist_wheels/ \( -name 'tfx_dev-*.whl' -o -name 'tfx-*.whl' \) | head -n 1)[docker-image]" \
	    tf_keras setuptools==78.1.1 ${ADDITIONAL_PACKAGES} ; \
    else \
	  python -m pip install --upgrade --upgrade-strategy=eager --no-cache-dir \
	    -c /tfx/src/tfx/tools/docker/requirements.txt \
	    -c /tfx/src/tfx/tools/docker/build_constraints.txt \
	    /tfx/src/dist_wheels/*.whl \
	    /tfx/src/tfx/tools/docker/wheels/*.whl \
	    "$(find /tfx/src/dist_wheels/ \( -name 'tfx_dev-*.whl' -o -name 'tfx-*.whl' \) | head -n 1)[docker-image]" \
	    tf_keras setuptools==78.1.1 ${ADDITIONAL_PACKAGES} ; \
    fi && \
    for pkg in golang-go golang git binutils wget policykit-1 packagekit gnupg2 gcc-12; do apt-get purge -y $pkg || true; done && \
    rm -rf /usr/local/go && \
    rm -rf /opt/apache/beam && \
    find /opt/conda -name "*go*" -delete || true && \
    find /opt/conda -name "*boot*" -delete || true && \
    apt-get autoremove -y && \
    apt-get clean && \
    python -m pip install --upgrade --no-cache-dir \
        pip \
        wheel \
        protobuf \
        lxml \
        cryptography \
        idna \
        python-dotenv \
        keras \
        tf_keras || true && \
    (python -m pip uninstall -y jupyter jupyter-server jupyterlab notebook nbconvert mistune pyopenssl pygments jaraco jaraco-context jaraco.context jaraco.classes jaraco.functools || true) && \
    rm -rf /opt/conda/pkgs/* && \
    find /opt/conda -name "*.tar.bz2" -type f -delete || true && \
    find /opt/conda -name "*.conda" -type f -delete || true && \
    find /usr/local/lib -name "*golang*" -delete || true && \
    find /root/.cache -type f -delete || true

RUN echo "Installed python packages:\n" && python -m pip list && \
    echo "Setuptools version:" && python -c "import setuptools; print(setuptools.__version__)"
