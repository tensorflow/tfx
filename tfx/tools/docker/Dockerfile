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

FROM gcr.io/tfx-oss-public/tfx_base:py310-20230921 as wheel-builder
# docker build command should be run under root directory of github checkout.
ENV TFX_DIR=/tfx
ADD . ${TFX_DIR}/src
WORKDIR ${TFX_DIR}

# Specify what version of dependent libraries will be used. See dependencies.py.
ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}
ENV USE_BAZEL_VERSION=6.5.0

RUN python -m pip install --upgrade pip wheel setuptools==70.0.0 tomli

# TODO(b/175089240): clean up conditional checks on whether ml-pipelines-sdk is
# built after TFX versions <= 0.25 are no longer eligible for cherry-picks.
RUN cd ${TFX_DIR}/src; \
    if [ -e "package_build" ]; then \
      bash -x package_build/initialize.sh; \
      cd package_build/ml-pipelines-sdk; \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python setup.py bdist_wheel; \
      cd ../../package_build/tfx; \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python setup.py bdist_wheel; \
      cd ../..; \
      MLSDK_WHEEL=$(find dist -name "ml_pipelines_sdk-*.whl"); \
      TFX_WHEEL=$(find dist -name "tfx-*.whl"); \
    else \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python setup.py bdist_wheel; \
      MLSDK_WHEEL=; \
      TFX_WHEEL=$(find dist -name "tfx-*.whl"); \
    fi; \
    if [ "${TFX_DEPENDENCY_SELECTOR}" = "NIGHTLY" ]; then \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python -m pip install \
          --extra-index-url https://pypi-nightly.tensorflow.org/simple \
          ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] -c tfx/tools/docker/requirements.txt -c tfx/tools/docker/build_constraints.txt; \
    else \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python -m pip install ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] -c tfx/tools/docker/requirements.txt -c tfx/tools/docker/build_constraints.txt; \
    fi;

# We need to name this step for the next COPY --from command.
FROM apache/beam_python3.10_sdk:${BEAM_VERSION} as beam-worker

# Build stage to extend DLVM image.
FROM ${BASE_IMAGE} as install

# Additional Python packages which will be installed.
ARG ADDITIONAL_PACKAGES

# Specify what version of dependent libraries will be used. See dependencies.py.
ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}

# Copy from image matching installed version of 'apache-beam'.
COPY --from=beam-worker /opt/apache/beam /opt/apache/beam
# Dataflow SDK worker image looks for pip at a different location by default
# without following pip environment variable.
# TODO(zhitaoli): Remove hack on pip environment after
# https://github.com/apache/beam/pull/13850 is widely available.
ENV pip=/opt/conda/bin/pip
ENTRYPOINT ["/opt/apache/beam/boot"]

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

RUN sed -i 's/python3/python/g' /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools==70.0.0 wheel

RUN CURRENT_DIR="$(pwd)" && \
  mkdir -p /usr/local/lib/bazel/bin && \
  cd "/usr/local/lib/bazel/bin" && \
  curl -fLO https://releases.bazel.build/6.5.0/release/bazel-6.5.0-linux-x86_64 && \
  chmod +x bazel-6.5.0-linux-x86_64 && \
  ln -sf /usr/local/lib/bazel/bin/bazel-6.5.0-linux-x86_64 /usr/local/bin/bazel && \
  cd "$CURRENT_DIR"

# Build tensorflow-data-validation wheels from source with patch
RUN  /opt/conda/bin/conda install -y -c conda-forge \
    gcc_linux-64=13 \
    gxx_linux-64=13 \
    binutils_linux-64=2.40 \
    ld_impl_linux-64=2.40

 ENV CC=/opt/conda/bin/x86_64-conda-linux-gnu-gcc

 ENV CXX=/opt/conda/bin/x86_64-conda-linux-gnu-g++

 ENV LD=/opt/conda/bin/x86_64-conda-linux-gnu-ld

COPY tfx/tools/docker/build_tfdv_wheels.sh /tmp/
COPY tfx/tools/docker/tfdv.patch /tmp/
RUN mkdir -p /tmp/wheels && \
    bash /tmp/build_tfdv_wheels.sh /tmp/wheels

# Build tfx-bsl wheels from source with patch
COPY tfx/tools/docker/build_tfx_bsl_wheels.sh /tmp/
COPY tfx/tools/docker/tfx_bsl.patch /tmp/
RUN bash /tmp/build_tfx_bsl_wheels.sh /tmp/wheels

# Copy constraint file and wheels
COPY tfx/tools/docker/build_constraints.txt /tmp/
COPY tfx/tools/docker/wheels/*.whl /tmp/wheels/

# Install tensorflow-data-validation, tfx-bsl, tensorflow-model-analysis, tensorflow-transform, and tensorflow-cloud first
RUN PIP_CONSTRAINT=/tmp/build_constraints.txt python -m pip install /tmp/wheels/tensorflow_data_validation-*.whl /tmp/wheels/tfx_bsl-*.whl /tmp/wheels/tensorflow_model_analysis-*.whl /tmp/wheels/tensorflow_transform-*.whl /tmp/wheels/tensorflow_cloud-*.whl

COPY --from=wheel-builder /tfx/src/dist/*.whl /tfx/src/dist/
WORKDIR /tfx/src

RUN MLSDK_WHEEL=$(find dist -name "ml_pipelines_sdk-*.whl"); \
    TFX_WHEEL=$(find dist -name "tfx-*.whl"); \
    if [ "${TFX_DEPENDENCY_SELECTOR}" = "NIGHTLY" ]; then \
      PIP_CONSTRAINT=/tmp/build_constraints.txt python -m pip install \
        --extra-index-url https://pypi-nightly.tensorflow.org/simple \
        ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] ${ADDITIONAL_PACKAGES} ; \
    else \
      PIP_CONSTRAINT=/tmp/build_constraints.txt python -m pip install ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] \
      ${ADDITIONAL_PACKAGES} ; \
    fi && \
    echo "Installed python packages:\n" && python -m pip list && \
    echo "Setuptools version:" && python -c "import setuptools; print(setuptools.__version__)"
