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
ARG BASE_IMAGE
ARG BEAM_VERSION

FROM gcr.io/tfx-oss-public/tfx_base:py37-20200729 as wheel-builder
# docker build command should be run under root directory of github checkout.
ENV TFX_DIR=/tfx
ADD . ${TFX_DIR}/src
WORKDIR ${TFX_DIR}

ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}

RUN python -m pip install --upgrade pip

# TODO(b/175089240): clean up conditional checks on whether ml-pipelines-sdk is
# built after TFX versions <= 0.25 are no longer eligible for cherry-picks.
RUN cd ${TFX_DIR}/src; \
    if [ -e "package_build" ]; then \
      bash -x package_build/initialize.sh; \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python package_build/ml-pipelines-sdk/setup.py bdist_wheel; \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python package_build/tfx/setup.py bdist_wheel; \
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
          ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] ; \
    else \
      CFLAGS=$(/usr/bin/python-config --cflags) \
        python -m pip install ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] ; \
    fi;

# We need to name this step for the next COPY --from command.
FROM apache/beam_python3.7_sdk:${BEAM_VERSION} as beam-worker

# Build stage to extend DLVM image.
FROM ${BASE_IMAGE} as install

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

RUN python -m pip install --upgrade pip

COPY --from=wheel-builder /tfx/src/dist/*.whl /tfx/src/dist/
WORKDIR /tfx/src

RUN MLSDK_WHEEL=$(find dist -name "ml_pipelines_sdk-*.whl"); \
    TFX_WHEEL=$(find dist -name "tfx-*.whl"); \
    if [ "${TFX_DEPENDENCY_SELECTOR}" = "NIGHTLY" ]; then \
      python -m pip install \
        --extra-index-url https://pypi-nightly.tensorflow.org/simple \
        ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] ; \
    else \
      python -m pip install ${MLSDK_WHEEL} ${TFX_WHEEL}[docker-image] ; \
    fi && \
    echo "Installed python packages:\n" && python -m pip list
