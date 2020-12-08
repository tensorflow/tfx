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

FROM gcr.io/tfx-oss-public/tfx_base:py37-20200729
# Change to following base image if docker hub is preferred.
# FROM tensorflow/tfx_base:latest

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

# TODO(zhitaoli): Remove pinned version of tensorflow and related packages here
# once we switch default tensorflow version in released image to TF 2.x.
# TODO(b/151392812): Remove `google-api-python-client` and `google-apitools`
#                    when patching is not needed any more.
RUN CFLAGS=$(/usr/bin/python-config --cflags) python -m pip install \
  "kubernetes>=10.0.1,<11" \
  "tensorflow>=1.15,<3" \
  "tensorflow-serving-api>=1.15,<3" \
  "google-api-python-client==1.8.0" \
  "google-apitools==0.5.30"

# docker build command should be run under root directory of github checkout.
ENV TFX_DIR=/tfx
ADD . ${TFX_DIR}/src
WORKDIR ${TFX_DIR}

ARG TFX_DEPENDENCY_SELECTOR
ENV TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}

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
    fi; \
    cd ${TFX_DIR}

RUN echo "Installed python packages:\n" && python -m pip list

# Patch http.py in googleapiclient and base_api.py in apitools
# to use our own UserAgent.
# TODO(b/151392812): Remove this when other telemetries become available.
RUN patch `python -c 'import googleapiclient; print(googleapiclient.__path__[0])'`/http.py \
  /tfx/src/tfx/tools/docker/patches/http.patch && \
 patch `python -c 'import apitools; print(apitools.__path__[0])'`/base/py/base_api.py \
  /tfx/src/tfx/tools/docker/patches/base_api.patch

# TODO(b/166202742): Consolidate container entrypoint with Kubeflow runner.
ENTRYPOINT ["python", "-m", "tfx.scripts.run_executor"]
