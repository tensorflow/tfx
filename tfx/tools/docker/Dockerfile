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

FROM gcr.io/tfx-oss-public/tfx_base:latest
# Change to following base image if docker hub is preferred.
# FROM tensorflow/tfx_base:latest

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

ENV VIRTUAL_ENV=/opt/venv
RUN python3.6 -m pip install --upgrade pip && \
  python3.6 -m virtualenv --python=/usr/bin/python3.6 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# TODO(zhitaoli): Remove pinned version of tensorflow and related packages here
# once we switch default tensorflow version in released image to TF 2.x.
# TODO(jxzheng): investigate why setuptools need to be re-installed
# TODO(b/151392812): Remove `google-api-python-client` and `google-apitools`
#                    when patching is not needed any more.
RUN CFLAGS=$(/usr/bin/python3.6-config --cflags) python3.6 -m pip install \
  "kubernetes>=10.0.1,<11" \
  "tensorflow>=1.15,<3" \
  "tensorflow-serving-api>=1.15,<3" \
  "google-api-python-client==1.8.0" \
  "google-apitools==0.5.30" \
  "setuptools==41.4.0"

# docker build command should be run under root directory of github checkout.
ENV TFX_SRC_DIR=/tfx-src
ADD . ${TFX_SRC_DIR}
WORKDIR ${TFX_SRC_DIR}

RUN CFLAGS=$(/usr/bin/python3.6-config --cflags) bazel run tfx:build_pip_package
RUN CFLAGS=$(/usr/bin/python3.6-config --cflags) python3.6 -m pip install -e .

# Patch http.py in googleapiclient and base_api.py in apitools
# to use our own UserAgent.
# TODO(b/151392812): Remove this when other telemetries become available.
RUN patch /opt/venv/lib/python3.6/site-packages/googleapiclient/http.py \
  /tfx-src/tfx/tools/docker/patches/http.patch && \
  patch /opt/venv/lib/python3.6/site-packages/apitools/base/py/base_api.py \
  /tfx-src/tfx/tools/docker/patches/base_api.patch

# TODO(zhitaoli): Ensure that license dependency is an exact match instead of
# subset check.
ARG gather_third_party_licenses=true
RUN if [ "${gather_third_party_licenses}" = "true" ]; then \
  /tfx-src/tfx/tools/docker/license.sh \
  /tfx-src/tfx/tools/docker/third_party_licenses.csv \
  /third_party/licenses; \
  fi

# TODO(b/139532550): Consolidate container entrypoint with Kubeflow runner.
ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
