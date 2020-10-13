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
FROM ${BASE_IMAGE}
ARG TFX_WHEEL_PATH

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

RUN echo "Installed python packages before TFX Override:\n" && \
  pip freeze | tee ./pip-freeze-before-tfx.txt

ADD ${TFX_WHEEL_PATH} .

RUN CFLAGS=$(/usr/bin/python-config --cflags) python -m pip install -e .[docker-image]
RUN python -m pip --use-feature=2020-resolver install \
  "$(basename ${TFX_WHEEL_PATH})[docker-image]"

ADD tfx/tools/docker/patches /tmp/tfx-src/tfx/tools/docker/patches
# Patch http.py in googleapiclient and base_api.py in apitools
# to use our own UserAgent.
# TODO(b/151392812): Remove this when other telemetries become available.
RUN patch `python -c 'import googleapiclient; print(googleapiclient.__path__[0])'`/http.py \
  /tmp/tfx-src/tfx/tools/docker/patches/http.patch && \
 patch `python -c 'import apitools; print(apitools.__path__[0])'`/base/py/base_api.py \
  /tmp/tfx-src/tfx/tools/docker/patches/base_api.patch

RUN echo "Installed python packages after TFX Override:\n" && \
  pip freeze | tee ./pip-freeze-after-tfx.txt && \
  echo "\nPython packages delta:\n" && \
  diff -u ./pip-freeze-before-tfx.txt ./pip-freeze-after-tfx.txt || true

# TODO(b/166202742): Consolidate container entrypoint with Kubeflow runner.
ENTRYPOINT ["python", "-m", "tfx.scripts.run_executor"]
