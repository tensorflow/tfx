# Copyright 2018 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


FROM ubuntu:16.04

RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  python-dev \
  python-setuptools \
  wget \
  unzip \
  git

# tensorflow dependency is not explicitly listed in TFX setup.py.
RUN easy_install pip && pip install \
  python-snappy==0.5.3 \
  # LINT.IfChange
  tensorflow==1.12.0
  # LINT.ThenChange(
  #    ../../examples/chicago_taxi/setup.py,
  #    ../../examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_large.py,
  #    ../../examples/chicago_taxi_pipeline/taxi_pipeline_gcp.py,
  #)

# docker build command should be run under root directory of github checkout.
ADD . /tfx-src
WORKDIR /tfx-src
RUN pip install -r ./requirements.txt

RUN /tfx-src/tfx/tools/docker/license.sh /tfx-src/tfx/tools/docker/third_party_licenses.csv /third_party/licenses

ENV PYTHONPATH "${PYTHONPATH}:/tfx-src"
ENTRYPOINT ["python", "/tfx-src/tfx/scripts/run_executor.py"]
