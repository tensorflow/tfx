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


FROM ubuntu:18.04

RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  python-dev \
  python-pip \
  python-setuptools \
  python-wheel \
  wget \
  unzip \
  git

RUN python -m pip install --upgrade pip==19.0.3 && \
  pip install strip-hints==0.1.4

# docker build command should be run under root directory of github checkout.
ADD . /tfx-src
WORKDIR /tfx-src
# TODO(zhitaoli): Remove this once we move to python 3.
RUN python /tfx-src/tfx/scripts/strip_type_hints.py
RUN pip install -e .[docker-image]

RUN /tfx-src/tfx/tools/docker/license.sh /tfx-src/tfx/tools/docker/third_party_licenses.csv /third_party/licenses

ENV PYTHONPATH "${PYTHONPATH}:/tfx-src"
ENTRYPOINT ["python", "/tfx-src/tfx/scripts/run_executor.py"]
