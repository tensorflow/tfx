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

# Need additional ppa since python 3.5 is not part of Ubuntu 18.04 by default.
RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  python3.5-dev \
  python3-pip \
  python3-setuptools \
  python3-virtualenv \
  python3-wheel \
  wget \
  unzip \
  git

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3.5 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# docker build command should be run under root directory of github checkout.
ENV TFX_SRC_DIR=/tfx-src
ADD . ${TFX_SRC_DIR}
WORKDIR ${TFX_SRC_DIR}
RUN python setup.py bdist_wheel
RUN CFLAGS=$(/usr/bin/python3.5-config --cflags) pip install $(find dist -name "tfx-*.whl")[docker-image]

RUN /tfx-src/tfx/tools/docker/license.sh /tfx-src/tfx/tools/docker/third_party_licenses.csv /third_party/licenses

ENTRYPOINT ["python3.5", "/tfx-src/tfx/scripts/run_executor.py"]
