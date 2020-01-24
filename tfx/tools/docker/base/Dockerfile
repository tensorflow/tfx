# Copyright 2020 Google LLC
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

# Base image used to facilitate docker building.
# This gets updated nightly.

FROM ubuntu:16.04
LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

# Need additional ppa since python 3.6 and protobuf 3
# are not part of Ubuntu 16.04 by default.
RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  add-apt-repository ppa:maarten-fonville/protobuf && \
  apt-get update -y && \
  apt-get install --no-install-recommends -y -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3.6-dev \
  python3-setuptools \
  python3-virtualenv \
  python3-wheel \
  wget \
  unzip \
  git && \
  add-apt-repository -r ppa:deadsnakes/ppa && \
  add-apt-repository -r ppa:maarten-fonville/protobuf && \
  apt-get autoclean && \
  apt-get autoremove --purge && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
# This is to address b/144805438
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py

