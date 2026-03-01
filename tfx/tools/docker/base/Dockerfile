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

# Use an ubuntu 20.04 image.
FROM ubuntu:20.04
LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

ARG DEBIAN_FRONTEND=noninteractive

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Install python 3.10 and additional dependencies.
RUN ${APT_COMMAND} update && \
  ${APT_COMMAND} install --no-install-recommends -q software-properties-common && \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  ${APT_COMMAND} update && \
  ${APT_COMMAND} install --no-install-recommends -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3.10-dev \
  python3-distutils \
  wget \
  unzip \
  git && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/python-config python-config /usr/bin/python3.10-config 1 && \
  ${APT_COMMAND} autoclean && \
  ${APT_COMMAND} autoremove --purge

# Pre-install pip so we can use the beta dependency resolver.
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
  pip install --upgrade --pre pip

# Install bazel
RUN wget -O /bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64 && \
  chmod +x /bin/bazel && \
  bazel version