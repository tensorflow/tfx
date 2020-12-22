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

# Use a pre-built GCR image including Bazel.
FROM gcr.io/cloud-marketplace-containers/google/bazel:3.0.0
LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Need additional ppa since python 3.7 and protobuf 3
# are not part of Ubuntu 16.04 by default.
# We also purge preinstalled python2.7 and 3.5.
RUN ${APT_COMMAND} update && \
  ${APT_COMMAND} install --no-install-recommends -q software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  add-apt-repository ppa:maarten-fonville/protobuf && \
  ${APT_COMMAND} update && \
  ${APT_COMMAND} install --no-install-recommends -q \
  build-essential \
  ca-certificates \
  libsnappy-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3.7-dev \
  wget \
  unzip \
  git && \
  add-apt-repository -r ppa:deadsnakes/ppa && \
  add-apt-repository -r ppa:maarten-fonville/protobuf && \
  ${APT_COMMAND} autoremove --purge python2.7-dev python2.7 libpython2.7 python2.7-minimal \
  python3.5-dev python3.5 libpython3.5 python3.5-minimal && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
  update-alternatives --install /usr/bin/python-config python-config /usr/bin/python3.7-config 1 && \
  ${APT_COMMAND} autoclean && \
  ${APT_COMMAND} autoremove --purge

# Pre-install pip so we can use the beta dependency resolver.
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
  pip install --upgrade --pre pip

