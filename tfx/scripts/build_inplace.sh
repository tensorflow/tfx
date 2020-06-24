#!/bin/bash
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

# Build non-python part of the TFX and copy the build output to the source
# directories (in-place).

source tfx/scripts/build_common.sh

set -u -x

# `BUILD_WORKSPACE_DIRECTORY` is provided by Bazel.
# See https://docs.bazel.build/versions/master/user-manual.html for details.
if [[ -z "${BUILD_WORKSPACE_DIRECTORY}" ]]; then
  echo "BUILD_WORKSPACE_DIRECTORY is unexpectedly empty."
  exit 1
fi

tfx::copy_proto_stubs "${PWD}" "${BUILD_WORKSPACE_DIRECTORY}"
