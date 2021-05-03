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

# Copy proto python stub files.
# Arguments:
#   src_dir: Root directory in which generated proto stubs exist.
#   dst_dir: Root directory of the project to which stubs will be copied.
function tfx::copy_proto_stubs() {
  local src_dir="$1"
  local dst_dir="$2"

  if [[ -z "${src_dir}" ]] || [[ -z "{dst_dir}" ]]; then
    echo "src_dir and dst_dir should not be empty."
    exit 1
  fi

  # Failure of copy should not be ignored.
  set -e
  find . -name "*_pb2.py" -exec cp -f {} ${dst_dir}/{} \;
  find . -name "*_pb2_grpc.py" -exec cp -f {} ${dst_dir}/{} \;
}

function tfx::gen_proto_main() {
  set -o nounset
  set -o xtrace

  # `BUILD_WORKSPACE_DIRECTORY` is provided by Bazel.
  # See https://docs.bazel.build/versions/master/user-manual.html for details.
  if [[ -z "${BUILD_WORKSPACE_DIRECTORY}" ]]; then
    echo "BUILD_WORKSPACE_DIRECTORY is unexpectedly empty."
    exit 1
  fi

  tfx::copy_proto_stubs "${PWD}" "${BUILD_WORKSPACE_DIRECTORY}"
}

tfx::gen_proto_main "$@"
