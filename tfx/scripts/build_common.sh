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

  # Copy all "*_pb2.py" file in src_dir to dst_dir, keeping directory structure.
  # There are multiple ways of copying files with directories:
  #
  # | Platforms      | uname  | cp --parents | cpio | rsync     | ditto |
  # :                :        : (gnu cp)     :      :           :       :
  # | -------------- | ------ | ------------ | ---- | --------- | ----- |
  # | TFX Base Image | Linux  |      O       |  X   |     X     |   X   |
  # | GCP_UBUNTU     | Linux  |      O       |  O   |     O     |   X   |
  # | MACOS_EXTERNAL | Darwin |      X       |  O   |     O     |   O   |
  #
  # Since there are no cross-platform options available, we should branch on
  # each platform to perform copy.
  local platform
  platform="$(uname -s)"

  if [[ "${platform}" =~ Darwin ]]; then
    pushd "${src_dir}" > /dev/null
    find . -name "*_pb2.py" | cpio -updm --quiet "${dst_dir}"
    popd
  else
    pushd "${src_dir}" > /dev/null
    find . -name "*_pb2.py" | xargs cp --parents -t "${dst_dir}"
    popd
  fi
}
