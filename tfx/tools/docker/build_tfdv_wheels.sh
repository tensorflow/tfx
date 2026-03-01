#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is used to build the TFDV wheel and perform basic checks.
set -e

# Build TFDV wheel.
# It is assumed that the following commands are being run from the docker
# container.
function build_tfdv_wheel() {
  # Activate the conda environment.
  # TODO(b/171583112): Use `conda run` once it is available.
  source /opt/conda/bin/activate tfdv-build

  # Check if the patch is being applied.
  # TODO(b/171583112): Find a better way to check the patch.
  if grep -q "googlesql" "third_party/zetasql_patched.BUILD"; then
    echo "Applying patch to tensorflow_data_validation."
    patch -p1 < /tmp/tfdv.patch
  fi

  # Build the wheel.
  python setup.py bdist_wheel

  # TODO(b/171583112): Add a step to repair the wheel.
  # TODO(b/171583112): Add a step to test the wheel.
}

build_tfdv_wheel
