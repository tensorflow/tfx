#!/bin/bash
#
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# A script to test a TFX installation in the current environment.
#
# Internally this script is used to test TFX installation on DLVM/DL Container
# images.
# - https://cloud.google.com/deep-learning-vm
# - https://cloud.google.com/ai-platform/deep-learning-containers
#
# The list of the container images can be found in:
# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
#
# You can also force TFX version by supplying optional INSTALL_TFX_VERSION
# environment variable.
#
# Example usage;
#  $ cat tfx/scripts/tfx_test_installed.sh | docker run --rm -i gcr.io/deeplearning-platform-release/tf2-cpu.2-4  bash -c 'source /dev/stdin'
#  $ cat tfx/scripts/tfx_test_installed.sh | docker run --rm -e 'INSTALL_TFX_VERSION=0.28.0' -i gcr.io/deeplearning-platform-release/tf2-cpu.2-4  bash -c 'source /dev/stdin'
#

# TFX should be installed with DLVM images for TF 1.15 or 2.1 ~ 2.4.
TFX_SUPPORTED_TF1_VERSION="1.15"
TFX_SUPPORTED_TF2_MIN_VERSION="1"
TFX_SUPPORTED_TF2_MAX_VERSION="4"

set -ex

PYTHON_BINARY=$(which python)
# We need to upgrade scipy to '>1.7.1' to avoid ImportError saying "version `GLIBCXX_3.4.26' not found"
${PYTHON_BINARY} -m pip install --upgrade "pip" "scipy>1.7.1"

if [[ -n "${INSTALL_TFX_VERSION}" ]]; then
  ${PYTHON_BINARY} -m pip install "tfx==${INSTALL_TFX_VERSION}"
fi
if [[ -n "${INSTALL_TF_VERSION}" ]]; then
  ${PYTHON_BINARY} -m pip install "tensorflow==${INSTALL_TF_VERSION}"
fi

TENSORFLOW_VERSION=$(${PYTHON_BINARY} -c 'import tensorflow; print(tensorflow.__version__)')

if ! python -c 'import tfx'; then
  tf_version_arr=(${TENSORFLOW_VERSION//./ })
  max_tf_version_arr=(${MAX_TFX_SUPPORTED_TF_VERSION//./ })
  if [[ "${tf_version_arr[0]}.${tf_version_arr[1]}" == $TFX_SUPPORTED_TF1_VERSION || \
         ${tf_version_arr[0]} == 2 && \
         ${tf_version_arr[1]} -ge $TFX_SUPPORTED_TF2_MIN_VERSION && \
         ${tf_version_arr[1]} -le $TFX_SUPPORTED_TF2_MAX_VERSION ]]; then
      echo "TFX should be installed with TF==${TENSORFLOW_VERSION} but missing."
      exit 1
  else
      echo "TFX does not exist."
      exit 0
  fi
fi

TFX_VERSION=$(${PYTHON_BINARY} -c 'from tfx import version; print(version.__version__)')

if [[ "${TFX_VERSION}" != *dev* ]]; then
  VERSION_TAG_FLAG="-b v${TFX_VERSION} --single-branch"
fi

git clone ${VERSION_TAG_FLAG} https://github.com/tensorflow/tfx.git
cd tfx

# Changes name to make sure we are running tests against installed copy.
mv tfx src

# All items must start with 'tfx/'.
SKIP_LIST=(
  # Skip tests which require additional packages.
  'tfx/examples/*'
  'tfx/*airflow*'
  'tfx/*kubeflow*'
  'tfx/*vertex*'
  'tfx/experimental/*'
  'tfx/orchestration/experimental/*'
  'tfx/orchestration/portable/*'
  'tfx/*e2e*'
  'tfx/tools/*'
  'tfx/*integration*'
  'tfx/components/trainer/*'
)

# TODO(b/179861879): Delete the following tests after TF1 images using TFX 0.28
#                    which includes skipIf branches for TF2 only tests.
if [[ "${TENSORFLOW_VERSION}" == 1.* ]]; then
  SKIP_LIST+=(
    "tfx/*"
  )
fi

# TODO(b/179328863): TF 2.1 is LTS and we should keep TFX 0.21.x until TF 2.1 retires.
if [[ "${TFX_VERSION}" == 0.21.* ]]; then
  SKIP_LIST+=(
    "tfx/*"
  )
fi

# TODO(b/189059446): Delete this after TF 2.2 is retired.
if [[ "${TFX_VERSION}" == 0.22.* ]]; then
  SKIP_LIST+=(
    "tfx/utils/dependency_utils_test.py"
    "tfx/components/transform/executor_with_tfxio_test.py"
    "tfx/components/statistics_gen/executor_test.py"
    "tfx/components/evaluator/executor_test.py"
    "tfx/orchestration/beam/beam_dag_runner_test.py"
  )
fi

if [[ "${TENSORFLOW_VERSION}" == 1.* && "${TFX_VERSION}" == 0.23.* ]]; then
  SKIP_LIST+=(
    "tfx/*"
  )
fi

# TODO(b/177609153): TF 2.3 is LTS and we should keep TFX 0.26.x until TF 2.3 retires
if [[ "${TFX_VERSION}" == 0.26.* ]]; then
  SKIP_LIST+=(
    'tfx/tools/cli/container_builder/dockerfile_test.py'
    'tfx/tools/cli/handler/beam_handler_test.py'
    'tfx/tools/cli/handler/local_handler_test.py'
  )
fi

# TODO(b/188658375): Delete the following test after TFX 1.0.0 released.
if [[ "${TFX_VERSION}" == 0.30.0 ]]; then
  SKIP_LIST+=(
    "tfx/orchestration/portable/execution_watcher_test.py"
  )
fi

# TODO(b/182435431): Delete the following test after the hanging issue resolved.
SKIP_LIST+=(
  "tfx/experimental/distributed_inference/graphdef_experiments/subgraph_partitioning/beam_pipeline_test.py"
)

# TODO(b/188223200): Add back following test for TFX 1.0 and later.
SKIP_LIST+=(
  "tfx/tools/cli/commands/pipeline_test.py"
)

# TODO(b/154871293): Migrate to pytest after fixing pytest issues.
# xargs stops only when the exit code is 255, so we convert any
# failure to exit code 255.

set -f  # Disable bash asterisk expansion.
find src -name '*_test.py' \
  ${SKIP_LIST[@]/#tfx/-not -path src} \
  |  xargs -I {} sh -c "${PYTHON_BINARY} {} || exit 255"
