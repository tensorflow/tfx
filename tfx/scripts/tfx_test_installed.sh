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

# TFX should be installed with DLVM images for 2.1 ~ 2.4.
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
  if [[ ${tf_version_arr[0]} == 2 && \
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
  # Following example code was not included in the package.
  'tfx/examples/bigquery_ml/taxi_utils_bqml_test.py'
  # Skip tests which require additional packages.
  'tfx/examples/custom_components/*'
  'tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple_test.py'
  'tfx/examples/penguin/experimental/penguin_pipeline_sklearn_gcp_test.py'
  'tfx/examples/ranking/*'
  'tfx/*airflow*'
  'tfx/*kubeflow*'
  'tfx/*vertex*'
  'tfx/*e2e*'
  'tfx/*integration*'
  'tfx/components/trainer/rewriting/rewriter_factory_test.py'
  'tfx/components/trainer/rewriting/tfjs_rewriter_test.py'
)

# TODO(b/177609153): TF 2.3 is LTS and we should keep TFX 0.26.x until TF 2.3 retires
if [[ "${TFX_VERSION}" == 0.26.* ]]; then
  SKIP_LIST+=(
    'tfx/tools/cli/container_builder/dockerfile_test.py'
    'tfx/tools/cli/handler/beam_handler_test.py'
    'tfx/tools/cli/handler/local_handler_test.py'
  )
fi

# TODO(b/182435431): Delete the following test after the hanging issue resolved.
SKIP_LIST+=(
  "tfx/experimental/distributed_inference/graphdef_experiments/subgraph_partitioning/beam_pipeline_test.py"
)

# TODO(b/154871293): Migrate to pytest after fixing pytest issues.
# xargs stops only when the exit code is 255, so we convert any
# failure to exit code 255.

set -f  # Disable bash asterisk expansion.
find src -name '*_test.py' \
  ${SKIP_LIST[@]/#tfx/-not -path src} \
  |  xargs -I {} sh -c "${PYTHON_BINARY} {} || exit 255"
