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
# Prerequites:
#  - Installed TFX package.
# Example usage;
#  $ cat tfx_test_installed.sh | docker run --rm -i gcr.io/deeplearning-platform-release/tf2-cpu.2-4  bash -c 'source /dev/stdin'
#

set -ex

PYTHON_BINARY=$(which python)

TFX_VERSION=$(${PYTHON_BINARY} -c 'from tfx import version; print(version.__version__)')

rm -rf tfx
if [[ "${TFX_VERSION}" != *dev* ]]; then
  VERSION_TAG_FLAG="-b v${TFX_VERSION} --single-branch"
fi

git clone ${VERSION_TAG_FLAG} https://github.com/tensorflow/tfx.git
cd tfx

# Changes name to make sure we are running tests against installed copy.
mv tfx src

# All items must start with 'tfx/'.
SKIP_LIST=(
  # TODO(b/175507983): Will be fixed in 0.26.0. Delete after 0.26.0 release.
  'tfx/orchestration/beam/beam_dag_runner_test.py'
  # TODO(b/175507983): Will be fixed in 0.26.0. Delete after 0.26.0 release.
  'tfx/utils/dependency_utils_test.py'
  # TODO(b/175507983): Will be fixed in 0.26.0. Delete after 0.26.0 release.
  'tfx/orchestration/metadata_test.py'
  # TODO(b/175507983): Will be fixed in 0.26.0. Delete after 0.26.0 release.
  'tfx/experimental/distributed_inference/*'

  # TODO(b/177609153): Will be fixed in 0.27.0. Delete after 0.27.0 release.
  'tfx/tools/cli/container_builder/dockerfile_test.py'
  'tfx/tools/cli/handler/beam_handler_test.py'
  'tfx/tools/cli/handler/local_handler_test.py'
  'tfx/tools/cli/kubeflow_v2/handler/kubeflow_v2_handler_test.py'

  # TODO(b/174968932): Delete after renaming this file.
  'tfx/orchestration/kubeflow/kubeflow_gcp_perf_test.py'

  # Following example code was not included in the package.
  'tfx/examples/bigquery_ml/taxi_utils_bqml_test.py'
  # Skip tests which require additional packages.
  'tfx/examples/custom_components/*'
  'tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple_test.py'
  'tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_*'
  'tfx/orchestration/airflow/*'
  'tfx/orchestration/kubeflow/*'
  'tfx/tools/cli/handler/airflow_handler_test.py'
  'tfx/tools/cli/handler/kubeflow_handler_test.py'
  'tfx/*e2e*'
  'tfx/*integration*'
  'tfx/components/trainer/rewriting/rewriter_factory_test.py'
  'tfx/components/trainer/rewriting/tfjs_rewriter_test.py'
)

# TODO(b/154871293): Migrate to pytest after fixing pytest issues.
# xargs stops only when the exit code is 255, so we convert any
# failure to exit code 255.

set -f  # Disable bash asterisk expansion.
find src -name '*_test.py' \
  ${SKIP_LIST[@]/#tfx/-not -path src} \
  |  xargs -I {} sh -c "${PYTHON_BINARY} {} || exit 255"
