#!/bin/bash
# Copyright 2019 Google LLC. All Rights Reserved.
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

# Convenience script to build TFX docker image.
set -ex

DOCKER_IMAGE_REPO=${DOCKER_IMAGE_REPO:-"tensorflow/tfx"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"latest"}
DOCKER_FILE=${DOCKER_FILE:-"Dockerfile"}

TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR:-""}
echo "Env for TFX_DEPENDENCY_SELECTOR is set as ${TFX_DEPENDENCY_SELECTOR}"

# Apply the patch before building
echo "Applying tfx.patch..."
if [[ -f patches/tfx.patch ]]; then
  git apply patches/tfx.patch
  patch_applied=true
else
  echo "Warning: patches/tfx.patch not found, skipping patch application"
  patch_applied=false
fi

mkdir -p tfx/tools/docker/wheels

# Download tensorflow-model-analysis wheel
echo "Downloading tensorflow-model-analysis wheel..."
TFMA_WHEEL_URL="https://files.pythonhosted.org/packages/a9/45/1ed03c0bd8168ebc8bdc5c15c206d2e3a7fb9269f8083492d17b995ac35f/tensorflow_model_analysis-0.48.0-py3-none-any.whl"
TFMA_WHEEL_FILE="tensorflow_model_analysis-0.48.0-py3-none-any.whl"
curl -L -o tfx/tools/docker/wheels/${TFMA_WHEEL_FILE} ${TFMA_WHEEL_URL}

# Download tensorflow-transform wheel
echo "Downloading tensorflow-transform wheel..."
TFT_WHEEL_URL="https://files.pythonhosted.org/packages/a2/b2/32d2ad3fbf16a67f7e91e125dca616a9e1b0d10588167ce3c19394a1811f/tensorflow_transform-1.17.0-py3-none-any.whl"
TFT_WHEEL_FILE="tensorflow_transform-1.17.0-py3-none-any.whl"
curl -L -o tfx/tools/docker/wheels/${TFT_WHEEL_FILE} ${TFT_WHEEL_URL}

# Download tensorflow-cloud wheel
echo "Downloading tensorflow-cloud wheel..."
TFC_WHEEL_URL="https://files.pythonhosted.org/packages/4b/bc/da205a15aaf22c1fda1f58552990d17d532a8573af6830e3663730ed485b/tensorflow_cloud-0.1.16-py3-none-any.whl"
TFC_WHEEL_FILE="tensorflow_cloud-0.1.16-py3-none-any.whl"
curl -L -o tfx/tools/docker/wheels/${TFC_WHEEL_FILE} ${TFC_WHEEL_URL}

function _get_tf_version_of_image() {
  local img="$1"
  docker run --rm --entrypoint=python ${img} -c 'import tensorflow as tf; print(tf.__version__)'
}

# Base image to extend: This should be a deep learning image with a compatible
# TensorFlow version. See
# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
# for possible images to use here.

# Use timestmap-rand for tag, to avoid collision of concurrent runs.
wheel_builder_tag="tfx-wheel-builder:$(date +%s)-$RANDOM"
# Run docker build command to build the wheel-builder first. We have to extract
# TF version from it.
docker build --target wheel-builder\
  -t ${wheel_builder_tag} \
  -f tfx/tools/docker/${DOCKER_FILE} \
  --build-arg TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR} \
  . "$@"

# TensorFlow current TFX code depends on here and use that instead.
if [[ -n "$BASE_IMAGE" ]]; then
  echo "Using override base image $BASE_IMAGE"
else
  tf_version=$(_get_tf_version_of_image "${wheel_builder_tag}")
  arr_version=(${tf_version//./ })
  echo "Detected TensorFlow version as ${tf_version}"
  DLVM_REPO=gcr.io/deeplearning-platform-release
  DLVM_PY_VERSION=py310
  BASE_IMAGE=${DLVM_REPO}/tf2-gpu.${arr_version[0]}-${arr_version[1]}.${DLVM_PY_VERSION}

  # Check the availability of the DLVM image.
  if gcloud container images list --repository=${DLVM_REPO} | grep -x "${BASE_IMAGE}" ; then
    # TF shouldn't be re-installed so we pin TF version in Pip install.
    installed_tf_version=$(_get_tf_version_of_image "${BASE_IMAGE}")
    # TODO(b/333895985): This should be rollbacked after the fix. The TF version
    # from the BASE_IMAGE is wrongly set (expected: 2.15.1, actually: 2.15.0).
    ADDITIONAL_PACKAGES="tensorflow==${tf_version}"
    # if [[ "${installed_tf_version}" =~ rc ]]; then
    #   # Overwrite the rc version with a latest regular version.
    #   ADDITIONAL_PACKAGES="tensorflow==${tf_version}"
    # else
    #   ADDITIONAL_PACKAGES="tensorflow==${installed_tf_version}"
    # fi
  else
    # Fallback to the image of the previous version but also install the newest
    # TF version.
    arr_version[1]=$((arr_version[1] - 1))
    BASE_IMAGE=${DLVM_REPO}/tf2-gpu.${arr_version[0]}-${arr_version[1]}.${DLVM_PY_VERSION}
    ADDITIONAL_PACKAGES="tensorflow==${tf_version}"
  fi

  echo "Using compatible tf2-gpu image $BASE_IMAGE as base"
fi

beam_version=$(docker run --rm --entrypoint=python ${wheel_builder_tag} -c 'import apache_beam as beam; print(beam.version.__version__)')
# Run docker build command.
docker build -t ${DOCKER_IMAGE_REPO}:${DOCKER_IMAGE_TAG} \
  -f tfx/tools/docker/${DOCKER_FILE} \
  --build-arg "TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BEAM_VERSION=${beam_version}" \
  --build-arg "ADDITIONAL_PACKAGES=${ADDITIONAL_PACKAGES}" \
  . "$@"

if [[ -n "${installed_tf_version}" && ! "${installed_tf_version}" =~ rc ]]; then
  # Double-check whether TF is re-installed.
  current_tf_version=$(_get_tf_version_of_image "${DOCKER_IMAGE_REPO}:${DOCKER_IMAGE_TAG}")
  # TODO(b/333895985): This should be rollbacked after the fix. The TF version
  # from the BASE_IMAGE is wrongly set (expected: 2.15.1, actually: 2.15.0).
  # if [[ "${installed_tf_version}" != "${current_tf_version}" ]]; then
  #   echo "Error: TF version has changed from ${installed_tf_version} to ${current_tf_version}."
  #   exit 1
  # fi
fi


# Remove the temp image.
docker rmi ${wheel_builder_tag}

# Cleanup: revert patch and remove downloaded wheel
if [[ "${patch_applied}" == "true" ]]; then
  echo "Reverting tfx.patch..."
  git apply -R patches/tfx.patch
fi

echo "Removing downloaded wheel..."
rm -rf tfx/tools/docker/wheels
