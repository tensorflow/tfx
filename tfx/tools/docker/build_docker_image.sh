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
  BASE_IMAGE=${DLVM_REPO}/tf2-gpu.${arr_version[0]}-${arr_version[1]}

  # Check the availability of the DLVM image.
  if gcloud container images list --repository=${DLVM_REPO} | grep "${BASE_IMAGE}" ; then
    # TF shouldn't be re-installed so we pin TF version in Pip install.
    installed_tf_version=$(_get_tf_version_of_image "${BASE_IMAGE}")
    ADDITIONAL_PACKAGES="tensorflow==${installed_tf_version}"
  else
    # Fallback to the image of the previous version but also install the newest
    # TF version.
    arr_version[1]=$((arr_version[1] - 1))
    BASE_IMAGE=${DLVM_REPO}/tf2-gpu.${arr_version[0]}-${arr_version[1]}
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

if [[ -n "${installed_tf_version}" ]]; then
  # Double-check whether TF is re-installed.
  current_tf_version=$(_get_tf_version_of_image "${DOCKER_IMAGE_REPO}:${DOCKER_IMAGE_TAG}")
  if [[ "${installed_tf_version}" != "${current_tf_version}" ]]; then
    echo "Error: TF version has changed from ${installed_tf_version} to ${current_tf_version}."
    exit 1
  fi
fi


# Remove the temp image.
docker rmi ${wheel_builder_tag}
