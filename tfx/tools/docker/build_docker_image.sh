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

# Parse arguments for USE_CPP_WHEELS_FROM_TEMP and other custom flags
USE_CPP_WHEELS_FROM_TEMP=false
CLEAN_CPP_TEMP_CACHE=false
NEW_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-rebuild|--skip-rebuild)
      USE_CPP_WHEELS_FROM_TEMP=true
      shift
      ;;
    --clean-cache)
      CLEAN_CPP_TEMP_CACHE=true
      shift
      ;;
    *)
      NEW_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${NEW_ARGS[@]}"

export BEAM_VERSION=${BEAM_VERSION}
export BASE_IMAGE=${BASE_IMAGE}


DOCKER_IMAGE_REPO=${DOCKER_IMAGE_REPO:-"tensorflow/tfx"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"latest"}
DOCKER_FILE=${DOCKER_FILE:-"Dockerfile"}

TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR:-""}
echo "Env for TFX_DEPENDENCY_SELECTOR is set as ${TFX_DEPENDENCY_SELECTOR}"


# Programmatically remove TFX sibling libraries from dependencies.py
echo "Programmatically editing tfx/dependencies.py to remove sibling dependencies..."
python3 -c "
import re
with open('tfx/dependencies.py', 'r') as f:
    content = f.read()
# Remove tfdv, tfma, tft, tfx-bsl blocks from make_required_install_packages
content = re.sub(r'\"tensorflow-data-validation\".*?\),', '', content, flags=re.DOTALL)
content = re.sub(r'\"tensorflow-model-analysis\".*?\),', '', content, flags=re.DOTALL)
content = re.sub(r'\"tensorflow-transform\".*?\),', '', content, flags=re.DOTALL)
content = re.sub(r'\"tfx-bsl\".*?\),', '', content, flags=re.DOTALL)
content = re.sub(r'\"ml-metadata\".*?\),', '', content, flags=re.DOTALL)
content = re.sub(r'\"tensorflow-cloud>=0.1,<0.2\",', '', content)
with open('tfx/dependencies.py', 'w') as f:
    f.write(content)
"

# Programmatically remove pins for components built from source or downloaded as wheels
for f in nightly_test_constraints.txt test_constraints.txt tfx/tools/docker/requirements.txt; do
  if [[ -f "$f" ]]; then
    echo "Removing pins from $f..."
    # Remove exact version pins or range constraints for the following packages
    sed -i '/tensorflow-cloud/d' "$f"
    sed -i '/tensorflow-data-validation/d' "$f"
    sed -i '/tensorflow-model-analysis/d' "$f"
    sed -i '/tensorflow-transform/d' "$f"
    sed -i '/tfx-bsl/d' "$f"
    sed -i '/ml-metadata/d' "$f"
    sed -i '/ml_metadata/d' "$f"
    sed -i '/tensorflow-metadata/d' "$f"
    sed -i '/absl-py/d' "$f"
    sed -i '/astunparse/d' "$f"
    sed -i '/flatbuffers/d' "$f"
    sed -i '/gast/d' "$f"
    sed -i '/google-/d' "$f"
    sed -i '/google_/d' "$f"
    sed -i '/grpcio/d' "$f"
    sed -i '/h5py/d' "$f"
    sed -i '/keras/d' "$f"
    sed -i '/libclang/d' "$f"
    sed -i '/ml-dtypes/d' "$f"
    sed -i '/ml_dtypes/d' "$f"
    sed -i '/numpy/d' "$f"
    sed -i '/opt-einsum/d' "$f"
    sed -i '/opt_einsum/d' "$f"
    sed -i '/packaging/d' "$f"
    sed -i '/protobuf/d' "$f"
    sed -i '/requests/d' "$f"
    sed -i '/six/d' "$f"
    sed -i '/termcolor/d' "$f"
    sed -i '/typing-extensions/d' "$f"
    sed -i '/typing_extensions/d' "$f"
    sed -i '/wrapt/d' "$f"
    sed -i '/kfp/d' "$f"
    sed -i '/kubernetes/d' "$f"
    sed -i '/urllib3/d' "$f"
    sed -i '/cryptography/d' "$f"
    sed -i '/proto-plus/d' "$f"
    sed -i '/proto_plus/d' "$f"
    sed -i '/opentelemetry/d' "$f"
    sed -i '/apache-/d' "$f"
  fi
done

mkdir -p tfx/tools/docker/wheels
rm -rf tfx/tools/docker/wheels/*

# Build tensorflow-model-analysis wheel from master
echo "Building tensorflow-model-analysis wheel from master..."
TFMA_BUILD_DIR="/tmp/tfma_build_$(date +%s)"
git clone --depth 1 https://github.com/tensorflow/model-analysis.git "${TFMA_BUILD_DIR}"
pushd "${TFMA_BUILD_DIR}"
TFX_DEPENDENCY_SELECTOR=NIGHTLY python setup.py bdist_wheel
popd
cp "${TFMA_BUILD_DIR}"/dist/*.whl tfx/tools/docker/wheels/
rm -rf "${TFMA_BUILD_DIR}"

# Build tensorflow-transform wheel from master
echo "Building tensorflow-transform wheel from master..."
TFT_BUILD_DIR="/tmp/tft_build_$(date +%s)"
git clone --depth 1 https://github.com/tensorflow/transform.git "${TFT_BUILD_DIR}"
pushd "${TFT_BUILD_DIR}"
# Loosen the hardcoded tfx-bsl git URL pin in setup.py to support installing our local compiled wheel
sed -i 's|tfx-bsl@git+https://github.com/tensorflow/tfx-bsl@master|tfx-bsl>=1.18.0.dev|g' setup.py
TFX_DEPENDENCY_SELECTOR=NIGHTLY python setup.py bdist_wheel
popd
cp "${TFT_BUILD_DIR}"/dist/*.whl tfx/tools/docker/wheels/
rm -rf "${TFT_BUILD_DIR}"

# Download tensorflow-cloud wheel
echo "Downloading tensorflow-cloud wheel..."
TFC_WHEEL_URL="https://files.pythonhosted.org/packages/4b/bc/da205a15aaf22c1fda1f58552990d17d532a8573af6830e3663730ed485b/tensorflow_cloud-0.1.16-py3-none-any.whl"
TFC_WHEEL_FILE="tensorflow_cloud-0.1.16-py3-none-any.whl"
curl -L -o tfx/tools/docker/wheels/${TFC_WHEEL_FILE} ${TFC_WHEEL_URL}

function _get_tf_version_of_image() {
  local img="$1"
  docker run --rm --entrypoint=python ${img} -c 'import tensorflow as tf; print(tf.__version__)'
}

function _get_beam_version_of_image() {
  local img="$1"
  docker run --rm --entrypoint=python ${img} -c 'import apache_beam as beam; print(beam.version.__version__)'
}

# Base image to extend: This should be a deep learning image with a compatible
# TensorFlow version. See
# https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
# for possible images to use here.

if [ "$CLEAN_CPP_TEMP_CACHE" = "true" ]; then
  echo "Pruning Docker builder cache..."
  docker builder prune -a -f
fi

# Use timestmap-rand for tag, to avoid collision of concurrent runs.
if [[ -z "$BASE_IMAGE" || -z "$BEAM_VERSION" ]]; then
  echo "Discovering versions using lightweight container..."
  discovery_tag="tfx-beam-discovery:$(date +%s)-$RANDOM"
  docker build -t ${discovery_tag} -f tfx/tools/docker/Dockerfile.beam_discovery .
  discovery_output=$(docker run --rm ${discovery_tag})
  tf_version=$(echo "${discovery_output}" | cut -d'|' -f1)
  beam_version_detected=$(echo "${discovery_output}" | cut -d'|' -f2)
  docker rmi ${discovery_tag}

  if [[ -z "$BEAM_VERSION" ]]; then
    BEAM_VERSION=${beam_version_detected}
  fi
  echo "Detected Beam version as ${BEAM_VERSION}"
else
  echo "Using override base image $BASE_IMAGE"
fi

if [[ -z "$BASE_IMAGE" ]]; then
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

# Run docker build command.
docker build --progress=plain -t ${DOCKER_IMAGE_REPO}:${DOCKER_IMAGE_TAG} \
  -f tfx/tools/docker/${DOCKER_FILE} \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BEAM_VERSION=${BEAM_VERSION}" \
  --build-arg "ADDITIONAL_PACKAGES=${ADDITIONAL_PACKAGES}" \
  --build-arg USE_CPP_WHEELS_FROM_TEMP=${USE_CPP_WHEELS_FROM_TEMP} \
  --build-arg CLEAN_CPP_TEMP_CACHE=${CLEAN_CPP_TEMP_CACHE} \
  --build-arg "TFX_DEPENDENCY_SELECTOR=${TFX_DEPENDENCY_SELECTOR}" \
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

# Cleanup: revert edits to dependencies.py and constraint files
echo "Reverting edits to dependencies.py and constraint files..."
git checkout tfx/dependencies.py test_constraints.txt nightly_test_constraints.txt tfx/tools/docker/requirements.txt

echo "Removing downloaded wheel..."
rm -rf tfx/tools/docker/wheels
