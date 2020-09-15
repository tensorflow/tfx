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

set -u -x

DOCKER_IMAGE_REPO=${DOCKER_IMAGE_REPO:-"tensorflow/tfx"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"latest"}
DOCKER_FILE=${DOCKER_FILE:-"Dockerfile"}
GATHER_THIRD_PARTY_LICENSES=${GATHER_THIRD_PARTY_LICENSES:-"true"}

# Run docker build command.
docker build -t ${DOCKER_IMAGE_REPO}:${DOCKER_IMAGE_TAG} -f tfx/tools/docker/${DOCKER_FILE} . --build-arg gather_third_party_licenses=${GATHER_THIRD_PARTY_LICENSES} "$@"
