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
"""Python wrapper for build_docker_image.sh."""

import os
import subprocess

from absl import logging
import docker
from tfx.utils import timer

# Set longer timeout when pushing an image. Default timeout is 60 seconds.
_DOCKER_TIMEOUT_SECONDS = 60 * 5


def build_docker_image(container_image: str,
                       repo_base: str,
                       push: bool = False):
  """Build docker image using `tfx/tools/docker/Dockerfile`.

  Args:
    container_image: Docker container image name.
    repo_base: The src path to use to build docker image.
    push: Push the built image to the registry and delete local image.
  """

  docker_image_repo, docker_image_tag = container_image.split(':')
  # Default to NIGHTLY. GIT_MASTER might be better to use the latest source,
  # But it takes too long (~1h) to build packages from scratch. If some changes
  # in a dependent package break tests, just run a nightly build of dependent
  # package again after fixing it.
  dependency_selector = os.getenv('TFX_DEPENDENCY_SELECTOR') or 'NIGHTLY'

  logging.info('Building image %s with %s dependency', container_image,
               dependency_selector)
  with timer.Timer('BuildingTFXContainerImage'):
    subprocess.run(
        ['tfx/tools/docker/build_docker_image.sh'],
        cwd=repo_base,
        env={
            'DOCKER_IMAGE_REPO': docker_image_repo,
            'DOCKER_IMAGE_TAG': docker_image_tag,
            'TFX_DEPENDENCY_SELECTOR': dependency_selector,
        },
        check=True)
  if not push:
    return

  client = docker.from_env(timeout=_DOCKER_TIMEOUT_SECONDS)
  logging.info('Pushing image %s', container_image)
  with timer.Timer('PushingTFXContainerImage'):
    client.images.push(repository=container_image)
  with timer.Timer('DeletingLocalTFXContainerImage'):
    client.images.remove(image=container_image)
