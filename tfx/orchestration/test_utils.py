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
"""Common utilities for testing various runners."""

import contextlib
import os
import random
import string
import subprocess
import time

from absl import logging
import docker


class Timer:
  """Helper class to time operations in pipeline e2e tests."""

  def __init__(self, operation: str):
    """Creates a context object to measure time taken.

    Args:
      operation: A description of the operation being measured.
    """
    self._operation = operation

  def __enter__(self):
    self._start = time.time()

  def __exit__(self, *unused_args):
    self._end = time.time()

    logging.info(
        'Timing Info >> Operation: %s Elapsed time in seconds: %d',
        self._operation, self._end - self._start)


def random_id() -> str:
  """Generates a random string that is also a valid Kubernetes DNS name.

  Returns:
    A random string valid for Kubernetes DNS name.
  """
  now = time.time()
  random.seed(now)
  alpha = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
  return f'{int(now)}-{alpha}'


# Set longer timeout when pushing an image. Default timeout is 60 seconds.
_DOCKER_TIMEOUT_SECONDS = 60 * 5


def build_docker_image(container_image: str, repo_base: str):
  """Build docker image using `tfx/tools/docker/Dockerfile`.

  Args:
    container_image: Docker container image name.
    repo_base: The src path to use to build docker image.
  """

  [docker_image_repo, docker_image_tag] = container_image.split(':')
  envs = {
      'DOCKER_IMAGE_REPO': docker_image_repo,
      'DOCKER_IMAGE_TAG': docker_image_tag
  }

  # UNCONSTRAINED means that we will use the latest version which is built from
  # latest source. But it takes too long (~1h) to build packages from
  # GIT_MASTER. So we fallback to NIGHTLY here. If some changes in a dependent
  # package break tests, just run a nightly build of dependent package again.
  # TODO(b/181290953): Use UNCONSTRAINED as it is if we can supply latest wheel
  # packages for TFX family libraries to the image build script.
  dependency_selector = os.getenv('TFX_DEPENDENCY_SELECTOR')
  if dependency_selector == 'UNCONSTRAINED':
    dependency_selector = 'NIGHTLY'

  if dependency_selector:
    envs['TFX_DEPENDENCY_SELECTOR'] = dependency_selector

  logging.info('Building image %s with env:%s', container_image, envs)
  with Timer('BuildingTFXContainerImage'), _chdir(repo_base):
    subprocess.check_call(
        args=[
            os.path.join(repo_base, 'tfx/tools/docker/build_docker_image.sh'),
        ],
        env=envs,
        shell=True,
    )


@contextlib.contextmanager
def _chdir(path):
  old_cwd = os.getcwd()
  try:
    os.chdir(path)
    logging.info('cwd changed from %s to %s', old_cwd, path)
    yield
  finally:
    os.chdir(old_cwd)
    logging.info('cwd changed back to %s', old_cwd)


def build_and_push_docker_image(container_image: str, repo_base: str):
  """Build and push docker image using `tfx/tools/docker/Dockerfile`.

  Note: The local copy of the image will be deleted after push.

  Args:
    container_image: Docker container image name.
    repo_base: The src path to use to build docker image.
  """
  build_docker_image(container_image, repo_base)

  client = docker.from_env(timeout=_DOCKER_TIMEOUT_SECONDS)
  logging.info('Pushing image %s', container_image)
  with Timer('PushingTFXContainerImage'):
    client.images.push(repository=container_image)
  with Timer('DeletingLocalTFXContainerImage'):
    client.images.remove(image=container_image)
