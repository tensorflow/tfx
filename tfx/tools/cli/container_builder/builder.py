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
"""Builds the container image using files in the current directory."""

import os
from typing import Any, Dict, Iterable, Optional

import click
import docker
from docker import utils as docker_utils

from tfx.tools.cli.container_builder import dockerfile
from tfx.tools.cli.container_builder import labels


def _get_image_repo(image: str) -> str:
  """Extracts image name before ':' which is REPO part of the image name."""
  image_fields = image.split(':')
  if len(image_fields) > 2:
    raise ValueError(f'Too many ":" in the image name: {image}')
  return image_fields[0]


def _print_docker_log_stream(stream: Iterable[Dict[str, Any]],
                             message_key: str,
                             newline: bool = False):
  for item in stream:
    if message_key in item:
      click.echo('[Docker] ' + item[message_key], nl=newline)


def build(target_image: str,
          base_image: Optional[str] = None,
          dockerfile_name: Optional[str] = None,
          setup_py_filename: Optional[str] = None) -> str:
  """Build containers.

  Generates a dockerfile if needed and build a container image using docker SDK.

  Args:
    target_image: the target image path to be built.
    base_image: the image path to use as the base image.
    dockerfile_name: the dockerfile name, which is stored in the workspace
      directory. The default workspace directory is '.' and cannot be changed
      for now.
    setup_py_filename: the setup.py file name, which is used to build a python
      package for the workspace directory. If not specified, the whole directory
      is copied and PYTHONPATH is configured.

  Returns:
    Built image name with sha256 id.
  """
  dockerfile_name = dockerfile_name or os.path.join(labels.BUILD_CONTEXT,
                                                    labels.DOCKERFILE_NAME)

  dockerfile.Dockerfile(
      filename=dockerfile_name,
      setup_py_filename=setup_py_filename,
      base_image=base_image)

  # Uses Low-level API for log streaming.
  docker_low_client = docker.APIClient(**docker_utils.kwargs_from_env())
  log_stream = docker_low_client.build(
      path=labels.BUILD_CONTEXT,
      dockerfile=dockerfile_name,
      tag=target_image,
      rm=True,
      decode=True,
  )
  _print_docker_log_stream(log_stream, 'stream')

  docker_client = docker.from_env()
  log_stream = docker_client.images.push(
      repository=target_image, stream=True, decode=True)
  _print_docker_log_stream(log_stream, 'status', newline=True)

  image_id = docker_client.images.get_registry_data(target_image).id
  return f'{_get_image_repo(target_image)}@{image_id}'
