# Lint as: python2, python3
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
"""Docker component launcher which launches a container in docker environment ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
from typing import Any, Dict, List, Text, cast

import absl
import docker
import tensorflow as tf

from tfx import types
from tfx.components.base import executor_spec
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import container_common


class DockerComponentLauncher(base_component_launcher.BaseComponentLauncher):
  """Responsible for launching a container executor."""

  @classmethod
  def can_launch(
      cls, component_executor_spec: executor_spec.ExecutorSpec,
      component_config: base_component_config.BaseComponentConfig) -> bool:
    """Checks if the launcher can launch the executor spec."""
    if component_config and not isinstance(
        component_config, docker_component_config.DockerComponentConfig):
      return False

    return isinstance(component_executor_spec,
                      executor_spec.ExecutorContainerSpec)

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""

    executor_container_spec = cast(executor_spec.ExecutorContainerSpec,
                                   self._component_executor_spec)
    if self._component_config:
      docker_config = cast(docker_component_config.DockerComponentConfig,
                           self._component_config)
    else:
      docker_config = docker_component_config.DockerComponentConfig()

    # Replace container spec with jinja2 template.
    executor_container_spec = container_common.resolve_container_template(
        executor_container_spec, input_dict, output_dict, exec_properties)

    absl.logging.info('Container spec: %s' % vars(executor_container_spec))
    absl.logging.info('Docker config: %s' % vars(docker_config))

    # Call client.containers.run and wait for completion.
    # ExecutorContainerSpec follows k8s container spec which has different
    # names to Docker's container spec. It's intended to set command to docker's
    # entrypoint and args to docker's command.
    if docker_config.docker_server_url:
      client = docker.DockerClient(base_url=docker_config.docker_server_url)
    else:
      client = docker.from_env()

    run_args = docker_config.to_run_args()

    # Preparing the volume mounts for input and output files
    volume_mounts = run_args.pop('volumes', {}) or {}
    container_path_to_host_path = {}
    host_artifact_dir = tempfile.mkdtemp()
    for path in (list((executor_container_spec.input_path_uris or {}).keys()) +
                 list((executor_container_spec.output_path_uris or {}).keys())):
      container_dir = os.path.dirname(path)  # TODO(avolkov) Fix for Windows
      container_filename = os.path.basename(path)
      host_dir = os.path.join(host_artifact_dir,
                              container_dir.replace('/', '_'))
      host_path = os.path.join(host_dir, container_filename)
      os.makedirs(host_dir)
      container_path_to_host_path[path] = host_path
      volume_mounts[host_dir] = dict(
          bind=container_dir,
          mode='rw',
      )

    # Downloading the input files
    for path, uri in (executor_container_spec.input_path_uris or {}).items():
      src = uri
      dst = container_path_to_host_path[path]
      absl.logging.info('Downloading from "{}" to "{}"'.format(src, dst))
      tf.io.gfile.copy(src, dst)

    container = client.containers.run(
        image=executor_container_spec.image,
        entrypoint=executor_container_spec.command,
        command=executor_container_spec.args,
        detach=True,
        volumes=volume_mounts,
        **run_args)

    # Streaming logs
    for log in container.logs(stream=True):
      absl.logging.info('Docker: ' + log.decode('utf-8'))
    exit_code = container.wait()['StatusCode']
    if exit_code != 0:
      raise RuntimeError(
          'Container exited with error code "{}"'.format(exit_code))

    # Uploading the output files
    for path, uri in (executor_container_spec.output_path_uris or {}).items():
      src = container_path_to_host_path[path]
      dst = uri
      absl.logging.info('Uploading from "{}" to "{}"'.format(src, dst))
      # Workaround for b/150515270
      if tf.io.gfile.exists(dst):
        if tf.io.gfile.isdir(dst):
          if tf.io.gfile.glob(dst + '/*'):
            absl.logging.error(
                'Output artifact URI "{}" is an existing non-empty directory.'
                .format(dst))
          else:
            tf.io.gfile.rmtree(dst)
        else:
          absl.logging.error(
              'Destination URI already exists: "{}"..'.format(dst))
      tf.io.gfile.copy(src, dst)

    # Cleaning up the temporary directory with artifact files.
    shutil.rmtree(host_artifact_dir)

    # TODO(b/141192583): Report data to publisher
    # - report container digest
    # - report replaced command line entrypoints
    # - report docker run args
