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

from typing import Any, Dict, List, cast

import absl

import docker

from tfx import types
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.components.base import executor_spec
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
                      (executor_spec.ExecutorContainerSpec,
                       executor_specs.TemplatedExecutorContainerSpec))

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[str, List[types.Artifact]],
                    output_dict: Dict[str, List[types.Artifact]],
                    exec_properties: Dict[str, Any]) -> None:
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
    container = client.containers.run(
        image=executor_container_spec.image,
        entrypoint=executor_container_spec.command,
        command=executor_container_spec.args,
        detach=True,
        **run_args)

    # Streaming logs
    for log in container.logs(stream=True):
      absl.logging.info('Docker: ' + log.decode('utf-8'))
    exit_code = container.wait()['StatusCode']
    if exit_code != 0:
      raise RuntimeError(
          'Container exited with error code "{}"'.format(exit_code))
    # TODO(b/141192583): Report data to publisher
    # - report container digest
    # - report replaced command line entrypoints
    # - report docker run args
