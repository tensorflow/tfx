# Lint as: python3
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
"""Component config for docker run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Text, Union

from tfx.orchestration.config import base_component_config


class DockerComponentConfig(base_component_config.BaseComponentConfig):
  """Component config which holds docker run args.

  Attributes:
    docker_server_url: URL to the Docker server. For example,
      `unix:///var/run/docker.sock` or `tcp://127.0.0.1:1234`. Uses environment
        viarable to initialize the docker client if this parameter is not set.
        Default: `None`.
    environment: Environment variables to set inside the container, as a
      dictionary or a list of strings in the format ["SOMEVARIABLE=xxx"].
    name: The name for this container.
    privileged: Give extended privileges to this container. Default: `False`.
    remove: Remove the container when it has finished running. Default: `False`.
    user: Username or UID to run commands as inside the container.
    volumes: A dictionary to configure volumes mounted inside the container. The
      key is either the host path or a volume name, and the value is a
      dictionary with the keys: {bind: mode}.
      For example:
      `{'/home/user1': {'bind': '/mnt/vol2', 'mode': 'rw'},
       '/var/www': {'bind': '/mnt/vol1', 'mode': 'ro'}}`
    additional_run_args: Additional run args to pass to
      `docker.client.containers.run`. See
      https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run.
  """

  def __init__(self,
               docker_server_url: Text = None,
               environment: Union[Dict[Text, Text], List[Text]] = None,
               name: Text = None,
               privileged: bool = False,
               user: Union[Text, int] = None,
               volumes: Union[Dict[Text, Dict[Text, Text]], List[Text]] = None,
               **kwargs):
    self.docker_server_url = docker_server_url
    self.environment = environment
    self.name = name
    self.privileged = privileged
    self.user = user
    self.volumes = volumes
    self.additional_run_args = kwargs

  def to_run_args(self):
    if self.additional_run_args:
      args = self.additional_run_args.copy()
    else:
      args = {}
    args.update(privileged=self.privileged)
    if self.environment:
      args.update(environment=self.environment)
    if self.name:
      args.update(name=self.name)
    if self.user:
      args.update(user=self.user)
    if self.volumes:
      args.update(volumes=self.volumes)
    return args
