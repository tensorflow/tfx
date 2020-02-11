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
"""Executor specifications for defining what to to execute."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Dict, List, Optional, Text, Type

from six import with_metaclass

from tfx.components.base import base_executor
from tfx.utils import json_utils


class ExecutorSpec(with_metaclass(abc.ABCMeta, json_utils.Jsonable)):
  """A specification for a component executor.

  An instance of ExecutorSpec describes the implementation of a component.
  """


class ExecutorClassSpec(ExecutorSpec):
  """A specification of executor class.

  Attributes:
    executor_class: a subclass of base_executor.BaseExecutor used to execute
      this component (required).
  """

  def __init__(self, executor_class: Type[base_executor.BaseExecutor]):
    if not executor_class:
      raise ValueError('executor_class is required')
    self.executor_class = executor_class
    super(ExecutorClassSpec, self).__init__()


class ExecutorContainerSpec(ExecutorSpec):
  """A specifcation of a container.

  The spec includes image, command line entrypoint and arguments for a
  container. For example:

  spec = ExecutorContainerSpec(
    image='docker/whalesay',
    command=['sh', '-c', 'cowsay "$0" > $1'],
    args=[
        'hello world',
        '/tmp/output.txt',
    ],
    output_path_uris={
        '/tmp/output.txt': '{{output_dict["text"][0].uri}}',
    },
  )

  Attributes:
    image: Container image that has executor application. Assumption is that
      this container image is separately release-managed, and tagged/versioned
      accordingly.
    command: Container entrypoint array. Not executed within a shell. The docker
      image's ENTRYPOINT is used if this is not provided. The Jinja templating
      mechanism is used for constructing a user-specified command-line
      invocation based on input and output metadata at runtime.
    args: Arguments to the container entrypoint. The docker image's CMD is used
      if this is not provided. The Jinja templating mechanism is used for
      constructing a user-specified command-line invocation based on input and
      output metadata at runtime.
    input_path_uris: Maps local container paths to input artifact URIs.
      The keys are container-local paths. The values are URIs.
      At this moment only GCS URIs are supported (also local paths when running
      locally).
      When container starts, the artifact data will be made available at those
      local paths (downloaded or mounted).
    output_path_uris: Maps local container paths to output artifact URIs.
      When container finishes, the data from the specified paths will be stored
      at the provided URIs (uploaded or mounted).
  """

  def __init__(
      self,
      image: Text,
      command: List[Text] = None,
      args: List[Text] = None,
      input_path_uris: Optional[Dict[Text, Text]] = None,
      output_path_uris: Optional[Dict[Text, Text]] = None,
  ):
    if not image:
      raise ValueError('image cannot be None or empty.')
    self.image = image
    self.command = command
    self.args = args
    self.input_path_uris = input_path_uris
    self.output_path_uris = output_path_uris
    super(ExecutorContainerSpec, self).__init__()
