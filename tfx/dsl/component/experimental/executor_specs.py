# Lint as: python2, python3
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
"""Executor specifications for components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator
from typing import List, Optional, Text, Union

from tfx import types
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import executor_spec
from tfx.dsl.placeholder import placeholder
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import placeholder_pb2

from google.protobuf import message


class TemplatedExecutorContainerSpec(executor_spec.ExecutorSpec):
  """Experimental: Describes a command-line program inside a container.

  This class is similar to ExecutorContainerSpec, but uses structured
  placeholders instead of jinja templates for constructing container commands
  based on input and output artifact metadata. See placeholders.py for a list of
  supported placeholders.
  The spec includes the container image name and the command line
  (entrypoint plus arguments) for a program inside the container.

  Example:

  class MyTrainer(base_component.BaseComponent)
    class MyTrainerSpec(types.ComponentSpec):
      INPUTS = {
          'training_data':
              component_spec.ChannelParameter(type=standard_artifacts.Dataset),
      }
      OUTPUTS = {
          'model':
              component_spec.ChannelParameter(type=standard_artifacts.Model),
      }
      PARAMETERS = {
          'num_training_steps': component_spec.ExecutionParameter(type=int),
      }

    SPEC_CLASS = MyTrainerSpec
    EXECUTOR_SPEC = executor_specs.TemplatedExecutorContainerSpec(
        image='gcr.io/my-project/my-trainer',
        command=[
            'python3', 'my_trainer',
            '--training_data_uri', InputUriPlaceholder('training_data'),
            '--model_uri', OutputUriPlaceholder('model'),
            '--num_training-steps', InputValuePlaceholder('num_training_steps'),
        ]
    )

  Attributes:
    image: Container image name.
    command: Container entrypoint command-line. Not executed within a shell.
      The command-line can use placeholder objects that will be replaced at
      the compilation time. Note: Jinja templates are not supported.
    args: Container entrypoint command-args.
  """

  def __init__(
      self,
      image: Text,
      command: Optional[List[placeholders.CommandlineArgumentType]] = None,
      args: Optional[List[placeholders.CommandlineArgumentType]] = None,
  ):
    self.image = image
    self.command = command or []
    self.args = args or []
    super(TemplatedExecutorContainerSpec, self).__init__()

  def __eq__(self, other) -> bool:
    return (isinstance(other, self.__class__) and self.image == other.image and
            self.command == other.command)

  def __ne__(self, other) -> bool:
    return not self.__eq__(other)

  def _recursively_encode(
      self, ph: placeholders.CommandlineArgumentType
  ) -> Union[str, placeholder.Placeholder]:
    if isinstance(ph, str):
      return ph
    elif isinstance(ph, placeholders.InputValuePlaceholder):
      return placeholder.input(ph.input_name)[0]
    elif isinstance(ph, placeholders.InputUriPlaceholder):
      return placeholder.input(ph.input_name)[0].uri
    elif isinstance(ph, placeholders.OutputUriPlaceholder):
      return placeholder.output(ph.output_name)[0].uri
    elif isinstance(ph, placeholders.ConcatPlaceholder):
      # operator.add wil use the overloaded __add__ operator for Placeholder
      # instances.
      return functools.reduce(
          operator.add,
          [self._recursively_encode(item) for item in ph.items])
    else:
      raise TypeError(
          ('Unsupported type of placeholder arguments: "{}".'
           ' Supported types are {}.')
          .format(type(ph), str(placeholders.CommandlineArgumentType)))

  def encode(
      self,
      component_spec: Optional[types.ComponentSpec] = None) -> message.Message:
    """Encodes ExecutorSpec into an IR proto for compiling.

    This method will be used by DSL compiler to generate the corresponding IR.

    Args:
      component_spec: Optional. The ComponentSpec to help with the encoding.

    Returns:
      An executor spec proto.
    """
    result = executable_spec_pb2.ContainerExecutableSpec()
    result.image = self.image
    for command in self.command:
      cmd = result.commands.add()
      str_or_placeholder = self._recursively_encode(command)
      if isinstance(str_or_placeholder, str):
        expression = placeholder_pb2.PlaceholderExpression()
        expression.value.string_value = str_or_placeholder
        cmd.CopyFrom(expression)
      else:
        cmd.CopyFrom(self._recursively_encode(command).encode())

    for arg in self.args:
      cmd = result.args.add()
      str_or_placeholder = self._recursively_encode(arg)
      if isinstance(str_or_placeholder, str):
        expression = placeholder_pb2.PlaceholderExpression()
        expression.value.string_value = str_or_placeholder
        cmd.CopyFrom(expression)
      else:
        cmd.CopyFrom(self._recursively_encode(arg).encode())
    return result
