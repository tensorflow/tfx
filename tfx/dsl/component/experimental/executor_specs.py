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

import functools
import operator
from typing import cast, List, Optional, Union

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
              component_spec.ChannelParameter(type=standard_artifacts.Examples),
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
      image: str,
      command: Optional[List[placeholders.CommandlineArgumentType]] = None,
      args: Optional[List[placeholders.CommandlineArgumentType]] = None,
  ):
    self.image = image
    self.command = command or []
    self.args = args or []
    super().__init__()

  def __eq__(self, other) -> bool:
    return (isinstance(other, self.__class__) and self.image == other.image and
            self.command == other.command)

  def __ne__(self, other) -> bool:
    return not self.__eq__(other)

  def _recursively_encode(
      self,
      ph: Union[placeholders.CommandlineArgumentType, placeholder.Placeholder,
                str],
      component_spec: Optional[types.ComponentSpec] = None
  ) -> Union[str, placeholder.Placeholder]:
    """This method recursively encodes placeholders.CommandlineArgumentType.

       The recursion ending condision is that the input ph is alerady a string
       or placeholder.Placeholder.

    Args:
      ph: The placeholder to encode.
      component_spec: Optional. The ComponentSpec to help with the encoding.

    Returns:
      The encoded placeholder in the type of string or placeholder.Placeholder.
    """
    if isinstance(ph, str) or isinstance(ph, placeholder.Placeholder):
      # If there is no place holder. Or if the placeholder is already a
      # new style placeholder.
      # No further encoding is needed.
      return cast(Union[str, placeholder.Placeholder], ph)
    elif isinstance(ph, placeholders.InputValuePlaceholder):
      if not component_spec:
        raise ValueError(
            'Requires component spec to encode InputValuePlaceholder.')
      if ph.input_name in component_spec.INPUTS:
        return placeholder.input(ph.input_name)[0].value
      elif ph.input_name in component_spec.PARAMETERS:
        return placeholder.exec_property(ph.input_name)
      else:
        raise ValueError(
            'For InputValuePlaceholder, input name must be in component\'s INPUTS or PARAMETERS.'
        )
    elif isinstance(ph, placeholders.InputUriPlaceholder):
      if component_spec and ph.input_name not in component_spec.INPUTS:
        raise ValueError(
            'For InputUriPlaceholder, input name must be in component\'s INPUTS.'
        )
      return placeholder.input(ph.input_name)[0].uri
    elif isinstance(ph, placeholders.OutputUriPlaceholder):
      if component_spec and ph.output_name not in component_spec.OUTPUTS:
        raise ValueError(
            'For OutputUriPlaceholder, output name must be in component\'s OUTPUTS.'
        )
      return placeholder.output(ph.output_name)[0].uri
    elif isinstance(ph, placeholders.ConcatPlaceholder):
      # operator.add wil use the overloaded __add__ operator for Placeholder
      # instances.
      return functools.reduce(
          operator.add,
          [self._recursively_encode(item, component_spec) for item in ph.items])
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
      str_or_placeholder = self._recursively_encode(command, component_spec)
      if isinstance(str_or_placeholder, str):
        expression = placeholder_pb2.PlaceholderExpression()
        expression.value.string_value = str_or_placeholder
        cmd.CopyFrom(expression)
      else:
        cmd.CopyFrom(str_or_placeholder.encode())

    for arg in self.args:
      cmd = result.args.add()
      str_or_placeholder = self._recursively_encode(arg, component_spec)
      if isinstance(str_or_placeholder, str):
        expression = placeholder_pb2.PlaceholderExpression()
        expression.value.string_value = str_or_placeholder
        cmd.CopyFrom(expression)
      else:
        cmd.CopyFrom(str_or_placeholder.encode())
    return result
