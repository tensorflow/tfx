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

from typing import List, Text, Union

from tfx.components.base import executor_spec
from tfx.dsl.component.experimental import placeholders


CommandlineArgumentType = Union[
    Text,
    placeholders.InputValuePlaceholder,
    placeholders.InputUriPlaceholder,
    placeholders.OutputUriPlaceholder,
    placeholders.ConcatPlaceholder,
]


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
  """

  # The "command" parameter holds the name of the program and its arguments.
  # The "command" parameter is required to enable instrumentation.
  # The command-line is often split into command+args, but here "args" would be
  # redundant since all items can just be added to "command".

  def __init__(
      self,
      image: Text,
      command: List[CommandlineArgumentType],
  ):
    self.image = image
    self.command = command
    super(TemplatedExecutorContainerSpec, self).__init__()

  def __eq__(self, other) -> bool:
    return (isinstance(other, self.__class__) and self.image == other.image and
            self.command == other.command)

  def __ne__(self, other) -> bool:
    return not self.__eq__(other)
