# Lint as: python3
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
"""Functions for creating container components."""

from typing import Any, Callable, Dict, List, Text

from tfx.dsl.component.experimental import component_utils
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.types import channel_utils
from tfx.types import component_spec


def create_container_component(
    name: Text,
    image: Text,
    command: List[placeholders.CommandlineArgumentType],
    inputs: Dict[Text, Any] = None,
    outputs: Dict[Text, Any] = None,
    parameters: Dict[Text, Any] = None,
) -> Callable[..., base_component.BaseComponent]:
  """Creates a container-based component.

  Args:
    name: The name of the component
    image: Container image name.
    command: Container entrypoint command-line. Not executed within a shell. The
      command-line can use placeholder objects that will be replaced at the
      compilation time. The placeholder objects can be imported from
      tfx.dsl.component.experimental.placeholders. Note that Jinja templates are
      not supported.
    inputs: The list of component inputs
    outputs: The list of component outputs
    parameters: The list of component parameters

  Returns:
    Component that can be instantiated and user inside pipeline.

  Example:

    component = create_container_component(
        name='TrainModel',
        inputs={
            'training_data': Dataset,
        },
        outputs={
            'model': Model,
        },
        parameters={
            'num_training_steps': int,
        },
        image='gcr.io/my-project/my-trainer',
        command=[
            'python3', 'my_trainer',
            '--training_data_uri', InputUriPlaceholder('training_data'),
            '--model_uri', OutputUriPlaceholder('model'),
            '--num_training-steps', InputValuePlaceholder('num_training_steps'),
        ]
    )
  """
  if not name:
    raise ValueError('Component name cannot be empty.')

  if inputs is None:
    inputs = {}
  if outputs is None:
    outputs = {}
  if parameters is None:
    parameters = {}

  input_channel_parameters = {}
  output_channel_parameters = {}
  output_channels = {}
  execution_parameters = {}

  for input_name, channel_type in inputs.items():
    # TODO(b/155804245) Sanitize the names so that they're valid python names
    input_channel_parameters[input_name] = (
        component_spec.ChannelParameter(type=channel_type,))

  for output_name, channel_type in outputs.items():
    # TODO(b/155804245) Sanitize the names so that they're valid python names
    output_channel_parameters[output_name] = (
        component_spec.ChannelParameter(type=channel_type))
    artifact = channel_type()
    channel = channel_utils.as_channel([artifact])
    output_channels[output_name] = channel

  for param_name, parameter_type in parameters.items():
    # TODO(b/155804245) Sanitize the names so that they're valid python names

    execution_parameters[param_name] = (
        component_spec.ExecutionParameter(type=parameter_type))

  default_init_args = {**output_channels}

  return component_utils.create_tfx_component_class(
      name=name,
      tfx_executor_spec=executor_specs.TemplatedExecutorContainerSpec(
          image=image,
          command=command,
      ),
      input_channel_parameters=input_channel_parameters,
      output_channel_parameters=output_channel_parameters,
      execution_parameters=execution_parameters,
      default_init_args=default_init_args)
