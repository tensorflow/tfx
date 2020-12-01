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
"""Component that launches CAIP custom training job with flexible interface."""

from typing import Any, Dict, List, Optional, Text

from tfx.dsl.component.experimental import component_utils
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration.kubeflow.v2.components.experimental import ai_platform_training_executor
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.utils import json_utils


def create_ai_platform_training(
    name: Text,
    project_id: Text,
    region: Optional[Text] = None,
    job_id: Optional[Text] = None,
    image_uri: Optional[Text] = None,
    args: Optional[List[placeholders.CommandlineArgumentType]] = None,
    # TODO(jxzheng): support Python training spec
    scale_tier: Optional[Text] = None,
    training_input: Optional[Dict[Text, Any]] = None,
    labels: Optional[Dict[Text, Text]] = None,
    inputs: Dict[Text, Any] = None,
    outputs: Dict[Text, Any] = None,
    parameters: Dict[Text, Any] = None,
) -> base_component.BaseComponent:
  """Creates a pipeline step that launches a AIP training job.

  The generated TFX component will have a component spec specified dynamically,
  through inputs/outputs/parameters in the following format:
  - inputs: A mapping from input name to the upstream channel connected. The
      artifact type of the channel will be automatically inferred.
  - outputs: A mapping from output name to the associated artifact type.
  - parameters: A mapping from execution property names to its associated value.
      Only primitive typed values are supported. Note that RuntimeParameter is
      not supported yet.

  For example:

  ```
  create_ai_platform_training(
    ...
    inputs: {
        # Assuming there is an upstream node example_gen, with an output
        # 'examples' of the type Examples.
        'examples': example_gen.outputs['examples'],
    },
    outputs: {
        'model': standard_artifacts.Model,
    },
    parameters: {
        'n_steps': 100,
        'optimizer': 'sgd',
    }
    ...
  )
  ```

  will generate a component instance with a component spec equivalent to:

  ```
  class MyComponentSpec(ComponentSpec):
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }
    OUTPUTS = {
        'model': ChannelParameter(type=standard_artifacts.Model)
    }
    PARAMETERS = {
        'n_steps': ExecutionParameter(type=int),
        'optimizer': ExecutionParameter(type=str)
    }
  ```

  with its input 'examples' is connected to the example_gen output, and
  execution properties specified as 100 and 'sgd' respectively.

  Example usage of the component:

  ```
  # A single node training job.
  my_train = create_ai_platform_training(
      name='my_training_step',
      project_id='my-project',
      region='us-central1',
      image_uri='gcr.io/my-project/caip-training-test:latest',
      'args': [
          '--examples',
          placeholders.InputUriPlaceholder('examples'),
          '--n-steps',
          placeholders.InputValuePlaceholder('n_step'),
          '--output-location',
          placeholders.OutputUriPlaceholder('model')
      ]
      scale_tier='BASIC_GPU',
      inputs={'examples': example_gen.outputs['examples']},
      outputs={
          'model': standard_artifacts.Model
      },
      parameters={'n_step': 100}
  )

  # More complex setting can be expressed by providing training_input
  # directly.
  my_distributed_train = create_ai_platform_training(
      name='my_training_step',
      project_id='my-project',
      training_input={
          'scaleTier':
              'CUSTOM',
          'region':
              'us-central1',
          'masterType': 'n1-standard-8',
          'masterConfig': {
              'imageUri': 'gcr.io/my-project/my-dist-training:latest'
          },
          'workerType': 'n1-standard-8',
          'workerCount': 8,
          'workerConfig': {
              'imageUri': 'gcr.io/my-project/my-dist-training:latest'
          },
          'args': [
              '--examples',
              placeholders.InputUriPlaceholder('examples'),
              '--n-steps',
              placeholders.InputValuePlaceholder('n_step'),
              '--output-location',
              placeholders.OutputUriPlaceholder('model')
          ]
      },
      inputs={'examples': example_gen.outputs['examples']},
      outputs={'model': Model},
      parameters={'n_step': 100}
  )
  ```

  Args:
    name: name of the component. This is needed to construct the component spec
      and component class dynamically as well.
    project_id: the GCP project under which the AIP training job will be
      running.
    region: GCE region where the AIP training job will be running.
    job_id: the unique ID of the job. Default to 'tfx_%Y%m%d%H%M%S'
    image_uri: the GCR location of the container image, which will be used to
      execute the training program. If the same field is specified in
      training_input, the latter overrides image_uri.
    args: command line arguments that will be passed into the training program.
      Users can use placeholder semantics as in
      tfx.dsl.component.experimental.container_component to wire the args with
      component inputs/outputs/parameters.
    scale_tier: Cloud ML resource requested by the job. See
      https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#ScaleTier
    training_input: full training job spec. This field overrides other
      specifications if applicable. This field follows the
      [TrainingInput](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput)
        schema.
    labels: user-specified label attached to the job.
    inputs: the dict of component inputs.
    outputs: the dict of component outputs.
    parameters: the dict of component parameters, aka, execution properties.

  Returns:
    A component instance that represents the AIP job in the DSL.

  Raises:
    ValueError: when image_uri is missing and masterConfig is not specified in
      training_input, or when region is missing and training_input
      does not provide region either.
    TypeError: when non-primitive parameters are specified.
  """
  training_input = training_input or {}

  if scale_tier and not training_input.get('scale_tier'):
    training_input['scaleTier'] = scale_tier

  if not training_input.get('masterConfig'):
    # If no replica config is specified, create a default one.
    if not image_uri:
      raise ValueError('image_uri is required when masterConfig is not '
                       'explicitly specified in training_input.')
    training_input['masterConfig'] = {'imageUri': image_uri}
    # Note: A custom entrypoint can be set to training_input['masterConfig']
    # through key 'container_command'.

  training_input['args'] = args

  if not training_input.get('region'):
    if not region:
      raise ValueError('region is required when it is not set in '
                       'training_input.')
    training_input['region'] = region

  # Squash training_input, project, job_id, and labels into an exec property
  # namely 'aip_training_config'.
  aip_training_config = {
      ai_platform_training_executor.PROJECT_CONFIG_KEY: project_id,
      ai_platform_training_executor.TRAINING_INPUT_CONFIG_KEY: training_input,
      ai_platform_training_executor.JOB_ID_CONFIG_KEY: job_id,
      ai_platform_training_executor.LABELS_CONFIG_KEY: labels,
  }

  aip_training_config_str = json_utils.dumps(aip_training_config)

  # Construct the component spec.
  if inputs is None:
    inputs = {}
  if outputs is None:
    outputs = {}
  if parameters is None:
    parameters = {}

  input_channel_parameters = {}
  output_channel_parameters = {}
  output_channels = {}
  execution_parameters = {
      ai_platform_training_executor.CONFIG_KEY:
          component_spec.ExecutionParameter(type=(str, Text))
  }

  for input_name, single_channel in inputs.items():
    # Infer the type of input channels based on the channels passed in.
    # TODO(b/155804245) Sanitize the names so that they're valid python names
    input_channel_parameters[input_name] = (
        component_spec.ChannelParameter(type=single_channel.type))

  for output_name, channel_type in outputs.items():
    # TODO(b/155804245) Sanitize the names so that they're valid python names
    output_channel_parameters[output_name] = (
        component_spec.ChannelParameter(type=channel_type))
    artifact = channel_type()
    channel = channel_utils.as_channel([artifact])
    output_channels[output_name] = channel

  # TODO(jxzheng): Support RuntimeParameter as parameters.
  for param_name, single_parameter in parameters.items():
    # Infer the type of parameters based on the parameters passed in.
    # TODO(b/155804245) Sanitize the names so that they're valid python names
    if not isinstance(single_parameter, (int, float, Text, bytes)):
      raise TypeError(
          'Parameter can only be int/float/str/bytes, got {}'.format(
              type(single_parameter)))
    execution_parameters[param_name] = (
        component_spec.ExecutionParameter(type=type(single_parameter)))

  default_init_args = {
      **inputs,
      **output_channels,
      **parameters, ai_platform_training_executor.CONFIG_KEY:
          aip_training_config_str
  }

  tfx_component_class = component_utils.create_tfx_component_class(
      name=name,
      tfx_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_training_executor.AiPlatformTrainingExecutor),
      input_channel_parameters=input_channel_parameters,
      output_channel_parameters=output_channel_parameters,
      execution_parameters=execution_parameters,
      default_init_args=default_init_args)

  return tfx_component_class()
