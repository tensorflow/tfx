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
"""TFX Trainer component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Type

from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.components.trainer import driver
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.utils import channel
from tfx.utils import types


class TrainerSpec(base_component.ComponentSpec):
  """Trainer component spec."""

  COMPONENT_NAME = 'Trainer'
  PARAMETERS = {
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'module_file': ExecutionParameter(type=(str, Text)),
      'custom_config': ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      'transformed_examples': ChannelParameter(type_name='ExamplesPath'),
      'transform_output': ChannelParameter(type_name='TransformPath'),
      'schema': ChannelParameter(type_name='SchemaPath'),
  }
  OUTPUTS = {
      'output': ChannelParameter(type_name='ModelExportPath')
  }


class Trainer(base_component.BaseComponent):
  """Official TFX Trainer component.

  The Trainer component is used to train and eval a model using given inputs.
  This component includes a custom driver to optionally grab previous model to
  warm start from.

  There are two executors provided for this component currently:
  - A default executor (in tfx.components.trainer.executor.py) provides local
    training;
  - A custom executor (in
    tfx.extensions.google_cloud_ai_platform.trainer.executor.py) provides
    training on Google Cloud AI Platform.
  """

  SPEC_CLASS = TrainerSpec
  EXECUTOR_CLASS = executor.Executor
  DRIVER_CLASS = driver.Driver

  def __init__(self,
               transformed_examples: channel.Channel = None,
               transform_output: channel.Channel = None,
               schema: channel.Channel = None,
               module_file: Text = None,
               train_args: trainer_pb2.TrainArgs = None,
               eval_args: trainer_pb2.EvalArgs = None,
               custom_config: Optional[Dict[Text, Any]] = None,
               executor_class: Optional[Type[
                   base_executor.BaseExecutor]] = None,
               output: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    """Construct a Trainer component.

    Args:
      transformed_examples: A Channel of 'ExamplesPath' type, serving as the
        source of transformed examples. This is usually an output of Transform.
      transform_output: A Channel of 'TransformPath' type, serving as the input
        transform graph.
      schema:  A Channel of 'SchemaPath' type, serving as the schema of training
        and eval data.
      module_file: A python module file containing UDF model definition.
      train_args: A trainer_pb2.TrainArgs instance, containing args used for
        training. Current only num_steps is available.
      eval_args: A trainer_pb2.EvalArgs instance, containing args used for
        eval. Current only num_steps is available.
      custom_config: A dict which contains the training job parameters to be
        passed to Google Cloud ML Engine.  For the full set of parameters
        supported by Google Cloud ML Engine, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      executor_class: Optional custom executor class.
      output: Optional 'ModelExportPath' channel for result of exported models.
      name: Optional unique name. Necessary iff multiple Trainer components are
        declared in the same pipeline.
    """
    output = output or channel.Channel(
        type_name='ModelExportPath',
        artifacts=[types.TfxArtifact('ModelExportPath')])
    spec = TrainerSpec(
        transformed_examples=channel.as_channel(transformed_examples),
        transform_output=channel.as_channel(transform_output),
        schema=channel.as_channel(schema),
        train_args=train_args,
        eval_args=eval_args,
        module_file=module_file,
        custom_config=custom_config,
        output=output)
    super(Trainer, self).__init__(
        spec=spec,
        custom_executor_class=executor_class,
        name=name)
