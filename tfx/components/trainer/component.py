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

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.components.trainer import driver
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class TrainerSpec(base_component.ComponentSpec):
  """Trainer component spec."""

  PARAMETERS = {
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'trainer_fn': ExecutionParameter(type=(str, Text), optional=True),
      'custom_config': ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'transform_output':
          ChannelParameter(
              type=standard_artifacts.TransformResult, optional=True),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {'output': ChannelParameter(type=standard_artifacts.Model)}


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

  def __init__(
      self,
      examples: types.Channel = None,
      transformed_examples: types.Channel = None,
      transform_output: Optional[types.Channel] = None,
      schema: types.Channel = None,
      module_file: Optional[Text] = None,
      trainer_fn: Optional[Text] = None,
      train_args: trainer_pb2.TrainArgs = None,
      eval_args: trainer_pb2.EvalArgs = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      executor_class: Optional[Type[base_executor.BaseExecutor]] = None,
      output: Optional[types.Channel] = None,
      name: Optional[Text] = None):
    """Construct a Trainer component.

    Args:
      examples: A Channel of 'ExamplesPath' type, serving as the source of
        examples that are used in training. May be raw or transformed.
      transformed_examples: Deprecated field. Please set 'examples' instead.
      transform_output: An optional Channel of 'TransformPath' type, serving as
        the input transform graph if present.
      schema:  A Channel of 'SchemaPath' type, serving as the schema of training
        and eval data.
      module_file: A path to python module file containing UDF model definition.
        The module_file must implement a function named `trainer_fn` at its
        top level. The function must have the following signature.

        def trainer_fn(tf.contrib.training.HParams,
                       tensorflow_metadata.proto.v0.schema_pb2) -> Dict:
          ...

        where the returned Dict has the following key-values.
          'estimator': an instance of tf.estimator.Estimator
          'train_spec': an instance of tf.estimator.TrainSpec
          'eval_spec': an instance of tf.estimator.EvalSpec
          'eval_input_receiver_fn': an instance of tfma.export.EvalInputReceiver

        Exactly one of 'module_file' or 'trainer_fn' must be supplied.
      trainer_fn:  A python path to UDF model definition function. See
        'module_file' for the required signature of the UDF.
        Exactly one of 'module_file' or 'trainer_fn' must be supplied.
      train_args: A trainer_pb2.TrainArgs instance, containing args used for
        training. Current only num_steps is available.
      eval_args: A trainer_pb2.EvalArgs instance, containing args used for eval.
        Current only num_steps is available.
      custom_config: A dict which contains the training job parameters to be
        passed to Google Cloud ML Engine.  For the full set of parameters
        supported by Google Cloud ML Engine, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      executor_class: Optional custom executor class.
      output: Optional 'ModelExportPath' channel for result of exported models.
      name: Optional unique name. Necessary iff multiple Trainer components are
        declared in the same pipeline.

    Raises:
      ValueError:
        - When both or neither of 'module_file' and 'trainer_fn' is supplied.
        - When both or neither of 'examples' and 'transformed_examples'
            is supplied.
        - When 'transformed_examples' is supplied but 'transform_output'
            is not supplied.
    """
    if bool(module_file) == bool(trainer_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'trainer_fn' must be supplied")

    if bool(examples) == bool(transformed_examples):
      raise ValueError(
          "Exactly one of 'example' or 'transformed_example' must be supplied.")

    if transformed_examples and not transform_output:
      raise ValueError("If 'transformed_examples' is supplied, "
                       "'transform_output' must be supplied too.")
    examples = examples or transformed_examples
    transform_output_channel = channel_utils.as_channel(
        transform_output) if transform_output else None
    output = output or channel.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    spec = TrainerSpec(
        examples=channel_utils.as_channel(examples),
        transform_output=transform_output_channel,
        schema=channel_utils.as_channel(schema),
        train_args=train_args,
        eval_args=eval_args,
        module_file=module_file,
        trainer_fn=trainer_fn,
        custom_config=custom_config,
        output=output)
    super(Trainer, self).__init__(
        spec=spec, custom_executor_class=executor_class, name=name)
