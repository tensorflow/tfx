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

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.trainer import driver
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.utils import channel
from tfx.utils import types
from google.protobuf import json_format


class Trainer(base_component.BaseComponent):
  """Official TFX Trainer component.

  The Trainer component is used to train and eval a model using given inputs.
  This component includes a custom driver to optionally grab previous model to
  warm start from.

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
    name: Optional unique name. Necessary iff multiple Trainer components are
      declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following members:
      - output: A channel of 'ModelExportPath' with result of exported models.
  """

  def __init__(self,
               transformed_examples,
               transform_output,
               schema,
               module_file,
               train_args,
               eval_args,
               custom_config = None,
               name = None,
               outputs = None):
    component_name = 'Trainer'
    input_dict = {
        'transformed_examples': channel.as_channel(transformed_examples),
        'transform_output': channel.as_channel(transform_output),
        'schema': channel.as_channel(schema),
    }
    exec_properties = {
        'train_args': json_format.MessageToJson(train_args),
        'eval_args': json_format.MessageToJson(eval_args),
        'module_file': module_file,
        'custom_config': custom_config,
    }
    super(Trainer, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=driver.Driver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for Trainer.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [
        types.TfxType('ModelExportPath'),
    ]
    return base_component.ComponentOutputs({
        'output':
            channel.Channel(
                type_name='ModelExportPath',
                static_artifact_collection=output_artifact_collection),
    })

  def _type_check(self, input_dict,
                  exec_properties):
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unused right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    input_dict['transformed_examples'].type_check('ExamplesPath')
    input_dict['transform_output'].type_check('TransformPath')
    input_dict['schema'].type_check('SchemaPath')
