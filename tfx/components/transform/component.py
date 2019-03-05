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
"""TFX Transform component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text
from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.transform import executor
from tfx.utils import channel
from tfx.utils import types


class Transform(base_component.BaseComponent):
  """Official TFX Transform component.

  The Transform component wraps TensorFlow Transform (tf.Transform) to
  preprocess data in a TFX pipeline. This component will load the
  preprocessing_fn from input module file, preprocess both 'train' and 'eval'
  splits of input examples, generate the `tf.Transform` output, and save both
  transform function and transformed examples to orchestrator desired locations.

  Please see https://www.tensorflow.org/tfx/transform for more details.

  Args:
    input_data: A Channel of 'ExamplesPath' type. This should contain two splits
      'train' and 'eval'.
    schema: A Channel of 'SchemaPath' type. This should contain a single schema
      artifact.
    module_file: The file path to a python module file, from which the
      'preprocessing_fn' function will be loaded.
    name: Optional unique name. Necessary iff multiple transform components are
      declared in the same pipeline.
    output: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - transform_output: Output of 'tf.Transform', which includes an exported
        Tensorflow graph suitable for both training and serving;
      - transformed_examples: Materialized transformed examples, which includes
        both 'train' and 'eval' splits.
  """

  def __init__(self,
               input_data,
               schema,
               module_file,
               name = None,
               outputs = None):
    component_name = 'Transform'
    input_dict = {
        'input_data': channel.as_channel(input_data),
        'schema': channel.as_channel(schema)
    }
    exec_properties = {
        'module_file': module_file,
    }
    super(Transform, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for Transform.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    transform_output_artifact_collection = [types.TfxType('TransformPath',)]
    transformed_examples_artifact_collection = [
        types.TfxType('ExamplesPath', split=split)
        for split in types.DEFAULT_EXAMPLE_SPLITS
    ]
    return base_component.ComponentOutputs({
        'transform_output':
            channel.Channel(
                type_name='TransformPath',
                static_artifact_collection=transform_output_artifact_collection
            ),
        'transformed_examples':
            channel.Channel(
                type_name='ExamplesPath',
                static_artifact_collection=transformed_examples_artifact_collection
            ),
    })

  def _type_check(self, input_dict,
                  exec_properties):
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unchecked right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    del exec_properties
    input_dict['input_data'].type_check('ExamplesPath')
    input_dict['schema'].type_check('SchemaPath')
