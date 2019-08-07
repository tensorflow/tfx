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

from typing import Optional, Text
from tfx import types
from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.components.transform import executor
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class TransformSpec(base_component.ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'preprocessing_fn': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'transform_output':
          ChannelParameter(type=standard_artifacts.TransformResult),
      'transformed_examples':
          ChannelParameter(type=standard_artifacts.Examples),
  }


class Transform(base_component.BaseComponent):
  """Official TFX Transform component.

  The Transform component wraps TensorFlow Transform (tf.Transform) to
  preprocess data in a TFX pipeline. This component will load the
  preprocessing_fn from input module file, preprocess both 'train' and 'eval'
  splits of input examples, generate the `tf.Transform` output, and save both
  transform function and transformed examples to orchestrator desired locations.

  Please see https://www.tensorflow.org/tfx/transform for more details.
  """

  SPEC_CLASS = TransformSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               input_data: types.Channel = None,
               schema: types.Channel = None,
               module_file: Optional[Text] = None,
               preprocessing_fn: Optional[Text] = None,
               transform_output: Optional[types.Channel] = None,
               transformed_examples: Optional[types.Channel] = None,
               name: Optional[Text] = None):
    """Construct a Transform component.

    Args:
      input_data: A Channel of 'ExamplesPath' type. This should contain two
        splits 'train' and 'eval'.
      schema: A Channel of 'SchemaPath' type. This should contain a single
        schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded. The function must have the
        following signature.

        def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
          ...

        where the values of input and returned Dict are either tf.Tensor or
        tf.SparseTensor.  Exactly one of 'module_file' or 'preprocessing_fn'
        must be supplied.
      preprocessing_fn: The path to python function that implements a
         'preprocessing_fn'. See 'module_file' for expected signature of the
         function. Exactly one of 'module_file' or 'preprocessing_fn' must
         be supplied.
      transform_output: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      name: Optional unique name. Necessary iff multiple transform components
        are declared in the same pipeline.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    if bool(module_file) == bool(preprocessing_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'preprocessing_fn' must be supplied."
      )

    transform_output = transform_output or types.Channel(
        type=standard_artifacts.TransformResult,
        artifacts=[standard_artifacts.TransformResult()])
    transformed_examples = transformed_examples or types.Channel(
        type=standard_artifacts.Examples,
        artifacts=[
            standard_artifacts.Examples(split=split)
            for split in types.DEFAULT_EXAMPLE_SPLITS
        ])
    spec = TransformSpec(
        input_data=channel_utils.as_channel(input_data),
        schema=channel_utils.as_channel(schema),
        module_file=module_file,
        preprocessing_fn=preprocessing_fn,
        transform_output=transform_output,
        transformed_examples=transformed_examples)
    super(Transform, self).__init__(spec=spec, name=name)
