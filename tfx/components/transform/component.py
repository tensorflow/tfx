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
from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.components.transform import executor
from tfx.utils import channel
from tfx.utils import types


class TransformSpec(base_component.ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text)),
  }
  INPUTS = {
      'input_data': ChannelParameter(type_name='ExamplesPath'),
      'schema': ChannelParameter(type_name='SchemaPath'),
  }
  OUTPUTS = {
      'transform_output': ChannelParameter(type_name='TransformPath'),
      'transformed_examples': ChannelParameter(type_name='ExamplesPath'),
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
               input_data: channel.Channel = None,
               schema: channel.Channel = None,
               module_file: Text = None,
               transform_output: Optional[channel.Channel] = None,
               transformed_examples: Optional[channel.Channel] = None,
               label: Optional[Text] = None):
    """Construct a Transform component.

    Args:
      input_data: A Channel of 'ExamplesPath' type. This should contain two
        splits 'train' and 'eval'.
      schema: A Channel of 'SchemaPath' type. This should contain a single
        schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded.
      transform_output: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      label: Optional unique label. Necessary iff multiple transform components
        are declared in the same pipeline.
    """
    transform_output = transform_output or channel.Channel(
        type_name='TransformPath',
        artifacts=[types.TfxArtifact('TransformPath')])
    transformed_examples = transformed_examples or channel.Channel(
        type_name='ExamplesPath',
        artifacts=[
            types.TfxArtifact('ExamplesPath', split=split)
            for split in types.DEFAULT_EXAMPLE_SPLITS
        ])
    spec = TransformSpec(
        input_data=channel.as_channel(input_data),
        schema=channel.as_channel(schema),
        module_file=module_file,
        transform_output=transform_output,
        transformed_examples=transformed_examples)
    super(Transform, self).__init__(spec=spec, label=label)
