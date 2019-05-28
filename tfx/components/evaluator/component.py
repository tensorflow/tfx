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
"""TFX Evaluator component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.base.base_component import Parameter
from tfx.components.evaluator import executor
from tfx.proto import evaluator_pb2
from tfx.utils import channel
from tfx.utils import types


class EvaluatorSpec(base_component.ComponentSpec):
  """Evaluator component spec."""

  COMPONENT_NAME = 'Evaluator'
  PARAMETERS = [
      Parameter('feature_slicing_spec', type=evaluator_pb2.FeatureSlicingSpec),
  ]
  INPUTS = [
      ChannelInput('examples', type='ExamplesPath'),
      ChannelInput('model_exports', type='ModelExportPath'),
  ]
  OUTPUTS = [
      ChannelOutput('output', type='ModelEvalPath'),
  ]


class Evaluator(base_component.BaseComponent):
  """Official TFX Evaluator component.

  The evaluator component can be used to perform model evaluations.

  Args:
    examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
      component.
    model_exports: A Channel of 'ModelExportPath' type, usually produced by
      Trainer component.
    feature_slicing_spec: A evaluator_pb2.FeatureSlicingSpec instance,
      providing the way to slice the data.
    name: Optional unique name. Necessary if multiple Evaluator components are
      declared in the same pipeline.
    outputs: Optional dict from name to output channel.
    output: Optional channel of 'ModelEvalPath' for result of evaluation.
  """

  def __init__(
      self,
      examples: channel.Channel,
      model_exports: channel.Channel,
      feature_slicing_spec: Optional[evaluator_pb2.FeatureSlicingSpec] = None,
      name: Optional[Text] = None,
      output: Optional[channel.Channel] = None):
    if not output:
      output = channel.Channel(
          type_name='ModelEvalPath',
          static_artifact_collection=[types.TfxArtifact('ModelEvalPath')])
    spec = EvaluatorSpec(
        examples=channel.as_channel(examples),
        model_exports=channel.as_channel(model_exports),
        feature_slicing_spec=(
            feature_slicing_spec or evaluator_pb2.FeatureSlicingSpec()),
        output=output)
    super(Evaluator, self).__init__(
        unique_name=name, spec=spec, executor=executor.Executor)
