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

from tfx import types
from tfx.components.base import base_component
from tfx.components.evaluator import executor
from tfx.proto import evaluator_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import EvaluatorSpec


class Evaluator(base_component.BaseComponent):
  """Official TFX Evaluator component.

  The evaluator component can be used to perform model evaluations.
  """

  SPEC_CLASS = EvaluatorSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(
      self,
      examples: types.Channel,
      model_exports: types.Channel,
      feature_slicing_spec: Optional[evaluator_pb2.FeatureSlicingSpec] = None,
      output: Optional[types.Channel] = None,
      name: Optional[Text] = None):
    """Construct an Evaluator component.

    Args:
      examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
        component.
      model_exports: A Channel of 'ModelExportPath' type, usually produced by
        Trainer component.
      feature_slicing_spec: Optional evaluator_pb2.FeatureSlicingSpec instance,
        providing the way to slice the data.
      output: Optional channel of 'ModelEvalPath' for result of evaluation.
      name: Optional unique name. Necessary if multiple Evaluator components are
        declared in the same pipeline.
    """
    output = output or types.Channel(
        type=standard_artifacts.ModelEvalResult,
        artifacts=[standard_artifacts.ModelEvalResult()])
    spec = EvaluatorSpec(
        examples=examples,
        model_exports=model_exports,
        feature_slicing_spec=(feature_slicing_spec or
                              evaluator_pb2.FeatureSlicingSpec()),
        output=output)
    super(Evaluator, self).__init__(spec=spec, name=name)
