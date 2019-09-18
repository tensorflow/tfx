# Copyright 2019 Google LLC. All Rights Reserved.
# noop change
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
from tfx.components.base import executor_spec
from tfx.components.evaluator import executor
from tfx.proto import evaluator_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import EvaluatorSpec


class Evaluator(base_component.BaseComponent):
  """A TFX component to evaluate models trained by a TFX Trainer component.

  The Evaluator component performs model evaluations in the TFX pipeline and
  the resultant metrics can be viewed in a Jupyter notebook.  It uses the
  input examples generated from the
  [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)
  component to evaluate the models.

  Specifically, it can provide:
    - metrics computed on entire training and eval dataset
    - tracking metrics over time
    - model quality performance on different feature slices

  ## Exporting the EvalSavedModel in Trainer

  In order to setup Evaluator in a TFX pipeline, an EvalSavedModel needs to be
  exported during training, which is a special SavedModel containing
  annotations for the metrics, features, labels, and so on in your model.
  Evaluator uses this EvalSavedModel to compute metrics.

  As part of this, the Trainer component creates eval_input_receiver_fn,
  analogous to the serving_input_receiver_fn, which will extract the features
  and labels from the input data. As with serving_input_receiver_fn, there are
  utility functions to help with this.

  Please see https://www.tensorflow.org/tfx/model_analysis for more details.

  ## Example
  ```
    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output)
  ```
  """

  SPEC_CLASS = EvaluatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      model_exports: types.Channel = None,
      feature_slicing_spec: Optional[evaluator_pb2.FeatureSlicingSpec] = None,
      output: Optional[types.Channel] = None,
      model: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct an Evaluator component.

    Args:
      examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
        component. _required_
      model_exports: A Channel of 'ModelExportPath' type, usually produced by
        Trainer component.  Will be deprecated in the future for the `model`
        parameter.
      feature_slicing_spec:
        [evaluator_pb2.FeatureSlicingSpec](https://github.com/tensorflow/tfx/blob/master/tfx/proto/evaluator.proto)
        instance that describes how Evaluator should slice the data.
      output: Channel of `ModelEvalPath` to store the evaluation results.
      model: Future replacement of the `model_exports` argument.
      instance_name: Optional name assigned to this specific instance of
        Evaluator. Required only if multiple Evaluator components are declared
        in the same pipeline.

      Either `model_exports` or `model` must be present in the input arguments.
    """
    model_exports = model_exports or model
    output = output or types.Channel(
        type=standard_artifacts.ModelEvaluation,
        artifacts=[standard_artifacts.ModelEvaluation()])
    spec = EvaluatorSpec(
        examples=examples,
        model_exports=model_exports,
        feature_slicing_spec=(feature_slicing_spec or
                              evaluator_pb2.FeatureSlicingSpec()),
        output=output)
    super(Evaluator, self).__init__(spec=spec, instance_name=instance_name)
