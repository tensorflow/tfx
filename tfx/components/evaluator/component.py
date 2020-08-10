# Lint as: python2, python3
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

from typing import Any, Dict, List, Optional, Text, Union
from absl import logging
import tensorflow_model_analysis as tfma

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.evaluator import executor
from tfx.orchestration import data_types
from tfx.proto import evaluator_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import EvaluatorSpec
from tfx.utils import json_utils


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
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=tfma.EvalConfig(...))
  ```
  """

  SPEC_CLASS = EvaluatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      model: types.Channel = None,
      baseline_model: Optional[types.Channel] = None,
      # TODO(b/148618405): deprecate feature_slicing_spec.
      feature_slicing_spec: Optional[Union[evaluator_pb2.FeatureSlicingSpec,
                                           Dict[Text, Any]]] = None,
      fairness_indicator_thresholds: Optional[List[Union[
          float, data_types.RuntimeParameter]]] = None,
      example_splits: Optional[List[Text]] = None,
      output: Optional[types.Channel] = None,
      model_exports: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None,
      eval_config: Optional[tfma.EvalConfig] = None,
      blessing: Optional[types.Channel] = None,
      schema: Optional[types.Channel] = None,
      module_file: Optional[Text] = None):
    """Construct an Evaluator component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, usually
        produced by an ExampleGen component. _required_
      model: A Channel of type `standard_artifacts.Model`, usually produced by
        a Trainer component.
      baseline_model: An optional channel of type 'standard_artifacts.Model' as
        the baseline model for model diff and model validation purpose.
      feature_slicing_spec:
        Deprecated, please use eval_config instead. Only support estimator.
        [evaluator_pb2.FeatureSlicingSpec](https://github.com/tensorflow/tfx/blob/master/tfx/proto/evaluator.proto)
          instance that describes how Evaluator should slice the data. If any
          field is provided as a RuntimeParameter, feature_slicing_spec should
          be constructed as a dict with the same field names as
          FeatureSlicingSpec proto message.
      fairness_indicator_thresholds: Optional list of float (or
        RuntimeParameter) threshold values for use with TFMA fairness
          indicators. Experimental functionality: this interface and
          functionality may change at any time. TODO(b/142653905): add a link
          to additional documentation for TFMA fairness indicators here.
      example_splits: Names of splits on which the metrics are computed.
        Default behavior (when example_splits is set to None or Empty) is using
        the 'eval' split.
      output: Channel of `ModelEvalPath` to store the evaluation results.
      model_exports: Backwards compatibility alias for the `model` argument.
      instance_name: Optional name assigned to this specific instance of
        Evaluator. Required only if multiple Evaluator components are declared
        in the same pipeline.  Either `model_exports` or `model` must be present
        in the input arguments.
      eval_config: Instance of tfma.EvalConfig containg configuration settings
        for running the evaluation. This config has options for both estimator
        and Keras.
      blessing: Output channel of 'ModelBlessingPath' that contains the
        blessing result.
      schema: A `Schema` channel to use for TFXIO.
      module_file: A path to python module file containing UDFs for Evaluator
        customization. The module_file can implement following functions at its
        top level.
          def custom_eval_shared_model(
             eval_saved_model_path, model_name, eval_config, **kwargs,
          ) -> tfma.EvalSharedModel:
          def custom_extractors(
            eval_shared_model, eval_config, tensor_adapter_config,
          ) -> List[tfma.extractors.Extractor]:
    """
    if eval_config is not None and feature_slicing_spec is not None:
      raise ValueError("Exactly one of 'eval_config' or 'feature_slicing_spec' "
                       "must be supplied.")
    if eval_config is None and feature_slicing_spec is None:
      feature_slicing_spec = evaluator_pb2.FeatureSlicingSpec()
      logging.info('Neither eval_config nor feature_slicing_spec is passed, '
                   'the model is treated as estimator.')

    if model_exports:
      logging.warning(
          'The "model_exports" argument to the Evaluator component has '
          'been renamed to "model" and is deprecated. Please update your '
          'usage as support for this argument will be removed soon.')
      model = model_exports

    if feature_slicing_spec:
      logging.warning('feature_slicing_spec is deprecated, please use '
                      'eval_config instead.')

    blessing = blessing or types.Channel(
        type=standard_artifacts.ModelBlessing,
        artifacts=[standard_artifacts.ModelBlessing()])

    evaluation = output or types.Channel(
        type=standard_artifacts.ModelEvaluation,
        artifacts=[standard_artifacts.ModelEvaluation()])
    spec = EvaluatorSpec(
        examples=examples,
        model=model,
        baseline_model=baseline_model,
        feature_slicing_spec=feature_slicing_spec,
        fairness_indicator_thresholds=fairness_indicator_thresholds,
        example_splits=json_utils.dumps(example_splits),
        evaluation=evaluation,
        eval_config=eval_config,
        blessing=blessing,
        schema=schema,
        module_file=module_file)
    super(Evaluator, self).__init__(spec=spec, instance_name=instance_name)
