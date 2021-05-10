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
from tfx.components.evaluator import executor
from tfx.components.util import udf_utils
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import evaluator_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class Evaluator(base_beam_component.BaseBeamComponent):
  """A TFX component to evaluate models trained by a TFX Trainer component.

  See [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) for more
  information on what this component's required inputs are, how to configure it,
  and what outputs it produces.

  Component `outputs` contains:
   - `evaluation`: Channel of type `standard_artifacts.ModelEvaluation` to store
                   the evaluation results.
   - `blessing`: Channel of type `standard_artifacts.ModelBlessing' that
                 contains the blessing result.
  """

  SPEC_CLASS = standard_component_specs.EvaluatorSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

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
      eval_config: Optional[tfma.EvalConfig] = None,
      schema: Optional[types.Channel] = None,
      module_file: Optional[Text] = None,
      module_path: Optional[Text] = None):
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
      eval_config: Instance of tfma.EvalConfig containg configuration settings
        for running the evaluation. This config has options for both estimator
        and Keras.
      schema: A `Schema` channel to use for TFXIO.
      module_file: A path to python module file containing UDFs for Evaluator
        customization. This functionality is experimental and may change at any
        time. The module_file can implement following functions at its top
        level.
          def custom_eval_shared_model(
             eval_saved_model_path, model_name, eval_config, **kwargs,
          ) -> tfma.EvalSharedModel:
          def custom_extractors(
            eval_shared_model, eval_config, tensor_adapter_config,
          ) -> List[tfma.extractors.Extractor]:
      module_path: A python path to the custom module that contains the UDFs.
        See 'module_file' for the required signature of UDFs. This functionality
        is experimental and this API may change at any time. Note this can
        not be set together with module_file.
    """
    if bool(module_file) and bool(module_path):
      raise ValueError(
          'Python module path can not be set together with module file path.')

    if eval_config is not None and feature_slicing_spec is not None:
      raise ValueError("Exactly one of 'eval_config' or 'feature_slicing_spec' "
                       "must be supplied.")
    if eval_config is None and feature_slicing_spec is None:
      feature_slicing_spec = evaluator_pb2.FeatureSlicingSpec()
      logging.info('Neither eval_config nor feature_slicing_spec is passed, '
                   'the model is treated as estimator.')

    if feature_slicing_spec:
      logging.warning('feature_slicing_spec is deprecated, please use '
                      'eval_config instead.')

    blessing = types.Channel(type=standard_artifacts.ModelBlessing)
    evaluation = types.Channel(type=standard_artifacts.ModelEvaluation)
    spec = standard_component_specs.EvaluatorSpec(
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
        module_file=module_file,
        module_path=module_path)
    super(Evaluator, self).__init__(spec=spec)

    if udf_utils.should_package_user_modules():
      # In this case, the `MODULE_PATH_KEY` execution property will be injected
      # as a reference to the given user module file after packaging, at which
      # point the `MODULE_FILE_KEY` execution property will be removed.
      udf_utils.add_user_module_dependency(
          self, standard_component_specs.MODULE_FILE_KEY,
          standard_component_specs.MODULE_PATH_KEY)
