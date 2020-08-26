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
"""TFX Tuner component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, NamedTuple
from kerastuner.engine import base_tuner

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.tuner import executor
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TunerSpec
from tfx.utils import json_utils

# tuner: A BaseTuner that will be used for tuning.
# fit_kwargs: Args to pass to tuner's run_trial function for fitting the
#             model , e.g., the training and validation dataset. Required
#             args depend on the tuner's implementation.
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])
TunerFnResult.__doc__ = """
tuner_fn returns a TunerFnResult that contains:
- tuner: A BaseTuner that will be used for tuning.
- fit_kwargs: Args to pass to tuner's run_trial function for fitting the
              model , e.g., the training and validation dataset. Required
              args depend on the tuner's implementation.
"""


class Tuner(base_component.BaseComponent):
  """A TFX component for model hyperparameter tuning."""

  SPEC_CLASS = TunerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel = None,
               schema: Optional[types.Channel] = None,
               transform_graph: Optional[types.Channel] = None,
               module_file: Optional[Text] = None,
               tuner_fn: Optional[Text] = None,
               train_args: trainer_pb2.TrainArgs = None,
               eval_args: trainer_pb2.EvalArgs = None,
               tune_args: Optional[tuner_pb2.TuneArgs] = None,
               custom_config: Optional[Dict[Text, Any]] = None,
               best_hyperparameters: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Construct a Tuner component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as the
        source of examples that are used in tuning (required).
      schema:  An optional Channel of type `standard_artifacts.Schema`, serving
        as the schema of training and eval data. This is used when raw examples
        are provided.
      transform_graph: An optional Channel of type
        `standard_artifacts.TransformGraph`, serving as the input transform
        graph if present. This is used when transformed examples are provided.
      module_file: A path to python module file containing UDF tuner definition.
        The module_file must implement a function named `tuner_fn` at its top
        level. The function must have the following signature.
            def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
        Exactly one of 'module_file' or 'tuner_fn' must be supplied.
      tuner_fn:  A python path to UDF model definition function. See
        'module_file' for the required signature of the UDF. Exactly one of
        'module_file' or 'tuner_fn' must be supplied.
      train_args: A trainer_pb2.TrainArgs instance, containing args used for
        training. Currently only splits and num_steps are available. Default
        behavior (when splits is empty) is train on `train` split.
      eval_args: A trainer_pb2.EvalArgs instance, containing args used for eval.
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is evaluate on `eval` split.
      tune_args: A tuner_pb2.TuneArgs instance, containing args used for tuning.
        Currently only num_parallel_trials is available.
      custom_config: A dict which contains addtional training job parameters
        that will be passed into user module.
      best_hyperparameters: Optional Channel of type
        `standard_artifacts.HyperParameters` for result of the best hparams.
      instance_name: Optional unique instance name. Necessary if multiple Tuner
        components are declared in the same pipeline.
    """
    if bool(module_file) == bool(tuner_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'tuner_fn' must be supplied")

    best_hyperparameters = best_hyperparameters or types.Channel(
        type=standard_artifacts.HyperParameters,
        artifacts=[standard_artifacts.HyperParameters()])
    spec = TunerSpec(
        examples=examples,
        schema=schema,
        transform_graph=transform_graph,
        module_file=module_file,
        tuner_fn=tuner_fn,
        train_args=train_args,
        eval_args=eval_args,
        tune_args=tune_args,
        best_hyperparameters=best_hyperparameters,
        custom_config=json_utils.dumps(custom_config),
    )
    super(Tuner, self).__init__(spec=spec, instance_name=instance_name)
