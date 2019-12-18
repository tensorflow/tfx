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

from typing import Optional, Text, NamedTuple
import kerastuner
import tensorflow.compat.v1 as tf

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.examples.custom_components.tuner.tuner_component import executor
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', kerastuner.Tuner),
                                             ('train_dataset', tf.data.Dataset),
                                             ('eval_dataset', tf.data.Dataset)])


# TODO(jyzhao): move to tfx/types/standard_component_specs.py.
class TunerSpec(ComponentSpec):
  """ComponentSpec for TFX Tuner Component."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'tuner_fn': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'model_export_path':
          ChannelParameter(type=standard_artifacts.Model),
      'study_best_hparams_path':
          ChannelParameter(type=standard_artifacts.HyperParameters),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'model': 'model_export_path',
      'best_hparams': 'study_best_hparams_path',
  }


class Tuner(base_component.BaseComponent):
  """A TFX component for model hyperparameter tuning."""

  SPEC_CLASS = TunerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel = None,
               schema: types.Channel = None,
               module_file: Optional[Text] = None,
               tuner_fn: Optional[Text] = None,
               model: Optional[types.Channel] = None,
               best_hparams: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Construct a Tuner component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as the
        source of examples that are used in tuning (required). Transformed
        examples are not yet supported.
      schema:  A Channel of type `standard_artifacts.Schema`, serving as the
        schema of training and eval data.
      module_file: A path to python module file containing UDF KerasTuner
        definition. Exactly one of 'module_file' or 'tuner_fn' must be supplied.
        The module_file must implement a function named `tuner_fn` at its top
        level. The function takes working dir path, train data path, eval data
        path and tensorflow_metadata.proto.v0.schema_pb2.Schema and generates a
        namedtuple TunerFnResult which contains:
        - 'tuner': A KerasTuner that will be used for tuning.
        - 'train_dataset': A tf.data.Dataset of training data.
        - 'eval_dataset': A tf.data.Dataset of eval data.
      tuner_fn:  A python path to UDF model definition function. See
        'module_file' for the required signature of the UDF. Exactly one of
        'module_file' or 'tuner_fn' must be supplied.
      model: Optional 'ModelExportPath' channel for result of best model.
      best_hparams: Optional 'StudyBestHParamsPath' channel for result of the
        best hparams.
      instance_name: Optional unique instance name. Necessary if multiple Tuner
        components are declared in the same pipeline.
    """
    if bool(module_file) == bool(tuner_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'tuner_fn' must be supplied")

    model = model or types.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    best_hparams = best_hparams or types.Channel(
        type=standard_artifacts.HyperParameters,
        artifacts=[standard_artifacts.HyperParameters()])
    spec = TunerSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        tuner_fn=tuner_fn,
        model_export_path=model,
        study_best_hparams_path=best_hparams)
    super(Tuner, self).__init__(spec=spec, instance_name=instance_name)
