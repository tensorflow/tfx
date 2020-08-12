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
"""Component specifications for the standard set of TFX Components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import tensorflow_model_analysis as tfma
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class BulkInferrerSpec(ComponentSpec):
  """BulkInferrer component spec."""

  PARAMETERS = {
      'model_spec':
          ExecutionParameter(type=bulk_inferrer_pb2.ModelSpec, optional=True),
      'data_spec':
          ExecutionParameter(type=bulk_inferrer_pb2.DataSpec, optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'model':
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      'model_blessing':
          ChannelParameter(
              type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      'inference_result':
          ChannelParameter(type=standard_artifacts.InferenceResult),
  }


class EvaluatorSpec(ComponentSpec):
  """Evaluator component spec."""

  PARAMETERS = {
      'eval_config':
          ExecutionParameter(type=tfma.EvalConfig, optional=True),
      # TODO(mdreves): Deprecated, use eval_config.slicing_specs.
      'feature_slicing_spec':
          ExecutionParameter(
              type=evaluator_pb2.FeatureSlicingSpec, optional=True),
      # This parameter is experimental: its interface and functionality may
      # change at any time.
      'fairness_indicator_thresholds':
          ExecutionParameter(type=List[float], optional=True),
      'example_splits':
          ExecutionParameter(type=(str, Text), optional=True),
      'module_file':
          ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'model':
          ChannelParameter(type=standard_artifacts.Model),
      'baseline_model':
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      'schema': ChannelParameter(type=standard_artifacts.Schema, optional=True),
  }
  OUTPUTS = {
      'evaluation': ChannelParameter(type=standard_artifacts.ModelEvaluation),
      'blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'model_exports': 'model',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'output': 'evaluation',
  }


class ExampleValidatorSpec(ComponentSpec):
  """ExampleValidator component spec."""

  PARAMETERS = {
      'exclude_splits': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'anomalies': ChannelParameter(type=standard_artifacts.ExampleAnomalies),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'stats': 'statistics',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'output': 'anomalies',
  }


class FileBasedExampleGenSpec(ComponentSpec):
  """File-based ExampleGen component spec."""

  PARAMETERS = {
      'input_base':
          ExecutionParameter(type=(str, Text)),
      'input_config':
          ExecutionParameter(type=example_gen_pb2.Input),
      'output_config':
          ExecutionParameter(type=example_gen_pb2.Output),
      'output_data_format':
          ExecutionParameter(type=int),  # example_gen_pb2.PayloadType enum.
      'custom_config':
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
  }
  INPUTS = {}
  OUTPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
  }


class QueryBasedExampleGenSpec(ComponentSpec):
  """Query-based ExampleGen component spec."""

  PARAMETERS = {
      'input_config':
          ExecutionParameter(type=example_gen_pb2.Input),
      'output_config':
          ExecutionParameter(type=example_gen_pb2.Output),
      'custom_config':
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
  }
  INPUTS = {}
  OUTPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
  }


class InfraValidatorSpec(ComponentSpec):
  """InfraValidator component spec."""

  PARAMETERS = {
      'serving_spec':
          ExecutionParameter(type=infra_validator_pb2.ServingSpec),
      'validation_spec':
          ExecutionParameter(type=infra_validator_pb2.ValidationSpec,
                             optional=True),
      'request_spec':
          ExecutionParameter(type=infra_validator_pb2.RequestSpec,
                             optional=True)
  }

  INPUTS = {
      'model':
          ChannelParameter(type=standard_artifacts.Model),
      'examples':
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }

  OUTPUTS = {
      'blessing': ChannelParameter(type=standard_artifacts.InfraBlessing),
  }


class ModelValidatorSpec(ComponentSpec):
  """ModelValidator component spec."""

  PARAMETERS = {}
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'model': ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
      'blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }


class PusherSpec(ComponentSpec):
  """Pusher component spec."""

  PARAMETERS = {
      'push_destination':
          ExecutionParameter(type=pusher_pb2.PushDestination, optional=True),
      'custom_config':
          ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'model': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing,
                                         optional=True),
      'infra_blessing': ChannelParameter(type=standard_artifacts.InfraBlessing,
                                         optional=True),
  }
  OUTPUTS = {
      'pushed_model': ChannelParameter(type=standard_artifacts.PushedModel),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'model_export': 'model',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'model_push': 'pushed_model',
  }


class SchemaGenSpec(ComponentSpec):
  """SchemaGen component spec."""

  PARAMETERS = {
      'infer_feature_shape': ExecutionParameter(type=bool, optional=True),
      'exclude_splits': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'stats': 'statistics',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'output': 'schema',
  }


class StatisticsGenSpec(ComponentSpec):
  """StatisticsGen component spec."""

  PARAMETERS = {
      'stats_options_json': ExecutionParameter(type=(str, Text), optional=True),
      'exclude_splits': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema, optional=True),
  }
  OUTPUTS = {
      'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'input_data': 'examples',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'output': 'statistics',
  }


class TrainerSpec(ComponentSpec):
  """Trainer component spec."""

  PARAMETERS = {
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'run_fn': ExecutionParameter(type=(str, Text), optional=True),
      'trainer_fn': ExecutionParameter(type=(str, Text), optional=True),
      'custom_config': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'transform_graph':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema),
      'base_model':
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      'hyperparameters':
          ChannelParameter(
              type=standard_artifacts.HyperParameters, optional=True),
  }
  OUTPUTS = {
      'model': ChannelParameter(type=standard_artifacts.Model),
      'model_run': ChannelParameter(type=standard_artifacts.ModelRun)
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'transform_output': 'transform_graph',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'output': 'model',
  }


class TunerSpec(ComponentSpec):
  """ComponentSpec for TFX Tuner Component."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'tuner_fn': ExecutionParameter(type=(str, Text), optional=True),
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'tune_args': ExecutionParameter(type=tuner_pb2.TuneArgs, optional=True),
      'custom_config': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      'transform_graph':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
  }
  OUTPUTS = {
      'best_hyperparameters':
          ChannelParameter(type=standard_artifacts.HyperParameters),
  }


class TransformSpec(ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'preprocessing_fn': ExecutionParameter(type=(str, Text), optional=True),
      'custom_config': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'transform_graph':
          ChannelParameter(type=standard_artifacts.TransformGraph),
      'transformed_examples':
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'input_data': 'examples',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'transform_output': 'transform_graph',
  }
