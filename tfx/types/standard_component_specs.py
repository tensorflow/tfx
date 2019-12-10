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

from typing import Any, Dict, List, Text

from tfx.proto import bulk_inferrer_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
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
      'feature_slicing_spec':
          ExecutionParameter(type=evaluator_pb2.FeatureSlicingSpec),
      # This parameter is experimental: its interface and functionality may
      # change at any time.
      'fairness_indicator_thresholds':
          ExecutionParameter(type=List[float], optional=True),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      # TODO(b/139281215): this will be renamed to 'model' in the future.
      'model_exports': ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ModelEvaluation),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'model': 'model_exports',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'evaluation': 'output',
  }


class ExampleValidatorSpec(ComponentSpec):
  """ExampleValidator component spec."""

  PARAMETERS = {}
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'statistics' in the future.
      'stats': ChannelParameter(type=standard_artifacts.ExampleStatistics),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ExampleAnomalies),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'statistics': 'stats',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'anomalies': 'output',
  }


class FileBasedExampleGenSpec(ComponentSpec):
  """File-based ExampleGen component spec."""

  PARAMETERS = {
      'input_config':
          ExecutionParameter(type=example_gen_pb2.Input),
      'output_config':
          ExecutionParameter(type=example_gen_pb2.Output),
      'custom_config':
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
  }
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'input' in the future.
      'input_base': ChannelParameter(type=standard_artifacts.ExternalArtifact),
  }
  OUTPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
  }
  # TODO(b/139281215): these input names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'input': 'input_base',
  }


class InfraValidatorSpec(ComponentSpec):
  """InfraValidator component spec."""

  PARAMETERS = {
      'serving_spec': ExecutionParameter(type=infra_validator_pb2.ServingSpec)
  }

  INPUTS = {
      'model': ChannelParameter(type=standard_artifacts.Model),
      'examples': ChannelParameter(type=standard_artifacts.Examples,
                                   optional=True),
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
          ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'model' in the future.
      'model_export': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  OUTPUTS = {
      'model_push': ChannelParameter(type=standard_artifacts.PushedModel),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'model': 'model_export',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'pushed_model': 'model_push',
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


class SchemaGenSpec(ComponentSpec):
  """SchemaGen component spec."""

  PARAMETERS = {
      'infer_feature_shape': ExecutionParameter(type=bool, optional=True)
  }
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'statistics' in the future.
      'stats': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.Schema),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'statistics': 'stats',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'schema': 'output',
  }


class StatisticsGenSpec(ComponentSpec):
  """StatisticsGen component spec."""

  PARAMETERS = {}
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'examples' in the future.
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'examples': 'input_data',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'statistics': 'output',
  }


class TrainerSpec(ComponentSpec):
  """Trainer component spec."""

  PARAMETERS = {
      'train_args': ExecutionParameter(type=trainer_pb2.TrainArgs),
      'eval_args': ExecutionParameter(type=trainer_pb2.EvalArgs),
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'trainer_fn': ExecutionParameter(type=(str, Text), optional=True),
      'custom_config': ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      # TODO(b/139281215): this will be renamed to 'transform_graph' in the
      # future.
      'transform_output':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema),
      'base_model':
          ChannelParameter(type=standard_artifacts.Model, optional=True),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.Model),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'transform_graph': 'transform_output',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'model': 'output',
  }


class TransformSpec(ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'preprocessing_fn': ExecutionParameter(type=(str, Text), optional=True),
      'custom_config': ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      # TODO(b/139281215): this will be renamed to 'examples' in the future.
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'transform_output':
          ChannelParameter(type=standard_artifacts.TransformGraph),
      'transformed_examples':
          ChannelParameter(type=standard_artifacts.Examples),
  }
  # TODO(b/139281215): these input / output names will be renamed in the future.
  # These compatibility aliases are provided for forwards compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'examples': 'input_data',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'transform_graph': 'transform_output',
  }
