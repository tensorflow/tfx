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

from typing import Any, Dict, Text

from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class EvaluatorSpec(ComponentSpec):
  """Evaluator component spec."""

  PARAMETERS = {
      'feature_slicing_spec':
          ExecutionParameter(type=evaluator_pb2.FeatureSlicingSpec),
  }
  INPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      'model_exports': ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ModelEvaluation),
  }


class ExampleValidatorSpec(ComponentSpec):
  """ExampleValidator component spec."""

  PARAMETERS = {}
  INPUTS = {
      'stats': ChannelParameter(type=standard_artifacts.ExampleStatistics),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ExampleAnomalies),
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
      'input_base': ChannelParameter(type=standard_artifacts.ExternalArtifact),
  }
  OUTPUTS = {
      'examples': ChannelParameter(type=standard_artifacts.Examples),
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


class PusherSpec(ComponentSpec):
  """Pusher component spec."""

  PARAMETERS = {
      'push_destination':
          ExecutionParameter(type=pusher_pb2.PushDestination, optional=True),
      'custom_config':
          ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
      'model_export': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  OUTPUTS = {
      'model_push': ChannelParameter(type=standard_artifacts.PushedModel),
  }


class SchemaGenSpec(ComponentSpec):
  """SchemaGen component spec."""

  PARAMETERS = {}
  INPUTS = {
      'stats': ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.Schema),
  }


class StatisticsGenSpec(ComponentSpec):
  """StatisticsGen component spec."""

  PARAMETERS = {}
  INPUTS = {
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=standard_artifacts.ExampleStatistics),
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
      'transform_output':
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      'schema':
          ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {'output': ChannelParameter(type=standard_artifacts.Model)}


class TransformSpec(ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      'module_file': ExecutionParameter(type=(str, Text), optional=True),
      'preprocessing_fn': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
      'schema': ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      'transform_output':
          ChannelParameter(type=standard_artifacts.TransformGraph),
      'transformed_examples':
          ChannelParameter(type=standard_artifacts.Examples),
  }
