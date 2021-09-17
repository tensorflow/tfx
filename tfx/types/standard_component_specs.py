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

import tensorflow_model_analysis as tfma
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import range_config_pb2
from tfx.proto import trainer_pb2
from tfx.proto import transform_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter

# Parameters keys for modules
# Shared Keys across components
SCHEMA_KEY = 'schema'
EXAMPLES_KEY = 'examples'
MODEL_KEY = 'model'
BLESSING_KEY = 'blessing'
TRAIN_ARGS_KEY = 'train_args'
CUSTOM_CONFIG_KEY = 'custom_config'
MODEL_BLESSING_KEY = 'model_blessing'
TRANSFORM_GRAPH_KEY = 'transform_graph'
EVAL_ARGS_KEY = 'eval_args'
MODULE_FILE_KEY = 'module_file'
EXCLUDE_SPLITS_KEY = 'exclude_splits'
STATISTICS_KEY = 'statistics'
# Key for example_validator
ANOMALIES_KEY = 'anomalies'
# Key for evaluator
EVAL_CONFIG_KEY = 'eval_config'
FEATURE_SLICING_SPEC_KEY = 'feature_slicing_spec'
FAIRNESS_INDICATOR_THRESHOLDS_KEY = 'fairness_indicator_thresholds'
EXAMPLE_SPLITS_KEY = 'example_splits'
MODULE_PATH_KEY = 'module_path'
BASELINE_MODEL_KEY = 'baseline_model'
EVALUATION_KEY = 'evaluation'
# Key for infra_validator
SERVING_SPEC_KEY = 'serving_spec'
VALIDATION_SPEC_KEY = 'validation_spec'
REQUEST_SPEC_KEY = 'request_spec'
# Key for tuner
TUNER_FN_KEY = 'tuner_fn'
TUNE_ARGS_KEY = 'tune_args'
BEST_HYPERPARAMETERS_KEY = 'best_hyperparameters'
# Key for bulk_inferer
MODEL_SPEC_KEY = 'model_spec'
DATA_SPEC_KEY = 'data_spec'
OUTPUT_EXAMPLE_SPEC_KEY = 'output_example_spec'
INFERENCE_RESULT_KEY = 'inference_result'
OUTPUT_EXAMPLES_KEY = 'output_examples'
# Key for schema_gen
INFER_FEATURE_SHAPE_KEY = 'infer_feature_shape'
SCHEMA_FILE_KEY = 'schema_file'
# Key for statistics_gen
STATS_OPTIONS_JSON_KEY = 'stats_options_json'
# Key for example_gen
INPUT_BASE_KEY = 'input_base'
INPUT_CONFIG_KEY = 'input_config'
OUTPUT_CONFIG_KEY = 'output_config'
OUTPUT_DATA_FORMAT_KEY = 'output_data_format'
OUTPUT_FILE_FORMAT_KEY = 'output_file_format'
RANGE_CONFIG_KEY = 'range_config'
# Key for pusher
PUSH_DESTINATION_KEY = 'push_destination'
INFRA_BLESSING_KEY = 'infra_blessing'
PUSHED_MODEL_KEY = 'pushed_model'
# Key for TrainerSpec
RUN_FN_KEY = 'run_fn'
TRAINER_FN_KEY = 'trainer_fn'
BASE_MODEL_KEY = 'base_model'
HYPERPARAMETERS_KEY = 'hyperparameters'
MODEL_RUN_KEY = 'model_run'
# Key for transform
PREPROCESSING_FN_KEY = 'preprocessing_fn'
STATS_OPTIONS_UPDATER_FN_KEY = 'stats_options_updater_fn'
FORCE_TF_COMPAT_V1_KEY = 'force_tf_compat_v1'
SPLITS_CONFIG_KEY = 'splits_config'
ANALYZER_CACHE_KEY = 'analyzer_cache'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'
UPDATED_ANALYZER_CACHE_KEY = 'updated_analyzer_cache'
DISABLE_STATISTICS_KEY = 'disable_statistics'
PRE_TRANSFORM_SCHEMA_KEY = 'pre_transform_schema'
PRE_TRANSFORM_STATS_KEY = 'pre_transform_stats'
POST_TRANSFORM_SCHEMA_KEY = 'post_transform_schema'
POST_TRANSFORM_STATS_KEY = 'post_transform_stats'
POST_TRANSFORM_ANOMALIES_KEY = 'post_transform_anomalies'


class BulkInferrerSpec(ComponentSpec):
  """BulkInferrer component spec."""

  PARAMETERS = {
      MODEL_SPEC_KEY:
          ExecutionParameter(type=bulk_inferrer_pb2.ModelSpec, optional=True),
      DATA_SPEC_KEY:
          ExecutionParameter(type=bulk_inferrer_pb2.DataSpec, optional=True),
      OUTPUT_EXAMPLE_SPEC_KEY:
          ExecutionParameter(
              type=bulk_inferrer_pb2.OutputExampleSpec, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      MODEL_BLESSING_KEY:
          ChannelParameter(
              type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      INFERENCE_RESULT_KEY:
          ChannelParameter(
              type=standard_artifacts.InferenceResult, optional=True),
      OUTPUT_EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }


class EvaluatorSpec(ComponentSpec):
  """Evaluator component spec."""

  PARAMETERS = {
      EVAL_CONFIG_KEY:
          ExecutionParameter(type=tfma.EvalConfig, optional=True),
      # TODO(b/181911822): Deprecated, use eval_config.slicing_specs.
      FEATURE_SLICING_SPEC_KEY:
          ExecutionParameter(
              type=evaluator_pb2.FeatureSlicingSpec, optional=True),
      # This parameter is experimental: its interface and functionality may
      # change at any time.
      FAIRNESS_INDICATOR_THRESHOLDS_KEY:
          ExecutionParameter(type=str, optional=True),
      EXAMPLE_SPLITS_KEY:
          ExecutionParameter(type=str, optional=True),
      MODULE_FILE_KEY:
          ExecutionParameter(type=str, optional=True),
      MODULE_PATH_KEY:
          ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      BASELINE_MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
  }
  OUTPUTS = {
      EVALUATION_KEY: ChannelParameter(type=standard_artifacts.ModelEvaluation),
      BLESSING_KEY: ChannelParameter(type=standard_artifacts.ModelBlessing),
  }


class ExampleValidatorSpec(ComponentSpec):
  """ExampleValidator component spec."""

  PARAMETERS = {
      EXCLUDE_SPLITS_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      STATISTICS_KEY:
          ChannelParameter(type=standard_artifacts.ExampleStatistics),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      ANOMALIES_KEY: ChannelParameter(type=standard_artifacts.ExampleAnomalies),
  }


class FileBasedExampleGenSpec(ComponentSpec):
  """File-based ExampleGen component spec."""

  PARAMETERS = {
      INPUT_BASE_KEY:
          ExecutionParameter(type=str),
      INPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Input),
      OUTPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Output),
      OUTPUT_DATA_FORMAT_KEY:
          ExecutionParameter(type=int),  # example_gen_pb2.PayloadFormat enum.
      OUTPUT_FILE_FORMAT_KEY:
          ExecutionParameter(type=int),  # example_gen_pb2.FileFormat enum.
      CUSTOM_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
      RANGE_CONFIG_KEY:
          ExecutionParameter(type=range_config_pb2.RangeConfig, optional=True),
  }
  INPUTS = {}
  OUTPUTS = {
      EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }


class QueryBasedExampleGenSpec(ComponentSpec):
  """Query-based ExampleGen component spec."""

  PARAMETERS = {
      INPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Input),
      OUTPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Output),
      OUTPUT_DATA_FORMAT_KEY:
          ExecutionParameter(type=int),  # example_gen_pb2.PayloadFormat enum.
      OUTPUT_FILE_FORMAT_KEY:
          ExecutionParameter(type=int),  # example_gen_pb2.FileFormat enum.
      CUSTOM_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
  }
  INPUTS = {}
  OUTPUTS = {
      EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }


class InfraValidatorSpec(ComponentSpec):
  """InfraValidator component spec."""

  PARAMETERS = {
      SERVING_SPEC_KEY:
          ExecutionParameter(type=infra_validator_pb2.ServingSpec),
      VALIDATION_SPEC_KEY:
          ExecutionParameter(
              type=infra_validator_pb2.ValidationSpec, optional=True),
      REQUEST_SPEC_KEY:
          ExecutionParameter(
              type=infra_validator_pb2.RequestSpec, optional=True)
  }

  INPUTS = {
      MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model),
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }

  OUTPUTS = {
      BLESSING_KEY: ChannelParameter(type=standard_artifacts.InfraBlessing),
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
      PUSH_DESTINATION_KEY:
          ExecutionParameter(type=pusher_pb2.PushDestination, optional=True),
      CUSTOM_CONFIG_KEY:
          ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      MODEL_BLESSING_KEY:
          ChannelParameter(
              type=standard_artifacts.ModelBlessing, optional=True),
      INFRA_BLESSING_KEY:
          ChannelParameter(
              type=standard_artifacts.InfraBlessing, optional=True),
  }
  OUTPUTS = {
      PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel),
  }


class SchemaGenSpec(ComponentSpec):
  """SchemaGen component spec."""

  PARAMETERS = {
      INFER_FEATURE_SHAPE_KEY: ExecutionParameter(type=int, optional=True),
      EXCLUDE_SPLITS_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      STATISTICS_KEY:
          ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {
      SCHEMA_KEY: ChannelParameter(type=standard_artifacts.Schema),
  }


class ImportSchemaGenSpec(ComponentSpec):
  """ImportSchemaGen component spec."""

  PARAMETERS = {
      SCHEMA_FILE_KEY: ExecutionParameter(type=str),
  }
  INPUTS = {}
  OUTPUTS = {
      SCHEMA_KEY: ChannelParameter(type=standard_artifacts.Schema),
  }


class StatisticsGenSpec(ComponentSpec):
  """StatisticsGen component spec."""

  PARAMETERS = {
      STATS_OPTIONS_JSON_KEY: ExecutionParameter(type=str, optional=True),
      EXCLUDE_SPLITS_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
  }
  OUTPUTS = {
      STATISTICS_KEY:
          ChannelParameter(type=standard_artifacts.ExampleStatistics),
  }


class TrainerSpec(ComponentSpec):
  """Trainer component spec."""

  PARAMETERS = {
      TRAIN_ARGS_KEY: ExecutionParameter(type=trainer_pb2.TrainArgs),
      EVAL_ARGS_KEY: ExecutionParameter(type=trainer_pb2.EvalArgs),
      MODULE_FILE_KEY: ExecutionParameter(type=str, optional=True),
      MODULE_PATH_KEY: ExecutionParameter(type=str, optional=True),
      RUN_FN_KEY: ExecutionParameter(type=str, optional=True),
      TRAINER_FN_KEY: ExecutionParameter(type=str, optional=True),
      CUSTOM_CONFIG_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      TRANSFORM_GRAPH_KEY:
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      BASE_MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      HYPERPARAMETERS_KEY:
          ChannelParameter(
              type=standard_artifacts.HyperParameters, optional=True),
  }
  OUTPUTS = {
      MODEL_KEY: ChannelParameter(type=standard_artifacts.Model),
      MODEL_RUN_KEY: ChannelParameter(type=standard_artifacts.ModelRun)
  }


class TunerSpec(ComponentSpec):
  """ComponentSpec for TFX Tuner Component."""

  PARAMETERS = {
      MODULE_FILE_KEY: ExecutionParameter(type=str, optional=True),
      TUNER_FN_KEY: ExecutionParameter(type=str, optional=True),
      TRAIN_ARGS_KEY: ExecutionParameter(type=trainer_pb2.TrainArgs),
      EVAL_ARGS_KEY: ExecutionParameter(type=trainer_pb2.EvalArgs),
      TUNE_ARGS_KEY: ExecutionParameter(type=tuner_pb2.TuneArgs, optional=True),
      CUSTOM_CONFIG_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      TRANSFORM_GRAPH_KEY:
          ChannelParameter(
              type=standard_artifacts.TransformGraph, optional=True),
      BASE_MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
  }
  OUTPUTS = {
      BEST_HYPERPARAMETERS_KEY:
          ChannelParameter(type=standard_artifacts.HyperParameters),
  }


class TransformSpec(ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
      MODULE_FILE_KEY:
          ExecutionParameter(type=str, optional=True),
      MODULE_PATH_KEY:
          ExecutionParameter(type=str, optional=True),
      PREPROCESSING_FN_KEY:
          ExecutionParameter(type=str, optional=True),
      STATS_OPTIONS_UPDATER_FN_KEY:
          ExecutionParameter(type=str, optional=True),
      FORCE_TF_COMPAT_V1_KEY:
          ExecutionParameter(type=int, optional=True),
      CUSTOM_CONFIG_KEY:
          ExecutionParameter(type=str, optional=True),
      SPLITS_CONFIG_KEY:
          ExecutionParameter(type=transform_pb2.SplitsConfig, optional=True),
      DISABLE_STATISTICS_KEY:
          ExecutionParameter(type=int, optional=True),
  }
  INPUTS = {
      EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples),
      SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema),
      ANALYZER_CACHE_KEY:
          ChannelParameter(
              type=standard_artifacts.TransformCache, optional=True),
  }
  OUTPUTS = {
      TRANSFORM_GRAPH_KEY:
          ChannelParameter(type=standard_artifacts.TransformGraph),
      TRANSFORMED_EXAMPLES_KEY:
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
      UPDATED_ANALYZER_CACHE_KEY:
          ChannelParameter(
              type=standard_artifacts.TransformCache, optional=True),
      PRE_TRANSFORM_SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      PRE_TRANSFORM_STATS_KEY:
          ChannelParameter(
              type=standard_artifacts.ExampleStatistics, optional=True),
      POST_TRANSFORM_SCHEMA_KEY:
          ChannelParameter(type=standard_artifacts.Schema, optional=True),
      POST_TRANSFORM_STATS_KEY:
          ChannelParameter(
              type=standard_artifacts.ExampleStatistics, optional=True),
      POST_TRANSFORM_ANOMALIES_KEY:
          ChannelParameter(
              type=standard_artifacts.ExampleAnomalies, optional=True)
  }
