# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Penguin example using TFX."""

import os
import sys
from typing import List, Optional, Text

import absl
from absl import flags

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

flags.DEFINE_enum(
    'model_framework', 'keras', ['keras', 'flax_experimental'],
    'The modeling framework.')


# This example assumes that penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(os.environ['HOME'], 'penguin')
_data_root = os.path.join(_penguin_root, 'data')

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

# Configs for ExampleGen and SpansResolver, e.g.,
#
# This will match the <input_base>/day3/* as ExampleGen's input and generate
# Examples artifact with Span equals to 3.
#   examplegen_input_config = tfx.proto.Input(splits=[
#       tfx.proto.Input.Split(name='input', pattern='day{SPAN}/*'),
#   ])
#   examplegen_range_config = tfx.proto.RangeConfig(
#       static_range=tfx.proto.StaticRange(
#           start_span_number=3, end_span_number=3))
#
# This will get the latest 2 Spans (Examples artifacts) from MLMD for training.
#   resolver_range_config = tfx.proto.RangeConfig(
#       rolling_range=tfx.proto.RollingRange(num_spans=2))
_examplegen_input_config = None
_examplegen_range_config = None
_resolver_range_config = None


def _create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    module_file: Text,
    accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_path: Text,
    enable_tuning: bool,
    examplegen_input_config: Optional[tfx.proto.Input],
    examplegen_range_config: Optional[tfx.proto.RangeConfig],
    resolver_range_config: Optional[tfx.proto.RangeConfig],
    beam_pipeline_args: List[Text],
) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline.
    data_root: directory containing the penguin data.
    module_file: path to files used in Trainer and Transform components.
    accuracy_threshold: minimum accuracy to push the model.
    serving_model_dir: filepath to write pipeline SavedModel to.
    metadata_path: path to local pipeline ML Metadata store.
    enable_tuning: If True, the hyperparameter tuning through KerasTuner is
      enabled.
    examplegen_input_config: ExampleGen's input_config.
    examplegen_range_config: ExampleGen's range_config.
    resolver_range_config: SpansResolver's range_config. Specify this will
      enable SpansResolver to get a window of ExampleGen's output Spans for
      transform and training.
    beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
      refer to https://beam.apache.org/documentation/runners/direct/.

  Returns:
    A TFX pipeline object.
  """

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(
      input_base=data_root,
      input_config=examplegen_input_config,
      range_config=examplegen_range_config)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Gets multiple Spans for transform and training.
  if resolver_range_config:
    examples_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.SpanRangeStrategy,
        config={
            'range_config': resolver_range_config
        },
        examples=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.Examples,
            producer_component_id=example_gen.id)).with_id('span_resolver')

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=(examples_resolver.outputs['examples']
                if resolver_range_config else example_gen.outputs['examples']),
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Tunes the hyperparameters for model training based on user-provided Python
  # function. Note that once the hyperparameters are tuned, you can drop the
  # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
  if enable_tuning:
    tuner = tfx.components.Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=20),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use ImporterNode to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of ImporterNode,
      #   hparams_importer = ImporterNode(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']
                       if enable_tuning else None),
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='species')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': accuracy_threshold}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  components_list = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      trainer,
      model_resolver,
      evaluator,
      pusher,
  ]
  if resolver_range_config:
    components_list.append(examples_resolver)
  if enable_tuning:
    components_list.append(tuner)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components_list,
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_local.py [--model_framework=flax_experimental]
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  absl.flags.FLAGS(sys.argv)
  _pipeline_name = f'penguin_local_{flags.FLAGS.model_framework}'

  # Python module file to inject customized logic into the TFX components. The
  # Transform, Trainer and Tuner all require user-defined functions to run
  # successfully.
  _module_file_name = f'penguin_utils_{flags.FLAGS.model_framework}.py'
  _module_file = os.path.join(_penguin_root, _module_file_name)
  # Path which can be listened to by the model server.  Pusher will output the
  # trained model here.
  _serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                    _pipeline_name)
  # Pipeline root for artifacts.
  _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
  # Sqlite ML-metadata db path.
  _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                'metadata.db')
  tfx.orchestration.LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          accuracy_threshold=0.6,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          # TODO(b/180723394): support tuning for Flax.
          enable_tuning=(flags.FLAGS.model_framework == 'keras'),
          examplegen_input_config=_examplegen_input_config,
          examplegen_range_config=_examplegen_range_config,
          resolver_range_config=_resolver_range_config,
          beam_pipeline_args=_beam_pipeline_args))
