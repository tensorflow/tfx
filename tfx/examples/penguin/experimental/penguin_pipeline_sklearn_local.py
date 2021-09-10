# Copyright 2021 Google LLC. All Rights Reserved.
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
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

_pipeline_name = 'penguin_sklearn_local'

# This example assumes that Penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(os.environ['HOME'], 'penguin')
_data_root = os.path.join(_penguin_root, 'data')

# Python module file to inject customized logic into the TFX components.
# Trainer requires user-defined functions to run successfully.
_trainer_module_file = os.path.join(
    _penguin_root, 'experimental', 'penguin_utils_sklearn.py')

# Python module file to inject customized logic into the TFX components. The
# Evaluator component needs a custom extractor in order to make predictions
# using the scikit-learn model.
_evaluator_module_file = os.path.join(
    _penguin_root, 'experimental', 'sklearn_predict_extractor.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                  _pipeline_name)

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
# TODO(b/171316320): Change direct_running_mode back to multi_processing and set
# direct_num_workers to 0.
_beam_pipeline_args = [
    '--direct_running_mode=multi_threading',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=1',
]


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    trainer_module_file: str,
    evaluator_module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    beam_pipeline_args: List[str],
) -> tfx.dsl.Pipeline:
  """Implements the Penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(
      input_base=os.path.join(data_root, 'labelled'))

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

  # TODO(humichael): Handle applying transformation component in Milestone 3.

  # Uses user-provided Python function that trains a model.
  # Num_steps is not provided during evaluation because the scikit-learn model
  # loads and evaluates the entire test set at once.
  trainer = tfx.components.Trainer(
      module_file=trainer_module_file,
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=2000),
      eval_args=tfx.proto.EvalArgs())

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
                  class_name='Accuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.6}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = tfx.components.Evaluator(
      module_file=evaluator_module_file,
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          trainer,
          model_resolver,
          evaluator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args,
  )


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_sklearn_local.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tfx.orchestration.LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          trainer_module_file=_trainer_module_file,
          evaluator_module_file=_evaluator_module_file,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))
