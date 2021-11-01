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
"""TFX/TFJS Page Prediction Pipeline."""

import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx.v1 import dsl
from tfx.v1 import orchestration
from tfx.v1 import proto
from tfx.v1 import types
from tfx.v1.components import Evaluator
from tfx.v1.components import ExampleValidator
from tfx.v1.components import ImportExampleGen
from tfx.v1.components import Pusher
from tfx.v1.components import SchemaGen
from tfx.v1.components import StatisticsGen
from tfx.v1.components import Trainer
from tfx.v1.components import Transform


_pipeline_name = 'tfx_tfjs_page_prediction'

# This example assumes that train set data is stored in
# ~/tfx_tfjs_page_prediction/data/. Feel free to customize and use
# google cloud storage paths if needed.
_page_prediction_root = os.path.join(os.environ['HOME'],
                                     'tfx_tfjs_page_prediction')
_data_root = os.path.join(_page_prediction_root, 'data')

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_page_prediction_root,
                            'tfjs_next_page_prediction_util.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_page_prediction_root, 'serving_model',
                                  _pipeline_name)

# Directory and data locations. This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(
    os.getenv('HOME'), 'metadata', _pipeline_name, 'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> dsl.Pipeline:
  """Implements the page prediction pipline with TFX."""
  input_config = proto.Input(
      splits=[proto.Input.Split(name='input', pattern='*.tfrecord.gz')])
  output_config = proto.Output(
      split_config=proto.SplitConfig(splits=[
          proto.SplitConfig.Split(name='train', hash_buckets=9),
          proto.SplitConfig.Split(name='eval', hash_buckets=1)
      ]))

  # Brings data in to the pipline
  example_gen = ImportExampleGen(
      input_base=data_root,
      input_config=input_config,
      output_config=output_config)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that trains a model.
  trainer = Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=proto.TrainArgs(num_steps=100000),
      eval_args=proto.EvalArgs(num_steps=200))

  # Get the latest blessed model for model validation.
  model_resolver = dsl.Resolver(
      strategy_class=dsl.experimental.LatestBlessedModelStrategy,
      model=dsl.Channel(type=types.standard_artifacts.Model),
      model_blessing=dsl.Channel(
          type=types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      # Directly evaluates the tfjs model.
      model_specs=[tfma.ModelSpec(label_key='label', model_type='tf_js')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          # Increase this threshold when training on complete
                          # dataset.
                          lower_bound={'value': 0.01}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-2})))
          ])
      ])

  evaluator = Evaluator(
      examples=transform.outputs['transformed_examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=proto.PushDestination(
          filesystem=proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  components = [
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
  return dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config=orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      enable_cache=True,
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
# $python imdb_pipeline_native_keras.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  orchestration.LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))
