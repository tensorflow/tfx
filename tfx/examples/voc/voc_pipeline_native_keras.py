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
"""VOC 2007 object detection example using TFX."""

from __future__ import absolute_import
from __future__ import division

import os
from typing import Text, List

import absl
import tensorflow_model_analysis as tfma

from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'voc_native_keras'

# This example assumes that VOC train set data is stored in
# ~/voc/data/train, test set data is stored in ~/voc/data/test, and
# the utility function is in ~/voc. Feel free to customize as needed.
_voc_root = os.path.join(os.environ['HOME'], 'voc')
_data_root = os.path.join(_voc_root, 'data')
# Python module files to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_voc_root, 'voc_utils_native_keras.py')

# Path which can be listened to by the model server. Pusher will output the
# trained model here.
_serving_model_dir_lite = os.path.join(_voc_root, 'serving_model',
                                       _pipeline_name)

# Directory and data locations.  This example assumes all of the images,
# example code, and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir_lite: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the VOC 2007 object detection pipeline using TFX."""
  # This is needed for datasets with pre-defined splits
  # Change the pattern argument to train_whole/* and test_whole/* to train
  # on the whole VOC 2007 dataset
  input_config = example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train_tiny/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='val_tiny/*')])

  examples = external_input(data_root)

  # Brings data into the pipeline.
  example_gen = ImportExampleGen(input=examples, input_config=input_config)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

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
  # The configuration below is for training on the dataset we provided in the data folder,
  # which has 256 train and 100 test samples. The 160 train steps correspond to 20 epochs
  # on this tiny train set. The number of eval steps doesn't matter here since we will not
  # do evaluation during training.
  trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=160),
      eval_args=trainer_pb2.EvalArgs(num_steps=4))

  # Uses TFMA to compute an evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compare to a baseline).
  # We use the custom TFMA Metric CalculateMAP to evaluation the trained detection model,
  # which is defined in the voc_utils_native_keras.py
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(model_type='tf_lite')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='CalculateMAP',
                  module='voc_utils_native_keras',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.5}))
              )
          ])
      ])

  # Uses TFMA to compute the evaluation statistics over features of a model.
  # We evaluate using the materialized examples that are output by Transform because
  # 1. the decoding_png function currently performed within Transform are not
  # compatible with TFLite.
  # 2. MediaPipe requires deserialized (float32) tensor image inputs
  # Note that for deployment, the same logic that is performed within Transform
  # must be reproduced client-side.
  evaluator = Evaluator(
      examples=transform.outputs['transformed_examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir_lite)))

  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      trainer,
      evaluator,
      pusher
  ]

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)

# To run this pipeline from the python CLI:
#   $python voc_pipeline_native_keras.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          serving_model_dir_lite=_serving_model_dir_lite,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))
