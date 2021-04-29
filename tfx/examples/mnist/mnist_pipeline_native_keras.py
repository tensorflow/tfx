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
"""MNIST handwritten digit classification example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

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
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

_pipeline_name = 'mnist_native_keras'

# This example assumes that MNIST data is stored in ~/mnist/data and the utility
# function is in ~/mnist. Feel free to customize as needed.
_mnist_root = os.path.join(os.environ['HOME'], 'mnist')
_data_root = os.path.join(_mnist_root, 'data')
# Python module files to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_mnist_root, 'mnist_utils_native_keras.py')
_module_file_lite = os.path.join(
    _mnist_root, 'mnist_utils_native_keras_lite.py')

# Path which can be listened to by the model server. Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_mnist_root, 'serving_model', _pipeline_name)
_serving_model_dir_lite = os.path.join(
    _mnist_root, 'serving_model_lite', _pipeline_name)

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
                     module_file: Text, module_file_lite: Text,
                     serving_model_dir: Text, serving_model_dir_lite: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the handwritten digit classification example using TFX."""
  # Brings data into the pipeline.
  example_gen = ImportExampleGen(input_base=data_root)

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

  def _create_trainer(module_file, component_id):
    return Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(num_steps=100)).with_id(component_id)

  # Uses user-provided Python function that trains a Keras model.
  trainer = _create_trainer(module_file, 'Trainer.mnist')

  # Trains the same model as the one above, but converts it into a TFLite one.
  trainer_lite = _create_trainer(module_file_lite, 'Trainer.mnist_lite')

  # TODO(b/150949276): Add resolver back once it supports two trainers.

  # Uses TFMA to compute evaluation statistics over features of a model and
  # performs quality validation of a candidate model.
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='image_class')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.config.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.8})))
          ])
      ])

  eval_config_lite = tfma.EvalConfig()
  eval_config_lite.CopyFrom(eval_config)
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  # Uses TFMA to compute the evaluation statistics over features of a model.
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config).with_id('Evaluator.mnist')

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  evaluator_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite).with_id('Evaluator.mnist_lite')

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir))).with_id('Pusher.mnist')

  # Checks whether the TFLite model passed the validation steps and pushes the
  # model to a file destination if check passed.
  pusher_lite = Pusher(
      model=trainer_lite.outputs['model'],
      model_blessing=evaluator_lite.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir_lite))).with_id(
                  'Pusher.mnist_lite')

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          transform,
          trainer,
          trainer_lite,
          evaluator,
          evaluator_lite,
          pusher,
          pusher_lite,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python mnist_pipeline_native_keras.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          module_file_lite=_module_file_lite,
          serving_model_dir=_serving_model_dir,
          serving_model_dir_lite=_serving_model_dir_lite,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))
