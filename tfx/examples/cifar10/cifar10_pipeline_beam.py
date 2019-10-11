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
"""CIFAR-10 example using TFX DSL on Beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
from typing import Text
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'cifar10_beam'

# This example assumes that the cifar10 data is stored in ~/cifar10/data and the
# cifar10 utility function is in ~/cifar10.
_cifar10_root = os.path.join(os.environ['HOME'], 'cifar10')
_data_root = os.path.join(_cifar10_root, 'data')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_cifar10_root, 'cifar10_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_cifar10_root, 'serving_model',
                                  _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago cifar10
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     metadata_path: Text) -> pipeline.Pipeline:
  """Implements the cifar10 pipeline with TFX."""
  examples = external_input(data_root)
  input_split = example_gen_pb2.Input(splits=[
      example_gen_pb2.Input.Split(name='train', pattern='train.tfrecord'),
      example_gen_pb2.Input.Split(name='eval', pattern='test.tfrecord')
  ])
  example_gen = ImportExampleGen(input=examples, input_config=input_split)
  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=infer_schema.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=1000),
      eval_args=trainer_pb2.EvalArgs(num_steps=500))

  # Uses TFMA to compute a evaluation statistics over features of a model.
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model_exports=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
          specs=[evaluator_pb2.SingleSlicingSpec()]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, infer_schema, validate_stats, transform,
          trainer, evaluator, model_validator, pusher
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      additional_pipeline_args={},
  )


# To run this pipeline from the python CLI:
#   $python cifar10_pipeline_beam.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path))
