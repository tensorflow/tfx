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
"""IMDB Sentiment Analysis example using TFX."""

from __future__ import absolute_import
from __future__ import print_function

import os
from typing import Text

import absl
import tensorflow_model_analysis as tfma

from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import GenericExecutor
from tfx.components.base import executor_spec 
from tfx.utils.dsl_utils import tfrecord_input

from tfx.proto import trainer_pb2 

from tfx.orchestration import metadata
from tfx.orchestration import pipeline 
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

_pipeline_name = 'bert_cola'

# This exmaple assumes the utility function is in ~/bert_cola
_imdb_root = os.path.join(os.environ['HOME'], 'bert_cola')
_data_root = os.path.join(_imdb_root, 'data') 
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_imdb_root, 'bert_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_imdb_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the flowers
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     metadata_path: Text,
                     direct_num_workers: int) -> pipeline.Pipeline:
    """Implements the imdb sentiment analysis pipline with TFX."""
    examples = tfrecord_input(data_root)
    # Brings data in to the pipline
    example_gen = ImportExampleGen(input=examples)

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
    
     # Uses user-provided Python function that trains a model using TF-Learn.
    trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=200),
      eval_args=trainer_pb2.EvalArgs(num_steps=100))
    
    
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
          #model_resolver,
          #valuator,
          #pusher,
      ],
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      enable_cache=True,
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers],
  ) 

if __name__ == '__main__':
   absl.logging.set_verbosity(absl.logging.INFO)
   BeamDagRunner().run(
           _create_pipeline(
               pipeline_name = _pipeline_name,
               pipeline_root = _pipeline_root,
               data_root = _data_root,
               module_file = _module_file,
               serving_model_dir = _serving_model_dir,
               metadata_path = _metadata_path,
               direct_num_workers = 1))
