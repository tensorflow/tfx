# Lint as: python3
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
"""Iris flowers example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Dict, Optional, Text

import absl

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input


# Identifier for the pipeline. This will also be used as the model name on AI
# Platform, so it should begin with a letter and only consist of letters,
# numbers, and underscores.
_pipeline_name = 'iris_sklearn'

# Google Cloud Platform project id to use when deploying this pipeline. Leave
# blank to run locally.
_project_id = 'project_id'

# Directory and data locations (uses Google Cloud Storage).
_bucket = 'gs://bucket'

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': _pipeline_name,
    'project_id': _project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [_gcp_region],
    # TODO(b/157646655): Update the version once sklearn support is added back
    # to CAIP in the next runtime release.
    'runtime_version': '1.15',
}

# This example assumes that Iris flowers data is stored in ~/iris/data and the
# utility function is in ~/iris. Feel free to customize as needed.
_iris_root = os.path.join(os.environ['HOME'], 'iris')
_data_root = os.path.join(_iris_root, 'data')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_iris_root, 'experimental', 'iris_utils_sklearn.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_iris_root, 'serving_model', _pipeline_name)

# Directory and data locations. This example assumes all of the flowers
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem. The AI Platform Pusher requires
# that pipeline outputs are stored in a GCS bucket.
_tfx_root = os.path.join(_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
# TODO(humichael) Beam dag runner expects this to be a local path. Switch to
# kubeflow dag runner when making cloud example.
_metadata_path = os.path.join(os.environ['HOME'], 'tfx', 'metadata',
                              _pipeline_name, 'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     metadata_path: Text,
                     ai_platform_serving_args: Optional[Dict[Text, Text]],
                     direct_num_workers: int) -> pipeline.Pipeline:
  """Implements the Iris flowers pipeline with TFX."""
  examples = external_input(data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering during training.
  # TODO(humichael): Handle applying transformations at serving time in
  # Milestone 3.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that trains a model using TF-Learn.
  # Num_steps is not provided during evaluation because the scikit-learn model
  # loads and evaluates the entire test set at once.
  # TODO(b/159470716): Make schema optional in Trainer.
  trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      eval_args=trainer_pb2.EvalArgs())

  # TODO(humichael): Add Evaluator once it's decided how to proceed with
  # Milestone 2.

  pusher = Pusher(
      model=trainer.outputs['model'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  # TODO(humichael): Split this example into local and cloud examples.
  cloud_pusher = Pusher(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_pusher_executor.Executor),
      model=trainer.outputs['model'],
      instance_name='cloud_pusher',
      custom_config={
          ai_platform_pusher_executor.SERVING_ARGS_KEY:
          ai_platform_serving_args,
      })

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
          pusher,
          cloud_pusher,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      # TODO(b/142684737): The multi-processing API might change.
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers],
  )


# To run this pipeline from the python CLI:
#   $python iris_pipeline_sklearn.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          ai_platform_serving_args=_ai_platform_serving_args,
          # 0 means auto-detect based on the number of CPUs available during
          # execution time.
          direct_num_workers=0))
