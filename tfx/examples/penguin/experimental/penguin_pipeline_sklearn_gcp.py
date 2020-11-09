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
"""Penguin example using TFX on GCP."""

import os
from typing import Dict, List, Optional, Text

import absl
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import trainer_pb2


# Identifier for the pipeline. This will also be used as the model name on AI
# Platform, so it should begin with a letter and only consist of letters,
# numbers, and underscores.
_pipeline_name = 'penguin_sklearn_gcp'

# Google Cloud Platform project id to use when deploying this pipeline. Leave
# blank to run locally.
_project_id = 'PROJECT_ID'

# Directory and data locations (uses Google Cloud Storage).
_bucket = 'gs://BUCKET'

# Custom container image in Google Container Registry (GCR) to use for training
# on Google Cloud AI Platform.
_tfx_image = f'gcr.io/{_project_id}/tfx-example-sklearn'

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    # Override the default TFX image used for training with one with the correct
    # scikit-learn version.
    'masterConfig': {
        'imageUri': _tfx_image,
    },
}

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

# This example assumes that Penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(_bucket, 'penguin')
_data_root = os.path.join(_penguin_root, 'data')
# Python module file to inject customized logic into the TFX components.
# Trainer requires user-defined functions to run successfully.
_module_file = os.path.join(_penguin_root, 'experimental',
                            'penguin_utils_sklearn.py')

# Directory and data locations. This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem. The AI Platform Pusher requires
# that pipeline outputs are stored in a GCS bucket.
_tfx_root = os.path.join(_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
# TODO(humichael): Beam dag runner expects this to be a local path. Switch to
# kubeflow dag runner when making cloud example.
_metadata_path = os.path.join(os.environ['HOME'], 'tfx', 'metadata',
                              _pipeline_name, 'metadata.db')

# Pipeline arguments for Beam powered Components.
# TODO(humichael): Use Dataflow runner once there is support for user modules
# for Evaluator and BulkInferrer.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, metadata_path: Text,
                     ai_platform_training_args: Optional[Dict[Text, Text]],
                     ai_platform_serving_args: Optional[Dict[Text, Text]],
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the Penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # TODO(humichael): Handle applying transformation component in Milestone 3.

  # Uses user-provided Python function that trains a model using TF-Learn.
  # Num_steps is not provided during evaluation because the scikit-learn model
  # loads and evaluates the entire test set at once.
  # TODO(b/159470716): Make schema optional in Trainer.
  trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_trainer_executor.GenericExecutor),
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      eval_args=trainer_pb2.EvalArgs(),
      custom_config={
          ai_platform_trainer_executor.TRAINING_ARGS_KEY:
          ai_platform_training_args,
      })

  # TODO(humichael): Add Evaluator once it's decided how to proceed with
  # Milestone 2.

  pusher = Pusher(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_pusher_executor.Executor),
      model=trainer.outputs['model'],
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
          trainer,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args,
  )


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_sklearn_gcp.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          metadata_path=_metadata_path,
          ai_platform_training_args=_ai_platform_training_args,
          ai_platform_serving_args=_ai_platform_serving_args,
          beam_pipeline_args=_beam_pipeline_args))
