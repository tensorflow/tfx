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
"""Common utility for testing CLI in Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

from tfx.components.base.base_component import BaseComponent
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils import dsl_utils

# The base container image name to use when building the image used in tests.
BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The project id to use to run tests.
GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

# The GCP bucket to use to write output artifacts.
BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# The input data root location on GCS. The input files are never modified and
# are safe for concurrent reads.
DATA_ROOT = os.environ['KFP_E2E_DATA_ROOT']

# Location of the input taxi module file to be used in the test pipeline.
TAXI_MODULE_FILE = os.environ['KFP_E2E_TAXI_MODULE_FILE']


def create_e2e_components(pipeline_root: Text, csv_input_location: Text,
                          taxi_module_file: Text) -> List[BaseComponent]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    csv_input_location: The location of the input data directory.
    taxi_module_file: The location of the module file for Transform/Trainer.

  Returns:
    A list of TFX components that constitutes an end-to-end test pipeline.
  """
  examples = dsl_utils.csv_input(csv_input_location)

  example_gen = CsvExampleGen(input_base=examples)
  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])
  infer_schema = SchemaGen(
      stats=statistics_gen.outputs['output'], infer_feature_shape=False)
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=infer_schema.outputs['output'])
  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=infer_schema.outputs['output'],
      module_file=taxi_module_file)
  trainer = Trainer(
      module_file=taxi_module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['output'],
      transform_output=transform.outputs['transform_output'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model_exports=trainer.outputs['output'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['output'])
  pusher = Pusher(
      model_export=trainer.outputs['output'],
      model_blessing=model_validator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


def get_kubeflow_metadata_config(
    pipeline_name: Text) -> kubeflow_pb2.KubeflowMetadataConfig:
  config = kubeflow_pb2.KubeflowMetadataConfig()
  config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
  config.mysql_db_service_port.environment_variable = 'MYSQL_SERVICE_PORT'
  # MySQL database name cannot exceed 64 characters.
  config.mysql_db_name.value = 'mlmd_{}'.format(pipeline_name[-59:])
  config.mysql_db_user.value = 'root'
  config.mysql_db_password.value = ''
  return config
