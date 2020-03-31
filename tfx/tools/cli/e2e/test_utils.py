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

# The location of test data. The input files are copied to a test-local
# location for each invocation, and cleaned up at the end of test.
TESTDATA_ROOT = os.environ['KFP_E2E_TEST_DATA_ROOT']

# The location of test user module
# It is retrieved from inside the container subject to testing.
MODULE_ROOT = '/tfx-src/tfx/components/testdata/module_file'


def get_test_output_dir():
  return 'gs://{}/test_output'.format(BUCKET_NAME)


def get_csv_input_location():
  return os.path.join(TESTDATA_ROOT, 'external', 'csv')


def get_transform_module():
  return os.path.join(MODULE_ROOT, 'transform_module.py')


def get_trainer_module():
  return os.path.join(MODULE_ROOT, 'trainer_module.py')


def create_e2e_components(
    pipeline_root: Text,
    csv_input_location: Text,
    transform_module: Text,
    trainer_module: Text,
) -> List[BaseComponent]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    csv_input_location: The location of the input data directory.
    transform_module: The location of the transform module file.
    trainer_module: The location of the trainer module file.

  Returns:
    A list of TFX components that constitutes an end-to-end test pipeline.
  """
  examples = dsl_utils.external_input(csv_input_location)

  example_gen = CsvExampleGen(input=examples)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  infer_schema = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=infer_schema.outputs['schema'])
  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      module_file=transform_module)
  trainer = Trainer(
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10),
      eval_args=trainer_pb2.EvalArgs(num_steps=5),
      module_file=trainer_module)
  model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]
