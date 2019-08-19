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
"""End to end tests for Kubeflow-based orchestrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys

import tensorflow as tf
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
from tfx.orchestration.kubeflow import test_utils
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils import dsl_utils


def _create_e2e_components(pipeline_root: Text, csv_input_location: Text,
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
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)
  infer_schema = SchemaGen(
      stats=statistics_gen.outputs.output, infer_feature_shape=False)
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=taxi_module_file)
  trainer = Trainer(
      module_file=taxi_module_file,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output,
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


class KubeflowEndToEndTest(test_utils.BaseKubeflowTest):

  def _delete_pipeline_output(self, unused_pipeline_name: Text):
    """Deletes output produced by the named pipeline."""
    pass

  def testSimpleEnd2EndPipeline(self):
    """End-to-End test for simple pipeline."""
    pipeline_name = 'kubeflow-e2e-test-fixed-{}'.format(self._random_id())
    components = _create_e2e_components(
        self._pipeline_root(pipeline_name), self._data_root,
        self._taxi_module_file)
    pipeline = self._create_pipeline(pipeline_name, components)

    self._compile_and_run_pipeline(pipeline)


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
