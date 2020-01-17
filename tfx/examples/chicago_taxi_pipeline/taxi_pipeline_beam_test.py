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
"""Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input


class TaxiPipelineBeamTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineBeamTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def testTaxiPipelineCheckDagConstruction(self):
    logical_pipeline = taxi_pipeline_beam._create_pipeline(
        pipeline_name='Test',
        pipeline_root=self._test_dir,
        data_root=self._test_dir,
        module_file=self._test_dir,
        serving_model_dir=self._test_dir,
        metadata_path=self._test_dir,
        direct_num_workers=1)
    self.assertEqual(9, len(logical_pipeline.components))

  def testTaxiPipelineNewStyleCompatibility(self):
    examples = external_input('/tmp/fake/path')
    example_gen = CsvExampleGen(input=examples)
    self.assertIs(example_gen.inputs['input'],
                  example_gen.inputs['input_base'])
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    self.assertIs(statistics_gen.inputs['examples'],
                  statistics_gen.inputs['input_data'])
    infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    self.assertIs(infer_schema.inputs['statistics'],
                  infer_schema.inputs['stats'])
    self.assertIs(infer_schema.outputs['schema'],
                  infer_schema.outputs['output'])
    validate_examples = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema'])
    self.assertIs(validate_examples.inputs['statistics'],
                  validate_examples.inputs['stats'])
    self.assertIs(validate_examples.outputs['anomalies'],
                  validate_examples.outputs['output'])
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=infer_schema.outputs['schema'],
        module_file='/tmp/fake/module/file')
    self.assertIs(transform.inputs['examples'],
                  transform.inputs['input_data'])
    self.assertIs(transform.outputs['transform_graph'],
                  transform.outputs['transform_output'])
    trainer = Trainer(
        module_file='/tmp/fake/module/file',
        transformed_examples=transform.outputs['transformed_examples'],
        schema=infer_schema.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))
    self.assertIs(trainer.inputs['transform_graph'],
                  trainer.inputs['transform_output'])
    self.assertIs(trainer.outputs['model'],
                  trainer.outputs['output'])
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))
    self.assertIs(evaluator.inputs['model'],
                  evaluator.inputs['model_exports'])
    self.assertIs(evaluator.outputs['evaluation'],
                  evaluator.outputs['output'])
    model_validator = ModelValidator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'])
    pusher = Pusher(
        model=trainer.outputs['output'],
        model_blessing=model_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='/fake/serving/dir')))
    self.assertIs(pusher.inputs['model'],
                  pusher.inputs['model_export'])
    self.assertIs(pusher.outputs['pushed_model'],
                  pusher.outputs['model_push'])


if __name__ == '__main__':
  tf.test.main()
