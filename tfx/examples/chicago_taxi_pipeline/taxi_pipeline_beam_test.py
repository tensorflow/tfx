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
import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing


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
        beam_pipeline_args=[])
    self.assertEqual(9, len(logical_pipeline.components))

  def testTaxiPipelineNewStyleCompatibility(self):
    example_gen = CsvExampleGen(input_base='/tmp/fake/path')
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    self.assertIs(statistics_gen.inputs['examples'],
                  statistics_gen.inputs['input_data'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    self.assertIs(schema_gen.inputs['statistics'],
                  schema_gen.inputs['stats'])
    self.assertIs(schema_gen.outputs['schema'],
                  schema_gen.outputs['output'])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    self.assertIs(example_validator.inputs['statistics'],
                  example_validator.inputs['stats'])
    self.assertIs(example_validator.outputs['anomalies'],
                  example_validator.outputs['output'])
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='/tmp/fake/module/file')
    self.assertIs(transform.inputs['examples'],
                  transform.inputs['input_data'])
    self.assertIs(transform.outputs['transform_graph'],
                  transform.outputs['transform_output'])
    trainer = Trainer(
        module_file='/tmp/fake/module/file',
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))
    self.assertIs(trainer.inputs['transform_graph'],
                  trainer.inputs['transform_output'])
    self.assertIs(trainer.outputs['model'],
                  trainer.outputs['output'])
    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(signature_name='eval')],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['trip_start_hour'])
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                thresholds={
                    'accuracy':
                        tfma.config.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.6}),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10}))
                })
        ])
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
    self.assertIs(evaluator.inputs['model'],
                  evaluator.inputs['model_exports'])
    self.assertIs(evaluator.outputs['evaluation'],
                  evaluator.outputs['output'])
    pusher = Pusher(
        model=trainer.outputs['output'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='/fake/serving/dir')))
    self.assertIs(pusher.inputs['model'],
                  pusher.inputs['model_export'])
    self.assertIs(pusher.outputs['pushed_model'],
                  pusher.outputs['model_push'])


if __name__ == '__main__':
  tf.test.main()
