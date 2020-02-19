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
"""Tests for tfx.components.evaluator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
from google.protobuf import json_format
from tfx.components.evaluator import constants
from tfx.components.evaluator import executor
from tfx.proto import evaluator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase, absl.testing.parameterized.TestCase):

  # TODO(jinhuang): add test for eval_saved_model when supported.
  @absl.testing.parameterized.named_parameters(('eval_config', {
      'eval_config':
          json_format.MessageToJson(
              tfma.EvalConfig(
                  slicing_specs=[
                      tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                      tfma.SlicingSpec(
                          feature_keys=['trip_start_day', 'trip_miles']),
                  ]),
              preserving_proto_field_name=True)
  }), ('eval_config_w_baseline', {
      'eval_config':
          json_format.MessageToJson(
              tfma.EvalConfig(
                  model_specs=[
                      tfma.ModelSpec(name='baseline', is_baseline=True),
                      tfma.ModelSpec(name='candidate'),
                  ],
                  slicing_specs=[
                      tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                      tfma.SlicingSpec(
                          feature_keys=['trip_start_day', 'trip_miles']),
                  ]),
              preserving_proto_field_name=True)
  }), ('legacy_feature_slicing', {
      'feature_slicing_spec':
          json_format.MessageToJson(
              evaluator_pb2.FeatureSlicingSpec(specs=[
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_hour']),
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_day', 'trip_miles']),
              ]),
              preserving_proto_field_name=True),
  }))
  def testDo(self, exec_properties):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    model = standard_artifacts.Model()
    baseline_model = standard_artifacts.Model()
    model.uri = os.path.join(source_data_dir, 'trainer/current')
    baseline_model.uri = os.path.join(source_data_dir, 'trainer/previous/')
    input_dict = {
        constants.EXAMPLES_KEY: [examples],
        constants.MODEL_KEY: [model],
        constants.BASELINE_MODEL_KEY: [baseline_model],
    }

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    output_dict = {
        constants.EVALUATION_KEY: [eval_output],
    }

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(eval_output.uri, 'plots')))

  @absl.testing.parameterized.named_parameters(
      (
          'eval_config_w_validation',
          {
              'eval_config':
                  json_format.MessageToJson(
                      tfma.EvalConfig(
                          model_specs=[
                              tfma.ModelSpec(name='baseline', is_baseline=True),
                              tfma.ModelSpec(name='candidate'),
                          ],
                          metrics_specs=[
                              tfma.MetricsSpec(metrics=[
                                  tfma.config.MetricConfig(
                                      class_name='ExampleCount',
                                      # Count > 0, OK.
                                      threshold=tfma.config.MetricThreshold(
                                          value_threshold=tfma
                                          .GenericValueThreshold(
                                              lower_bound={'value': 0}))),
                              ]),
                          ],
                          slicing_specs=[tfma.SlicingSpec()]),
                      preserving_proto_field_name=True)
          },
          True),
      (
          'eval_config_w_validation_fail',
          {
              'eval_config':
                  json_format.MessageToJson(
                      tfma.EvalConfig(
                          model_specs=[
                              tfma.ModelSpec(name='baseline', is_baseline=True),
                              tfma.ModelSpec(name='candidate'),
                          ],
                          metrics_specs=[
                              tfma.MetricsSpec(metrics=[
                                  tfma.config.MetricConfig(
                                      class_name='ExampleCount',
                                      # Count < -1, NOT OK.
                                      threshold=tfma.config.MetricThreshold(
                                          value_threshold=tfma
                                          .GenericValueThreshold(
                                              upper_bound={'value': -1}))),
                              ]),
                          ],
                          slicing_specs=[tfma.SlicingSpec()]),
                      preserving_proto_field_name=True)
          },
          False))
  def testDoValidation(self, exec_properties, blessed):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    model = standard_artifacts.Model()
    baseline_model = standard_artifacts.Model()
    model.uri = os.path.join(source_data_dir, 'trainer/current')
    baseline_model.uri = os.path.join(source_data_dir, 'trainer/previous/')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    input_dict = {
        constants.EXAMPLES_KEY: [examples],
        constants.MODEL_KEY: [model],
        constants.BASELINE_MODEL_KEY: [baseline_model],
    }

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    output_dict = {
        constants.EVALUATION_KEY: [eval_output],
        constants.BLESSING_KEY: [blessing_output],
    }

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(eval_output.uri, 'plots')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'validations')))
    if blessed:
      self.assertTrue(
          tf.io.gfile.exists(os.path.join(blessing_output.uri, 'BLESSED')))
    else:
      self.assertFalse(
          tf.io.gfile.exists(os.path.join(blessing_output.uri, 'BLESSED')))

  @absl.testing.parameterized.named_parameters(('legacy_feature_slicing', {
      'feature_slicing_spec':
          json_format.MessageToJson(
              evaluator_pb2.FeatureSlicingSpec(specs=[
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_hour']),
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_day', 'trip_miles']),
              ]),
              preserving_proto_field_name=True),
  }))
  def testDoLegacySingleEvalSavedModelWFairness(self, exec_properties):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    model = standard_artifacts.Model()
    baseline_model = standard_artifacts.Model()
    model.uri = os.path.join(source_data_dir, 'trainer/current')
    baseline_model.uri = os.path.join(source_data_dir, 'trainer/previous/')
    input_dict = {
        constants.EXAMPLES_KEY: [examples],
        constants.MODEL_KEY: [model],
    }

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    output_dict = {
        constants.EVALUATION_KEY: [eval_output],
        constants.BLESSING_KEY: [blessing_output],
    }

    try:
      # Need to import the following module so that the fairness indicator
      # post-export metric is registered.  This may raise an ImportError if the
      # currently-installed version of TFMA does not support fairness
      # indicators.
      import tensorflow_model_analysis.addons.fairness.post_export_metrics.fairness_indicators  # pylint: disable=g-import-not-at-top, unused-variable
      exec_properties['fairness_indicator_thresholds'] = [
          0.1, 0.3, 0.5, 0.7, 0.9
      ]
    except ImportError:
      absl.logging.warning(
          'Not testing fairness indicators because a compatible TFMA version '
          'is not installed.')

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(eval_output.uri, 'plots')))
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(blessing_output.uri, 'BLESSED')))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
