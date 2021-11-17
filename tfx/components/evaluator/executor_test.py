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

import os
import unittest

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components.evaluator import executor
from tfx.components.testdata.module_file import evaluator_module
from tfx.dsl.io import fileio
from tfx.proto import evaluator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.utils import proto_utils


@unittest.skipIf(tf.__version__ < '2',
                 'This test uses testdata only compatible with TF 2.x')
class ExecutorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('evaluation_w_eval_config', {
          standard_component_specs.EVAL_CONFIG_KEY:
              proto_utils.proto_to_json(
                  tfma.EvalConfig(slicing_specs=[
                      tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                      tfma.SlicingSpec(
                          feature_keys=['trip_start_day', 'trip_miles']),
                  ]))
      }),
      ('evaluation_w_module_file', {
          standard_component_specs.EVAL_CONFIG_KEY:
              proto_utils.proto_to_json(
                  tfma.EvalConfig(slicing_specs=[
                      tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                      tfma.SlicingSpec(
                          feature_keys=['trip_start_day', 'trip_miles']),
                  ])),
          standard_component_specs.MODULE_FILE_KEY:
              None
      }),
      ('evaluation_w_module_path', {
          standard_component_specs.EVAL_CONFIG_KEY:
              proto_utils.proto_to_json(
                  tfma.EvalConfig(slicing_specs=[
                      tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                      tfma.SlicingSpec(
                          feature_keys=['trip_start_day', 'trip_miles']),
                  ])),
          standard_component_specs.MODULE_PATH_KEY:
              evaluator_module.__name__,
      }),
      ('model_agnostic_evaluation', {
          standard_component_specs.EVAL_CONFIG_KEY:
              proto_utils.proto_to_json(
                  tfma.EvalConfig(
                      model_specs=[
                          tfma.ModelSpec(
                              label_key='tips', prediction_key='tips'),
                      ],
                      slicing_specs=[
                          tfma.SlicingSpec(feature_keys=['trip_start_hour']),
                          tfma.SlicingSpec(
                              feature_keys=['trip_start_day', 'trip_miles']),
                      ]))
      }, True),
  )
  def testEvalution(self, exec_properties, model_agnostic=False):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    baseline_model = standard_artifacts.Model()
    baseline_model.uri = os.path.join(source_data_dir, 'trainer/previous/')
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')
    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples, examples],
        standard_component_specs.SCHEMA_KEY: [schema],
    }
    if not model_agnostic:
      model = standard_artifacts.Model()
      model.uri = os.path.join(source_data_dir, 'trainer/current')
      input_dict[standard_component_specs.MODEL_KEY] = [model]

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    output_dict = {
        standard_component_specs.EVALUATION_KEY: [eval_output],
        standard_component_specs.BLESSING_KEY: [blessing_output],
    }

    # Test multiple splits.
    exec_properties[
        standard_component_specs.EXAMPLE_SPLITS_KEY] = json_utils.dumps(
            ['train', 'eval'])

    if standard_component_specs.MODULE_FILE_KEY in exec_properties:
      exec_properties[standard_component_specs.MODULE_FILE_KEY] = os.path.join(
          source_data_dir, 'module_file', 'evaluator_module.py')

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        fileio.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'plots')))
    self.assertFalse(
        fileio.exists(os.path.join(blessing_output.uri, 'BLESSED')))

  @parameterized.named_parameters(('legacy_feature_slicing', {
      standard_component_specs.FEATURE_SLICING_SPEC_KEY:
          proto_utils.proto_to_json(
              evaluator_pb2.FeatureSlicingSpec(specs=[
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_hour']),
                  evaluator_pb2.SingleSlicingSpec(
                      column_for_slicing=['trip_start_day', 'trip_miles']),
              ])),
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
    model.uri = os.path.join(source_data_dir, 'trainer/current')
    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.MODEL_KEY: [model],
    }

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    output_dict = {
        standard_component_specs.EVALUATION_KEY: [eval_output],
        standard_component_specs.BLESSING_KEY: [blessing_output],
    }

    try:
      # Need to import the following module so that the fairness indicator
      # post-export metric is registered.  This may raise an ImportError if the
      # currently-installed version of TFMA does not support fairness
      # indicators.
      import tensorflow_model_analysis.addons.fairness.post_export_metrics.fairness_indicators  # pylint: disable=g-import-not-at-top, unused-variable
      exec_properties[
          standard_component_specs
          .FAIRNESS_INDICATOR_THRESHOLDS_KEY] = '[0.1, 0.3, 0.5, 0.7, 0.9]'
    except ImportError:
      logging.warning(
          'Not testing fairness indicators because a compatible TFMA version '
          'is not installed.')

    # List needs to be serialized before being passed into Do function.
    exec_properties[
        standard_component_specs.EXAMPLE_SPLITS_KEY] = json_utils.dumps(None)

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        fileio.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'plots')))
    self.assertFalse(
        fileio.exists(os.path.join(blessing_output.uri, 'BLESSED')))

  @parameterized.named_parameters(
      (
          'eval_config_w_validation',
          {
              standard_component_specs.EVAL_CONFIG_KEY:
                  proto_utils.proto_to_json(
                      tfma.EvalConfig(
                          model_specs=[
                              tfma.ModelSpec(label_key='tips'),
                          ],
                          metrics_specs=[
                              tfma.MetricsSpec(metrics=[
                                  tfma.MetricConfig(
                                      class_name='ExampleCount',
                                      # Count > 0, OK.
                                      threshold=tfma.MetricThreshold(
                                          value_threshold=tfma
                                          .GenericValueThreshold(
                                              lower_bound={'value': 0}))),
                              ]),
                          ],
                          slicing_specs=[tfma.SlicingSpec()]))
          },
          True,
          True),
      (
          'eval_config_w_validation_fail',
          {
              standard_component_specs.EVAL_CONFIG_KEY:
                  proto_utils.proto_to_json(
                      tfma.EvalConfig(
                          model_specs=[
                              tfma.ModelSpec(
                                  name='baseline1',
                                  label_key='tips',
                                  is_baseline=True),
                              tfma.ModelSpec(
                                  name='candidate1', label_key='tips'),
                          ],
                          metrics_specs=[
                              tfma.MetricsSpec(metrics=[
                                  tfma.MetricConfig(
                                      class_name='ExampleCount',
                                      # Count < -1, NOT OK.
                                      threshold=tfma.MetricThreshold(
                                          value_threshold=tfma
                                          .GenericValueThreshold(
                                              upper_bound={'value': -1}))),
                              ]),
                          ],
                          slicing_specs=[tfma.SlicingSpec()]))
          },
          False,
          True),
      (
          'no_baseline_model_ignore_change_threshold_validation_pass',
          {
              standard_component_specs.EVAL_CONFIG_KEY:
                  proto_utils.proto_to_json(
                      tfma.EvalConfig(
                          model_specs=[
                              tfma.ModelSpec(
                                  name='baseline',
                                  label_key='tips',
                                  is_baseline=True),
                              tfma.ModelSpec(
                                  name='candidate', label_key='tips'),
                          ],
                          metrics_specs=[
                              tfma.MetricsSpec(metrics=[
                                  tfma.MetricConfig(
                                      class_name='ExampleCount',
                                      # Count > 0, OK.
                                      threshold=tfma.MetricThreshold(
                                          value_threshold=tfma
                                          .GenericValueThreshold(
                                              lower_bound={'value': 0}))),
                                  tfma.MetricConfig(
                                      class_name='Accuracy',
                                      # Should be ignored due to no baseline.
                                      threshold=tfma.MetricThreshold(
                                          change_threshold=tfma
                                          .GenericChangeThreshold(
                                              relative={'value': 0},
                                              direction=tfma.MetricDirection
                                              .LOWER_IS_BETTER))),
                              ]),
                          ],
                          slicing_specs=[tfma.SlicingSpec()]))
          },
          True,
          False))
  def testDoValidation(self, exec_properties, blessed, has_baseline):
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
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')
    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples, examples],
        standard_component_specs.MODEL_KEY: [model],
        standard_component_specs.SCHEMA_KEY: [schema],
    }
    if has_baseline:
      input_dict[standard_component_specs.BASELINE_MODEL_KEY] = [baseline_model]

    # Create output dict.
    eval_output = standard_artifacts.ModelEvaluation()
    eval_output.uri = os.path.join(output_data_dir, 'eval_output')
    blessing_output = standard_artifacts.ModelBlessing()
    blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
    output_dict = {
        standard_component_specs.EVALUATION_KEY: [eval_output],
        standard_component_specs.BLESSING_KEY: [blessing_output],
    }

    # List needs to be serialized before being passed into Do function.
    exec_properties[
        standard_component_specs.EXAMPLE_SPLITS_KEY] = json_utils.dumps(None)

    # Run executor.
    evaluator = executor.Executor()
    evaluator.Do(input_dict, output_dict, exec_properties)

    # Check evaluator outputs.
    self.assertTrue(
        fileio.exists(os.path.join(eval_output.uri, 'eval_config.json')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'metrics')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'plots')))
    self.assertTrue(fileio.exists(os.path.join(eval_output.uri, 'validations')))
    if blessed:
      self.assertTrue(
          fileio.exists(os.path.join(blessing_output.uri, 'BLESSED')))
    else:
      self.assertTrue(
          fileio.exists(os.path.join(blessing_output.uri, 'NOT_BLESSED')))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
