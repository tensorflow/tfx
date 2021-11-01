# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.scripts.run_component."""

import os
import pathlib
import tempfile

from absl.testing import absltest
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.scripts import run_component
from tfx.types import artifact_utils


class RunComponentTest(absltest.TestCase):

  def testRunStatisticsGen(self):
    # Prepare the paths
    test_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'components', 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName)
    statistics_split_names_path = os.path.join(output_data_dir,
                                               'statistics.properties',
                                               'split_names')
    fileio.makedirs(output_data_dir)

    # Run StatisticsGen
    run_component.run_component(
        full_component_class_name='tfx.components.StatisticsGen',
        examples_uri=os.path.join(test_data_dir, 'csv_example_gen'),
        examples_split_names=artifact_utils.encode_split_names(
            ['train', 'eval']),
        # Testing that we can set non-string artifact properties
        examples_version='1',
        statistics_path=output_data_dir,
        statistics_split_names_path=statistics_split_names_path,
    )

    # Check the statistics_gen outputs
    self.assertTrue(
        fileio.exists(
            os.path.join(output_data_dir, 'Split-train', 'FeatureStats.pb')))
    self.assertTrue(
        fileio.exists(
            os.path.join(output_data_dir, 'Split-eval', 'FeatureStats.pb')))
    self.assertTrue(os.path.exists(statistics_split_names_path))
    self.assertEqual(
        pathlib.Path(statistics_split_names_path).read_text(),
        '["train", "eval"]')

  def testRunSchemaGen(self):
    # Prepare the paths
    test_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'components', 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    # Run SchemaGen
    run_component.run_component(
        full_component_class_name='tfx.components.SchemaGen',
        # Testing that we can specify input artifact paths
        statistics_path=os.path.join(test_data_dir, 'statistics_gen'),
        # Testing that we can specify artifact properties
        statistics_split_names=artifact_utils.encode_split_names(
            ['train', 'eval']),
        # Testing that we can pass arguments for non-string properties
        infer_feature_shape='1',
        # Testing that we can specify output artifact paths
        schema_path=os.path.join(output_data_dir),
    )

    # Checking the schema_gen outputs
    self.assertTrue(
        fileio.exists(os.path.join(output_data_dir, 'schema.pbtxt')))

if __name__ == '__main__':
  tf.test.main()
