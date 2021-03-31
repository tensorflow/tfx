# Lint as: python3
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

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


if __name__ == '__main__':
  tf.test.main()
