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
"""Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


class TaxiPipelineKubeflowTest(tf.test.TestCase):

  def setUp(self):
    self._tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                   self.get_temp_dir())
    self._olddir = os.getcwd()
    os.chdir(self._tmp_dir)

  def tearDown(self):
    os.chdir(self._olddir)

  def test_taxi_pipeline_construction_and_definition_file_exists(self):
    # Import creates the pipeline.
    from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_kubeflow  # pylint: disable=g-import-not-at-top
    logical_pipeline = taxi_pipeline_kubeflow._create_pipeline()
    self.assertEqual(9, len(logical_pipeline.components))

    file_path = os.path.join(self._tmp_dir,
                             'chicago_taxi_pipeline_kubeflow.tar.gz')
    self.assertTrue(tf.gfile.Exists(file_path))


if __name__ == '__main__':
  tf.test.main()
