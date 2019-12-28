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
"""Tests for tfx.extensions.google_cloud_bigquery_ml.pusher.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports
import mock
import tensorflow as tf
from google.cloud import bigquery
from tfx.extensions.google_cloud_big_query_ml.pusher.executor import Executor
from tfx.types import standard_artifacts
from tfx.utils import path_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'components', 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        'model_export': [self._model_export],
        'model_blessing': [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    tf.io.gfile.makedirs(self._model_push.uri)
    self._output_dict = {
        'model_push': [self._model_push],
    }
    self._exec_properties = {
        'custom_config': {
            'bigquery_serving_args': {
                'model_name': 'model_name',
                'project_id': 'project_id',
                'bq_dataset_id': 'bq_dataset_id',
            },
        },
        'push_destination': None,
    }
    self._executor = Executor()

    # Setting up Mock for external services
    self.addCleanup(mock.patch.stopall)
    self.mock_bq = mock.patch.object(bigquery, 'Client', autospec=True).start()
    self.mock_check_blessing = mock.patch.object(
        Executor, 'CheckBlessing', autospec=True).start()
    self.mock_path_utils = mock.patch.object(
        path_utils,
        'serving_model_path',
        return_value='gs://test_model_path',
        autospec=True).start()

  def testPipelineRoot(self):
    self.mock_path_utils.return_value = '/none_gcs_pipeline_root'
    with self.assertRaises(ValueError):
      self._executor.Do(self._input_dict, self._output_dict,
                        self._exec_properties)

  def testBigQueryServingArgs(self):
    temp_exec_properties = {
        'custom_config': {},
        'push_destination': None,
    }
    with self.assertRaises(ValueError):
      self._executor.Do(self._input_dict, self._output_dict,
                        temp_exec_properties)

  def testDoBlessed(self):
    self.mock_check_blessing.return_value = True
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    self.mock_bq.assert_called_once()
    self.assertEqual(
        1, self._model_push.mlmd_artifact.custom_properties['pushed'].int_value)

  def testDoNotBlessed(self):
    self.mock_check_blessing.return_value = False
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    self.mock_bq.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
