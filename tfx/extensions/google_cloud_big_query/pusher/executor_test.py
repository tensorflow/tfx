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
"""Tests for tfx.extensions.google_cloud_big_query.ml.pusher.executor."""

import copy
import os
from typing import Any, Dict
from unittest import mock

from google.cloud import bigquery
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_big_query.pusher import executor
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import json_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'components', 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        'model': [self._model_export],
        'model_blessing': [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = 'gs://bucket/test_model_path'
    self._output_dict = {
        'pushed_model': [self._model_push],
    }
    self._exec_properties = {
        'custom_config': {
            'bigquery_serving_args': {
                'model_name': 'model_name',
                'project_id': 'project_id',
                'bq_dataset_id': 'bq_dataset_id',
                'compute_project_id': 'compute_project_id',
            },
        },
        'push_destination': None,
    }
    self._executor = executor.Executor()

    # Setting up Mock for external services
    self.addCleanup(mock.patch.stopall)
    self.mock_bq = mock.patch.object(bigquery, 'Client', autospec=True).start()
    self.mock_check_blessing = mock.patch.object(
        executor.Executor, 'CheckBlessing', autospec=True).start()
    self.mock_copy_dir = mock.patch.object(
        io_utils, 'copy_dir', autospec=True).start()

  def _serialize_custom_config_under_test(self) -> Dict[str, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def assertPushed(self):
    self.mock_copy_dir.assert_called_with(
        src=mock.ANY, dst=self._model_push.uri)
    self.assertEqual(1, self._model_push.get_int_custom_property('pushed'))

  def assertNotPushed(self):
    self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

  def testPipelineRoot(self):
    self._model_push.uri = '/none_gcs_pipeline_root'
    with self.assertRaises(ValueError):
      self._executor.Do(self._input_dict, self._output_dict,
                        self._serialize_custom_config_under_test())

  def testBigQueryServingArgs(self):
    temp_exec_properties = {
        'custom_config': json_utils.dumps({}),
        'push_destination': None,
    }
    with self.assertRaises(ValueError):
      self._executor.Do(self._input_dict, self._output_dict,
                        temp_exec_properties)

  def testDoBlessed(self):
    self.mock_check_blessing.return_value = True
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    self.mock_bq.assert_called_once()
    self.assertPushed()

  def testDoNotBlessed(self):
    self.mock_check_blessing.return_value = False
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    self.mock_bq.assert_not_called()
    self.assertNotPushed()

if __name__ == '__main__':
  tf.test.main()
