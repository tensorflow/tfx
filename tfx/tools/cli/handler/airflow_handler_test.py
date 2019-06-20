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
"""Tests for tfx.tools.cli.handler.airflow_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import mock
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import airflow_handler


class AirflowHandlerTest(tf.test.TestCase):

  def setUp(self):
    self._home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_airflow_home_value = os.environ.get('AIRFLOW_HOME', '')
    os.environ['AIRFLOW_HOME'] = os.path.join(os.environ['HOME'], 'airflow')

    self.engine = 'airflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'taxi_pipeline_simple.py')

    # Pipeline args for mocking subprocess
    self.pipeline_args = {'pipeline_name': 'chicago_taxi_simple'}

  def tearDown(self):
    os.environ['HOME'] = self._original_home_value
    os.environ['AIRFLOW_HOME'] = self._original_airflow_home_value

  def _MockSubprocess(self, env):
    # Store pipeline_args in a pickle file
    pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
    pipeline_args = {'pipeline_name': 'chicago_taxi_simple'}
    with open(pipeline_args_path, 'w') as f:
      json.dump(pipeline_args, f)

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_extract_pipeline_args(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    self.assertEqual(pipeline_args, self.pipeline_args)

  def test_get_handler_home(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    handler_home_dir = 'AIRFLOW_HOME'
    self.assertEqual(os.environ[handler_home_dir], handler._handler_home_dir)

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_save_pipeline(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    handler._save_pipeline(pipeline_args)
    self.assertTrue(tf.io.gfile.exists(os.path.join(
        handler._handler_home_dir,
        'dags',
        self.pipeline_args[labels.PIPELINE_NAME])))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_create_pipeline(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    handler.create_pipeline()
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertTrue(tf.io.gfile.exists(os.path.join(
        handler_pipeline_path, 'taxi_pipeline_simple.py')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(
        handler_pipeline_path, 'pipeline_args.json')))


if __name__ == '__main__':
  tf.test.main()
