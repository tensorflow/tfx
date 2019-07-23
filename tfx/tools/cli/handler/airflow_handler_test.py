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
import sys
import click
import mock
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import airflow_handler


def _MockSubprocess(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a pickle file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_args = {'pipeline_name': 'chicago_taxi_simple'}
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)


def _MockSubprocess2(cmd):  # pylint: disable=invalid-name, unused-argument
  click.echo(cmd)


def _MockSubprocess3(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a pickle file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_args = {}
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)


def _MockSubprocess4(cmd):  # pylint: disable=invalid-name, unused-argument
  list_dags_output_path = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata',
      'test_airflow_list_dags_output.txt')
  with open(list_dags_output_path, 'rb') as f:
    list_dags_output = f.read()
  return list_dags_output


class AirflowHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(AirflowHandlerTest, self).setUp()
    self._home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_airflow_home_value = os.environ.get('AIRFLOW_HOME', '')
    os.environ['AIRFLOW_HOME'] = os.path.join(os.environ['HOME'], 'airflow')

    # Flags for handler.
    self.engine = 'airflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_airflow_1.py')
    self.pipeline_name = 'chicago_taxi_simple'
    self.run_id = 'manual__2019-07-19T19:56:02+00:00'

    # Pipeline args for mocking subprocess
    self.pipeline_args = {'pipeline_name': 'chicago_taxi_simple'}

  def tearDown(self):
    super(AirflowHandlerTest, self).tearDown()
    os.environ['HOME'] = self._original_home_value
    os.environ['AIRFLOW_HOME'] = self._original_airflow_home_value

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
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'test_pipeline_airflow_1.py')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(
        handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_create_pipeline_existent_pipeline(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(str(err.exception), 'Pipeline {} already exists.'
                     .format(self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_update_pipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_airflow_1.py')
    flags_dict_1 = {labels.ENGINE_FLAG: self.engine,
                    labels.PIPELINE_DSL_PATH: pipeline_path_1}
    handler = airflow_handler.AirflowHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_airflow_2.py')
    flags_dict_2 = {labels.ENGINE_FLAG: self.engine,
                    labels.PIPELINE_DSL_PATH: pipeline_path_2}
    handler = airflow_handler.AirflowHandler(flags_dict_2)
    handler.update_pipeline()
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'test_pipeline_airflow_2.py')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(
        handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_update_pipeline_no_pipeline(self):
    # Update pipeline without craeting one.
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(str(err.exception), 'Pipeline {} does not exist.'
                     .format(self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_delete_pipeline(self):
    # First create a pipeline.
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: self.pipeline_path}
    handler = airflow_handler.AirflowHandler(flags_dict)
    handler.create_pipeline()

    # Now delete the pipeline created aand check if pipeline folder is deleted.
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_NAME: self.pipeline_name}
    handler = airflow_handler.AirflowHandler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_delete_pipeline_non_existent_pipeline(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_NAME: self.pipeline_name}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(str(err.exception), 'Pipeline {} does not exist.'
                     .format(flags_dict[labels.PIPELINE_NAME]))

  def test_list_pipelines_non_empty(self):
    # First create two pipelines in the dags folder.
    handler_pipeline_path_1 = os.path.join(os.environ['AIRFLOW_HOME'],
                                           'dags',
                                           'pipeline_1')
    handler_pipeline_path_2 = os.path.join(os.environ['AIRFLOW_HOME'],
                                           'dags',
                                           'pipeline_2')
    tf.io.gfile.makedirs(handler_pipeline_path_1)
    tf.io.gfile.makedirs(handler_pipeline_path_2)

    # Now, list the pipelines
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = airflow_handler.AirflowHandler(flags_dict)

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('pipeline_1', captured.contents())
    self.assertIn('pipeline_2', captured.contents())

  def test_list_pipelines_empty(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('No pipelines to display.', captured.contents())

  @mock.patch('subprocess.call', _MockSubprocess)
  def test_compile_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())

  @mock.patch('subprocess.call', _MockSubprocess3)
  def test_compile_pipeline_no_pipeline_args(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.compile_pipeline()
    self.assertEqual(
        str(err.exception),
        'Unable to compile pipeline. Check your pipeline dsl.')

  @mock.patch('subprocess.call', _MockSubprocess2)
  def test_create_run(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(os.environ['AIRFLOW_HOME'],
                                         'dags',
                                         self.pipeline_name)
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_NAME: self.pipeline_name}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.create_run()
    self.assertIn("['airflow', 'unpause', '" + self.pipeline_name + "']",
                  captured.contents())
    self.assertIn("['airflow', 'trigger_dag', '" + self.pipeline_name + "']",
                  captured.contents())

  def test_create_run_no_pipeline(self):
    # Run pipeline without creating one.
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_NAME: self.pipeline_name}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.create_run()
    self.assertEqual(str(err.exception), 'Pipeline {} does not exist.'
                     .format(flags_dict[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess2)
  def test_list_runs(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(os.environ['AIRFLOW_HOME'], 'dags',
                                         self.pipeline_name)
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_runs()
    self.assertIn("['airflow', 'list_dag_runs', '" + self.pipeline_name + "']",
                  captured.contents())

  def test_list_runs_wrong_pipeline(self):
    # Run pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: 'chicago_taxi'
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.list_runs()
    self.assertEqual(
        str(err.exception),
        'Pipeline {} does not exist.'.format(flags_dict[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.check_output', _MockSubprocess4)
  def test_get_run(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(os.environ['AIRFLOW_HOME'], 'dags',
                                         self.pipeline_name)
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.RUN_ID: self.run_id,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_run()
    self.assertIn('run_id : ' + self.run_id, captured.contents())
    self.assertIn('state : running', captured.contents())

  def test_get_run_wrong_pipeline(self):
    # Run pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.RUN_ID: self.run_id,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.get_run()
    self.assertEqual(
        str(err.exception),
        'Pipeline {} does not exist.'.format(flags_dict[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _MockSubprocess2)
  def test_delete_run(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(os.environ['AIRFLOW_HOME'], 'dags',
                                         self.pipeline_name)
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.delete_run()
    self.assertIn('Not supported for Airflow.', captured.contents())

  @mock.patch('subprocess.call', _MockSubprocess2)
  def test_terminate_run(self):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(os.environ['AIRFLOW_HOME'], 'dags',
                                         self.pipeline_name)
    tf.io.gfile.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = airflow_handler.AirflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.terminate_run()
    self.assertIn('Not supported for Airflow.', captured.contents())


if __name__ == '__main__':
  tf.test.main()
