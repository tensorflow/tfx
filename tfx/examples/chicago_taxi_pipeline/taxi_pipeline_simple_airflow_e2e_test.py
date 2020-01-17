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
"""End to end test for tfx.orchestration.airflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import tempfile
import time
from typing import Sequence, Set, Text

import absl
import pytest
import tensorflow.compat.v1 as tf

import unittest
from tfx.utils import io_utils


class AirflowSubprocess(object):
  """Launch an Airflow command."""

  def __init__(self, airflow_args):
    self._args = ['airflow'] + airflow_args
    self._sub_process = None

  def __enter__(self):
    self._sub_process = subprocess.Popen(self._args)
    return self

  def __exit__(self, exception_type, exception_value, traceback):  # pylint: disable=unused-argument
    if self._sub_process:
      self._sub_process.terminate()


# Number of seconds between polling pending task states.
_TASK_POLLING_INTERVAL_SEC = 10
# Maximum duration to allow no task state change.
_MAX_TASK_STATE_CHANGE_SEC = 120

# Any task state not listed as success or pending will be considered as failure.
_SUCCESS_TASK_STATES = set(['success'])
_PENDING_TASK_STATES = set(['queued', 'scheduled', 'running', 'none'])


@pytest.mark.end_to_end
class AirflowEndToEndTest(unittest.TestCase):
  """An end to end test using fully orchestrated Airflow."""

  def _GetState(self, task_name: Text) -> Text:
    """Get a task state as a string."""
    try:
      output = subprocess.check_output([
          'airflow', 'task_state', self._dag_id, task_name, self._execution_date
      ]).split()
      # Some logs are emitted to stdout, so we take the last word as state.
      return tf.compat.as_str(output[-1])
    except subprocess.CalledProcessError:
      # For multi-processing, state checking might fail because database lock
      # has not been released. 'none' will be treated as a pending state, so
      # this state checking will be retried later.
      return 'none'

  # TODO(b/130882241): Add validation on output artifact type and content.
  def _CheckOutputArtifacts(self, task: Text) -> None:
    pass

  def _PrintTaskLogsOnError(self, task):
    task_log_dir = os.path.join(self._airflow_home, 'logs',
                                '%s.%s' % (self._dag_id, task))
    for dir_name, _, leaf_files in tf.io.gfile.walk(task_log_dir):
      for leaf_file in leaf_files:
        leaf_file_path = os.path.join(dir_name, leaf_file)
        absl.logging.error('Print task log %s:', leaf_file_path)
        with tf.io.gfile.GFile(leaf_file_path, 'r') as f:
          lines = f.readlines()
          for line in lines:
            absl.logging.error(line)

  def _CheckPendingTasks(self, pending_task_names: Sequence[Text]) -> Set[Text]:
    unknown_tasks = set(pending_task_names) - set(self._all_tasks)
    assert not unknown_tasks, 'Unknown task name {}'.format(unknown_tasks)
    still_pending = set()
    failed = dict()
    for task in pending_task_names:
      task_state = self._GetState(task).lower()
      if task_state in _SUCCESS_TASK_STATES:
        absl.logging.info('Task %s succeeded, checking output artifacts', task)
        self._CheckOutputArtifacts(task)
      elif task_state in _PENDING_TASK_STATES:
        still_pending.add(task)
      else:
        failed[task] = task_state
    for task, state in failed.items():
      absl.logging.error('Retrieving logs for %s task %s', state, task)
      self._PrintTaskLogsOnError(task)
    self.assertFalse(failed)
    return still_pending

  def setUp(self):
    super(AirflowEndToEndTest, self).setUp()
    # setup airflow_home in a temp directory, config and init db.
    self._airflow_home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName)
    self._old_airflow_home = os.environ.get('AIRFLOW_HOME')
    os.environ['AIRFLOW_HOME'] = self._airflow_home
    self._old_home = os.environ.get('HOME')
    os.environ['HOME'] = self._airflow_home
    absl.logging.info('Using %s as AIRFLOW_HOME and HOME in this e2e test',
                      self._airflow_home)
    # Set a couple of important environment variables. See
    # https://airflow.apache.org/howto/set-config.html for details.
    os.environ['AIRFLOW__CORE__AIRFLOW_HOME'] = self._airflow_home
    os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = os.path.join(
        self._airflow_home, 'dags')
    os.environ['AIRFLOW__CORE__BASE_LOG_FOLDER'] = os.path.join(
        self._airflow_home, 'logs')
    os.environ['AIRFLOW__CORE__SQL_ALCHEMY_CONN'] = ('sqlite:///%s/airflow.db' %
                                                     self._airflow_home)
    # Do not load examples to make this a bit faster.
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    # Following environment variables make scheduler process dags faster.
    os.environ['AIRFLOW__SCHEDULER__JOB_HEARTBEAT_SEC'] = '1'
    os.environ['AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC'] = '1'
    os.environ['AIRFLOW__SCHEDULER__RUN_DURATION'] = '-1'
    os.environ['AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL'] = '1'
    os.environ['AIRFLOW__SCHEDULER__PRINT_STATS_INTERVAL'] = '30'
    # Using more than one thread results in a warning for sqlite backend.
    # See https://github.com/tensorflow/tfx/issues/141
    os.environ['AIRFLOW__SCHEDULER__MAX_THREADS'] = '1'

    # Following fields are specific to the chicago_taxi_simple example.
    self._dag_id = 'chicago_taxi_simple'
    self._run_id = 'manual_run_id_1'
    # This execution date must be after the start_date in chicago_taxi_simple
    # but before current execution date.
    self._execution_date = '2019-02-01T01:01:01+01:01'
    self._all_tasks = [
        'CsvExampleGen',
        'Evaluator',
        'ExampleValidator',
        'ModelValidator',
        'Pusher',
        'SchemaGen',
        'StatisticsGen',
        'Trainer',
        'Transform',
    ]
    # Copy dag file and data.
    chicago_taxi_pipeline_dir = os.path.dirname(__file__)
    simple_pipeline_file = os.path.join(chicago_taxi_pipeline_dir,
                                        'taxi_pipeline_simple.py')

    io_utils.copy_file(
        simple_pipeline_file,
        os.path.join(self._airflow_home, 'dags', 'taxi_pipeline_simple.py'))

    data_dir = os.path.join(chicago_taxi_pipeline_dir, 'data', 'simple')
    content = tf.io.gfile.listdir(data_dir)
    assert content, 'content in {} is empty'.format(data_dir)
    target_data_dir = os.path.join(self._airflow_home, 'taxi', 'data', 'simple')
    io_utils.copy_dir(data_dir, target_data_dir)
    assert tf.io.gfile.isdir(target_data_dir)
    content = tf.io.gfile.listdir(target_data_dir)
    assert content, 'content in {} is {}'.format(target_data_dir, content)
    io_utils.copy_file(
        os.path.join(chicago_taxi_pipeline_dir, 'taxi_utils.py'),
        os.path.join(self._airflow_home, 'taxi', 'taxi_utils.py'))

    # Initialize database.
    _ = subprocess.check_output(['airflow', 'initdb'])
    _ = subprocess.check_output(['airflow', 'unpause', self._dag_id])

  def testSimplePipeline(self):
    # We will use subprocess to start the DAG instead of webserver, so only
    # need to start a scheduler on the background.
    with AirflowSubprocess(['scheduler']):
      _ = subprocess.check_output([
          'airflow',
          'trigger_dag',
          self._dag_id,
          '-r',
          self._run_id,
          '-e',
          self._execution_date,
      ])
      pending_tasks = set(self._all_tasks)
      attempts = int(
          _MAX_TASK_STATE_CHANGE_SEC / _TASK_POLLING_INTERVAL_SEC) + 1
      while True:
        if not pending_tasks:
          absl.logging.info('No pending task left anymore')
          return
        for _ in range(attempts):
          absl.logging.debug('Polling task state')
          still_pending = self._CheckPendingTasks(pending_tasks)
          if len(still_pending) != len(pending_tasks):
            pending_tasks = still_pending
            break
          absl.logging.info('Polling task state after %d secs',
                            _TASK_POLLING_INTERVAL_SEC)
          time.sleep(_TASK_POLLING_INTERVAL_SEC)
        else:
          self.fail('No pending tasks in %s finished within %d secs' %
                    (pending_tasks, _MAX_TASK_STATE_CHANGE_SEC))

  def tearDown(self):
    super(AirflowEndToEndTest, self).tearDown()
    if self._old_airflow_home:
      os.environ['AIRFLOW_HOME'] = self._old_airflow_home
    if self._old_home:
      os.environ['HOME'] = self._old_home


if __name__ == '__main__':
  unittest.main()
