# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.tools.cli.handler.local_dag_runner_patcher."""

from unittest import mock

import tensorflow as tf
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.local import local_dag_runner
from tfx.tools.cli.handler import local_dag_runner_patcher

_PIPELINE_NAME = 'pipeline1'


class LocalDagRunnerPatcherTest(tf.test.TestCase):

  @mock.patch.object(local_dag_runner.LocalDagRunner, 'run', autospec=True)
  def testPatcher(self, mock_run):
    patcher = local_dag_runner_patcher.LocalDagRunnerPatcher()
    with patcher.patch() as context:
      local_dag_runner.LocalDagRunner().run(
          tfx_pipeline.Pipeline(_PIPELINE_NAME, ''))
      mock_run.assert_not_called()
      self.assertEqual(context[patcher.PIPELINE_NAME], _PIPELINE_NAME)


if __name__ == '__main__':
  tf.test.main()
