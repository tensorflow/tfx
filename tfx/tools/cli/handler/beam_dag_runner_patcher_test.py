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
"""Tests for tfx.tools.cli.handler.beam_dag_runner_patcher."""

from unittest import mock

import tensorflow as tf
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.tools.cli.handler import beam_dag_runner_patcher

_PIPELINE_NAME = 'pipeline1'


class BeamDagRunnerPatcherTest(tf.test.TestCase):

  @mock.patch.object(beam_dag_runner.BeamDagRunner, 'run', autospec=True)
  def testPatcher(self, mock_run):
    patcher = beam_dag_runner_patcher.BeamDagRunnerPatcher()
    with patcher.patch() as context:
      beam_dag_runner.BeamDagRunner().run(
          tfx_pipeline.Pipeline(_PIPELINE_NAME, ''))
      mock_run.assert_not_called()
      self.assertEqual(context[patcher.PIPELINE_NAME], _PIPELINE_NAME)


if __name__ == '__main__':
  tf.test.main()
