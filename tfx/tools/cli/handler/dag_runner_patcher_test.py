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
"""Tests for tfx.tools.cli.handler.dag_runner_patcher."""

from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.tools.cli.handler import dag_runner_patcher

_PIPELINE_NAME = 'pipeline1'


class _DummyDagRunner(tfx_runner.TfxRunner):

  def __init__(self):
    super().__init__()
    self.bar = 'baz'

  def run(self, pipeline):
    pass


class _DummyDagRunnerPatcher(dag_runner_patcher.DagRunnerPatcher):

  def __init__(self, test_case, call_real_run=True):
    super().__init__(call_real_run)
    self._test_case = test_case

  def get_runner_class(self):
    return _DummyDagRunner

  def _before_run(self, runner, pipeline, context):
    context['foo'] = 42
    self._test_case.assertEqual(runner.bar, 'baz')
    self._test_case.assertEqual(self._context[self.PIPELINE_NAME],
                                _PIPELINE_NAME)

  def _after_run(self, runner, pipeline, context):
    self._test_case.assertEqual(context['foo'], 42)
    context['foo'] = 24


class DagRunnerPatcherTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('PipelineObject', False),
      ('PipelineProto', True),
  )
  @mock.patch.object(_DummyDagRunner, 'run', autospec=True)
  def testPatcher(self, use_pipeline_proto, mock_run):
    patcher = _DummyDagRunnerPatcher(self)
    pipeline = tfx_pipeline.Pipeline(_PIPELINE_NAME, 'dummy_root')
    if use_pipeline_proto:
      pipeline = compiler.Compiler().compile(pipeline)
    runner = _DummyDagRunner()

    with patcher.patch() as context:
      self.assertNotIn('foo', context)
      self.assertFalse(patcher.run_called)
      runner.run(pipeline)
      print(context)
      self.assertEqual(context['foo'], 24)
      self.assertTrue(patcher.run_called)
      mock_run.assert_called_once()

  @mock.patch.object(_DummyDagRunner, 'run', autospec=True)
  def testPatcherWithoutRealRun(self, mock_run):
    patcher = _DummyDagRunnerPatcher(self, False)
    with patcher.patch() as _:
      _DummyDagRunner().run(tfx_pipeline.Pipeline(_PIPELINE_NAME, ''))
      mock_run.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
