# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.input_resolution.ops.latest_pipeline_run_op."""

import tensorflow as tf
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.utils import test_case_utils


class LatestPipelineRunTest(tf.test.TestCase, test_case_utils.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()
    self.enter_context(self.mlmd_handler)

  def testLatestPipelineRun_Empty(self):
    context = resolver_op.Context(store=self.mlmd_handler.store)
    self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
    self._latest_pipeline_run_op.set_context(context)

    # Tests LatestPipelineRun with empty input.
    result = self._latest_pipeline_run_op.apply({})
    self.assertAllEqual(result, {})

  def testLatestPipelineRun_NotEmpty(self):
    context = resolver_op.Context(store=self.mlmd_handler.store)
    self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
    self._latest_pipeline_run_op.set_context(context)

    # Makes the first pipeline run and artifact.
    input_artifact_1 = self.put_artifact('Example')
    pipeline_ctx = self.put_context('pipeline', 'my-pipeline')
    pipeline_run_ctx_1 = self.put_context('pipeline_run', 'run-001')
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [input_artifact_1]},
        contexts=[pipeline_ctx, pipeline_run_ctx_1])

    # Tests LatestPipelineRun with the first artifact.
    result = self._latest_pipeline_run_op.apply({'key': [input_artifact_1]})
    self.assertAllEqual(result, {'key': [input_artifact_1]})

    # Makes the second pipeline run and artifact.
    input_artifact_2 = self.put_artifact('Example')
    pipeline_run_ctx_2 = self.put_context('pipeline_run', 'run-002')
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [input_artifact_2]},
        contexts=[pipeline_ctx, pipeline_run_ctx_2])

    # Tests LatestPipelineRun with the first and second artifacts.
    result = self._latest_pipeline_run_op.apply(
        {'key': [input_artifact_1, input_artifact_2]})
    self.assertAllEqual(result, {'key': [input_artifact_2]})


if __name__ == '__main__':
  tf.test.main()
