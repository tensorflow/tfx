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
from tfx.orchestration.portable.input_resolution import mlmd_mixins_test_util

from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRunTest(tf.test.TestCase, mlmd_mixins_test_util.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()
    # with self.mlmd_handler as m:
    #   self.latest_pipeline_run_op = ops.LatestPipelineRun.create()
    #   self.latest_pipeline_run_op.set_context(
    #       resolver_op.Context(store=m.store))

  def testLatestCreateTime_Empty(self):
    with self.mlmd_handler as m:
      self.latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self.latest_pipeline_run_op.set_context(
          resolver_op.Context(store=m.store))

      actual = self.latest_pipeline_run_op.apply([])
      self.assertEqual(actual, [])

  def testLatestCreateTime_NotEmpty(self):
    with self.mlmd_handler as m:
      self.latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self.latest_pipeline_run_op.set_context(
          resolver_op.Context(store=m.store))

      # Prepares artifacts
      artifact_1 = self.put_artifact('example')
      artifact_2 = self.put_artifact('example')
      # Prepares contexts
      pipeline_ctx = self.put_context(
          context_type='pipeline', context_name='pipeline')
      pipeline_run_ctx_1 = self.put_context(
          context_type='pipeline_run', context_name='pipeline_run_1')
      pipeline_run_ctx_2 = self.put_context(
          context_type='pipeline_run', context_name='pipeline_run_2')
      self.mlmd_handler.store.put_parent_contexts([
          metadata_store_pb2.ParentContext(
              child_id=pipeline_run_ctx_1.id, parent_id=pipeline_ctx.id),
          metadata_store_pb2.ParentContext(
              child_id=pipeline_run_ctx_2.id, parent_id=pipeline_ctx.id)
      ])
      # Prepares executions
      self.put_execution(
          execution_type='ExampleGen',
          inputs={},
          outputs={'x': [artifact_1]},
          contexts=[pipeline_ctx, pipeline_run_ctx_1])
      self.put_execution(
          execution_type='ExampleGen',
          inputs={},
          outputs={'x': [artifact_2]},
          contexts=[pipeline_ctx, pipeline_run_ctx_2])

      # Tests LatestPipelineRun
      actual = self.latest_pipeline_run_op.apply([artifact_1, artifact_2])
      self.assertAllEqual(actual, [artifact_2])


if __name__ == '__main__':
  tf.test.main()
