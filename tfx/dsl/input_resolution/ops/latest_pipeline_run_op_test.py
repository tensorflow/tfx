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

  def testLatestPipelineRun_Empty(self):
    with self.mlmd_handler as mlmd_handler:
      context = resolver_op.Context(store=mlmd_handler.store)
      self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self._latest_pipeline_run_op.set_context(context)

      # Tests LatestPipelineRun with empty input.
      result = self._latest_pipeline_run_op.apply({})
      self.assertAllEqual(result, {})

  def testLatestPipelineRun_OneKey(self):
    with self.mlmd_handler as mlmd_handler:
      context = resolver_op.Context(store=mlmd_handler.store)
      self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self._latest_pipeline_run_op.set_context(context)

      node_context = self.put_context('node', 'example-gen')

      # Creates the first pipeline run and artifact.
      input_artifact_1 = self.put_artifact('Example')
      pipeline_run_ctx_1 = self.put_context('pipeline_run', 'run-001')
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': [input_artifact_1]},
          contexts=[node_context, pipeline_run_ctx_1])

      # Tests LatestPipelineRun with the first artifact.
      result = self._latest_pipeline_run_op.apply({'key': [input_artifact_1]})
      self.assertAllEqual(result, {'key': [input_artifact_1]})

      # Creates the second pipeline run and artifact.
      input_artifact_2 = self.put_artifact('Example')
      pipeline_run_ctx_2 = self.put_context('pipeline_run', 'run-002')
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': [input_artifact_2]},
          contexts=[node_context, pipeline_run_ctx_2])

      # Tests LatestPipelineRun with the first and second artifacts.
      result = self._latest_pipeline_run_op.apply(
          {'key': [input_artifact_1, input_artifact_2]})
      self.assertAllEqual(result, {'key': [input_artifact_2]})

  def testLatestPipelineRun_TwoKeys(self):
    with self.mlmd_handler as mlmd_handler:
      context = resolver_op.Context(store=mlmd_handler.store)
      self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self._latest_pipeline_run_op.set_context(context)

      example_gen_node_context = self.put_context('node', 'example-gen')
      statistics_gen_node_context = self.put_context('node', 'statistics-gen')

      # First pipeline run generates two artifacts
      example_artifact_1 = self.put_artifact('Example')
      statistics_artifact_1 = self.put_artifact('Statistics')
      pipeline_run_ctx_1 = self.put_context('pipeline_run', 'run-001')
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': [example_artifact_1]},
          contexts=[example_gen_node_context, pipeline_run_ctx_1])
      self.put_execution(
          'StatisticsGen',
          inputs={},
          outputs={'statistics': [statistics_artifact_1]},
          contexts=[statistics_gen_node_context, pipeline_run_ctx_1])

      # Second pipeline run generates one artifact
      example_artifact_2 = self.put_artifact('Example')
      pipeline_run_ctx_2 = self.put_context('pipeline_run', 'run-002')
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': [example_artifact_2]},
          contexts=[example_gen_node_context, pipeline_run_ctx_2])

      # Only Examples are input
      result = self._latest_pipeline_run_op.apply(
          {'examples': [example_artifact_1, example_artifact_2]})
      self.assertAllEqual(result, {'examples': [example_artifact_2]})

      # Both Examples and Statistics are input
      result = self._latest_pipeline_run_op.apply({
          'examples': [example_artifact_1, example_artifact_2],
          'statistics': [statistics_artifact_1]
      })
      self.assertAllEqual(result, {
          'examples': [example_artifact_1],
          'statistics': [statistics_artifact_1]
      })

      # If any of the input value is empty, returns empty dict.
      result = self._latest_pipeline_run_op.apply({
          'examples': [example_artifact_1, example_artifact_2],
          'statistics': []
      })
      self.assertAllEqual(result, {})

  def testLatestPipelineRun_PartialComplete(self):
    with self.mlmd_handler as mlmd_handler:
      context = resolver_op.Context(store=mlmd_handler.store)
      self._latest_pipeline_run_op = ops.LatestPipelineRun.create()
      self._latest_pipeline_run_op.set_context(context)

      node_context = self.put_context('node', 'example-gen')

      # Creates a pipeline run and artifact.
      input_artifact_1 = self.put_artifact('Example')
      pipeline_run_ctx_1 = self.put_context('pipeline_run', 'run-001')

      # There are two exectuions in the pipeline run, but only one of them is
      # completed and creates an artifact.
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': [input_artifact_1]},
          contexts=[node_context, pipeline_run_ctx_1])
      self.put_execution(
          'ExampleGen',
          inputs={},
          outputs={'examples': []},
          contexts=[node_context, pipeline_run_ctx_1])

      # Tests LatestPipelineRun with the first artifact.
      result = self._latest_pipeline_run_op.apply({'key': [input_artifact_1]})
      self.assertAllEqual(result, {})


if __name__ == '__main__':
  tf.test.main()
