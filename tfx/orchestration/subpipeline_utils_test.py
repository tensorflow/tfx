# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.subpipeline_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler import constants
from tfx.orchestration import pipeline as dsl_pipeline
from tfx.orchestration import subpipeline_utils
from tfx.orchestration.experimental.core.testing import test_sync_pipeline
from tfx.orchestration.portable import runtime_parameter_utils

_PIPELINE_NAME = 'test_pipeline'
_TEST_PIPELINE = dsl_pipeline.Pipeline(pipeline_name=_PIPELINE_NAME)


class SubpipelineUtilsTest(parameterized.TestCase):

  def test_is_subpipeline_with_subpipeline(self):
    subpipeline = dsl_pipeline.Pipeline(pipeline_name='subpipeline')
    pipeline = dsl_pipeline.Pipeline(
        pipeline_name=_PIPELINE_NAME, components=[subpipeline]
    )
    pipeline_ir = compiler.Compiler().compile(pipeline)
    subpipeline_ir = pipeline_ir.nodes[0].sub_pipeline
    self.assertTrue(subpipeline_utils.is_subpipeline(subpipeline_ir))

  def test_is_subpipeline_with_parent_pipelines(self):
    subpipeline = dsl_pipeline.Pipeline(pipeline_name='subpipeline')
    pipeline = dsl_pipeline.Pipeline(
        pipeline_name=_PIPELINE_NAME, components=[subpipeline]
    )
    pipeline_ir = compiler.Compiler().compile(pipeline)
    self.assertFalse(subpipeline_utils.is_subpipeline(pipeline_ir))

  def test_run_id_for_execution(self):
    run_id = 'run0'
    execution_id = 123
    self.assertEqual(
        subpipeline_utils.run_id_for_execution(run_id, execution_id),
        'run0_123',
    )

  def test_subpipeline_ir_rewrite(self):
    pipeline = test_sync_pipeline.create_pipeline_with_subpipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'run0',
        },
    )
    subpipeline = pipeline.nodes[1].sub_pipeline
    rewritten_pipeline = subpipeline_utils.subpipeline_ir_rewrite(
        subpipeline, 123
    )
    self.assertEqual(
        rewritten_pipeline.runtime_spec.pipeline_run_id.field_value.string_value,
        'sub-pipeline_run0_123',
    )
    self.assertEmpty(rewritten_pipeline.nodes[0].pipeline_node.upstream_nodes)
    self.assertEmpty(
        rewritten_pipeline.nodes[-1].pipeline_node.downstream_nodes
    )
    # New run id should be <old_run_id>_<execution_id>.
    old_run_id = (
        subpipeline.runtime_spec.pipeline_run_id.field_value.string_value
    )
    new_run_id = (
        rewritten_pipeline.runtime_spec.pipeline_run_id.field_value.string_value
    )
    self.assertEqual(new_run_id, old_run_id + '_123')

    # All nodes should associate with the new pipeline run id.
    for node in rewritten_pipeline.nodes:
      pipeline_run_context_names = set()
      for c in node.pipeline_node.contexts.contexts:
        if c.type.name == 'pipeline_run':
          pipeline_run_context_names.add(c.name.field_value.string_value)
      self.assertIn(new_run_id, pipeline_run_context_names)
      self.assertNotIn(old_run_id, pipeline_run_context_names)

    # All inputs except those of PipelineBeginNode's should associate with the
    # new pipeline run id.
    for node in rewritten_pipeline.nodes[1:]:
      for input_spec in node.pipeline_node.inputs.inputs.values():
        for channel in input_spec.channels:
          pipeline_run_context_names = set()
          for context_query in channel.context_queries:
            if context_query.type.name == 'pipeline_run':
              pipeline_run_context_names.add(
                  context_query.name.field_value.string_value
              )
          self.assertIn(new_run_id, pipeline_run_context_names)
          self.assertNotIn(old_run_id, pipeline_run_context_names)


if __name__ == '__main__':
  absltest.main()
