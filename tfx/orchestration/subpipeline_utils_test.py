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
from tfx.orchestration import pipeline as dsl_pipeline
from tfx.orchestration import subpipeline_utils

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


if __name__ == '__main__':
  absltest.main()
