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
"""Test pipeline with a subpipeline inside."""

from tfx.dsl.compiler import compiler
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel
from tfx.types import standard_artifacts


@component
def _example_gen(examples: OutputArtifact[standard_artifacts.Examples]):
  del examples


@component
def _statistics_gen(
    examples: InputArtifact[standard_artifacts.Examples],
    statistics: OutputArtifact[standard_artifacts.ExampleStatistics]):
  del examples, statistics


@component
def _schema_gen(statistics: InputArtifact[standard_artifacts.ExampleStatistics],
                schema: OutputArtifact[standard_artifacts.Schema]):
  del statistics, schema


@component
def _transform(
    examples: InputArtifact[standard_artifacts.Examples],
    schema: InputArtifact[standard_artifacts.Schema],
    transform_graph: OutputArtifact[standard_artifacts.TransformGraph]):
  del examples, schema, transform_graph


def create_sub_pipeline(examples: channel.Channel) -> pipeline_lib.Pipeline:
  """A test sub pipeline."""
  # pylint: disable=no-value-for-parameter
  p_in = pipeline_lib.PipelineInputs(inputs={'examples': examples})
  stats_gen = _statistics_gen(
      examples=p_in.inputs['examples']).with_id('my_statistics_gen')
  schema_gen = _schema_gen(
      statistics=stats_gen.outputs['statistics']).with_id('my_schema_gen')

  return pipeline_lib.Pipeline(
      pipeline_name='my_sub_pipeline',
      components=[stats_gen, schema_gen],
      inputs=p_in,
      outputs={'schema': schema_gen.outputs['schema']})


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Builds a test pipeline."""
  # pylint: disable=no-value-for-parameter
  example_gen = _example_gen().with_id('my_example_gen')
  sub_pipeline = create_sub_pipeline(example_gen.outputs['examples'])
  transform = _transform(
      examples=example_gen.outputs['examples'],
      schema=sub_pipeline.outputs['schema']).with_id('my_transform')

  my_pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[example_gen, sub_pipeline, transform])
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(my_pipeline)
