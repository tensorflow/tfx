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
"""Sync pipeline for testing."""

from tfx.dsl.compiler import compiler
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts


@component
def my_example_gen(examples: OutputArtifact[standard_artifacts.Examples]):
  del examples


@component
def my_statistics_gen(
    examples: InputArtifact[standard_artifacts.Examples],
    statistics: OutputArtifact[standard_artifacts.ExampleStatistics]):
  del examples, statistics


@component
def my_schema_gen(
    statistics: InputArtifact[standard_artifacts.ExampleStatistics],
    schema: OutputArtifact[standard_artifacts.Schema]):
  del statistics, schema


@component
def my_example_validator(
    statistics: InputArtifact[standard_artifacts.ExampleStatistics],
    schema: InputArtifact[standard_artifacts.Schema],
    anomalies: OutputArtifact[standard_artifacts.ExampleAnomalies]):
  del statistics, schema, anomalies


@component
def my_transform(
    examples: InputArtifact[standard_artifacts.Examples],
    schema: InputArtifact[standard_artifacts.Schema],
    transform_graph: OutputArtifact[standard_artifacts.TransformGraph]):
  del examples, schema, transform_graph


@component
def my_trainer(
    examples: InputArtifact[standard_artifacts.Examples],
    schema: InputArtifact[standard_artifacts.Schema],
    transform_graph: InputArtifact[standard_artifacts.TransformGraph],
    model: OutputArtifact[standard_artifacts.Model]):
  del examples, schema, transform_graph, model


@component
def chore_a():
  pass


@component
def chore_b():
  pass


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Builds a test pipeline."""
  # pylint: disable=no-value-for-parameter
  example_gen = my_example_gen()
  stats_gen = my_statistics_gen(examples=example_gen.outputs['examples'])
  schema_gen = my_schema_gen(statistics=stats_gen.outputs['statistics'])
  example_validator = my_example_validator(
      statistics=stats_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  transform = my_transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'])
  trainer = my_trainer(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'])

  # Nodes with no input or output specs for testing task only dependencies.
  chore_a_component = chore_a()
  chore_a_component.add_upstream_node(trainer)
  chore_b_component = chore_b()
  chore_b_component.add_upstream_node(chore_a_component)
  # pylint: enable=no-value-for-parameter

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          example_gen,
          stats_gen,
          schema_gen,
          example_validator,
          transform,
          trainer,
          chore_a_component,
          chore_b_component,
      ],
      enable_cache=True)
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
