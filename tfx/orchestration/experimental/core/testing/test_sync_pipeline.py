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
from tfx.dsl.control_flow.for_each import ForEach
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.experimental.node_execution_options import utils
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
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
def _example_validator(
    statistics: InputArtifact[standard_artifacts.ExampleStatistics],
    schema: InputArtifact[standard_artifacts.Schema],
    anomalies: OutputArtifact[standard_artifacts.ExampleAnomalies]):
  del (
      statistics,
      schema,
      anomalies,
  )


@component
def _transform(
    examples: InputArtifact[standard_artifacts.Examples],
    schema: InputArtifact[standard_artifacts.Schema],
    transform_graph: OutputArtifact[standard_artifacts.TransformGraph]):
  del examples, schema, transform_graph


@component
def _trainer(examples: InputArtifact[standard_artifacts.Examples],
             schema: InputArtifact[standard_artifacts.Schema],
             transform_graph: InputArtifact[standard_artifacts.TransformGraph],
             model: OutputArtifact[standard_artifacts.Model]):
  del examples, schema, transform_graph, model


@component
def _evaluator(model: InputArtifact[standard_artifacts.Model],
               evals: OutputArtifact[standard_artifacts.ModelEvaluation]):
  del model, evals


@component
def _chore():
  pass


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Builds a test pipeline.

    ┌───────────┐
    │example_gen│
    └┬─┬─┬──────┘
     │ │┌▽──────────────┐
     │ ││stats_gen       │
     │ │└┬─────────────┬─┘
     │ │┌▽───────────┐│
     │ ││schema_gen   ││
     │ │└┬───────┬─┬──┘│
     │┌▽─▽────┐│┌▽──▽─────────────┐
     ││transform │││example_validator │
     │└┬────────┘│└───────────────────┘
    ┌▽─▽───────▽┐
    │trainer       │
    └┬─────────┬───┘
    ┌▽─────┐┌▽─────────┐
    │chore_a││evaluator  │
    └┬──────┘└───────────┘
    ┌▽──────┐
    │chore_b │
    └────────┘

  Returns:
    A pipeline proto for the above DAG
  """
  # pylint: disable=no-value-for-parameter
  example_gen = _example_gen().with_id('my_example_gen')
  stats_gen = _statistics_gen(
      examples=example_gen.outputs['examples']).with_id('my_statistics_gen')
  schema_gen = _schema_gen(
      statistics=stats_gen.outputs['statistics']).with_id('my_schema_gen')
  example_validator = _example_validator(
      statistics=stats_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']).with_id('my_example_validator')
  transform = _transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema']).with_id('my_transform')
  trainer = _trainer(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph']).with_id(
          'my_trainer')

  # Nodes with no input or output specs for testing task only dependencies.
  chore_a = _chore().with_id('chore_a')
  chore_a.add_upstream_node(trainer)
  chore_b = _chore().with_id('chore_b')
  chore_b.add_upstream_node(chore_a)

  with conditional.Cond(
      trainer.outputs['model'].future()[0].custom_property('evaluate') == 1):
    evaluator = _evaluator(
        model=trainer.outputs['model']).with_id('my_evaluator')
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
          evaluator,
          chore_a,
          chore_b,
      ],
      enable_cache=True)
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)


def create_pipeline_with_foreach() -> pipeline_pb2.Pipeline:
  """Builds a test pipeline with ForEach."""
  # pylint: disable=no-value-for-parameter
  example_gen = _example_gen().with_id('my_example_gen')
  with ForEach(example_gen.outputs['examples']) as examples:
    stats_gen = _statistics_gen(examples=examples).with_id(
        'my_statistics_gen_in_foreach'
    )

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          example_gen,
          stats_gen,
      ],
      enable_cache=True,
  )
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)


def create_chore_pipeline() -> pipeline_pb2.Pipeline:
  """Creates a pipeline full of chores.

    ┌─────────────┐┌──────────────┐
    │example_gen_1││example_gen_2 │
    └┬────────────┘└┬───────┬─────┘
    ┌▽──────┐┌──────▽───┐┌▽──────┐
    │chore_a ││chore_d    ││chore_e │
    └┬───────┘└┬─────────┬┘└┬───────┘
    ┌▽──────┐┌▽──────┐┌▽──▽───┐
    │chore_b ││chore_f││chore_g   │
    └┬───────┘└┬───────┘└─────────┘
    ┌▽────────▽┐
    │chore_c     │
    └────────────┘
  Returns:
    A pipeline for the above DAG
  """

  # pylint: disable=no-value-for-parameter
  example_gen_1 = _example_gen().with_id('my_example_gen_1')
  example_gen_2 = _example_gen().with_id('my_example_gen_2')

  chore_a = _chore().with_id('chore_a')
  chore_a.add_upstream_node(example_gen_1)
  chore_b = _chore().with_id('chore_b')
  chore_b.add_upstream_node(chore_a)
  chore_c = _chore().with_id('chore_c')
  chore_c.add_upstream_node(chore_b)

  chore_d = _chore().with_id('chore_d')
  chore_d.add_upstream_node(example_gen_2)
  chore_e = _chore().with_id('chore_e')
  chore_e.add_upstream_node(example_gen_2)
  chore_f = _chore().with_id('chore_f')
  chore_f.add_upstream_node(chore_d)
  chore_g = _chore().with_id('chore_g')
  chore_g.add_upstream_node(chore_d)
  chore_g.add_upstream_node(chore_e)
  chore_f.add_downstream_node(chore_c)

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          example_gen_1,
          example_gen_2,
          chore_a,
          chore_b,
          chore_d,
          chore_e,
          chore_f,
          chore_g,
          chore_c,
      ],
      enable_cache=True,
  )
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)


def create_resource_lifetime_pipeline() -> pipeline_pb2.Pipeline:
  """Creates a pipeline full of chores to be used for testing resource lifetime.

    ┌───────┐
    │start_a│
    └┬──────┘
    ┌▽──────┐
    │start_b │
    └┬───────┘
    ┌▽─────┐
    │worker │
    └┬──────┘
    ┌▽────┐
    │end_b │
    └┬─────┘
    ┌▽────┐
    │end_a │
    └──────┘

  Returns:
    A pipeline for the above DAG
  """

  # pylint: disable=no-value-for-parameter
  start_a = _example_gen().with_id('start_a')
  start_b = _chore().with_id('start_b')
  start_b.add_upstream_node(start_a)
  worker = _chore().with_id('worker')
  worker.add_upstream_node(start_b)
  end_b = _chore().with_id('end_b')
  end_b.add_upstream_nodes([worker, start_b])
  end_b.node_execution_options = utils.NodeExecutionOptions(
      trigger_strategy=pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS,
      lifetime_start=start_b.id,
  )
  end_a = _chore().with_id('end_a')
  end_a.add_upstream_nodes([start_a, start_b, worker, end_b])
  end_a.node_execution_options = utils.NodeExecutionOptions(
      trigger_strategy=pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS,
      lifetime_start=start_a.id,
  )

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          start_a,
          start_b,
          worker,
          end_b,
          end_a,
      ],
      enable_cache=True,
  )
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)


def create_pipeline_with_subpipeline() -> pipeline_pb2.Pipeline:
  """Creates a pipeline with a subpipeline."""
  # pylint: disable=no-value-for-parameter
  example_gen = _example_gen().with_id('my_example_gen')

  p_in = pipeline_lib.PipelineInputs(
      {'examples': example_gen.outputs['examples']}
  )
  stats_gen = _statistics_gen(examples=p_in['examples']).with_id(
      'my_statistics_gen'
  )
  schema_gen = _schema_gen(statistics=stats_gen.outputs['statistics']).with_id(
      'my_schema_gen'
  )
  p_out = {'schema': schema_gen.outputs['schema']}

  componsable_pipeline = pipeline_lib.Pipeline(
      pipeline_name='sub-pipeline',
      pipeline_root='/path/to/root/sub',
      components=[stats_gen, schema_gen],
      enable_cache=True,
      inputs=p_in,
      outputs=p_out,
  )

  transform = _transform(
      examples=example_gen.outputs['examples'],
      schema=componsable_pipeline.outputs['schema'],
  ).with_id('my_transform')

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          example_gen,
          componsable_pipeline,
          transform,
      ],
      enable_cache=True,
  )
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
