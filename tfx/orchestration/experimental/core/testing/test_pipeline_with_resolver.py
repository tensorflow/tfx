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
"""Pipeline with a resolver node for testing."""

from tfx import types
from tfx.dsl.compiler import compiler
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts


@component
def _trainer(model: OutputArtifact[standard_artifacts.Model]):
  del model


@component
def _consumer(resolved_model: InputArtifact[standard_artifacts.Model]):
  del resolved_model


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Creates a pipeline with a resolver node for testing."""
  trainer = _trainer().with_id('my_trainer')  # pylint: disable=no-value-for-parameter
  rnode = resolver.Resolver(
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
      config={
          'desired_num_of_artifacts': 1
      },
      resolved_model=types.Channel(
          type=standard_artifacts.Model,
          producer_component_id=trainer.id,
          output_key='model')).with_id('my_resolver')
  rnode.add_upstream_node(trainer)
  consumer = _consumer(
      resolved_model=rnode.outputs['resolved_model']).with_id('my_consumer')
  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          trainer,
          rnode,
          consumer,
      ],
      execution_mode=pipeline_lib.ExecutionMode.SYNC)
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
