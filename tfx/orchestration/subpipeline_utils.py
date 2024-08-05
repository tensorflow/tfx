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
"""Generic utilities for orchestrating subpipelines."""
import copy
from typing import Callable
from tfx.proto.orchestration import pipeline_pb2


def is_subpipeline(pipeline: pipeline_pb2.Pipeline) -> bool:
  """Returns True if the pipeline is a subpipeline."""
  return bool(pipeline.pipeline_info.parent_ids)


def run_id_for_execution(run_id: str, execution_id: int) -> str:
  """Returns the pipeline run id for a given subpipeline execution."""
  return f'{run_id}_{execution_id}'


def subpipeline_ir_rewrite(
    original_ir: pipeline_pb2.Pipeline, execution_id: int
) -> pipeline_pb2.Pipeline:
  """Rewrites the subpipeline IR so that it can be run independently.

  Args:
    original_ir: Original subpipeline IR that is produced by compiler.
    execution_id: The ID of Subpipeline task scheduler Execution. It is used to
      generated a new pipeline run id.

  Returns:
    An updated subpipeline IR that can be run independently.
  """
  pipeline = copy.deepcopy(original_ir)
  pipeline.nodes[0].pipeline_node.ClearField('upstream_nodes')
  pipeline.nodes[-1].pipeline_node.ClearField('downstream_nodes')
  _update_pipeline_run_id(pipeline, execution_id)
  return pipeline


def _visit_pipeline_nodes_recursively(
    p: pipeline_pb2.Pipeline,
    visitor: Callable[[pipeline_pb2.PipelineNode], None],
):
  """Helper function to visit every node inside a possibly nested pipeline."""
  for pipeline_or_node in p.nodes:
    if pipeline_or_node.WhichOneof('node') == 'pipeline_node':
      visitor(pipeline_or_node.pipeline_node)
    else:
      _visit_pipeline_nodes_recursively(pipeline_or_node.sub_pipeline, visitor)


def _update_pipeline_run_id(pipeline: pipeline_pb2.Pipeline, execution_id: int):
  """Rewrites pipeline run id in a given pipeline IR."""
  old_pipeline_run_id = (
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value
  )
  new_pipeline_run_id = run_id_for_execution(old_pipeline_run_id, execution_id)

  def _node_updater(node: pipeline_pb2.PipelineNode):
    for context_spec in node.contexts.contexts:
      if (
          context_spec.type.name == 'pipeline_run'
          and context_spec.name.field_value.string_value == old_pipeline_run_id
      ):
        context_spec.name.field_value.string_value = new_pipeline_run_id
    for input_spec in node.inputs.inputs.values():
      for channel in input_spec.channels:
        for context_query in channel.context_queries:
          if (
              context_query.type.name == 'pipeline_run'
              and context_query.name.field_value.string_value
              == old_pipeline_run_id
          ):
            context_query.name.field_value.string_value = new_pipeline_run_id

  _visit_pipeline_nodes_recursively(pipeline, _node_updater)
  pipeline.runtime_spec.pipeline_run_id.field_value.string_value = (
      new_pipeline_run_id
  )
