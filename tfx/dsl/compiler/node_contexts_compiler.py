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
"""Compiles NodeContexts."""

from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2


def compile_node_contexts(
    pipeline_ctx: compiler_context.PipelineContext,
    node_id: str,
) -> pipeline_pb2.NodeContexts:
  """Compiles the node contexts of a pipeline node."""

  if pipeline_ctx.pipeline_info is None:
    return pipeline_pb2.NodeContexts()
  if maybe_contexts := pipeline_ctx.node_context_protos_cache.get(node_id):
    return maybe_contexts

  node_contexts = pipeline_pb2.NodeContexts()
  # Context for the pipeline, across pipeline runs.
  pipeline_context_pb = node_contexts.contexts.add()
  pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
  pipeline_context_pb.name.field_value.string_value = (
      pipeline_ctx.pipeline_info.pipeline_context_name
  )

  # Context for the current pipeline run.
  if pipeline_ctx.is_sync_mode:
    pipeline_run_context_pb = node_contexts.contexts.add()
    pipeline_run_context_pb.type.name = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
    # TODO(kennethyang): Miragte pipeline run id to structural_runtime_parameter
    # To keep existing IR textprotos used in tests unchanged, we only use
    # structural_runtime_parameter for subpipelines. After the subpipeline being
    # implemented, we will need to migrate normal pipelines to
    # structural_runtime_parameter as well for consistency. Similar for below.
    if pipeline_ctx.is_subpipeline:
      compiler_utils.set_structural_runtime_parameter_pb(
          pipeline_run_context_pb.name.structural_runtime_parameter,
          [
              f"{pipeline_ctx.pipeline_info.pipeline_context_name}_",
              (constants.PIPELINE_RUN_ID_PARAMETER_NAME, str),
          ],
      )
    else:
      compiler_utils.set_runtime_parameter_pb(
          pipeline_run_context_pb.name.runtime_parameter,
          constants.PIPELINE_RUN_ID_PARAMETER_NAME,
          str,
      )
  # If this is a subpipline then set the subpipeline as node context.
  if pipeline_ctx.is_subpipeline:
    subpipeline_context_pb = node_contexts.contexts.add()
    subpipeline_context_pb.type.name = constants.NODE_CONTEXT_TYPE_NAME
    subpipeline_context_pb.name.field_value.string_value = (
        compiler_utils.node_context_name(
            pipeline_ctx.parent.pipeline_info.pipeline_context_name,
            pipeline_ctx.pipeline_info.pipeline_context_name,
        )
    )
  # Contexts inherited from the parent pipelines.
  for i, parent_pipeline in enumerate(pipeline_ctx.parent_pipelines[::-1]):
    parent_pipeline_context_pb = node_contexts.contexts.add()
    parent_pipeline_context_pb.type.name = constants.PIPELINE_CONTEXT_TYPE_NAME
    parent_pipeline_context_pb.name.field_value.string_value = (
        parent_pipeline.pipeline_info.pipeline_context_name
    )

    if parent_pipeline.execution_mode == pipeline.ExecutionMode.SYNC:
      pipeline_run_context_pb = node_contexts.contexts.add()
      pipeline_run_context_pb.type.name = (
          constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
      )

      # TODO(kennethyang): Miragte pipeline run id to structural runtime
      # parameter for the similar reason mentioned above.
      # Use structural runtime parameter to represent pipeline_run_id except
      # for the root level pipeline, for backward compatibility.
      if i == len(pipeline_ctx.parent_pipelines) - 1:
        compiler_utils.set_runtime_parameter_pb(
            pipeline_run_context_pb.name.runtime_parameter,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME,
            str,
        )
      else:
        compiler_utils.set_structural_runtime_parameter_pb(
            pipeline_run_context_pb.name.structural_runtime_parameter,
            [
                f"{parent_pipeline.pipeline_info.pipeline_context_name}_",
                (constants.PIPELINE_RUN_ID_PARAMETER_NAME, str),
            ],
        )

  # Context for the node, across pipeline runs.
  node_context_pb = node_contexts.contexts.add()
  node_context_pb.type.name = constants.NODE_CONTEXT_TYPE_NAME
  node_context_pb.name.field_value.string_value = (
      compiler_utils.node_context_name(
          pipeline_ctx.pipeline_info.pipeline_context_name, node_id
      )
  )
  pipeline_ctx.node_context_protos_cache[node_id] = node_contexts
  return node_contexts
