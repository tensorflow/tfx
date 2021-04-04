# Copyright 2020 Google LLC. All Rights Reserved.
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
"""This module defines the handler for resolver node."""

from typing import Any, Dict

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import system_node_handler
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import pipeline_pb2


class ResolverNodeHandler(system_node_handler.SystemNodeHandler):
  """The handler for the system Resolver node."""

  def _extract_proto_map(
      self,
      # The actual type of proto message of map<str, pipeline_pb2.Value>.
      proto_map: Any) -> Dict[str, Any]:
    extract_mlmd_value = lambda v: getattr(v, v.WhichOneof('value'))
    return {k: extract_mlmd_value(v.field_value) for k, v in proto_map.items()}

  def run(
      self, mlmd_connection: metadata.Metadata,
      pipeline_node: pipeline_pb2.PipelineNode,
      pipeline_info: pipeline_pb2.PipelineInfo,
      pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec
  ) -> data_types.ExecutionInfo:
    """Runs Resolver specific logic.

    Args:
      mlmd_connection: ML metadata connection.
      pipeline_node: The specification of the node that this launcher lauches.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.

    Returns:
      The execution of the run.
    """
    logging.info('Running as an resolver node.')
    with mlmd_connection as m:
      # 1.Prepares all contexts.
      contexts = context_lib.prepare_contexts(
          metadata_handler=m, node_contexts=pipeline_node.contexts)

      # 2. Resolves inputs an execution properties.
      exec_properties = inputs_utils.resolve_parameters(
          node_parameters=pipeline_node.parameters)
      input_artifacts = inputs_utils.resolve_input_artifacts(
          metadata_handler=m, node_inputs=pipeline_node.inputs)

      # 3. Registers execution in metadata.
      execution = execution_publish_utils.register_execution(
          metadata_handler=m,
          execution_type=pipeline_node.node_info.type,
          contexts=contexts,
          exec_properties=exec_properties)

      # 4. Publish the execution as a cached execution with
      # resolved input artifact as the output artifacts.
      execution_publish_utils.publish_internal_execution(
          metadata_handler=m,
          contexts=contexts,
          execution_id=execution.id,
          output_artifacts=input_artifacts)

      return data_types.ExecutionInfo(
          execution_id=execution.id,
          input_dict=input_artifacts,
          output_dict=input_artifacts,
          exec_properties=exec_properties,
          pipeline_node=pipeline_node,
          pipeline_info=pipeline_info)
