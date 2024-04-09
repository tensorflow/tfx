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


from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants as compiler_constants
from tfx.orchestration import pipeline as dsl_pipeline
from tfx.proto.orchestration import pipeline_pb2

# This pipeline *only* exists so that we can correctly infer the correct node
# types for pipeline begin and end nodes, as the compiler uses a Python Pipeline
# object to generate the names.
# This pipeline *should not* be used otherwise.
_DUMMY_PIPELINE = dsl_pipeline.Pipeline(pipeline_name="UNUSED")


def is_subpipeline(pipeline: pipeline_pb2.Pipeline) -> bool:
  """Returns True if the pipeline is a subpipeline."""
  nodes = pipeline.nodes
  if len(nodes) < 2:
    return False
  maybe_begin_node = nodes[0]
  maybe_end_node = nodes[-1]
  if (
      maybe_begin_node.WhichOneof("node") != "pipeline_node"
      or maybe_begin_node.pipeline_node.node_info.id
      != f"{pipeline.pipeline_info.id}{compiler_constants.PIPELINE_BEGIN_NODE_SUFFIX}"
      or maybe_begin_node.pipeline_node.node_info.type.name
      != compiler_utils.pipeline_begin_node_type_name(_DUMMY_PIPELINE)
  ):
    return False
  if (
      maybe_end_node.WhichOneof("node") != "pipeline_node"
      or maybe_end_node.pipeline_node.node_info.id
      != compiler_utils.pipeline_end_node_id_from_pipeline_id(
          pipeline.pipeline_info.id
      )
      or maybe_end_node.pipeline_node.node_info.type.name
      != compiler_utils.pipeline_end_node_type_name(_DUMMY_PIPELINE)
  ):
    return False
  return True
