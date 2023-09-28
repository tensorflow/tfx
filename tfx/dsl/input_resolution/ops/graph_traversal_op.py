# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Module for GraphTraversal operator."""

from typing import Sequence

from absl import logging
from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration.portable.mlmd import filter_query_builder as q
from tfx.types import artifact_utils

from ml_metadata.tools.mlmd_resolver import metadata_resolver


class GraphTraversal(
    resolver_op.ResolverOp,
    canonical_name='tfx.GraphTraversal',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """GraphTraversal operator."""

  # Whether to search for artifacts upstream or downstream. Required.
  traverse_upstream = resolver_op.Property(type=bool)

  # The artifact type names to search for, e.g. "ModelBlessing",
  # "TransformGraph". Should match Tflex standard artifact type names or user
  # defined custom artifact type names.
  artifact_type_names = resolver_op.Property(type=Sequence[str])

  # TODO(b/299985043): Add a node_ids field that only returns artifacts produced
  # by components with a specific node ID.

  def apply(self, input_list: Sequence[types.Artifact]):
    """Returns a dict with the upstream (or downstream) and root artifacts.

    Args:
      input_list: A list with exactly one Artifact to use as the root.

    Returns:
      A dictionary with the upstream (or downstream) artifacts, and the root
      artifact.

      For example, consider: Examples -> Model -> ModelBlessing.

      Calling GraphTraversal with [ModelBlessing], traverse_upstream=True, and
      artifact_type_names=["Examples"] will return:

      {
          "root_artifact": [ModelBlessing],
          "examples": [Examples].
      }

      Note the key "root_artifact" is set with the original artifact inside
      input_list. This makes input synchronzation easier in an ASYNC pipeline.
    """
    if not input_list:
      return {}

    if not self.artifact_type_names:
      raise ValueError(
          'At least one artifact type name must be provided, but '
          'artifact_type_names was empty.'
      )

    # TODO(b/299985043): Support batch traversal.
    if len(input_list) != 1:
      raise ValueError(
          'GraphTraversal ResolverOp does not support batch traversal.'
      )
    root_artifact = input_list[0]

    # Query MLMD to get the upstream (or downstream) artifacts.
    filter_query = f'type IN {q.to_sql_string(self.artifact_type_names)}'
    mlmd_resolver = metadata_resolver.MetadataResolver(self.context.store)
    mlmd_resolver_fn = (
        mlmd_resolver.get_upstream_artifacts_by_artifact_ids
        if self.traverse_upstream
        else mlmd_resolver.get_downstream_artifacts_by_artifact_ids
    )
    related_artifacts = mlmd_resolver_fn(
        [root_artifact.id],
        max_num_hops=ops_utils.GRAPH_TRAVERSAL_OP_MAX_NUM_HOPS,
        filter_query=filter_query,
    )

    result = {ops_utils.ROOT_ARTIFACT_KEY: [root_artifact]}
    for artifact_type in self.artifact_type_names:
      result[artifact_type] = []
    if not related_artifacts.get(root_artifact.id):
      logging.info(
          'No neighboring artifacts were found for root artifact %s and '
          'artifact_type_names %s.',
          root_artifact,
          self.artifact_type_names,
      )
      return result
    related_artifacts = related_artifacts[root_artifact.id]

    # Get the ArtifactType for the related artifacts.
    type_ids = set(a.type_id for a in related_artifacts)
    artifact_types = self.context.store.get_artifact_types_by_id(type_ids)
    artifact_type_by_artifact_id = {}
    for artifact in related_artifacts:
      for artifact_type in artifact_types:
        if artifact.type_id == artifact_type.id:
          artifact_type_by_artifact_id[artifact.id] = artifact_type
          break

    # Build the result dictionary, with a separate key for each ArtifactType.
    for artifact in related_artifacts:
      if artifact.type in self.artifact_type_names:
        deserialized_artifact = artifact_utils.deserialize_artifact(
            artifact_type_by_artifact_id[artifact.id], artifact
        )
        result[artifact.type].append(deserialized_artifact)

    return ops_utils.sort_artifact_dict(result)
