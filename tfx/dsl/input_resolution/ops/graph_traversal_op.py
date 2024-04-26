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
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import filter_query_builder as q
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2


# Valid artifact states for GraphTraversal.
_VALID_ARTIFACT_STATES = [metadata_store_pb2.Artifact.State.LIVE]


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
  # defined custom artifact type names. Can not be empty.
  artifact_type_names = resolver_op.Property(type=Sequence[str])

  # The producer component node IDs to match by, e.g.
  # "example-gen.import-example". Optional.
  node_ids = resolver_op.Property(type=Sequence[str], default=[])

  # The Event output key(s) to match by. Optional.
  output_keys = resolver_op.Property(type=Sequence[str], default=[])

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
    artifact_states_filter_query = (
        ops_utils.get_valid_artifact_states_filter_query(_VALID_ARTIFACT_STATES)
    )
    filter_query = (
        f'type IN {q.to_sql_string(self.artifact_type_names)} AND '
        f'{artifact_states_filter_query}'
    )

    if self.node_ids:
      for context in self.context.store.get_contexts_by_artifact(
          root_artifact.id
      ):
        if context.type == constants.PIPELINE_CONTEXT_TYPE_NAME:
          pipeline_name = context.name
          break
      else:
        raise ValueError('No pipeline context was found.')

      # We match against the node Context's name, which has the format
      # <pipeline-name>.<node-id>
      node_context_names = [
          compiler_utils.node_context_name(pipeline_name, ni)
          for ni in self.node_ids
      ]
      query = (
          f'contexts_a.name IN {q.to_sql_string(node_context_names)} '
          'AND contexts_a.type = '
          f'{q.to_sql_string(constants.NODE_CONTEXT_TYPE_NAME)}'
      )
      filter_query += ' AND ' + query

    mlmd_resolver = metadata_resolver.MetadataResolver(self.context.store)
    mlmd_resolver_fn = (
        mlmd_resolver.get_upstream_artifacts_by_artifact_ids
        if self.traverse_upstream
        else mlmd_resolver.get_downstream_artifacts_by_artifact_ids
    )
    related_artifact_and_type = mlmd_resolver_fn(
        [root_artifact.id],
        max_num_hops=ops_utils.GRAPH_TRAVERSAL_OP_MAX_NUM_HOPS,
        filter_query=filter_query,
    )
    artifact_type_by_id = {}
    related_artifacts = {}
    for artifact_id, artifacts_and_types in related_artifact_and_type.items():
      related_artifacts[artifact_id], artifact_types = zip(*artifacts_and_types)
      artifact_type_by_id.update({t.id: t for t in artifact_types})

    # Build the result dict to return. We include the root_artifact to help with
    # input synchronization in ASYNC mode. Note, Python dicts preserve key
    # insertion order, so when a user gets the unrolled dict values, they will
    # first get the root artifact, followed by ancestor/descendant artifacts in
    # the same order as self.artifact_type_names.
    result = {ops_utils.ROOT_ARTIFACT_KEY: [root_artifact]}
    for artifact_type in self.artifact_type_names:
      result[artifact_type] = []

    if not related_artifacts.get(root_artifact.id):
      logging.info(
          'No neighboring artifacts were found for root artifact %s and '
          'artifact_type_names %s node_ids %s output_keys %s.',
          root_artifact,
          self.artifact_type_names,
          self.node_ids,
          self.output_keys,
      )
      return result
    related_artifacts = related_artifacts[root_artifact.id]

    # Get the ArtifactType for the related artifacts.
    artifact_type_by_artifact_id = {}
    for artifact in related_artifacts:
      artifact_type_by_artifact_id[artifact.id] = artifact_type_by_id[
          artifact.type_id
      ]

    # Build the result dictionary, with a separate key for each ArtifactType.
    artifact_ids = set(a.id for a in related_artifacts)
    events = self.context.store.get_events_by_artifact_ids(artifact_ids)
    events_by_artifact_id = {
        e.artifact_id: e for e in events if event_lib.is_valid_output_event(e)
    }
    for artifact in related_artifacts:
      # MLMD does not support filter querying by event.paths, so we manually
      # check for matching output key.
      # TODO(b/302394845): Once MLMD supports filtering by the last event, then
      # add this check inside the filter_query or event_filter.
      if self.output_keys and not any(
          event_lib.contains_key(events_by_artifact_id[artifact.id], k)
          for k in self.output_keys
      ):
        continue

      deserialized_artifact = artifact_utils.deserialize_artifact(
          artifact_type_by_artifact_id[artifact.id], artifact
      )
      result[artifact.type].append(deserialized_artifact)

    return ops_utils.sort_artifact_dict(result)
