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
"""Module for Siblings operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2


# Valid artifact states for Siblings.
_VALID_ARTIFACT_STATES = [metadata_store_pb2.Artifact.State.LIVE]


class Siblings(
    resolver_op.ResolverOp,
    canonical_name='tfx.Siblings',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """Siblings operator."""

  # The output key(s) that the sibling Artifact should be associated with. If
  # empty, defaults to all output keys, except for keys that produced the
  # root artifact.
  output_keys = resolver_op.Property(type=Sequence[str], default=[])

  def apply(self, input_list: Sequence[types.Artifact]):
    """Returns output artifacts produced in the same execution."""
    if not input_list:
      return {}

    # TODO(b/299985043): Support batch traversal.
    if len(input_list) != 1:
      raise ValueError('Siblings ResolverOp does not support batch queries.')
    root_artifact = input_list[0]

    artifact_states_filter_query = (
        ops_utils.get_valid_artifact_states_filter_query(_VALID_ARTIFACT_STATES)
    )
    lineage_graph = self.context.store.get_lineage_subgraph(
        query_options=metadata_store_pb2.LineageSubgraphQueryOptions(
            starting_artifacts=(
                metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
                    filter_query=(
                        f'id = {root_artifact.id} AND '
                        f'{artifact_states_filter_query}'
                    ),
                )
            ),
            ending_executions=(
                metadata_store_pb2.LineageSubgraphQueryOptions.EndingNodes(
                    # NOTE: This query assumes that an artifact will never be
                    # the input of an execution and the output of another (or
                    # the same) execution. This is always the case in Tflex,
                    # because the orchestrator produces new output artifacts
                    # for every execution.
                    filter_query=(
                        f'events_0.artifact_id = {root_artifact.id} AND'
                        ' events_0.type = INPUT'
                    )
                )
            ),
            max_num_hops=2,
            direction=metadata_store_pb2.LineageSubgraphQueryOptions.BIDIRECTIONAL,
        ),
        field_mask_paths=[
            'artifacts',
            'artifact_types',
            'events',
        ],
    )

    if not self.output_keys:
      # Find all output keys.
      output_keys = set()
      for event in lineage_graph.events:
        if (
            event_lib.is_valid_output_event(event)
            # We exclude output keys associated with the root artifact. This
            # ensures the root artifact will only be associated with the key
            # "root_artifact" in the returned dictionary.
            and event.artifact_id != root_artifact.id
        ):
          keys_and_indexes = event_lib._parse_path(event)  # pylint: disable=protected-access
          for key, _ in keys_and_indexes:
            output_keys.add(key)
      self.output_keys = list(output_keys)

    # Build the result dict to return. We include the root_artifact to help with
    # input synchronization in ASYNC mode. Note, Python dicts preserve key
    # insertion order, so when a user gets the unrolled dict values, they will
    # first get the root artifact, followed by sibling artifacts in the same
    # order as self.output_keys.
    result = {ops_utils.ROOT_ARTIFACT_KEY: [root_artifact]}
    for output_key in self.output_keys:
      result[output_key] = []

    # Get output Artifact IDs associated with each output key.
    artifact_by_id = {a.id: a for a in lineage_graph.artifacts}
    artifact_type_by_id = {at.id: at for at in lineage_graph.artifact_types}
    for event in lineage_graph.events:
      if not event_lib.is_valid_output_event(event):
        continue
      for output_key in self.output_keys:
        if event_lib.contains_key(event, output_key):
          artifact = artifact_by_id[event.artifact_id]
          artifact_type = artifact_type_by_id[artifact.type_id]
          deserialized_artifact = artifact_utils.deserialize_artifact(
              artifact_type, artifact
          )
          result[output_key].append(deserialized_artifact)

    return ops_utils.sort_artifact_dict(result)
