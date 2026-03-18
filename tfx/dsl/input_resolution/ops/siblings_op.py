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

    # Check that the root artifact is in a valid state.
    if root_artifact.mlmd_artifact.state not in _VALID_ARTIFACT_STATES:
      return {}

    # Find the execution that produced the root artifact (OUTPUT event).
    root_events = self.context.store.get_events_by_artifact_ids(
        [root_artifact.id]
    )
    producing_execution_ids = [
        e.execution_id
        for e in root_events
        if event_lib.is_valid_output_event(e)
    ]
    if not producing_execution_ids:
      return {ops_utils.ROOT_ARTIFACT_KEY: [root_artifact]}

    # Get all events for those executions and keep only output events.
    all_execution_events = self.context.store.get_events_by_execution_ids(
        producing_execution_ids
    )
    output_events = [
        e for e in all_execution_events if event_lib.is_valid_output_event(e)
    ]

    # Fetch the artifacts and filter to LIVE state only.
    sibling_artifact_ids = list({e.artifact_id for e in output_events})
    sibling_artifacts = self.context.store.get_artifacts_by_id(
        sibling_artifact_ids
    )
    live_artifact_ids = {
        a.id
        for a in sibling_artifacts
        if a.state in _VALID_ARTIFACT_STATES
    }
    output_events = [
        e for e in output_events if e.artifact_id in live_artifact_ids
    ]
    artifact_type_ids = list({a.type_id for a in sibling_artifacts})
    artifact_types = self.context.store.get_artifact_types_by_id(
        artifact_type_ids
    )
    artifact_by_id = {a.id: a for a in sibling_artifacts}
    artifact_type_by_id = {t.id: t for t in artifact_types}

    if not self.output_keys:
      # Find all output keys, excluding the key(s) associated with root artifact.
      output_keys = set()
      for event in output_events:
        if event.artifact_id != root_artifact.id:
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

    for event in output_events:
      for output_key in self.output_keys:
        if event_lib.contains_key(event, output_key):
          artifact = artifact_by_id[event.artifact_id]
          artifact_type = artifact_type_by_id[artifact.type_id]
          deserialized_artifact = artifact_utils.deserialize_artifact(
              artifact_type, artifact
          )
          result[output_key].append(deserialized_artifact)

    return ops_utils.sort_artifact_dict(result)
