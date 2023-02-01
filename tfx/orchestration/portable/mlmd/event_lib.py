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
"""Portable libraries for event related APIs."""

import collections
import contextlib
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

from tfx import types

from ml_metadata.proto import metadata_store_pb2


_VALID_OUTPUT_EVENT_TYPES = frozenset([
    metadata_store_pb2.Event.OUTPUT, metadata_store_pb2.Event.INTERNAL_OUTPUT,
    metadata_store_pb2.Event.DECLARED_OUTPUT
])
_VALID_INPUT_EVENT_TYPES = frozenset([
    metadata_store_pb2.Event.INPUT, metadata_store_pb2.Event.INTERNAL_INPUT,
    metadata_store_pb2.Event.DECLARED_INPUT
])
_Artifact = TypeVar(
    '_Artifact', metadata_store_pb2.Artifact, types.Artifact)
_ArtifactMultiDict = Dict[str, List[_Artifact]]


def _parse_path(event: metadata_store_pb2.Event) -> List[Tuple[str, int]]:
  """Parses event.path to the list of (key, index)."""
  if len(event.path.steps) % 2 != 0:
    raise ValueError(
        'len(event.path.steps) should be even number but got: '
        f'{event.path.steps}.')
  result = []
  steps = list(event.path.steps)
  i = 0
  while steps:
    s0, s1, *steps = steps
    if s0.WhichOneof('value') != 'key':
      raise ValueError(
          f'steps[{i}] should have "key" but got {event.path.steps}.')
    if s1.WhichOneof('value') != 'index':
      raise ValueError(
          f'steps[{i+1}] should have "index" but got {event.path.steps}.')
    i += 2
    result.append((s0.key, s1.index))
  return result


def reconstruct_artifact_id_multimap(
    events: Sequence[metadata_store_pb2.Event],
) -> Dict[str, Tuple[int, ...]]:
  """Reconstructs events to the {key: [artifact_ids]} multimap."""
  key_to_index_and_id = collections.defaultdict(list)
  for event in events:
    for key, index in _parse_path(event):
      key_to_index_and_id[key].append((index, event.artifact_id))
  result = {}
  for key in key_to_index_and_id:
    indices, artifact_ids = zip(*sorted(key_to_index_and_id[key]))
    if tuple(indices) != tuple(range(len(indices))):
      raise ValueError(
          f'Index values for key "{key}" are not consecutive. '
          'Maybe some events are missing?')
    result[key] = artifact_ids
  return result


def reconstruct_artifact_multimap(
    artifacts: Sequence[_Artifact],
    events: Sequence[metadata_store_pb2.Event]) -> _ArtifactMultiDict:
  """Reconstructs input or output artifact maps from events."""
  execution_ids = {e.execution_id for e in events}
  events_by_artifact_id = {e.artifact_id: e for e in events}
  if len(execution_ids) > 1:
    raise ValueError(
        'All events should be from the same execution but got: '
        f'{execution_ids}.')

  artifacts_by_id = {a.id: a for a in artifacts}
  artifact_id_multimap = reconstruct_artifact_id_multimap(events)
  result = {
      key: [artifacts_by_id[i] for i in artifact_ids]
      for key, artifact_ids in artifact_id_multimap.items()
  }
  for key, artifacts in result.items():
    artifact_types = {a.type_id for a in artifacts}
    if len(artifact_types) != 1:
      raise ValueError(
          f'Artifact type of key "{key}" is heterogeneous: {artifact_types}')
    event_types = {events_by_artifact_id[a.id].type for a in artifacts}
    if len(event_types) != 1:
      raise ValueError(
          f'Event type of key "{key}" is heterogeneous: {event_types}')
  return result


def reconstruct_inputs_and_outputs(
    artifacts: Sequence[_Artifact],
    events: Sequence[metadata_store_pb2.Event],
) -> Tuple[_ArtifactMultiDict, _ArtifactMultiDict]:
  """Reconstructs input and output artifact maps from events."""
  execution_ids = {event.execution_id for event in events}
  if len(execution_ids) > 1:
    raise ValueError(
        'All events should be from the same execution but got: '
        f'{execution_ids}.')

  input_events = [e for e in events if e.type in _VALID_INPUT_EVENT_TYPES]
  output_events = [e for e in events if e.type in _VALID_OUTPUT_EVENT_TYPES]
  return (
      reconstruct_artifact_multimap(artifacts, input_events),
      reconstruct_artifact_multimap(artifacts, output_events),
  )


def is_valid_output_event(event: metadata_store_pb2.Event,
                          expected_output_key: Optional[str] = None) -> bool:
  """Evaluates whether an event is an output event with the right output key.

  This function only returns true if the event type produces a finalized output,
  which excludes events of type PENDING_OUTPUT.

  Args:
    event: The event to evaluate.
    expected_output_key: The expected output key.

  Returns:
    A bool value indicating result
  """
  if event.type not in _VALID_OUTPUT_EVENT_TYPES:
    return False
  if expected_output_key:
    # Ignores errors during event.path parsing which indicates the event is
    # invalid, and returns False.
    with contextlib.suppress(ValueError):
      for key, _ in _parse_path(event):
        if key == expected_output_key:
          return True
    return False
  return True


def is_pending_output_event(event: metadata_store_pb2.Event) -> bool:
  """Returns true if the event represents a pending (not finalized) output."""
  return event.type == metadata_store_pb2.Event.PENDING_OUTPUT


def is_valid_input_event(event: metadata_store_pb2.Event,
                         expected_input_key: Optional[str] = None) -> bool:
  """Evaluates whether an event is an input event with the right input key.

  Args:
    event: The event to evaluate.
    expected_input_key: The expected input key.

  Returns:
    A bool value indicating result
  """
  if event.type not in _VALID_INPUT_EVENT_TYPES:
    return False
  if expected_input_key:
    # Ignores errors during event.path parsing which indicates the event is
    # invalid, and returns False.
    with contextlib.suppress(ValueError):
      for key, _ in _parse_path(event):
        if key == expected_input_key:
          return True
    return False
  return True


def add_event_path(
    event: metadata_store_pb2.Event,
    key: str,
    index: int) -> None:
  """Adds event path to a given MLMD event."""
  # The order matters, we always use the first step to store key and the second
  # step to store index.
  event.path.steps.add().key = key
  event.path.steps.add().index = index


def generate_event(
    event_type: metadata_store_pb2.Event.Type,
    key: str,
    index: int,
    artifact_id: Optional[int] = None,
    execution_id: Optional[int] = None) -> metadata_store_pb2.Event:
  """Generates a MLMD event given type, key and index.

  Args:
    event_type: The type of the event. e.g., INPUT, OUTPUT, etc.
    key: The key of the input or output channel. Usually a key can uniquely
      identify a channel of a TFX node.
    index: The index of the artifact in a channel. For example, a trainer might
      take more than one Example artifacts in one of its input channels. We need
      to distinguish each artifact when creating events.
    artifact_id: Optional artifact id for the event.
    execution_id: Optional execution id for the event.

  Returns:
    A metadata_store_pb2.Event message.
  """
  event = metadata_store_pb2.Event()
  event.type = event_type
  # The order matters, we always use the first step to store key and the second
  # step to store index.
  event.path.steps.add().key = key
  event.path.steps.add().index = index
  if artifact_id:
    event.artifact_id = artifact_id
  if execution_id:
    event.execution_id = execution_id

  return event
