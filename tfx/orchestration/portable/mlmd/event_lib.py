# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from ml_metadata.proto import metadata_store_pb2


def validate_output_event(event: metadata_store_pb2.Event,
                          key: Optional[Text] = None) -> bool:
  """Evaluates whether an event is an output event with the right output key.

  Args:
    event: The event to evaluate.
    key: The expected output key.

  Returns:
    A bool value indicating result
  """
  if key:
    return (len(event.path.steps) == 2 and  # Valid event should have 2 steps.
            event.type == metadata_store_pb2.Event.OUTPUT
            and event.path.steps[0].key == key)
  else:
    return event.type == metadata_store_pb2.Event.OUTPUT


def generate_event(
    event_type: metadata_store_pb2.Event.Type,
    key: Text,
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
