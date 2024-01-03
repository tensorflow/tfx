# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Utilities for working with MLMD state."""

import contextlib
import copy
from typing import Callable, Iterator, Optional

from tfx.orchestration import metadata

from google.protobuf.internal import containers
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


@contextlib.contextmanager
def mlmd_execution_atomic_op(
    mlmd_handle: metadata.Metadata,
    execution_id: int,
    on_commit: Optional[
        Callable[[metadata_store_pb2.Execution, metadata_store_pb2.Execution],
                 None]] = None,
) -> Iterator[metadata_store_pb2.Execution]:
  """Context manager for accessing or mutating an execution atomically.

  The idea of using this context manager is to ensure that the in-memory state
  of an MLMD execution is centrally managed so that it stays in sync with the
  execution in MLMD even when multiple threads in the process may be mutating.

  If execution for given execution id exists in MLMD, it is locked before being
  yielded so that no other thread in the process can make conflicting updates if
  the yielded execution is mutated within the context. Mutated executions are
  also automatically committed to MLMD when exiting the context.

  Args:
    mlmd_handle: A handle to MLMD db.
    execution_id: Id of the execution to yield.
    on_commit: An optional callback function which is invoked post successful
      MLMD execution commit operation. This won't be invoked if execution is not
      mutated within the context and hence MLMD commit is not needed. The
      callback is passed copies of the pre-commit and post-commit executions.

  Yields:
    If execution with given id exists in MLMD, the execution is yielded under
    an exclusive lock context.

  Raises:
    RuntimeError: If execution id is changed within the context.
    ValueError: If execution having given execution id is not found in MLMD.
  """
  with mlmd.execution_id_locks.lock(execution_id):
    execution = mlmd.execution_cache.get_execution(
        mlmd_handle.store, execution_id
    )
    execution_copy = copy.deepcopy(execution)
    yield execution_copy
    if execution != execution_copy:
      if execution.id != execution_copy.id:
        raise RuntimeError(
            'Execution id should not be changed within mlmd_execution_atomic_op'
            ' context.'
        )

      # Orchestrator code will only update top-level fields and properties/
      # custom properties with diffs.

      # Motivation: to allow non-orchestrator code (specifically, pipeline tags
      # and labels) to modify execution custom properties while the orchestrator
      # is running. Delta changes are only applied for masked properties /
      # custom properties. execution.last_known_state will always be updated.

      # It enables orchestrator and non-orchestrator codes to run concurrently
      # as long as there are no overlaps in the modified fields.

      # Make a copy before writing to cache as the yielded `execution_copy`
      # object may be modified even after exiting the contextmanager.
      mlmd.execution_cache.put_execution(
          mlmd_handle.store,
          copy.deepcopy(execution_copy),
          get_field_mask_paths(execution, execution_copy),
      )
      if on_commit is not None:
        pre_commit_execution = copy.deepcopy(execution)
        post_commit_execution = copy.deepcopy(
            mlmd.execution_cache.get_execution(
                mlmd_handle.store, execution_copy.id
            )
        )
        on_commit(pre_commit_execution, post_commit_execution)


@contextlib.contextmanager
def evict_from_cache(execution_id: int) -> Iterator[None]:
  """Context manager for mutating an MLMD execution using cache unaware functions.

  It is preferable to use `mlmd_execution_atomic_op` for mutating MLMD
  executions but sometimes it may be necessary to use third party functions
  which are not cache aware. Such functions should be invoked within this
  context for proper locking and cache eviction to prevent stale entries.

  Args:
    execution_id: Id of the execution to be evicted from cache.

  Yields:
    Nothing
  """
  with mlmd.execution_id_locks.lock(execution_id):
    mlmd.execution_cache.evict([execution_id])
    yield


def clear_in_memory_state():
  """Clears cached state. Useful in tests."""
  mlmd.execution_cache.clear_cache()


def get_field_mask_paths(
    execution: metadata_store_pb2.Execution,
    execution_copy: metadata_store_pb2.Execution,
) -> list[str]:
  """Get Execution field mask paths for mutations.

  Args:
    execution: original in-memory state of an MLMD execution.
    execution_copy: in-memory state of an MLMD execution after mutations.

  Returns:
    All top-level field paths, and property / custom property fields with diffs.
    Only field paths in the mask will be updated during MLMD commits.
  """
  field_mask_paths = []

  # Get all non-property field paths.
  for field in metadata_store_pb2.Execution.DESCRIPTOR.fields:
    # Skip property fields.
    if field.name not in ['properties', 'custom_properties']:
      field_mask_paths.append(field.name)

  # Get property names with diffs.  Note that Python supports == operator for
  # proto messages.
  def _get_property_names_with_diff(
      properties: containers.MessageMap[str, metadata_store_pb2.Value],
      copy_properties: containers.MessageMap[str, metadata_store_pb2.Value],
  ) -> list[str]:
    property_names_with_diff = []
    for name in set(properties.keys()).union(set(copy_properties.keys())):
      if (
          name in properties
          and name in copy_properties
          and properties[name] == copy_properties[name]
      ):
        continue
      property_names_with_diff.append(name)
    return property_names_with_diff

  property_names_with_diff = _get_property_names_with_diff(
      execution.properties, execution_copy.properties
  )
  custom_property_names_with_diff = _get_property_names_with_diff(
      execution.custom_properties, execution_copy.custom_properties
  )

  field_mask_paths.extend(
      [f'properties.{name}' for name in property_names_with_diff]
  )
  field_mask_paths.extend(
      [f'custom_properties.{name}' for name in custom_property_names_with_diff]
  )
  return field_mask_paths
