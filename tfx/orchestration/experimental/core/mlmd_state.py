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
import threading
from typing import Iterator, Optional

import cachetools
from tfx.orchestration import metadata

from ml_metadata.proto import metadata_store_pb2

_execution_by_id = cachetools.LRUCache(maxsize=1024)
_execution_by_id_lock = threading.Lock()


@contextlib.contextmanager
def mlmd_execution_atomic_op(
    mlmd_handle: metadata.Metadata,
    execution_id: int) -> Iterator[Optional[metadata_store_pb2.Execution]]:
  """Yields an execution proto for the given execution id if one exists.

  The idea of using this context manager is to ensure that the in-memory state
  of an MLMD execution is centrally managed so that it stays in sync with the
  execution in MLMD even when multiple threads in the process may be mutating.

  If execution for given execution id exists in MLMD, it is locked before being
  yielded so that no other thread in the process can make conflicting updates if
  the yielded execution is mutated within the context. Mutated executions are
  also automatically committed to MLMD when exiting the context.

  NOTE: currently a global lock is held within the context, so callers should
  only make lightweight operations such as mutating execution fields within the
  context. TODO(goutham): Remove this restriction.

  Args:
    mlmd_handle: A handle to MLMD db.
    execution_id: Id of the execution to yield.

  Yields:
    If execution with given id exists in MLMD, the execution is yielded under
    an exclusive lock context. `None` if no execution exists.

  Raises:
    RuntimeError: If execution id is changed within the context.
  """
  # TODO(goutham): Finer granularity (execution level) locking is desirable.
  with _execution_by_id_lock:
    execution = _execution_by_id.get(execution_id)
    if not execution:
      executions = mlmd_handle.store.get_executions_by_id([execution_id])
      if executions:
        execution = executions[0]
        _execution_by_id[execution_id] = execution
    execution_copy = copy.deepcopy(execution)
    yield execution_copy
    if execution and execution != execution_copy:
      if execution.id != execution_copy.id:
        raise RuntimeError(
            'Execution id should not be changed within mlmd_execution_atomic_op '
            'context.')
      mlmd_handle.store.put_executions([execution_copy])
      execution.CopyFrom(execution_copy)


def clear_in_memory_state():
  """Clears cached state. Useful in tests."""
  with _execution_by_id_lock:
    _execution_by_id.clear()
