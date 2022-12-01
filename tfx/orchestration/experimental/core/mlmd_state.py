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

import collections
import contextlib
import copy
import threading
import typing
from typing import Callable, Iterator, MutableMapping, Optional

from tfx.orchestration import metadata

from ml_metadata.proto import metadata_store_pb2


class _LocksManager:
  """Class for managing value based locking."""

  def __init__(self):
    self._main_lock = threading.Lock()
    self._locks: MutableMapping[typing.Hashable, threading.Lock] = {}
    self._refcounts = collections.defaultdict(int)

  @contextlib.contextmanager
  def lock(self, value: typing.Hashable) -> Iterator[None]:
    """Context manager for input value based locking.

    Only one thread can enter the context for a given value.

    Args:
      value: Value of any hashable type.

    Yields:
      Nothing.
    """
    with self._main_lock:
      lock = self._locks.setdefault(value, threading.Lock())
      self._refcounts[value] += 1
    try:
      with lock:
        yield
    finally:
      with self._main_lock:
        self._refcounts[value] -= 1
        if self._refcounts[value] <= 0:
          del self._refcounts[value]
          del self._locks[value]


_execution_id_locks = _LocksManager()


def _get_execution(mlmd_handle: metadata.Metadata,
                   execution_id: int) -> metadata_store_pb2.Execution:
  executions = mlmd_handle.store.get_executions_by_id([execution_id])
  if not executions:
    raise ValueError(f'Execution not found for execution id: {execution_id}')
  return executions[0]


def _put_execution(mlmd_handle: metadata.Metadata,
                   execution: metadata_store_pb2.Execution) -> None:
  mlmd_handle.store.put_executions([execution])


@contextlib.contextmanager
def mlmd_execution_atomic_op(
    mlmd_handle: metadata.Metadata,
    execution_id: int,
    on_commit: Optional[
        Callable[[metadata_store_pb2.Execution, metadata_store_pb2.Execution],
                 None]] = None,
) -> Iterator[metadata_store_pb2.Execution]:
  """Context manager for accessing or mutating an execution atomically.

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
  with _execution_id_locks.lock(execution_id):
    execution = _get_execution(mlmd_handle, execution_id)
    execution_copy = copy.deepcopy(execution)
    yield execution_copy
    if execution != execution_copy:
      if execution.id != execution_copy.id:
        raise RuntimeError(
            'Execution id should not be changed within mlmd_execution_atomic_op '
            'context.')
      # Make a copy before writing to cache as the yielded `execution_copy`
      # object may be modified even after exiting the contextmanager.
      _put_execution(mlmd_handle, copy.deepcopy(execution_copy))
      if on_commit is not None:
        pre_commit_execution = copy.deepcopy(execution)
        post_commit_execution = copy.deepcopy(
            _get_execution(mlmd_handle, execution_copy.id))
        on_commit(pre_commit_execution, post_commit_execution)
