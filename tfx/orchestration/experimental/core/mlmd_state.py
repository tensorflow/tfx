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
import sys
import threading
import typing
from typing import Callable, Iterator, MutableMapping, Optional

from absl import logging
import cachetools
from tfx.orchestration import metadata

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_CACHE_MAX_SIZE = 1024


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


class _ExecutionCache:
  """Read-through / write-through cache for MLMD executions."""

  def __init__(self):
    self._cache: MutableMapping[int, metadata_store_pb2.Execution] = (
        cachetools.LRUCache(maxsize=_CACHE_MAX_SIZE)
    )
    self._lock = threading.Lock()

  def get_execution(
      self, mlmd_handle: metadata.Metadata, execution_id: int
  ) -> metadata_store_pb2.Execution:
    """Gets execution either from cache or, upon cache miss, from MLMD."""
    with self._lock:
      execution = self._cache.get(execution_id)
    if not execution:
      executions = mlmd_handle.store.get_executions_by_id([execution_id])
      if executions:
        execution = executions[0]
        with self._lock:
          self._cache[execution_id] = execution
    if not execution:
      raise ValueError(f'Execution not found for execution id: {execution_id}')
    return execution

  def put_execution(self, mlmd_handle: metadata.Metadata,
                    execution: metadata_store_pb2.Execution) -> None:
    """Writes execution to MLMD and updates cache."""
    mlmd_handle.store.put_executions([execution])
    # The execution is fetched from MLMD again to ensure that the in-memory
    # value of `last_update_time_since_epoch` of the execution is same as the
    # one stored in MLMD.
    [execution] = mlmd_handle.store.get_executions_by_id([execution.id])
    with self._lock:
      self._cache[execution.id] = execution

  def evict(self, execution_id: int) -> None:
    """Evicts execution with the given execution_id from the cache if one exists."""
    self._cache.pop(execution_id, None)

  def clear_cache(self):
    """Clears underlying cache; MLMD is untouched."""
    with self._lock:
      self._cache.clear()


class _LiveOrchestratorContextsCache:
  """Read-through / write-through cache for MLMD live orchestrator contexts.

  Caches the key value pairs of live orchestrator context ID : context name.
  """

  def __init__(self):
    self._cache: MutableMapping[int, str] = cachetools.LRUCache(
        maxsize=_CACHE_MAX_SIZE
    )
    self.synced = False
    self._lock = threading.Lock()

  def get(self, mlmd_handle: metadata.Metadata) -> MutableMapping[int, str]:
    """get live __ORCHESTRATOR__ cache."""
    with self._lock:
      if self.synced:
        return self._cache
    return self.sync(mlmd_handle)

  def sync(self, mlmd_handle: metadata.Metadata) -> MutableMapping[int, str]:
    """sync to MLMD to reserve the active executions and contexts."""
    with self._lock:
      contexts = mlmd_handle.store.get_contexts_by_type(
          _ORCHESTRATOR_RESERVED_ID
      )
      for context in contexts:
        self._cache[context.id] = context.name
      if sys.getsizeof(self._cache) < _CACHE_MAX_SIZE:
        self.synced = True
        return self._cache
      else:
        self.synced = False
        warning_str = (
            'Too many pipelines, cache is not opt in to avoid intensive'
            'memory usage, orchestrator performance might be'
            'slowing down. Please restrict one pipeline under your project if'
            'using legacy API.'
        )
        logging.warning(warning_str)
        raise RuntimeError(warning_str)

  def clear(self) -> None:
    with self._lock:
      self._cache.clear()
      self.synced = False


class _ContextToExecutionsCache:
  """Cache from context to live executions.

  Cache live executions associated with orchestrator contexts.
  """

  def __init__(self):
    self._cache: MutableMapping[int, list[metadata_store_pb2.Execution]] = (
        cachetools.LRUCache(maxsize=_CACHE_MAX_SIZE)
    )
    self._lock = threading.Lock()

  def get_active_executions(
      self, mlmd_handle: metadata.Metadata, context_id: int
  ) -> list[metadata_store_pb2.Execution]:
    """Cache from context to live executions."""
    with self._lock:
      if context_id in self._cache:
        return self._cache[context_id]
    return self.sync_execution(mlmd_handle, context_id)

  def clear(self) -> None:
    with self._lock:
      self._cache.clear()

  def sync_execution(
      self, mlmd_handle: metadata.Metadata, context_id: int
  ) -> list[metadata_store_pb2.Execution]:
    """Cache from context to live executions."""
    with self._lock:
      active_executions = mlmd_handle.store.get_executions_by_context(
          context_id,
          list_options=mlmd.ListOptions(
              filter_query=(
                  'last_known_state = NEW OR last_known_state = RUNNING'
              )
          ),
      )
      self._cache[context_id] = active_executions
      return active_executions


_execution_cache = _ExecutionCache()
_execution_id_locks = _LocksManager()
live_orchestrator_contexts_cache = _LiveOrchestratorContextsCache()
context_to_executions_cache = _ContextToExecutionsCache()


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
  with _execution_id_locks.lock(execution_id):
    execution = _execution_cache.get_execution(mlmd_handle, execution_id)
    execution_copy = copy.deepcopy(execution)
    yield execution_copy
    if execution != execution_copy:
      if execution.id != execution_copy.id:
        raise RuntimeError(
            'Execution id should not be changed within mlmd_execution_atomic_op '
            'context.')
      # Make a copy before writing to cache as the yielded `execution_copy`
      # object may be modified even after exiting the contextmanager.
      _execution_cache.put_execution(mlmd_handle, copy.deepcopy(execution_copy))
      if on_commit is not None:
        pre_commit_execution = copy.deepcopy(execution)
        post_commit_execution = copy.deepcopy(
            _execution_cache.get_execution(mlmd_handle, execution_copy.id))
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
  with _execution_id_locks.lock(execution_id):
    _execution_cache.evict(execution_id)
    yield


def clear_in_memory_state():
  """Clears cached state. Useful in tests."""
  _execution_cache.clear_cache()
