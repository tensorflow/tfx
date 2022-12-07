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
"""Tests for tfx.orchestration.experimental.core.mlmd_state."""

from concurrent import futures
import os
import threading

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import test_utils

from ml_metadata.proto import metadata_store_pb2


def _write_test_execution(mlmd_handle):
  execution_type = metadata_store_pb2.ExecutionType(name='foo', version='bar')
  execution_type_id = mlmd_handle.store.put_execution_type(execution_type)
  [execution_id] = mlmd_handle.store.put_executions(
      [metadata_store_pb2.Execution(type_id=execution_type_id)])
  [execution] = mlmd_handle.store.get_executions_by_id([execution_id])
  return execution


class LocksManagerTest(test_utils.TfxTest):

  def test_locking_different_values(self):
    locks = mlmd_state._LocksManager()
    barrier = threading.Barrier(3)

    def _func(value):
      with locks.lock(value):
        barrier.wait()
        self.assertDictEqual({0: 1, 1: 1, 2: 1}, locks._refcounts)
        barrier.wait()

    futs = []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
      for i in range(3):
        futs.append(pool.submit(_func, i))

    # Raises any exceptions raised in the threads.
    for fut in futs:
      fut.result()
    self.assertEmpty(locks._refcounts)

  def test_locking_same_value(self):
    locks = mlmd_state._LocksManager()
    barrier = threading.Barrier(3, timeout=3.0)

    def _func():
      with locks.lock(1):
        barrier.wait()

    futs = []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
      for _ in range(3):
        futs.append(pool.submit(_func))

    with self.assertRaises(threading.BrokenBarrierError):
      for fut in futs:
        fut.result()
    self.assertEmpty(locks._refcounts)


class MlmdStateTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

  def test_mlmd_execution_update(self):
    event_on_commit = threading.Event()
    got_pre_commit_execution = None
    got_post_commit_execution = None

    def on_commit(pre_commit_execution, post_commit_execution):
      nonlocal got_pre_commit_execution
      nonlocal got_post_commit_execution
      got_pre_commit_execution = pre_commit_execution
      got_post_commit_execution = post_commit_execution
      event_on_commit.set()

    with self._mlmd_connection as m:
      expected_execution = _write_test_execution(m)
      # Mutate execution.
      with mlmd_state.mlmd_execution_atomic_op(
          m, expected_execution.id, on_commit=on_commit) as execution:
        self.assertEqual(expected_execution, execution)
        execution.last_known_state = metadata_store_pb2.Execution.CANCELED
        self.assertFalse(event_on_commit.is_set())  # not yet invoked.
      self.assertEqual(expected_execution, got_pre_commit_execution)
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       got_post_commit_execution.last_known_state)

      # Test that we made a deep copy of the executions, so mutating them
      # doesn't mutate the values in the cache.
      got_pre_commit_execution.last_known_state = (
          metadata_store_pb2.Execution.UNKNOWN)
      got_post_commit_execution.last_known_state = (
          metadata_store_pb2.Execution.UNKNOWN)

      # Test that updated execution is committed to MLMD.
      [execution] = m.store.get_executions_by_id([execution.id])
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       execution.last_known_state)
      # Test that in-memory state is also in sync.
      self.assertEqual(execution,
                       mlmd_state._execution_cache._cache[execution.id])
      # Test that on_commit callback was invoked.
      self.assertTrue(event_on_commit.is_set())
      # Sanity checks that the updated execution is yielded in the next call.
      with mlmd_state.mlmd_execution_atomic_op(
          m, expected_execution.id) as execution2:
        self.assertEqual(execution, execution2)

  def test_mlmd_execution_absent(self):
    with self._mlmd_connection as m:
      with self.assertRaisesRegex(ValueError,
                                  'Execution not found for execution id'):
        with mlmd_state.mlmd_execution_atomic_op(m, 1):
          pass

  def test_evict_from_cache(self):
    with self._mlmd_connection as m:
      expected_execution = _write_test_execution(m)
      # Load the execution in cache.
      with mlmd_state.mlmd_execution_atomic_op(m, expected_execution.id):
        pass
      # Test that execution is in cache.
      self.assertEqual(
          expected_execution,
          mlmd_state._execution_cache._cache.get(expected_execution.id))
      # Evict from cache and test.
      with mlmd_state.evict_from_cache(expected_execution.id):
        self.assertIsNone(
            mlmd_state._execution_cache._cache.get(expected_execution.id))
      # Execution should stay evicted.
      self.assertIsNone(
          mlmd_state._execution_cache._cache.get(expected_execution.id))
      # Evicting a non-existent execution should not raise any errors.
      with mlmd_state.evict_from_cache(expected_execution.id):
        pass


if __name__ == '__main__':
  tf.test.main()
