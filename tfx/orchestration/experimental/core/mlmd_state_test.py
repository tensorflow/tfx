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

import os
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
    with self._mlmd_connection as m:
      expected_execution = _write_test_execution(m)
      # Mutate execution.
      with mlmd_state.mlmd_execution_atomic_op(
          m, expected_execution.id) as execution:
        self.assertEqual(expected_execution, execution)
        execution.last_known_state = metadata_store_pb2.Execution.CANCELED
      # Test that updated execution is committed to MLMD.
      [execution] = m.store.get_executions_by_id([execution.id])
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       execution.last_known_state)
      # Test that in-memory state is also in sync.
      with mlmd_state.mlmd_execution_atomic_op(
          m, expected_execution.id) as execution:
        self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                         execution.last_known_state)

  def test_mlmd_execution_absent(self):
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(m, 1) as execution:
        self.assertIsNone(execution)


if __name__ == '__main__':
  tf.test.main()
