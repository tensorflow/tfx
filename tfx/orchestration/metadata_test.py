# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.metadata."""

from unittest import mock

from absl import logging
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration import metadata_test_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class SqliteMetadataTest(metadata_test_utils.MetadataTest):
  """Test Metadata operations using a SQLite-backed store."""

  def metadata(self) -> metadata.Metadata:
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    return metadata.Metadata(connection_config=connection_config)

  def testInvalidConnection(self):
    # read only connection to a unknown file
    invalid_config = metadata_store_pb2.ConnectionConfig()
    invalid_config.sqlite.filename_uri = 'unknown_file'
    invalid_config.sqlite.connection_mode = 1
    # test the runtime error contains detailed information
    with self.assertRaisesRegex(RuntimeError, 'unable to open database file'):
      with metadata.Metadata(connection_config=invalid_config) as m:
        m.store()

  def testPatchedStore(self):
    m = self.enter_context(self.metadata())
    mock_warning = self.enter_context(
        mock.patch.object(logging, 'log_every_n_seconds', autospec=True)
    )
    filter_query = 'contexts_0.type = "x" AND contexts_0.name = "y"'

    with self.subTest('filter_query with is_asc = false is warned.'):
      m.store.get_artifacts(
          list_options=mlmd.ListOptions(filter_query=filter_query, is_asc=False)
      )
      mock_warning.assert_called_once()

    mock_warning.reset_mock()

    with self.subTest('Note is_asc = false is implicit.'):
      m.store.get_artifacts(
          list_options=mlmd.ListOptions(filter_query=filter_query)
      )
      mock_warning.assert_called_once()

    mock_warning.reset_mock()

    with self.subTest('Also for get_executions()'):
      m.store.get_executions(
          list_options=mlmd.ListOptions(filter_query=filter_query)
      )
      mock_warning.assert_called_once()

    mock_warning.reset_mock()

    with self.subTest('Valid list_options'):
      m.store.get_artifacts(list_options=mlmd.ListOptions(is_asc=False))
      m.store.get_artifacts(
          list_options=mlmd.ListOptions(filter_query=filter_query, is_asc=True)
      )
      mock_warning.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
