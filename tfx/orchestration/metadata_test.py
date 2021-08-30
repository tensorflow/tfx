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

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration import metadata_test_utils

from ml_metadata.proto import metadata_store_pb2


class SqliteMetadataTest(metadata_test_utils.MetadataTest):
  """Test Metadata operations using a SQLite-backed store."""

  def metadata(self):
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


if __name__ == '__main__':
  tf.test.main()
