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
"""Tests for mlmd_connection_manager."""

import contextlib

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.experimental.core import test_utils


def _fake_create_reader_mlmd_connection_fn(unused_args):
  return contextlib.nullcontext()


class MlmdConnectionManagerTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()

    self._mock_primary_metadata_handle = mock.create_autospec(
        metadata.Metadata, instance=True)
    self._mock_reader_metadata_handle = mock.create_autospec(
        metadata.Metadata, instance=True)
    primary_metadata_handle_config = mlmd_cm.MLMDConnectionConfig(
        'owner', 'primary', 'prod')
    self._mlmd_connection_manager = mlmd_cm.MLMDConnectionManager(
        self._mock_primary_metadata_handle, primary_metadata_handle_config,
        _fake_create_reader_mlmd_connection_fn)
    self._mlmd_connection_manager._primary_mlmd_handle = (
        self._mock_primary_metadata_handle)
    self._mlmd_connection_manager._reader_mlmd_handles[
        mlmd_cm.MLMDConnectionConfig(
            'owner', 'reader', 'prod')] = self._mock_reader_metadata_handle

  @mock.patch.object(metadata, 'Metadata')
  def test_exit_context(self, mock_metadata):
    original_reader_handles = self._mlmd_connection_manager._reader_mlmd_handles
    self._mlmd_connection_manager.__exit__()
    self.assertNotEmpty(original_reader_handles)

    self._mock_primary_metadata_handle.__exit__.assert_called_once()
    for _, handle in original_reader_handles.items():
      handle.__exit__.assert_called_once()
    self.assertEmpty(self._mlmd_connection_manager._reader_mlmd_handles)

  def test_primary_mlmd_handle(self):
    self.assertEqual(self._mock_primary_metadata_handle,
                     self._mlmd_connection_manager.primary_mlmd_handle)

  def test_get_mlmd_handle(self):
    self.assertEqual(
        self._mock_primary_metadata_handle,
        self._mlmd_connection_manager.get_mlmd_handle('owner', 'primary',
                                                      'prod'))

    self.assertEqual(
        self._mock_reader_metadata_handle,
        self._mlmd_connection_manager.get_mlmd_handle('owner', 'reader',
                                                      'prod'))

    self._mlmd_connection_manager.get_mlmd_handle('new', 'new', 'prod')
    self.assertIn(
        mlmd_cm.MLMDConnectionConfig('new', 'new', 'prod'),
        self._mlmd_connection_manager._reader_mlmd_handles)


if __name__ == '__main__':
  tf.test.main()
