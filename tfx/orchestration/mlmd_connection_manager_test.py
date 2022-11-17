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

from unittest import mock

import tensorflow as tf
from tfx.orchestration import mlmd_connection_manager as mlmd_cm

import ml_metadata as mlmd


class MlmdConnectionManagerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_store_factory = mock.patch.object(
        mlmd, 'MetadataStore', autospec=True).start()
    self.addCleanup(mock.patch.stopall)

  def test_primary_handle(self):
    with mlmd_cm.MLMDConnectionManager.fake() as cm:
      connection_config = cm.primary_mlmd_handle.connection_config
      self.assertEqual(connection_config.WhichOneof('config'), 'fake_database')
      self.mock_store_factory.assert_called_with(connection_config)

    self.mock_store_factory.reset_mock()

    with mlmd_cm.MLMDConnectionManager.sqlite('foo') as cm:
      connection_config = cm.primary_mlmd_handle.connection_config
      self.assertEqual(connection_config.WhichOneof('config'), 'sqlite')
      self.mock_store_factory.assert_called_with(connection_config)

  def test_unusable_without_enter(self):
    cm = mlmd_cm.MLMDConnectionManager.fake()
    with self.assertRaisesRegex(RuntimeError, 'not entered yet'):
      cm.primary_mlmd_handle  # pylint: disable=pointless-statement

  def test_enter_synced_with_handle(self):
    cm = mlmd_cm.MLMDConnectionManager.fake()
    with cm:
      handle = cm.primary_mlmd_handle
      self.assertIsNotNone(handle.store)
    with self.assertRaisesRegex(
        RuntimeError, 'Metadata object is not in enter state'):
      handle.store  # pylint: disable=pointless-statement

  def test_multiple_enterable(self):
    cm = mlmd_cm.MLMDConnectionManager.fake()
    with cm:
      with cm:
        m1 = cm.primary_mlmd_handle
      m2 = cm.primary_mlmd_handle
      self.assertIs(m1, m2)
    with self.assertRaises(RuntimeError):
      cm.primary_mlmd_handle  # pylint: disable=pointless-statement


if __name__ == '__main__':
  tf.test.main()
