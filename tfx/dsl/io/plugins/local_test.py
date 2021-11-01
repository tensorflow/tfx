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
"""Tests for tfx.dsl.io.plugins.local."""

import os
import tempfile

import tensorflow as tf
from tfx.dsl.io.filesystem import NotFoundError
from tfx.dsl.io.plugins.local import LocalFilesystem


class LocalTest(tf.test.TestCase):

  def testNotFound(self):
    temp_dir = tempfile.mkdtemp()

    with self.assertRaises(NotFoundError):
      LocalFilesystem.open(os.path.join(temp_dir, 'foo')).read()

    with self.assertRaises(NotFoundError):
      LocalFilesystem.copy(
          os.path.join(temp_dir, 'foo'), os.path.join(temp_dir, 'bar'))

    # No exception raised.
    self.assertEqual(LocalFilesystem.glob(os.path.join(temp_dir, 'foo/*')), [])

    # No exception raised.
    self.assertEqual(
        LocalFilesystem.isdir(os.path.join(temp_dir, 'foo/bar')), False)

    with self.assertRaises(NotFoundError):
      LocalFilesystem.listdir(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      LocalFilesystem.mkdir(os.path.join(temp_dir, 'foo/bar'))

    with self.assertRaises(NotFoundError):
      LocalFilesystem.remove(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      LocalFilesystem.rmtree(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      LocalFilesystem.stat(os.path.join(temp_dir, 'foo'))

    # No exception raised.
    self.assertEqual(
        list(LocalFilesystem.walk(os.path.join(temp_dir, 'foo'))), [])


if __name__ == '__main__':
  tf.test.main()
