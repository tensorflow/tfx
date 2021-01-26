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
"""Tests for tfx.dsl.io.plugins.tensorflow_gfile."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from tfx.dsl.io.filesystem import NotFoundError
from tfx.dsl.io.plugins.tensorflow_gfile import TensorflowFilesystem


class TensorflowGfileTest(tf.test.TestCase):

  def testNotFound(self):
    temp_dir = tempfile.mkdtemp()

    # Because the GFile implementation delays I/O until necessary, we cannot
    # catch `NotFoundError` here, so this does not raise an error.
    TensorflowFilesystem.open(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.copy(
          os.path.join(temp_dir, 'foo'), os.path.join(temp_dir, 'bar'))

    # No exception raised.
    self.assertEqual(
        TensorflowFilesystem.glob(os.path.join(temp_dir, 'foo/*')), [])

    # No exception raised.
    self.assertEqual(
        TensorflowFilesystem.isdir(os.path.join(temp_dir, 'foo/bar')), False)

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.listdir(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.mkdir(os.path.join(temp_dir, 'foo/bar'))

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.remove(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.rmtree(os.path.join(temp_dir, 'foo'))

    with self.assertRaises(NotFoundError):
      TensorflowFilesystem.stat(os.path.join(temp_dir, 'foo'))

    # No exception raised.
    self.assertEqual(
        list(TensorflowFilesystem.walk(os.path.join(temp_dir, 'foo'))), [])


if __name__ == '__main__':
  tf.test.main()
