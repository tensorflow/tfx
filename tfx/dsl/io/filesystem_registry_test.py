# Lint as: python2, python3
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
"""Tests for tfx.dsl.io.filesystem_registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfx.dsl.io import filesystem
from tfx.dsl.io import filesystem_registry
from tfx.dsl.io.plugins import local
from tfx.dsl.io.plugins import tensorflow_gfile


class FakeFilesystemA(filesystem.Filesystem):
  pass


class FakeFilesystemB(filesystem.Filesystem):
  pass


class FakeFilesystemC(filesystem.Filesystem):
  pass


class FilesystemRegistryTest(tf.test.TestCase):

  def testRegistry(self):
    registry = filesystem_registry.FilesystemRegistry()

    # Test exceptions properly raised when schemes not registered.
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_scheme('')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('/tmp/my/file')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_scheme('gs://')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('gs://bucket/tmp/my/file')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_scheme('s3://')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('s3://bucket/tmp/my/file')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('unknown://bucket/tmp/my/file')

    # Test after local filesystem is registered.
    registry.register(local.LocalFilesystem, 20)
    self.assertIs(local.LocalFilesystem, registry.get_filesystem_for_scheme(''))
    self.assertIs(local.LocalFilesystem,
                  registry.get_filesystem_for_path('/tmp/my/file'))
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_scheme('gs://')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('gs://bucket/tmp/my/file')
    with self.assertRaisesRegexp(Exception, 'is not available for use'):
      registry.get_filesystem_for_path('unknown://bucket/tmp/my/file')

    # Test after Tensorflow filesystems are registered with higher priority.
    registry.register(
        tensorflow_gfile.TensorflowFilesystem, 10, use_as_fallback=True)
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_scheme(''))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path('/tmp/my/file'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_scheme('gs://'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path('gs://bucket/tmp/my/file'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_scheme('s3://'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path('s3://bucket/tmp/my/file'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_scheme('hdfs://'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path('hdfs://bucket/tmp/my/file'))
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path(
                      'unknown://bucket/tmp/my/file'))

    # Test that fallback filesystems are correctly prioritized. `FilesystemA`
    # should not be used as the fallback since it has lower priority than
    # `TensorflowFilesystem`.
    registry.register(FakeFilesystemA, 15, use_as_fallback=True)
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path(
                      'unknown://bucket/tmp/my/file'))
    # A filesystem registered without `use_as_fallback=True` should not be used
    # as a fallback.
    registry.register(FakeFilesystemB, 5)
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_path(
                      'unknown://bucket/tmp/my/file'))
    # `FakeFilesystemC` is a fallback with higher priority than
    # `TensorflowFilesystem` and so should be used as the fallback.
    registry.register(FakeFilesystemC, 5, use_as_fallback=True)
    self.assertIs(FakeFilesystemC,
                  registry.get_filesystem_for_path(
                      'unknown://bucket/tmp/my/file'))

    # Test usage of byte paths.
    self.assertIs(tensorflow_gfile.TensorflowFilesystem,
                  registry.get_filesystem_for_scheme(b'hdfs://'))
    self.assertIs(
        tensorflow_gfile.TensorflowFilesystem,
        registry.get_filesystem_for_path(b'hdfs://bucket/tmp/my/file'))
    with self.assertRaisesRegexp(ValueError, 'Invalid path type'):
      registry.get_filesystem_for_path(123)


if __name__ == '__main__':
  tf.test.main()
