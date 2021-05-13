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
"""Tests for tfx.utils.model_paths.tf_serving_flavor."""

import tensorflow as tf

from tfx.utils.model_paths import tf_serving_flavor as tfs_flavor


class TFServingFlavorTest(tf.test.TestCase):

  def testRoundTrip(self):
    self.assertEqual(
        tfs_flavor.parse_model_path(
            tfs_flavor.make_model_path('/foo/bar', 'my-model', 123)),
        ('/foo/bar', 'my-model', 123))

    self.assertEqual(
        tfs_flavor.make_model_path(
            *tfs_flavor.parse_model_path('/foo/bar/my-model/123')),
        '/foo/bar/my-model/123')

  def testMakeModelPath(self):
    self.assertEqual(
        tfs_flavor.make_model_path(
            model_base_path='/foo/bar',
            model_name='my-model',
            version=123),
        '/foo/bar/my-model/123')

    self.assertEqual(
        tfs_flavor.make_model_path(
            model_base_path='s3://bucket-name/foo/bar',
            model_name='my-model',
            version=123),
        's3://bucket-name/foo/bar/my-model/123')

    self.assertEqual(
        tfs_flavor.make_model_path(
            model_base_path='gs://bucket-name/foo/bar',
            model_name='my-model',
            version=123),
        'gs://bucket-name/foo/bar/my-model/123')

  def testParseModelPath(self):
    self.assertEqual(
        tfs_flavor.parse_model_path('/foo/bar/my-model/123',),
        ('/foo/bar', 'my-model', 123))

    self.assertEqual(
        tfs_flavor.parse_model_path('s3://bucket-name/foo/bar/my-model/123'),
        ('s3://bucket-name/foo/bar', 'my-model', 123))

    self.assertEqual(
        tfs_flavor.parse_model_path('gs://bucket-name/foo/bar/my-model/123'),
        ('gs://bucket-name/foo/bar', 'my-model', 123))

  def testParseModelPath_Fail(self):
    with self.assertRaises(ValueError):
      tfs_flavor.parse_model_path('too-short')

    with self.assertRaises(ValueError):
      tfs_flavor.parse_model_path('/foo/bar/my-model/not-an-int-version')

    with self.assertRaises(ValueError):
      tfs_flavor.parse_model_path('/foo/bar/other-model/123',
                                  expected_model_name='my-model')

if __name__ == '__main__':
  tf.test.main()
