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
"""Tests for tfx.utils.path_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized

import tensorflow as tf
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import path_utils

from ml_metadata.proto import metadata_store_pb2


class PathUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testEstimatorModelPath(self, is_old_artifact):
    # Create folders based on Estimator based Trainer output model directory,
    # after Executor performs cleaning.
    output_uri = os.path.join(self.get_temp_dir(), 'model_dir')
    eval_model_path = path_utils.eval_model_dir(output_uri, is_old_artifact)
    eval_model = os.path.join(eval_model_path, 'saved_model.pb')
    io_utils.write_string_file(eval_model, 'testing')
    serving_model_path = path_utils.serving_model_dir(output_uri,
                                                      is_old_artifact)
    serving_model = os.path.join(eval_model_path, 'saved_model.pb')
    io_utils.write_string_file(serving_model, 'testing')

    # Test retrieving model folder.
    self.assertEqual(eval_model_path,
                     path_utils.eval_model_path(output_uri, is_old_artifact))
    self.assertEqual(serving_model_path,
                     path_utils.serving_model_path(output_uri, is_old_artifact))

  @parameterized.parameters(True, False)
  def testKerasModelPath(self, is_old_artifact):
    # Create folders based on Keras based Trainer output model directory.
    output_uri = os.path.join(self.get_temp_dir(), 'model_dir')
    serving_model_path = path_utils.serving_model_dir(output_uri,
                                                      is_old_artifact)
    serving_model = os.path.join(serving_model_path, 'saved_model.pb')
    io_utils.write_string_file(serving_model, 'testing')

    # Test retrieving model folder.
    self.assertEqual(serving_model_path,
                     path_utils.eval_model_path(output_uri, is_old_artifact))
    self.assertEqual(serving_model_path,
                     path_utils.serving_model_path(output_uri, is_old_artifact))

  def testIsOldModelArtifact(self):
    artifact = standard_artifacts.Examples()
    with self.assertRaisesRegex(AssertionError, 'Wrong artifact type'):
      path_utils.is_old_model_artifact(artifact)

    artifact = standard_artifacts.Model()
    self.assertFalse(path_utils.is_old_model_artifact(artifact))
    artifact.mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    self.assertTrue(path_utils.is_old_model_artifact(artifact))


if __name__ == '__main__':
  tf.test.main()
