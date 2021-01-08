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
"""Tests for tfx.components.infra_validator.serving_bins."""

import tensorflow as tf

from tfx.components.infra_validator import serving_bins
from tfx.proto import infra_validator_pb2


class ServingBinsTest(tf.test.TestCase):

  def testParseServingBinaries_TensorFlowServing(self):
    spec = infra_validator_pb2.ServingSpec(
        tensorflow_serving=infra_validator_pb2.TensorFlowServing(
            image_name='gcr.io/my_project/my_serving_image',
            tags=['t1', 't2'],
            digests=['sha256:d1', 'sha256:d2']))
    result = serving_bins.parse_serving_binaries(spec)

    self.assertLen(result, 4)
    for item in result:
      self.assertIsInstance(item, serving_bins.TensorFlowServing)
    self.assertCountEqual([item.image for item in result], [
        'gcr.io/my_project/my_serving_image:t1',
        'gcr.io/my_project/my_serving_image:t2',
        'gcr.io/my_project/my_serving_image@sha256:d1',
        'gcr.io/my_project/my_serving_image@sha256:d2',
    ])

  def testParseServingBinaries_TensorFlowServing_DefaultImageName(self):
    spec = infra_validator_pb2.ServingSpec(
        tensorflow_serving=infra_validator_pb2.TensorFlowServing(
            tags=['latest']))
    result = serving_bins.parse_serving_binaries(spec)

    self.assertLen(result, 1)
    self.assertIsInstance(result[0], serving_bins.TensorFlowServing)
    self.assertEqual(result[0].image, 'tensorflow/serving:latest')


if __name__ == '__main__':
  tf.test.main()
