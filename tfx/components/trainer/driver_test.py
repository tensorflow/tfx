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
"""Tests for tfx.components.trainer.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.trainer import driver
from tfx.types import standard_artifacts


class DriverTest(tf.test.TestCase):

  def testFetchWarmStartingModel(self):
    mock_metadata = tf.compat.v1.test.mock.Mock()
    artifacts = []
    for aid in [3, 2, 1]:
      model = standard_artifacts.Model()
      model.id = aid
      model.uri = 'uri-%d' % aid
      artifacts.append(model.mlmd_artifact)
    mock_metadata.get_artifacts_by_type.return_value = artifacts
    trainer_driver = driver.Driver(mock_metadata)
    result = trainer_driver._fetch_latest_model()
    self.assertEqual('uri-3', result)


if __name__ == '__main__':
  tf.test.main()
