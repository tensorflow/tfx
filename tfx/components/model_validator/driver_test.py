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
"""Tests for tfx.components.model_validator.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Text
from tfx.components.model_validator import driver
from tfx.types import standard_artifacts


class DriverTest(tf.test.TestCase):

  def _create_mock_artifact(self, aid: int, is_blessed: bool,
                            pipeline_name: Text, component_id: Text):
    model_blessing = standard_artifacts.ModelBlessing()
    model_blessing.id = aid
    model_blessing.pipeline_name = pipeline_name
    model_blessing.set_string_custom_property('current_model', 'uri-%d' % aid)
    model_blessing.set_int_custom_property('current_model_id', aid)
    model_blessing.set_string_custom_property('component_id', component_id)
    model_blessing.set_int_custom_property('blessed', is_blessed)
    return model_blessing.artifact

  def testFetchLastBlessedModel(self):
    # Mock metadata.
    mock_metadata = tf.compat.v1.test.mock.Mock()
    model_validator_driver = driver.Driver(mock_metadata)
    component_id = 'test_component'
    pipeline_name = 'test_pipeline'

    # No blessed model.
    mock_metadata.get_artifacts_by_type.return_value = []
    self.assertEqual((None, None),
                     model_validator_driver._fetch_last_blessed_model(
                         pipeline_name, component_id))

    # Mock blessing artifacts.
    artifacts = [
        self._create_mock_artifact(aid, aid % 2, pipeline_name, component_id)
        for aid in [4, 3, 2, 1]
    ]

    # Mock blessing artifact produced by another component and another pipeline.
    artifacts.extend([
        self._create_mock_artifact(True, 5, pipeline_name,
                                   'different_component'),
        self._create_mock_artifact(True, 6, 'different_pipeline', component_id)
    ])

    mock_metadata.get_artifacts_by_type.return_value = artifacts
    self.assertEqual(('uri-3', 3),
                     model_validator_driver._fetch_last_blessed_model(
                         pipeline_name, component_id))


if __name__ == '__main__':
  tf.test.main()
