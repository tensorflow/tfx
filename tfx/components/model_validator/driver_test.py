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
                            component_id: Text):
    model_blessing = standard_artifacts.ModelBlessing()
    model_blessing.id = aid
    model_blessing.set_string_custom_property('current_model', 'uri-%d' % aid)
    model_blessing.set_int_custom_property('current_model_id', aid)
    model_blessing.set_string_custom_property('component_id', component_id)
    model_blessing.set_int_custom_property('blessed', is_blessed)
    return model_blessing

  def testFetchLastBlessedModel(self):
    # Mock metadata.
    mock_metadata = tf.compat.v1.test.mock.Mock()
    model_validator_driver = driver.Driver(mock_metadata)
    component_id = 'test_component'

    # No blessed model.
    mock_metadata.get_artifacts_by_type.return_value = []
    self.assertEqual(
        (None, None),
        model_validator_driver._fetch_last_blessed_model(component_id))

    # Mock blessing artifacts.
    artifacts = []
    for aid in [4, 3, 2, 1]:
      model_blessing = self._create_mock_artifact(aid, aid % 2, component_id)
      artifacts.append(model_blessing.artifact)

    # Mock blessing artifact produced by another component.
    model_blessing = self._create_mock_artifact(True, 5, 'different_component')
    artifacts.append(model_blessing.artifact)

    mock_metadata.get_artifacts_by_type.return_value = artifacts
    self.assertEqual(
        ('uri-3', 3),
        model_validator_driver._fetch_last_blessed_model(component_id))


if __name__ == '__main__':
  tf.test.main()
