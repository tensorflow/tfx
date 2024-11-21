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

import tensorflow as tf
import pytest

from tfx.experimental.templates.taxi.models.keras_model import model


@pytest.mark.xfail(run=False, reason="_build_keras_model is not compatible with Keras3.")
class ModelTest(tf.test.TestCase):

  def testBuildKerasModel(self):
    built_model = model._build_keras_model(
        hidden_units=[1, 1], learning_rate=0.1)  # pylint: disable=protected-access
    self.assertEqual(len(built_model.layers), 10)

    built_model = model._build_keras_model(hidden_units=[1], learning_rate=0.1)  # pylint: disable=protected-access
    self.assertEqual(len(built_model.layers), 9)
