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
"""Tests for bulk_inferrer component of Cloud AI platform."""

import tensorflow as tf

from tfx.extensions.google_cloud_ai_platform.bulk_inferrer import component
from tfx.proto import bulk_inferrer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self._model = channel_utils.as_channel([standard_artifacts.Model()])
    self._model_blessing = channel_utils.as_channel(
        [standard_artifacts.ModelBlessing()])

  def testConstructInferenceResult(self):
    bulk_inferrer = component.CloudAIBulkInferrerComponent(
        examples=self._examples,
        model=self._model,
        model_blessing=self._model_blessing)
    self.assertEqual('InferenceResult',
                     bulk_inferrer.outputs['inference_result'].type_name)
    self.assertNotIn('output_examples', bulk_inferrer.outputs.keys())

  def testConstructOutputExample(self):
    bulk_inferrer = component.CloudAIBulkInferrerComponent(
        examples=self._examples,
        model=self._model,
        model_blessing=self._model_blessing,
        output_example_spec=bulk_inferrer_pb2.OutputExampleSpec())
    self.assertEqual('Examples',
                     bulk_inferrer.outputs['output_examples'].type_name)
    self.assertNotIn('inference_result', bulk_inferrer.outputs.keys())


if __name__ == '__main__':
  tf.test.main()
