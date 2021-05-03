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
"""Tests for tfx.components.bulk_inferrer.component."""

import tensorflow as tf

from tfx.components.bulk_inferrer import component
from tfx.proto import bulk_inferrer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()
    self._examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self._model = channel_utils.as_channel([standard_artifacts.Model()])
    self._model_blessing = channel_utils.as_channel(
        [standard_artifacts.ModelBlessing()])

  def testConstructInferenceResult(self):
    bulk_inferrer = component.BulkInferrer(
        examples=self._examples,
        model=self._model,
        model_blessing=self._model_blessing)
    self.assertEqual(
        'InferenceResult', bulk_inferrer.outputs[
            standard_component_specs.INFERENCE_RESULT_KEY].type_name)
    self.assertNotIn('output_examples', bulk_inferrer.outputs.keys())

  def testConstructOutputExample(self):
    bulk_inferrer = component.BulkInferrer(
        examples=self._examples,
        model=self._model,
        model_blessing=self._model_blessing,
        output_example_spec=bulk_inferrer_pb2.OutputExampleSpec())
    self.assertEqual(
        'Examples', bulk_inferrer.outputs[
            standard_component_specs.OUTPUT_EXAMPLES_KEY].type_name)
    self.assertNotIn('inference_result', bulk_inferrer.outputs.keys())


if __name__ == '__main__':
  tf.test.main()
