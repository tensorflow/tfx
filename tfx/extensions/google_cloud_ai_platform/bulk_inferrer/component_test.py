# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfx.extensions.google_cloud_ai_platform.bulk_inferrer import component
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    model = standard_artifacts.Model()
    model_blessing = standard_artifacts.ModelBlessing()
    bulk_inferrer = component.CloudAIBulkInferrerComponent(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model]),
        model_blessing=channel_utils.as_channel([model_blessing]),
        custom_config={})
    self.assertEqual('InferenceResult',
                     bulk_inferrer.outputs['inference_result'].type_name)


if __name__ == '__main__':
  tf.test.main()
