# Lint as: python3
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
"""Tests for HelloComponent."""

import json

import tensorflow as tf

from tfx.examples.custom_components.hello_world.hello_component import component
from tfx.types import artifact
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class HelloComponentTest(tf.test.TestCase):

  def setUp(self):
    super(HelloComponentTest, self).setUp()
    self.name = 'HelloWorld'

  def testConstruct(self):
    input_data = standard_artifacts.Examples()
    input_data.split_names = json.dumps(artifact.DEFAULT_EXAMPLE_SPLITS)
    output_data = standard_artifacts.Examples()
    output_data.split_names = json.dumps(artifact.DEFAULT_EXAMPLE_SPLITS)
    this_component = component.HelloComponent(
        input_data=channel_utils.as_channel([input_data]),
        output_data=channel_utils.as_channel([output_data]),
        name=u'Testing123')
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     this_component.outputs['output_data'].type_name)
    artifact_collection = this_component.outputs['output_data'].get()
    for artifacts in artifact_collection:
      split_list = json.loads(artifacts.split_names)
      self.assertEqual(artifact.DEFAULT_EXAMPLE_SPLITS.sort(),
                       split_list.sort())


if __name__ == '__main__':
  tf.test.main()
