# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.experimental.filter.component."""

import tensorflow as tf
from tfx.components.experimental.filter import component
from tfx.types import standard_artifacts
from tfx.types import channel


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = channel.Channel(type=standard_artifacts.Examples)
    filter_fn_path = 'my_module.my_filter_fn'

    filter_component = component.FilterComponent(
        examples=examples,
        filter_fn_path=filter_fn_path
    )

    # Verify input channel
    self.assertEqual(
        filter_component.inputs['examples'].type,
        standard_artifacts.Examples
    )

    # Verify output channel
    self.assertEqual(
        filter_component.outputs['filtered_examples'].type,
        standard_artifacts.Examples
    )

    # Verify parameter
    self.assertEqual(
        filter_component.exec_properties['filter_fn_path'],
        filter_fn_path
    )


if __name__ == '__main__':
  tf.test.main()
