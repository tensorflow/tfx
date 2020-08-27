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
"""Tests for tfx.components.data_view.binder_component."""
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tfx.components.experimental.data_view import binder_component
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class BinderComponentTest(tf.test.TestCase):

  def _assert_example_artifact_equal(self, lhs, rhs):
    self.assertEqual(lhs.span, rhs.span)
    self.assertEqual(lhs.split_names, rhs.split_names)

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    binder = binder_component.DataViewBinder(
        input_examples=channel_utils.as_channel([examples]),
        data_view=channel_utils.as_channel([standard_artifacts.DataView()])
    )
    self.assertIsNotNone(binder.outputs['output_examples'])


if __name__ == '__main__':
  tf.test.main()
