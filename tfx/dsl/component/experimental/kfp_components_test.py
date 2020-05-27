# Lint as: python3
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
"""Tests ability to use KFP components in a TFX pipeline."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest

from tfx.components.base import base_component
from tfx.dsl.component.experimental import kfp_components


# A sample component function for tests
def filter_text(text_uri, filtered_text_uri, pattern):
  """Filters the text keeping the lines that contain the pattern."""
  import re  # pylint: disable=g-import-not-at-top
  import tensorflow  # pylint: disable=g-import-not-at-top
  with tensorflow.io.gfile.GFile(text_uri, 'r') as source:
    with tensorflow.io.gfile.GFile(filtered_text_uri, 'w') as dest:
      for line in source:
        if re.search(pattern, line):
          dest.write(line)


class KfpComponentsTest(absltest.TestCase):

  def testEnableKfpComponents(self):
    try:
      from kfp import components  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest('The kfp package is not installed')

    kfp_components.enable_kfp_components()

    filter_text_op = components.create_component_from_func(
        func=filter_text,
        base_image='tensorflow/tensorflow-2.2.0',
    )

    component_instance = filter_text_op(
        source_uri='gs://bucket/text.txt',
        filtered_text_uri='gs://bucket/filtered.txt',
        pattern='secret',
    )

    self.assertIsInstance(component_instance, base_component.BaseComponent)
    self.assertEqual(component_instance.executor_spec.image,
                     'tensorflow/tensorflow-2.2.0')
