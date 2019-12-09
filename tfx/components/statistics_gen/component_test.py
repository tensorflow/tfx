# Lint as: python2, python3
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
"""Tests for tfx.components.statistics_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.statistics_gen import component
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    train_examples = standard_artifacts.Examples(split='train')
    eval_examples = standard_artifacts.Examples(split='eval')
    statistics_gen = component.StatisticsGen(
        examples=channel_utils.as_channel([train_examples, eval_examples]))
    self.assertEqual(standard_artifacts.ExampleStatistics.TYPE_NAME,
                     statistics_gen.outputs['statistics'].type_name)


if __name__ == '__main__':
  tf.test.main()
