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
"""Tests for tfx.orchestration.pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import tensorflow as tf

from tfx.orchestration import pipeline


class PipelineTest(tf.test.TestCase):

  def test_pipeline(self):

    @pipeline.PipelineDecorator(
        pipeline_name='a', pipeline_root='b', log_root='c')
    def create_pipeline():
      component_a = 'component_a'
      component_b = 'component_b'

      return [component_a, component_b]

    my_pipeline = create_pipeline()

    self.assertItemsEqual(my_pipeline.components,
                          ['component_a', 'component_b'])
    self.assertDictEqual(my_pipeline.pipeline_args, {
        'pipeline_name': 'a',
        'pipeline_root': 'b',
        'log_root': 'c'
    })


if __name__ == '__main__':
  tf.test.main()
