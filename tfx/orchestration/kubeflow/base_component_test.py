# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.orchestration.kubeflow.base_component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kfp import dsl
import tensorflow as tf
from tfx.orchestration.kubeflow import base_component
from tfx.utils import types


class BaseComponentTest(tf.test.TestCase):

  def setUp(self):
    output_dict = {'output_name': [types.TfxType(type_name='ExamplesPath')]}

    with dsl.Pipeline('test_pipeline'):
      self.component = base_component.BaseComponent(
          component_name='TFXComponent',
          input_dict={
              'input_data': 'input-data-contents',
              'train_steps': 300,
              'accuracy_threshold': 0.3,
          },
          output_dict=output_dict,
          exec_properties={'module_file': '/path/to/module.py'},
      )

  def test_container_op_arguments(self):
    self.assertEqual(self.component.container_op.arguments[:3], [
        '--exec_properties',
        '{"module_file": "/path/to/module.py"}',
        '--outputs',
    ])
    self.assertItemsEqual(self.component.container_op.arguments[-7:], [
        'TFXComponent',
        '--input_data',
        'input-data-contents',
        '--train_steps',
        '300',
        '--accuracy_threshold',
        '0.3',
    ])

  def test_container_op_output_parameters(self):
    self.assertEqual(self.component.container_op.file_outputs,
                     {'output_name': '/output/ml_metadata/output_name'})


if __name__ == '__main__':
  tf.test.main()
