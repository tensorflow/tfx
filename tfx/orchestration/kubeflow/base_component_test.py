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

import collections
import json
from kfp import dsl
import tensorflow as tf
from tfx.orchestration.kubeflow import base_component
from tfx.utils import types


class BaseComponentTest(tf.test.TestCase):
  maxDiff = None  # pylint: disable=invalid-name

  def setUp(self):
    self._output_dict = {
        'output_name': [types.TfxType(type_name='ExamplesPath')]
    }
    self._pipeline_properties = base_component.PipelineProperties(
        output_dir='output_dir',
        log_root='log_root',
    )

    with dsl.Pipeline('test_pipeline'):
      self.component = base_component.BaseComponent(
          component_name='TFXComponent',
          input_dict=collections.OrderedDict([
              ('input_data', 'input-data-contents'),
              ('train_steps', 300),
              ('accuracy_threshold', 0.3),
          ]),
          output_dict=self._output_dict,
          exec_properties=collections.OrderedDict([('module_file',
                                                    '/path/to/module.py')]),
          executor_class_path='some.executor.Class',
          pipeline_properties=self._pipeline_properties,
      )

  def test_container_op_arguments(self):
    self.assertEqual(self.component.container_op.arguments[0],
                     '--exec_properties')
    self.assertDictEqual(
        {
            'output_dir': 'output_dir',
            'log_root': 'log_root',
            'module_file': '/path/to/module.py'
        }, json.loads(self.component.container_op.arguments[1]))

    self.assertEqual(self.component.container_op.arguments[2:], [
        '--outputs',
        types.jsonify_tfx_type_dict(self._output_dict),
        '--executor_class_path',
        'some.executor.Class',
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
