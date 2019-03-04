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
"""Tests for tfx.components.pusher.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.components.pusher import executor
from tfx.proto import pusher_pb2
from tfx.utils import types
from google.protobuf import json_format


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.gfile.MakeDirs(self._output_data_dir)
    self._model_export = types.TfxType(type_name='ModelExportPath')
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current/')
    self._model_blessing = types.TfxType(type_name='ModelBlessingPath')
    self._input_dict = {
        'model_export': [self._model_export],
        'model_blessing': [self._model_blessing],
    }

    self._model_push = types.TfxType(type_name='ModelPushPath')
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    tf.gfile.MakeDirs(self._model_push.uri)
    self._output_dict = {
        'model_push': [self._model_push],
    }
    self._serving_model_dir = os.path.join(self._output_data_dir,
                                           'serving_model_dir')
    tf.gfile.MakeDirs(self._serving_model_dir)
    self._exec_properties = {
        'push_destination':
            json_format.MessageToJson(
                pusher_pb2.PushDestination(
                    filesystem=pusher_pb2.PushDestination.Filesystem(
                        base_directory=self._serving_model_dir))),
    }
    self._executor = executor.Executor()

  def test_do_blessed(self):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed/')
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    self.assertNotEqual(0, len(tf.gfile.ListDirectory(self._serving_model_dir)))
    self.assertNotEqual(0, len(tf.gfile.ListDirectory(self._model_push.uri)))
    self.assertEqual(
        1, self._model_push.artifact.custom_properties['pushed'].int_value)

  def test_do_not_blessed(self):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/not_blessed/')
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    self.assertEqual(0, len(tf.gfile.ListDirectory(self._serving_model_dir)))
    self.assertEqual(0, len(tf.gfile.ListDirectory(self._model_push.uri)))
    self.assertEqual(
        0, self._model_push.artifact.custom_properties['pushed'].int_value)


if __name__ == '__main__':
  tf.test.main()
