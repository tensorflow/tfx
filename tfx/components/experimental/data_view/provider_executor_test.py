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
"""Tests for tfx.components.data_view.provider_executor."""
import os
import unittest

import tensorflow as tf
from tfx.components.experimental.data_view import provider_executor
from tfx.components.testdata.module_file import data_view_module
from tfx.types import standard_artifacts
from tfx_bsl.coders import tf_graph_record_decoder


@unittest.skipIf(tf.__version__ < '2',
                 'tfx-bsl installed does not have modules required to run this '
                 'test or Tensorflow is not 2.x.')
class DataViewProviderExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(DataViewProviderExecutorTest, self).setUp()
    # ../../../testdata
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def testExecutorModuleFileProvided(self):
    input_dict = {}
    output = standard_artifacts.DataView()
    output.uri = os.path.join(self._output_data_dir, 'output_data_view')
    output_dict = {'data_view': [output]}
    exec_properties = {
        'module_file':
            os.path.join(self._source_data_dir,
                         'module_file/data_view_module.py'),
        'create_decoder_func':
            'create_simple_decoder',
    }
    executor = provider_executor.TfGraphDataViewProviderExecutor()
    executor.Do(input_dict, output_dict, exec_properties)
    loaded_decoder = tf_graph_record_decoder.load_decoder(output.uri)
    self.assertIsInstance(
        loaded_decoder, tf_graph_record_decoder.LoadedDecoder)

  def testExecutorModuleFileNotProvided(self):
    input_dict = {}
    output = standard_artifacts.DataView()
    output.uri = os.path.join(self._output_data_dir, 'output_data_view')
    output_dict = {'data_view': [output]}
    exec_properties = {
        'module_file': None,
        'create_decoder_func':
            '%s.%s' % (data_view_module.create_simple_decoder.__module__,
                       data_view_module.create_simple_decoder.__name__),
    }
    executor = provider_executor.TfGraphDataViewProviderExecutor()
    executor.Do(input_dict, output_dict, exec_properties)
    loaded_decoder = tf_graph_record_decoder.load_decoder(output.uri)
    self.assertIsInstance(
        loaded_decoder, tf_graph_record_decoder.LoadedDecoder)


if __name__ == '__main__':
  tf.test.main()
