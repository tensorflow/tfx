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
"""Tests for tfx.components.data_view.provider_component."""
import tensorflow as tf
from tfx.components.experimental.data_view import provider_component
from tfx.types import standard_artifacts


class ProviderComponentTest(tf.test.TestCase):

  def testConstructModuleFileProvided(self):
    module_file = '/path/to/module_file.py'
    create_decoder_func = 'my_func'
    provider = provider_component.TfGraphDataViewProvider(
        module_file=module_file, create_decoder_func=create_decoder_func)
    self.assertEqual(module_file,
                     provider.spec.exec_properties['module_file'])
    self.assertEqual(
        create_decoder_func,
        provider.spec.exec_properties['create_decoder_func'])
    self.assertEqual(standard_artifacts.DataView.TYPE_NAME,
                     provider.outputs['data_view'].type_name)

  def testConstructModuleFileNotProvided(self):
    create_decoder_func = 'some_package.some_module.my_func'
    provider = provider_component.TfGraphDataViewProvider(
        create_decoder_func=create_decoder_func)
    self.assertIsNone(provider.spec.exec_properties['module_file'])
    self.assertEqual(
        create_decoder_func,
        provider.spec.exec_properties['create_decoder_func'])
    self.assertEqual(standard_artifacts.DataView.TYPE_NAME,
                     provider.outputs['data_view'].type_name)


if __name__ == '__main__':
  tf.test.main()
