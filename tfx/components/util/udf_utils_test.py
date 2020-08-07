# Lint as: python2, python3
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
"""Tests for tfx.components.util.udf_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow as tf

from tfx.components.util import udf_utils
from tfx.utils import import_utils


class UdfUtilsTest(tf.test.TestCase):

  @mock.patch.object(import_utils, 'import_func_from_source')
  def testGetFnFromSource(self, mock_import_func):
    exec_properties = {'module_file': 'path/to/module_file.py'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path/to/module_file.py',
                                             'test_fn')

  @mock.patch.object(import_utils, 'import_func_from_module')
  def testGetFnFromModule(self, mock_import_func):
    exec_properties = {'test_fn': 'path.to.test_fn'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path.to', 'test_fn')

  def testGetFnFailure(self):
    with self.assertRaises(ValueError):
      udf_utils.get_fn({}, 'test_fn')


if __name__ == '__main__':
  tf.test.main()
