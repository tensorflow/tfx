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

import hashlib
import os
import tempfile

from unittest import mock
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
    exec_properties = {'module_path': 'path.to.module'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path.to.module', 'test_fn')

  @mock.patch.object(import_utils, 'import_func_from_module')
  def testGetFnFromModuleFn(self, mock_import_func):
    exec_properties = {'test_fn': 'path.to.module.test_fn'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path.to.module', 'test_fn')

  def testGetFnFailure(self):
    with self.assertRaises(ValueError):
      udf_utils.get_fn({}, 'test_fn')

  def test_ephemeral_setup_py_contents(self):
    contents = udf_utils._get_ephemeral_setup_py_contents(
        'my_pkg', '0.0+xyz', ['a', 'abc', 'xyz'])
    self.assertIn("name='my_pkg',", contents)
    self.assertIn("version='0.0+xyz',", contents)
    self.assertIn("py_modules=['a', 'abc', 'xyz'],", contents)

  def test_version_hash(self):

    def _write_temp_file(user_module_dir, file_name, contents):
      with open(os.path.join(user_module_dir, file_name), 'w') as f:
        f.write(contents)

    user_module_dir = tempfile.mkdtemp()
    _write_temp_file(user_module_dir, 'a.py', 'aa1')
    _write_temp_file(user_module_dir, 'bb.py', 'bbb2')
    _write_temp_file(user_module_dir, 'ccc.py', 'cccc3')
    _write_temp_file(user_module_dir, 'dddd.py', 'ddddd4')

    expected_plaintext = (
        # Length and encoding of "a.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x04a.py'
        # Length and encoding of contents of "a.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x03aa1'
        # Length and encoding of "ccc.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x06ccc.py'
        # Length and encoding of contents of "ccc.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x05cccc3'
        # Length and encoding of "dddd.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x07dddd.py'
        # Length and encoding of contents of "dddd.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x06ddddd4')
    h = hashlib.sha256()
    h.update(expected_plaintext)
    expected_version_hash = h.hexdigest()
    self.assertEqual(
        expected_version_hash,
        '4fecd9af212c76ee4097037caf78c6ba02a2e82584837f2031bcffa0f21df43e')
    self.assertEqual(
        udf_utils._get_version_hash(user_module_dir,
                                    ['dddd.py', 'a.py', 'ccc.py']),
        expected_version_hash)


if __name__ == '__main__':
  tf.test.main()
