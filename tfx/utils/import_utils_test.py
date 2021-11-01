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
"""Tests for tfx.utils.import_utils."""

import importlib
import os
import sys

import tensorflow as tf
from tfx.utils import import_utils
from tfx.utils.testdata import test_fn


class ImportUtilsTest(tf.test.TestCase):

  def testImportClassByPath(self):
    test_class = test_fn.TestClass
    class_path = '%s.%s' % (test_class.__module__, test_class.__name__)
    imported_class = import_utils.import_class_by_path(class_path)
    self.assertEqual(test_class, imported_class)

  def testImportFuncFromSource(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'test_fn.ext')
    fn_1 = import_utils.import_func_from_source(test_fn_file, 'test_fn')
    fn_2 = import_utils.import_func_from_source(test_fn_file, 'test_fn')
    self.assertEqual(10, fn_1([1, 2, 3, 4]))
    self.assertEqual(10, fn_2([1, 2, 3, 4]))

  def testImportFuncFromSourceMissingFile(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'non_existing.py')
    with self.assertRaises(ImportError):
      import_utils.import_func_from_source(test_fn_file, 'test_fn')

  def testImportFuncFromSourceMissingFunction(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'test_fn.ext')
    with self.assertRaises(AttributeError):
      import_utils.import_func_from_source(test_fn_file, 'non_existing')

  def testImportFuncFromModule(self):
    imported_fn = import_utils.import_func_from_module(
        test_fn.test_fn.__module__, test_fn.test_fn.__name__)
    self.assertEqual(10, imported_fn([1, 2, 3, 4]))

  def testImportFuncFromModuleUnknownModule(self):
    with self.assertRaises(ImportError):
      _ = import_utils.import_func_from_module('non_existing_module', 'test_fn')

  def testImportFuncFromModuleModuleMissingFunction(self):
    with self.assertRaises(AttributeError):
      _ = import_utils.import_func_from_module(test_fn.test_fn.__module__,
                                               'non_existing_fn')

  def testtestImportFuncFromModuleReload(self):
    temp_dir = self.create_tempdir().full_path
    test_fn_file = os.path.join(temp_dir, 'fn.py')
    with tf.io.gfile.GFile(test_fn_file, mode='w') as f:
      f.write(
          """def test_fn(inputs):
            return sum(inputs)
          """)
    count_registered = import_utils._tfx_module_finder.count_registered
    fn_1 = import_utils.import_func_from_source(test_fn_file, 'test_fn')
    self.assertEqual(10, fn_1([1, 2, 3, 4]))
    with tf.io.gfile.GFile(test_fn_file, mode='w') as f:
      f.write(
          """def test_fn(inputs):
            return 1+sum(inputs)
          """)
    fn_2 = import_utils.import_func_from_source(test_fn_file, 'test_fn')
    self.assertEqual(11, fn_2([1, 2, 3, 4]))
    fn_3 = getattr(
        importlib.reload(sys.modules['user_module_%d' % count_registered]),
        'test_fn')
    self.assertEqual(11, fn_3([1, 2, 3, 4]))

if __name__ == '__main__':
  tf.test.main()
