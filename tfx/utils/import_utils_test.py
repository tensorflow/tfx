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
"""Tests for tfx.utils.import_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
# Standard Imports

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
    fn = import_utils.import_func_from_source(test_fn_file, 'test_fn')
    self.assertEqual(10, fn([1, 2, 3, 4]))

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


if __name__ == '__main__':
  tf.test.main()
