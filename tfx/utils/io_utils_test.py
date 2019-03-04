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
"""Tests for tfx.utils.io_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports

import tensorflow as tf
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.utils import io_utils


class IoUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'base_dir')
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)

  def testImportFunc(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'test_fn.py')
    test_fn = io_utils.import_func(test_fn_file, 'test_fn')
    self.assertEqual(10, test_fn([1, 2, 3, 4]))

  def testImportFuncMissingFile(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'non_existing.py')
    with self.assertRaises(IOError):
      io_utils.import_func(test_fn_file, 'test_fn')

  def testImportFuncMissingFunction(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_fn_file = os.path.join(source_data_dir, 'test_fn.py')
    with self.assertRaises(AttributeError):
      io_utils.import_func(test_fn_file, 'non_existing')

  def testCopyFile(self):
    file_path = os.path.join(self._base_dir, 'temp_file')
    io_utils.write_string_file(file_path, 'testing')
    copy_path = os.path.join(self._base_dir, 'copy_file')
    io_utils.copy_file(file_path, copy_path)
    self.assertTrue(file_io.file_exists(copy_path))
    f = file_io.FileIO(file_path, mode='r')
    self.assertEqual('testing', f.read())
    self.assertEqual(7, f.tell())

  def testCopyDir(self):
    old_path = os.path.join(self._base_dir, 'old', 'path')
    new_path = os.path.join(self._base_dir, 'new', 'path')
    io_utils.write_string_file(old_path, 'testing')
    io_utils.copy_dir(os.path.dirname(old_path), os.path.dirname(new_path))
    self.assertTrue(file_io.file_exists(new_path))
    f = file_io.FileIO(new_path, mode='r')
    self.assertEqual('testing', f.read())
    self.assertEqual(7, f.tell())

  def testGetOnlyFileInDir(self):
    file_path = os.path.join(self._base_dir, 'file', 'path')
    io_utils.write_string_file(file_path, 'testing')
    self.assertEqual(file_path,
                     io_utils.get_only_uri_in_dir(os.path.dirname(file_path)))

  def testGetOnlyDirInDir(self):
    top_level_dir = os.path.join(self._base_dir, 'dir_1')
    dir_path = os.path.join(top_level_dir, 'dir_2')
    file_path = os.path.join(dir_path, 'file')
    io_utils.write_string_file(file_path, 'testing')
    self.assertEqual('dir_2', os.path.basename(
        io_utils.get_only_uri_in_dir(top_level_dir)))

  def testDeleteDir(self):
    file_path = os.path.join(self._base_dir, 'file', 'path')
    io_utils.write_string_file(file_path, 'testing')
    self.assertTrue(tf.gfile.Exists(file_path))
    io_utils.delete_dir(os.path.dirname(file_path))
    self.assertFalse(tf.gfile.Exists(file_path))

  def testAllFilesPattern(self):
    self.assertEqual('model*', io_utils.all_files_pattern('model'))

  def testLoadCsvColumnNames(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_file = os.path.join(source_data_dir, 'test.csv')
    column_names = io_utils.load_csv_column_names(test_file)
    self.assertListEqual(['a', 'b', 'c', 'd'], column_names)


if __name__ == '__main__':
  tf.test.main()
