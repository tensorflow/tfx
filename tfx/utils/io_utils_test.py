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
"""Tests for tfx.utils.io_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports
import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.utils import io_utils

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import


class IoUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'base_dir')
    file_io.create_dir(self._base_dir)
    super(IoUtilsTest, self).setUp()

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)
    super(IoUtilsTest, self).tearDown()

  def testEnsureLocal(self):
    file_path = os.path.join(
        os.path.dirname(__file__), 'testdata', 'test_fn.py')
    self.assertEqual(file_path, io_utils.ensure_local(file_path))

  @mock.patch.object(io_utils, 'copy_file')
  def testEnsureLocalFromGCS(self, mock_copy_file):
    file_path = 'gs://path/to/testdata/test_fn.py'
    self.assertEqual('test_fn.py', io_utils.ensure_local(file_path))
    mock_copy_file.assert_called_once_with(file_path, 'test_fn.py', True)

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
    old_path = os.path.join(self._base_dir, 'old')
    old_path_file1 = os.path.join(old_path, 'file1')
    old_path_file2 = os.path.join(old_path, 'dir', 'dir2', 'file2')
    new_path = os.path.join(self._base_dir, 'new')
    new_path_file1 = os.path.join(new_path, 'file1')
    new_path_file2 = os.path.join(new_path, 'dir', 'dir2', 'file2')

    io_utils.write_string_file(old_path_file1, 'testing')
    io_utils.write_string_file(old_path_file2, 'testing2')
    io_utils.copy_dir(old_path, new_path)

    self.assertTrue(file_io.file_exists(new_path_file1))
    f = file_io.FileIO(new_path_file1, mode='r')
    self.assertEqual('testing', f.readline())

    self.assertTrue(file_io.file_exists(new_path_file2))
    f = file_io.FileIO(new_path_file2, mode='r')
    self.assertEqual('testing2', f.readline())

  def testCopyDirWithTrailingSlashes(self):
    old_path1 = os.path.join(self._base_dir, 'old1', '')
    old_path_file1 = os.path.join(old_path1, 'child', 'file')
    new_path1 = os.path.join(self._base_dir, 'new1')
    new_path_file1 = os.path.join(new_path1, 'child', 'file')

    io_utils.write_string_file(old_path_file1, 'testing')
    io_utils.copy_dir(old_path1, new_path1)
    self.assertTrue(file_io.file_exists(new_path_file1))

    old_path2 = os.path.join(self._base_dir, 'old2')
    old_path_file2 = os.path.join(old_path2, 'child', 'file')
    new_path2 = os.path.join(self._base_dir, 'new2', '')
    new_path_file2 = os.path.join(new_path2, 'child', 'file')

    io_utils.write_string_file(old_path_file2, 'testing')
    io_utils.copy_dir(old_path2, new_path2)
    self.assertTrue(file_io.file_exists(new_path_file2))

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
    self.assertTrue(fileio.exists(file_path))
    io_utils.delete_dir(os.path.dirname(file_path))
    self.assertFalse(fileio.exists(file_path))

  def testAllFilesPattern(self):
    self.assertEqual('model/*', io_utils.all_files_pattern('model'))

  def testLoadCsvColumnNames(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_file = os.path.join(source_data_dir, 'test.csv')
    column_names = io_utils.load_csv_column_names(test_file)
    self.assertListEqual(['a', 'b', 'c', 'd'], column_names)

  def testGeneratesFingerprint(self):
    d1_path = os.path.join(self._base_dir, 'fp', 'data1')
    io_utils.write_string_file(d1_path, 'testing')
    os.utime(d1_path, (0, 1))
    d2_path = os.path.join(self._base_dir, 'fp', 'data2')
    io_utils.write_string_file(d2_path, 'testing2')
    os.utime(d2_path, (0, 3))
    fingerprint = io_utils.generate_fingerprint(
        'split', os.path.join(self._base_dir, 'fp', '*'))
    self.assertEqual(
        'split:split,num_files:2,total_bytes:15,xor_checksum:2,sum_checksum:4',
        fingerprint)

  def testReadWriteString(self):
    file_path = os.path.join(self._base_dir, 'test_file')
    content = 'testing read/write'
    io_utils.write_string_file(file_path, content)
    read_content = io_utils.read_string_file(file_path)
    self.assertEqual(content, read_content)

  def testReadWriteBytes(self):
    file_path = os.path.join(self._base_dir, 'test_file')
    content = b'testing read/write'
    io_utils.write_bytes_file(file_path, content)
    read_content = io_utils.read_bytes_file(file_path)
    self.assertEqual(content, read_content)


if __name__ == '__main__':
  tf.test.main()
