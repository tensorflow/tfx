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

import os
from unittest import mock


import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.utils import io_utils

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import


class IoUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'base_dir')
    file_io.create_dir(self._base_dir)
    super().setUp()

  def relpath(self, *segments):
    return os.path.join(self._base_dir, *segments)

  def createFiles(self, dir_spec, base_dir=None):
    if base_dir is None:
      base_dir = self._base_dir
    for key, value in dir_spec.items():
      full_path = os.path.join(base_dir, key)
      if isinstance(value, str):
        io_utils.write_string_file(full_path, value)
      elif isinstance(value, dict):
        fileio.makedirs(full_path)
        self.createFiles(value, base_dir=full_path)
      else:
        raise TypeError(f'Invalid directory spec: {dir_spec}')

  def extractDirectorySpec(self, path):
    if fileio.isdir(path):
      result = {}
      for name in fileio.listdir(path):
        result[name] = self.extractDirectorySpec(os.path.join(path, name))
      return result
    elif fileio.exists(path):
      return file_io.FileIO(path, mode='r').read()
    else:
      raise ValueError(f'{path} does not exist.')

  def assertDirectoryEqual(self, path, expected_spec) -> None:
    self.assertEqual(self.extractDirectorySpec(path), expected_spec)

  def testCreateFilesAndDirectoryEqual(self):
    spec = {
        'file1.txt': 'testing1',
        'dir1': {
            'file2.txt': 'testing2',
            'dir2': {
                'file3.txt': 'testing3'
            },
            'dir3': {}
        }
    }
    self.createFiles(spec)
    self.assertDirectoryEqual(self._base_dir, spec)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)
    super().tearDown()

  def testEnsureLocal(self):
    file_path = os.path.join(
        os.path.dirname(__file__), 'testdata', 'test_fn.py')
    self.assertEqual(file_path, io_utils.ensure_local(file_path))

  @mock.patch.object(io_utils, 'copy_file')
  def testEnsureLocalFromGCS(self, mock_copy_file):
    file_path = 'gs://path/to/testdata/test_fn.py'
    local_file_path = io_utils.ensure_local(file_path)
    self.assertEndsWith(local_file_path, '/test_fn.py')
    self.assertFalse(
        any([
            local_file_path.startswith(prefix)
            for prefix in io_utils._REMOTE_FS_PREFIX
        ]))
    mock_copy_file.assert_called_once_with(file_path, local_file_path, True)

  def testCopyFile(self):
    self.createFiles({
        'file1.txt': 'testing'
    })
    io_utils.copy_file(self.relpath('file1.txt'), self.relpath('file2.txt'))
    self.assertDirectoryEqual(self._base_dir, {
        'file1.txt': 'testing',
        'file2.txt': 'testing'
    })

  def testCopyDir(self):
    self.createFiles({
        'old': {
            'file1.txt': 'testing',
            'dir1': {
                'dir2': {
                    'file2.txt': 'testing2'
                }
            }
        }
    })
    io_utils.copy_dir(self.relpath('old'), self.relpath('new'))
    self.assertDirectoryEqual(self.relpath('new'), {
        'file1.txt': 'testing',
        'dir1': {
            'dir2': {
                'file2.txt': 'testing2'
            }
        }
    })

  def testCopyDirWithTrailingSlashes(self):
    self.createFiles({
        'old': {
            'dir': {
                'file.txt': 'testing'
            }
        }
    })

    with self.subTest('Copy old/ to new1'):
      io_utils.copy_dir(self.relpath('old', ''), self.relpath('new1'))
      self.assertDirectoryEqual(self.relpath('new1'), {
          'dir': {
              'file.txt': 'testing'
          }
      })

    with self.subTest('Copy old to new2/'):
      io_utils.copy_dir(self.relpath('old'), self.relpath('new2', ''))
      self.assertDirectoryEqual(self.relpath('new2'), {
          'dir': {
              'file.txt': 'testing'
          }
      })

  def testCopyDirWithAllowlistAndDenylist(self):
    old = os.path.join(self._base_dir, 'old')
    self.createFiles({
        'old': {
            'file1.txt': 'testing1',
            'dir1': {
                'file2.txt': 'testing2',
                'dir2': {
                    'file3.txt': 'testing3'
                },
                'dir3': {}
            },
        }
    })

    with self.subTest('Allowlist filenames'):
      new1 = os.path.join(self._base_dir, 'new1')
      io_utils.copy_dir(old, new1, allow_regex_patterns=[r'file[2-3]\.txt'])
      self.assertDirectoryEqual(new1, {
          'dir1': {
              'file2.txt': 'testing2',
              'dir2': {
                  'file3.txt': 'testing3'
              }
          }
      })

    with self.subTest('Allowlist dir1'):
      new2 = os.path.join(self._base_dir, 'new2')
      io_utils.copy_dir(old, new2, allow_regex_patterns=['dir1'])
      self.assertDirectoryEqual(new2, {
          'dir1': {
              'file2.txt': 'testing2',
              'dir2': {
                  'file3.txt': 'testing3'
              },
              'dir3': {}
          }
      })

    with self.subTest('Allowlist dir2'):
      new3 = os.path.join(self._base_dir, 'new3')
      io_utils.copy_dir(old, new3, allow_regex_patterns=['dir2'])
      self.assertDirectoryEqual(new3, {
          'dir1': {
              'dir2': {
                  'file3.txt': 'testing3'
              }
          }
      })

    with self.subTest('Multiple allowlist is unioned'):
      new4 = os.path.join(self._base_dir, 'new4')
      io_utils.copy_dir(old, new4, allow_regex_patterns=[
          r'file1\.txt', r'file2\.txt'])
      self.assertDirectoryEqual(new4, {
          'file1.txt': 'testing1',
          'dir1': {
              'file2.txt': 'testing2'
          }
      })

    with self.subTest('Denylist filenames'):
      new5 = os.path.join(self._base_dir, 'new5')
      io_utils.copy_dir(old, new5, deny_regex_patterns=[r'file[2-3]\.txt'])
      self.assertDirectoryEqual(new5, {
          'file1.txt': 'testing1',
          'dir1': {
              'dir2': {},
              'dir3': {}
          }
      })

    with self.subTest('Denylist dir1'):
      new6 = os.path.join(self._base_dir, 'new6')
      io_utils.copy_dir(old, new6, deny_regex_patterns=['dir1'])
      self.assertDirectoryEqual(new6, {
          'file1.txt': 'testing1'
      })

    with self.subTest('Denylist dir2'):
      new7 = os.path.join(self._base_dir, 'new7')
      io_utils.copy_dir(old, new7, deny_regex_patterns=['dir2'])
      self.assertDirectoryEqual(new7, {
          'file1.txt': 'testing1',
          'dir1': {
              'file2.txt': 'testing2',
              'dir3': {}
          }
      })

    with self.subTest('Multiple denylist is unioned'):
      new8 = os.path.join(self._base_dir, 'new8')
      io_utils.copy_dir(old, new8, deny_regex_patterns=[
          r'file1\.txt', r'file2\.txt'])
      self.assertDirectoryEqual(new8, {
          'dir1': {
              'dir2': {
                  'file3.txt': 'testing3',
              },
              'dir3': {}
          }
      })

    with self.subTest('Allowlist and denylist is AND clause.'):
      new9 = os.path.join(self._base_dir, 'new9')
      io_utils.copy_dir(
          old, new9,
          allow_regex_patterns=['dir1'],
          deny_regex_patterns=[r'file2\.txt'])
      self.assertDirectoryEqual(new9, {
          'dir1': {
              'dir2': {
                  'file3.txt': 'testing3',
              },
              'dir3': {}
          }
      })

  def testGetOnlyFileInDir(self):
    self.createFiles({
        'dir': {
            'file.txt': 'testing'
        }
    })
    self.assertEqual(
        self.relpath('dir', 'file.txt'),
        io_utils.get_only_uri_in_dir(self.relpath('dir')))

  def testGetOnlyDirInDir(self):
    self.createFiles({
        'dir1': {
            'dir2': {
                'file.txt': 'testing'
            }
        }
    })
    self.assertEqual(
        self.relpath('dir1', 'dir2'),
        io_utils.get_only_uri_in_dir(self.relpath('dir1')))

  def testDeleteDir(self):
    self.createFiles({
        'dir': {
            'file.txt': 'testing'
        }
    })
    io_utils.delete_dir(self.relpath('dir'))
    self.assertDirectoryEqual(self._base_dir, {})

  def testAllFilesPattern(self):
    self.assertEqual('model/*', io_utils.all_files_pattern('model'))

  def testLoadCsvColumnNames(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_file = os.path.join(source_data_dir, 'test.csv')
    column_names = io_utils.load_csv_column_names(test_file)
    self.assertListEqual(['a', 'b', 'c', 'd'], column_names)

  def testGeneratesFingerprint(self):
    self.createFiles({
        'fp': {
            'data1': 'testing',
            'data2': 'testing2'
        },
    })
    os.utime(self.relpath('fp', 'data1'), (0, 1))
    os.utime(self.relpath('fp', 'data2'), (0, 3))
    fingerprint = io_utils.generate_fingerprint(
        'split', os.path.join(self.relpath('fp'), '*'))
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
