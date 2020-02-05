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
"""Tests for tfx.utils.dependency_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
# Standard Imports

import absl
import mock
import tensorflow.compat.v1 as tf
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.utils import dependency_utils


class DepsUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    self._tmp_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  @mock.patch('tempfile.mkdtemp')
  def testEphemeralPackage(self, mock_mkdtemp):
    mock_mkdtemp.return_value = self._tmp_dir
    if os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'):
      # This test requires setuptools which is not available.
      absl.logging.info('Skipping testEphemeralPackage')
      return
    package = dependency_utils.build_ephemeral_package()
    self.assertRegexpMatches(
        os.path.basename(package), r'tfx_ephemeral-.*\.tar.gz')

  @mock.patch('tempfile.mkdtemp')
  @mock.patch('subprocess.call')
  def testEphemeralPackageMocked(self, mock_subprocess_call, mock_mkdtemp):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    test_file = os.path.join(source_data_dir, 'test.csv')
    expected_package = 'mypackage.tar.gz'

    def side_effect(cmd):
      self.assertEqual(3, len(cmd))
      self.assertEqual(sys.executable, cmd[0])
      self.assertEqual('sdist', cmd[2])
      setup_file = cmd[1]
      dist_dir = os.path.join(os.path.dirname(setup_file), 'dist')
      tf.io.gfile.makedirs(dist_dir)
      dest_file = os.path.join(dist_dir, expected_package)
      tf.io.gfile.copy(test_file, dest_file)

    mock_subprocess_call.side_effect = side_effect
    mock_mkdtemp.return_value = self._tmp_dir
    package = dependency_utils.build_ephemeral_package()
    self.assertEqual(expected_package, os.path.basename(package))

  @mock.patch('tempfile.mkdtemp')
  def testRequirementFile(self, mock_mkdtemp):
    mock_mkdtemp.return_value = self._tmp_dir
    requirements_file = dependency_utils._build_requirements_file()
    content = file_io.read_file_to_string(requirements_file)
    self.assertRegexpMatches(content, 'tfx==.*')


if __name__ == '__main__':
  tf.test.main()
