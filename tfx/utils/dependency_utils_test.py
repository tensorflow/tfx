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

import os
import sys
from unittest import mock


from absl import logging
from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.utils import dependency_utils


class DependencyUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  @mock.patch('tfx.utils.dependency_utils.build_ephemeral_package')
  def testMakeBeamDependencyFlags(self, mock_build_ephemeral_package):
    mock_build_ephemeral_package.return_value = 'mock_file'
    beam_flags = dependency_utils.make_beam_dependency_flags(
        beam_pipeline_args=[])
    self.assertListEqual(['--extra_package=mock_file'], beam_flags)
    mock_build_ephemeral_package.assert_called_with()

  # TODO(zhitaoli): Add check on 'sdk_container_image' once supported version of
  #                 Beam converges.
  @parameterized.named_parameters(
      ('ExtraPackages', '--extra_packages=foo'),
      ('SetupFile', '--setup_file=foo'),
      ('RequirementsFile', '--requirements_file=foo'),
  )
  def testNoActionOnFlag(self, flag_value):
    beam_pipeline_args = [flag_value]
    self.assertListEqual(
        [flag_value],
        dependency_utils.make_beam_dependency_flags(beam_pipeline_args),
    )

  @mock.patch('tempfile.mkdtemp')
  def testEphemeralPackage(self, mock_mkdtemp):
    mock_mkdtemp.return_value = self._tmp_dir
    if os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'):
      # This test requires setuptools which is not available.
      logging.info('Skipping testEphemeralPackage')
      return
    package = dependency_utils.build_ephemeral_package()
    self.assertRegex(os.path.basename(package), r'tfx_ephemeral-.*\.tar.gz')

  @mock.patch('tempfile.mkdtemp')
  @mock.patch('subprocess.call')
  def testEphemeralPackageMocked(self, mock_subprocess_call, mock_mkdtemp):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'testdata')
    test_file = os.path.join(source_data_dir, 'test.csv')
    expected_package = 'mypackage.tar.gz'

    def side_effect(cmd, stdout, stderr):
      self.assertLen(cmd, 3)
      self.assertEqual(sys.executable, cmd[0])
      self.assertEqual('sdist', cmd[2])
      self.assertEqual(stdout, stderr)
      setup_file = cmd[1]
      dist_dir = os.path.join(os.path.dirname(setup_file), 'dist')
      fileio.makedirs(dist_dir)
      dest_file = os.path.join(dist_dir, expected_package)
      fileio.copy(test_file, dest_file)

    mock_subprocess_call.side_effect = side_effect
    mock_mkdtemp.return_value = self._tmp_dir
    package = dependency_utils.build_ephemeral_package()
    self.assertEqual(expected_package, os.path.basename(package))


if __name__ == '__main__':
  tf.test.main()
