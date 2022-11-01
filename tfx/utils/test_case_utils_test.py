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
"""Tests for tfx.utils.test_case_utils."""

import copy
import os
import unittest

import tensorflow as tf
from tfx import types
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils


def _create_artifact(uri: str) -> types.Artifact:
  artifact = standard_artifacts.Model()
  artifact.uri = uri
  return artifact


class TempWorkingDirTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.override_env_var('NEW_ENV', 'foo'))
    self.enter_context(test_case_utils.override_env_var('OVERWRITE_ENV', 'baz'))
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  # This test method will be invoked manually in TestCaseUtilsTest.
  def successfulTest(self):
    self.assertEqual(
        os.path.realpath(self.tmp_dir), os.path.realpath(os.getcwd()))
    self.assertEqual(os.getenv('NEW_ENV'), 'foo')
    self.assertEqual(os.getenv('OVERWRITE_ENV'), 'baz')


class FailingTempWorkingDirTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  # This test method will be invoked manually in TestCaseUtilsTest.
  def failingTest(self):
    self.assertTrue(False)


class TestCaseUtilsTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.enter_context(test_case_utils.override_env_var('OVERWRITE_ENV', 'bar'))

  def _run_test_case_class(self, cls, prefix, check=False):
    test_loader = unittest.TestLoader()
    test_loader.testMethodPrefix = prefix
    suite = test_loader.loadTestsFromTestCase(cls)
    result = unittest.TextTestRunner().run(suite)
    if check:
      self.assertTrue(result.wasSuccessful())

  def testTempWorkingDirWithTestCaseClass(self):
    old_cwd = os.getcwd()
    self._run_test_case_class(TempWorkingDirTest, 'successful', check=True)
    self.assertEqual(os.getcwd(), old_cwd)
    self.assertNotIn('NEW_ENV', os.environ)
    self.assertEqual(os.getenv('OVERWRITE_ENV'), 'bar')

    self._run_test_case_class(FailingTempWorkingDirTest, 'failing')
    self.assertEqual(os.getcwd(), old_cwd)

  def testChangeWorkingDir(self):
    cwd = os.getcwd()
    new_cwd = os.path.join(self.tmp_dir, 'new')
    os.makedirs(new_cwd)
    with test_case_utils.change_working_dir(new_cwd) as old_cwd:
      self.assertEqual(os.path.realpath(old_cwd), os.path.realpath(cwd))
      self.assertEqual(os.path.realpath(new_cwd), os.path.realpath(os.getcwd()))

    self.assertEqual(os.path.realpath(cwd), os.path.realpath(os.getcwd()))

  def testOverrideEnvVar(self):
    old_home = os.getenv('HOME')
    new_home = self.get_temp_dir()
    with test_case_utils.override_env_var('HOME', new_home):
      self.assertEqual(os.environ['HOME'], new_home)
    self.assertEqual(os.getenv('HOME'), old_home)

  def testAssertArtifactMapsEqual_equalMapsPassesAssertion(self):
    expected_artifacts = {
        'artifact1': [_create_artifact('uri1a'),
                      _create_artifact('uri1b')],
        'artifact2': [_create_artifact('uri2')],
    }
    actual_artifacts = copy.deepcopy(expected_artifacts)
    self.assertArtifactMapsEqual(expected_artifacts, actual_artifacts)

  def testAssertArtifactMapsEqual_differingMapsFailsAssertion(self):
    expected_artifacts = {
        'artifact1': [_create_artifact('uri1a'),
                      _create_artifact('uri1b')],
        'artifact2': [_create_artifact('uri2')],
    }
    actual_artifacts = copy.deepcopy(expected_artifacts)
    actual_artifacts['artifact1'][1].set_int_custom_property('key', 5)
    with self.assertRaises(AssertionError):
      self.assertArtifactMapsEqual(expected_artifacts, actual_artifacts)

if __name__ == '__main__':
  tf.test.main()
