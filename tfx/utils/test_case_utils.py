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
"""Utilities for customizing tf.test.TestCase class."""

import contextlib
import os

import tensorflow as tf


@contextlib.contextmanager
def override_env_var(name: str, value: str):
  """Overrides an environment variable and returns a context manager.

  Example:
    with test_case_utils.override_env_var('HOME', new_home_dir):

    or

    self.enter_context(test_case_utils.override_env_var('HOME', new_home_dir))

  Args:
    name: Name of the environment variable.
    value: Overriding value.

  Yields:
    None.
  """
  old_value = os.getenv(name)
  os.environ[name] = value

  yield

  if old_value is None:
    del os.environ[name]
  else:
    os.environ[name] = old_value


class TempWorkingDirTestCase(tf.test.TestCase):
  """TestCase class which uses a temporary directory as working directory.

  Inherit this class to make tests to use a temporary directory as a working
  directory.
  """

  def setUp(self):
    super().setUp()
    self._temp_working_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                            self.get_temp_dir())
    old_dir = os.getcwd()
    os.chdir(self.temp_working_dir)
    self.addCleanup(os.chdir, old_dir)

  @property
  def temp_working_dir(self):
    return self._temp_working_dir
