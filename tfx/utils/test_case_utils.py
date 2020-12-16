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
import copy
import os

from typing import Iterable, Optional, Text, Union

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.utils import io_utils
from google.protobuf import message
from google.protobuf import text_format


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


class TfxTest(tf.test.TestCase):
  """Convenient wrapper for tfx test cases."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(self.tmp_dir)

  def load_proto_from_text(self, path: Text,
                           proto_message: message.Message) -> message.Message:
    """Loads proto message from serialized text."""
    return io_utils.parse_pbtxt_file(path, proto_message)

  def assertProtoPartiallyEquals(
      self,
      expected: Union[str, message.Message],
      actual: message.Message,
      ignored_fields: Optional[Iterable[str]] = None,
  ):
    """Asserts proto messages are equal except the ignored fields."""
    if isinstance(expected, str):
      expected = text_format.Parse(expected, actual.__class__())
      actual = copy.deepcopy(actual)
    else:
      expected = copy.deepcopy(expected)
      actual = copy.deepcopy(actual)

    # Currently only supports one-level for ignored fields.
    for ignored_field in ignored_fields or []:
      expected.ClearField(ignored_field)
      actual.ClearField(ignored_field)

    return self.assertProtoEquals(expected, actual)


class TempWorkingDirTestCase(TfxTest):
  """TestCase class which uses a temporary directory as working directory.

  Inherit this class to make tests to use a temporary directory as a working
  directory.
  """

  def setUp(self):
    super().setUp()

    old_dir = os.getcwd()
    os.chdir(self.tmp_dir)
    self.addCleanup(os.chdir, old_dir)
