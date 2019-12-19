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
"""Common functions for component tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, cast, Text, Type

import tensorflow as tf

TEST_UNDECLARED_OUTPUTS_DIR = 'TEST_UNDECLARED_OUTPUTS_DIR'


def as_tensorflow_testcase(type_: Type[Any]) -> Type[tf.test.TestCase]:
  """Cast type for correct pytype recognition."""
  return cast(Type[tf.test.TestCase], type_)


class ComponentTestMixin(as_tensorflow_testcase(object)):
  """Mixin of commonly used properties and methods in component test."""

  @property
  def testdata_dir(self) -> Text:
    """Absolute path to tfx/components/testdata where input artifacts stay."""
    return os.path.join(os.path.dirname(__file__), '../testdata')

  @property
  def output_data_dir(self) -> Text:
    """Output temp path to write artifacts per test method."""
    return os.path.join(
        os.environ.get(TEST_UNDECLARED_OUTPUTS_DIR, self.get_temp_dir()),
        self._testMethodName)
