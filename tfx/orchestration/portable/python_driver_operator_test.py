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
"""Tests for tfx.orchestration.portable.python_driver_operator."""

from typing import Any, Dict, List, Text

import tensorflow as tf
from tfx import types
from tfx.orchestration.portable import base_driver
from tfx.orchestration.portable import python_driver_operator
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import executable_spec_pb2

_DEFAULT_DRIVER_OUTPUT = driver_output_pb2.DriverOutput()


class _FakeNoopDriver(base_driver.BaseDriver):

  def run(self, input_dict: Dict[Text, List[types.Artifact]],
          output_dict: Dict[Text, List[types.Artifact]],
          exec_properties: Dict[Text, Any]) -> driver_output_pb2.DriverOutput:
    return _DEFAULT_DRIVER_OUTPUT


class PythonDriverOperatorTest(tf.test.TestCase):

  def succeed(self):
    custom_driver_spec = (executable_spec_pb2.PythonClassExecutableSpec())
    custom_driver_spec.class_path = 'tfx.orchestration.portable.python_driver_operator._FakeNoopDriver'
    driver_operator = python_driver_operator.PythonDriverOperator(
        custom_driver_spec, None, None, None)
    driver_output = driver_operator.run_driver(None, None, None)
    self.assertEqual(driver_output, _DEFAULT_DRIVER_OUTPUT)


if __name__ == '__main__':
  tf.test.main()
