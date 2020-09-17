# Lint as: python3
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
"""Tests for tfx.scripts.run_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Any, Dict, List, Text

import tensorflow as tf

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.scripts import run_executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ArgsCapture(object):
  instance = None

  def __enter__(self):
    ArgsCapture.instance = self
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    ArgsCapture.instance = None


class FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Overrides BaseExecutor.Do()."""
    args_capture = ArgsCapture.instance
    args_capture.input_dict = input_dict
    args_capture.output_dict = output_dict
    args_capture.exec_properties = exec_properties


class RunExecutorTest(tf.test.TestCase):

  def testMainEmptyInputs(self):
    """Test executor class import under empty inputs/outputs."""
    inputs = {
        'x': [
            standard_artifacts.ExternalArtifact(),
            standard_artifacts.ExternalArtifact()
        ]
    }
    outputs = {'y': [standard_artifacts.Examples()]}
    exec_properties = {'a': 'b'}
    args = [
        '--executor_class_path=%s.%s' %
        (FakeExecutor.__module__, FakeExecutor.__name__),
        '--inputs=%s' % artifact_utils.jsonify_artifact_dict(inputs),
        '--outputs=%s' % artifact_utils.jsonify_artifact_dict(outputs),
        '--exec-properties=%s' % json.dumps(exec_properties),
    ]
    with ArgsCapture() as args_capture:
      run_executor.main(args)
      # TODO(b/131417512): Add equal comparison to types.Artifact class so we
      # can use asserters.
      self.assertSetEqual(
          set(args_capture.input_dict.keys()), set(inputs.keys()))
      self.assertSetEqual(
          set(args_capture.output_dict.keys()), set(outputs.keys()))
      self.assertDictEqual(args_capture.exec_properties, exec_properties)


# TODO(zhitaoli): Add tests for:
# - base64 decoding of flags;
# - write output.

if __name__ == '__main__':
  tf.test.main()
