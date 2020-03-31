# Lint as: python2, python3
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
"""Tests for tfx.scripts.run_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Any, Dict, List, Text
import mock
import tensorflow as tf

from tfx.components.base import base_executor
from tfx.scripts import ai_platform_run_executor
from tfx.types import artifact


class _ArgsCapture(object):
  instance = None

  def __enter__(self):
    _ArgsCapture.instance = self
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    _ArgsCapture.instance = None


class _FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[Text, List[artifact.Artifact]],
         output_dict: Dict[Text, List[artifact.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Overrides BaseExecutor.Do()."""
    args_capture = _ArgsCapture.instance
    args_capture.input_dict = input_dict
    args_capture.output_dict = output_dict
    args_capture.exec_properties = exec_properties


_INPUT_DICT = {
    "input_1": {
        "artifacts": [{
            "uri": "gs://root/input_1/",
            "typeId": 1,
            "type": "ExternalArtifact",
        }]
    },
    "input_2": {
        "artifacts": [{
            "uri": "gs://root/input_2/",
            "typeId": 1,
            "type": "ExternalArtifact"
        }]
    },
}

_OUTPUT_DICT = {
    "output": {
        "artifacts": [{
            "uri": "gs://root/output/",
            "typeId": 2,
            "type": "Examples"
        }]
    }
}

_EXEC_PROPERTIES_PB = {
    "key_1": {
        "stringValue": "value_1"
    },
    "key_2": {
        "intValue": 42
    },
}

_EXEC_PROPERTIES = {"key_1": "value_1", "key_2": 42}

_JSON_SERIALIZED_METADATA = {
    "inputs": _INPUT_DICT,
    "outputs": _OUTPUT_DICT,
    "execution_properties": _EXEC_PROPERTIES_PB,
    "output_metadata_uri": "gs://root/output_metadata/"
}


class RunExecutorTest(tf.test.TestCase):

  @mock.patch.object(
      tf.io.gfile.GFile, "write", return_value=True, autospec=True)
  def testEntryPoint(self, fake_write_fn):
    """Test the entrypoint with toy inputs."""
    with _ArgsCapture() as args_capture:
      args = [
          "--executor_class_path",
          "%s.%s" %
          (_FakeExecutor.__module__, _FakeExecutor.__name__),
          "--json_serialized_metadata",
          json.dumps(_JSON_SERIALIZED_METADATA)
      ]
      ai_platform_run_executor.main(args)
      # TODO(b/131417512): Add equal comparison to types.Artifact class so we
      # can use asserters.
      self.assertSetEqual(
          set(args_capture.input_dict.keys()), set(_INPUT_DICT.keys()))
      self.assertSetEqual(
          set(args_capture.output_dict.keys()), set(_OUTPUT_DICT.keys()))
      self.assertDictEqual(args_capture.exec_properties, _EXEC_PROPERTIES)


if __name__ == "__main__":
  tf.test.main()
