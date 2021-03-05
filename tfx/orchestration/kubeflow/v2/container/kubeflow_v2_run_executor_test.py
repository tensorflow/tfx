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
"""Tests for kubeflow_v2_run_executor.py."""

import json
import os
from typing import Any, Mapping, Sequence

import tensorflow as tf

from tfx.components.evaluator import constants
from tfx.components.evaluator import executor as evaluator_executor
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2.container import kubeflow_v2_run_executor
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types.standard_component_specs import BLESSING_KEY
from tfx.utils import test_case_utils


_TEST_OUTPUT_METADATA_JSON = "testdir/outputmetadata.json"

_TEST_OUTPUT_PROPERTY_KEY = "my_property"

_TEST_OUTPUT_PROPERTY_VALUE = "my_value"


class _ArgsCapture(object):
  instance = None

  def __enter__(self):
    _ArgsCapture.instance = self
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    _ArgsCapture.instance = None


class _FakeExecutor(evaluator_executor.Executor):

  def Do(self, input_dict: Mapping[str, Sequence[artifact.Artifact]],
         output_dict: Mapping[str, Sequence[artifact.Artifact]],
         exec_properties: Mapping[str, Any]) -> None:
    """Overrides BaseExecutor.Do()."""
    args_capture = _ArgsCapture.instance
    args_capture.input_dict = input_dict
    args_capture.output_dict = output_dict
    args_capture.exec_properties = exec_properties
    artifact_utils.get_single_instance(
        output_dict["output"]).set_string_custom_property(
            _TEST_OUTPUT_PROPERTY_KEY, _TEST_OUTPUT_PROPERTY_VALUE)
    blessing = artifact_utils.get_single_instance(output_dict["blessing"])
    # Set to the hash value of
    # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/1'
    blessing.set_int_custom_property(
        constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY, 5743745765020341227)
    # Set to the hash value of
    # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/2'
    blessing.set_int_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY, 7228748496289751000)
    # Set the blessing result
    blessing.set_int_custom_property(constants.ARTIFACT_PROPERTY_BLESSED_KEY,
                                     constants.BLESSED_VALUE)


_EXEC_PROPERTIES = {"key_1": "value_1", "key_2": 536870911}


class KubeflowV2RunExecutorTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()

    # Prepare executor input.
    serialized_metadata = self._get_text_from_test_data(
        "executor_invocation.json")
    metadata_json = json.loads(serialized_metadata)
    # Mutate the outputFile field.
    metadata_json["outputs"]["outputFile"] = _TEST_OUTPUT_METADATA_JSON
    self._serialized_metadata = json.dumps(metadata_json)

    self._expected_output = json.loads(
        self._get_text_from_test_data("expected_output_metadata.json"))

    # Change working directory after the testdata files have been read.
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  def _get_text_from_test_data(self, filename: str) -> str:
    filepath = os.path.join(os.path.dirname(__file__), "testdata", filename)
    return fileio.open(filepath, "rb").read().decode("utf-8")

  def testEntryPoint(self):
    """Test the entrypoint with toy inputs."""
    with _ArgsCapture() as args_capture:
      args = [
          "--executor_class_path",
          "%s.%s" % (_FakeExecutor.__module__, _FakeExecutor.__name__),
          "--json_serialized_invocation_args", self._serialized_metadata
      ]
      kubeflow_v2_run_executor.main(kubeflow_v2_run_executor._parse_flags(args))
      # TODO(b/131417512): Add equal comparison to types.Artifact class so we
      # can use asserters.
      self.assertEqual(
          set(args_capture.input_dict.keys()), set(["input_1", "input_2"]))
      self.assertEqual(
          set(args_capture.output_dict.keys()),
          set(["output", BLESSING_KEY]))
      self.assertEqual(args_capture.exec_properties, _EXEC_PROPERTIES)

    # Test what's been output.
    with open(_TEST_OUTPUT_METADATA_JSON) as output_meta_json:
      actual_output = json.load(output_meta_json)

    self.assertDictEqual(actual_output, self._expected_output)


if __name__ == "__main__":
  tf.test.main()
