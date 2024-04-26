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

from unittest import mock
from kfp.pipeline_spec import pipeline_spec_pb2
import tensorflow as tf

from tfx import version
from tfx.components.evaluator import constants
from tfx.components.evaluator import executor as evaluator_executor
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2.container import kubeflow_v2_run_executor
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import name_utils
from tfx.utils import test_case_utils

from google.protobuf import struct_pb2
from google.protobuf import json_format

_TEST_OUTPUT_METADATA_JSON = "testdir/outputmetadata.json"

_TEST_OUTPUT_PROPERTY_KEY = "my_property"

_TEST_OUTPUT_PROPERTY_VALUE = "my_value"


class _ArgsCapture:
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

    for key in args_capture.output_dict:
      output_artifact = args_capture.output_dict[key][0]
      if isinstance(output_artifact, standard_artifacts.String):
        output_artifact.value = "String ValueArtifact"
      elif isinstance(output_artifact, standard_artifacts.Float):
        output_artifact.value = 1.1
      elif isinstance(output_artifact, standard_artifacts.Integer):
        output_artifact.value = 1
      elif isinstance(output_artifact, standard_artifacts.Boolean):
        output_artifact.value = True

    args_capture.exec_properties = exec_properties
    artifact_utils.get_single_instance(
        output_dict["output"]).set_string_custom_property(
            _TEST_OUTPUT_PROPERTY_KEY, _TEST_OUTPUT_PROPERTY_VALUE)
    if "blessing" in output_dict:
      blessing = artifact_utils.get_single_instance(output_dict["blessing"])
      # Set to the hash value of
      # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/1'
      blessing.set_int_custom_property(
          constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY,
          5743745765020341227)
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

    # Set a constant version for artifact version tag.
    patcher = mock.patch("tfx.version.__version__")
    patcher.start()
    version.__version__ = "0.123.4.dev"
    self.addCleanup(patcher.stop)

    # Prepare executor input.
    serialized_metadata = self._get_text_from_test_data(
        "executor_invocation.json")
    metadata_json = json.loads(serialized_metadata)
    # Mutate the outputFile field.
    metadata_json["outputs"]["outputFile"] = _TEST_OUTPUT_METADATA_JSON
    self._serialized_metadata = json.dumps(metadata_json)

    # Prepare executor input using output parameters
    serialized_metadata_dynamic_execution = self._get_text_from_test_data(
        "executor_invocation_with_output_parameters.json")
    self._metadata_json_dynamic_execution = json.loads(
        serialized_metadata_dynamic_execution)
    # Mutate the outputFile field.
    self._metadata_json_dynamic_execution["outputs"][
        "outputFile"] = _TEST_OUTPUT_METADATA_JSON

    # Prepare executor input using legacy properties and custom properties.
    serialized_metadata_legacy = self._get_text_from_test_data(
        "executor_invocation_legacy.json")
    metadata_json_legacy = json.loads(serialized_metadata_legacy)
    # Mutate the outputFile field.
    metadata_json_legacy["outputs"]["outputFile"] = _TEST_OUTPUT_METADATA_JSON
    self._serialized_metadata_legacy = json.dumps(metadata_json_legacy)

    self._expected_output = (
        self._get_text_from_test_data("expected_output_metadata.json").strip())

    # Change working directory after the testdata files have been read.
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  def _get_text_from_test_data(self, filename: str) -> str:
    filepath = os.path.join(os.path.dirname(__file__), "testdata", filename)
    return fileio.open(filepath, "r").read()

  def testEntryPoint(self):
    """Test the entrypoint with toy inputs."""
    # Test both current version metadata and legacy property/custom property
    # metadata styles.
    for serialized_metadata in [
        self._serialized_metadata, self._serialized_metadata_legacy
    ]:
      with _ArgsCapture() as args_capture:
        args = [
            "--executor_class_path",
            name_utils.get_full_name(_FakeExecutor),
            "--json_serialized_invocation_args", serialized_metadata
        ]
        kubeflow_v2_run_executor.main(
            kubeflow_v2_run_executor._parse_flags(args))
        # TODO(b/131417512): Add equal comparison to types.Artifact class so we
        # can use asserters.
        self.assertEqual(
            set(args_capture.input_dict.keys()), set(["input_1", "input_2"]))
        self.assertEqual(
            set(args_capture.output_dict.keys()),
            set(["output", standard_component_specs.BLESSING_KEY]))
        self.assertEqual(args_capture.exec_properties, _EXEC_PROPERTIES)

      # Test what's been output.
      with open(_TEST_OUTPUT_METADATA_JSON) as output_meta_json:
        actual_output = json.dumps(
            json.load(output_meta_json), indent=2, sort_keys=True)

      self.assertEqual(actual_output, self._expected_output)
      os.remove(_TEST_OUTPUT_METADATA_JSON)

  def testDynamicExecutionProperties(self):
    """Test the entrypoint with dynamic execution properties."""

    test_value_artifact_float_dir = os.path.join(self.tmp_dir,
                                                 "test_value_artifact_float")
    test_value_artifact_string_dir = os.path.join(self.tmp_dir,
                                                  "test_value_artifact_string")
    test_value_artifact_boolean_dir = os.path.join(
        self.tmp_dir, "test_value_artifact_boolean")
    test_value_artifact_integer_dir = os.path.join(
        self.tmp_dir, "test_value_artifact_integer")
    test_executor_output_dir = os.path.join(self.tmp_dir,
                                            "test_executor_output")
    self._metadata_json_dynamic_execution["outputs"]["artifacts"][
        "test_value_artifact_float"]["artifacts"][0][
            "uri"] = test_value_artifact_float_dir
    self._metadata_json_dynamic_execution["outputs"]["artifacts"][
        "test_value_artifact_string"]["artifacts"][0][
            "uri"] = test_value_artifact_string_dir
    self._metadata_json_dynamic_execution["outputs"]["artifacts"][
        "test_value_artifact_boolean"]["artifacts"][0][
            "uri"] = test_value_artifact_boolean_dir
    self._metadata_json_dynamic_execution["outputs"]["artifacts"][
        "test_value_artifact_integer"]["artifacts"][0][
            "uri"] = test_value_artifact_integer_dir
    self._metadata_json_dynamic_execution["outputs"][
        "output_file"] = test_executor_output_dir
    serialized_metadata_dynamic_execution = json.dumps(
        self._metadata_json_dynamic_execution)

    with _ArgsCapture() as args_capture:
      args = [
          "--executor_class_path",
          name_utils.get_full_name(_FakeExecutor),
          "--json_serialized_invocation_args",
          serialized_metadata_dynamic_execution
      ]
      kubeflow_v2_run_executor.main(kubeflow_v2_run_executor._parse_flags(args))

      self.assertEqual(
          set(args_capture.output_dict.keys()),
          set([
              "output", "test_value_artifact_float",
              "test_value_artifact_string", "test_value_artifact_boolean",
              "test_value_artifact_integer"
          ]))
      executor_output = pipeline_spec_pb2.ExecutorOutput()
      with fileio.open(test_executor_output_dir, "rb") as f:
        json_format.Parse(f.read(), executor_output, ignore_unknown_fields=True)
      self.assertDictEqual(
          dict(executor_output.parameter_values), {
              "test_value_artifact_boolean":
                  struct_pb2.Value(bool_value=True),
              "test_value_artifact_float":
                  struct_pb2.Value(number_value=1.1),
              "test_value_artifact_integer":
                  struct_pb2.Value(number_value=1),
              "test_value_artifact_string":
                  struct_pb2.Value(string_value="String ValueArtifact")
          })
      self.assertEqual(
          io_utils.read_string_file(test_value_artifact_float_dir), "1.1")
      self.assertEqual(
          io_utils.read_string_file(test_value_artifact_string_dir),
          "String ValueArtifact")
      self.assertEqual(
          io_utils.read_string_file(test_value_artifact_boolean_dir), "1")
      self.assertEqual(
          io_utils.read_string_file(test_value_artifact_integer_dir), "1")

  def testEntryPointWithDriver(self):
    """Test the entrypoint with Driver's output metadata."""
    # Mock the driver's output metadata.
    output_metadata = pipeline_spec_pb2.ExecutorOutput()
    output_metadata.parameters["key_1"].string_value = "driver"
    output_metadata.parameters["key_3"].string_value = "driver3"
    fileio.makedirs(os.path.dirname(_TEST_OUTPUT_METADATA_JSON))
    with fileio.open(_TEST_OUTPUT_METADATA_JSON, "wb") as f:
      f.write(json_format.MessageToJson(output_metadata, sort_keys=True))

    with _ArgsCapture() as args_capture:
      args = [
          "--executor_class_path",
          name_utils.get_full_name(_FakeExecutor),
          "--json_serialized_invocation_args", self._serialized_metadata
      ]
      kubeflow_v2_run_executor.main(kubeflow_v2_run_executor._parse_flags(args))
      # TODO(b/131417512): Add equal comparison to types.Artifact class so we
      # can use asserters.
      self.assertEqual(
          set(args_capture.input_dict.keys()), set(["input_1", "input_2"]))
      self.assertEqual(
          set(args_capture.output_dict.keys()),
          set(["output", standard_component_specs.BLESSING_KEY]))
      # Verify that exec_properties use driver's output metadata.
      self.assertEqual(
          args_capture.exec_properties,
          {
              "key_1": "driver",  # Overwrite.
              "key_2": 536870911,
              "key_3": "driver3"  # Append.
          })

    # Test what's been output.
    with open(_TEST_OUTPUT_METADATA_JSON) as output_meta_json:
      actual_output = json.dumps(
          json.load(output_meta_json), indent=2, sort_keys=True)

    self.assertEqual(actual_output, self._expected_output)
    os.remove(_TEST_OUTPUT_METADATA_JSON)


if __name__ == "__main__":
  tf.test.main()
