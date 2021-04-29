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
"""Tests for kubeflow_v2_entrypoint_utils.py."""

import os
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow as tf
from tfx.components.evaluator import constants
from tfx.orchestration.kubeflow.v2.container import kubeflow_v2_entrypoint_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils

_ARTIFACT_1 = standard_artifacts.String()
_KEY_1 = 'input_1'

_ARTIFACT_2 = standard_artifacts.ModelBlessing()
_KEY_2 = 'input_2'

_ARTIFACT_3 = standard_artifacts.Examples()
_KEY_3 = 'input_3'

_EXEC_PROPERTIES = {
    'input_config': 'input config string',
    'output_config':
        '{ \"split_config\": { \"splits\": [ { \"hash_buckets\": 2, \"name\": '
        '\"train\" }, { \"hash_buckets\": 1, \"name\": \"eval\" } ] } }',
}

_ARTIFACT_INVALID_NAME = r"""
inputs {
  artifacts {
    key: "artifact"
    value {
      artifacts {
        name: "invalid_runtime_name"
        uri: "gs://path/to/my/artifact"
        type {
          instance_schema: "title: tfx.String\ntype: object\nproperties:\n"
        }
      }
    }
  }
}
"""

_EXPECTED_CURRENT_MODEL_INT_ID = 123
_EXPECTED_CURRENT_MODEL_STRING_ID = 'current_model_string_id'
_EXPECTED_BASELINE_MODEL_INT_ID = 321
_EXPECTED_BASELINE_MODEL_STRING_ID = 'baseline_model_string_id'

_TEST_NAME_FROM_ID = {
    _EXPECTED_BASELINE_MODEL_INT_ID: _EXPECTED_BASELINE_MODEL_STRING_ID,
    _EXPECTED_CURRENT_MODEL_INT_ID: _EXPECTED_CURRENT_MODEL_STRING_ID
}


class KubeflowV2EntrypointUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(KubeflowV2EntrypointUtilsTest, self).setUp()
    _ARTIFACT_1.uri = 'gs://root/string/'
    # Hash value of
    # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/11111'
    _ARTIFACT_1.id = 9171918664759481579
    _ARTIFACT_1.set_string_custom_property(
        key='my_property_1', value='Test string.')
    _ARTIFACT_2.uri = 'gs://root/model/'
    # Hash value of
    # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/22222'
    _ARTIFACT_2.id = 6826273797600318744
    _ARTIFACT_2.set_float_custom_property(key='my_property_2', value=42.0)
    _ARTIFACT_3.uri = 'gs://root/examples/'
    _ARTIFACT_3.span = 9000
    # Hash value of
    # 'projects/123456789/locations/us-central1/metadataStores/default/artifacts/33333'
    _ARTIFACT_3.id = 27709763105391302
    self._expected_dict = {
        _KEY_1: [_ARTIFACT_1],
        _KEY_2: [_ARTIFACT_2],
        _KEY_3: [_ARTIFACT_3],
    }
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    # Use two protos to store the testdata.
    artifacts_pb = pipeline_pb2.ExecutorInput()
    io_utils.parse_json_file(
        os.path.join(source_data_dir, 'artifacts.json'), artifacts_pb)
    self._artifacts = artifacts_pb.inputs.artifacts

    # Test legacy properties/custom properties deserialization.
    artifacts_legacy_pb = pipeline_pb2.ExecutorInput()
    io_utils.parse_json_file(
        os.path.join(source_data_dir, 'artifacts_legacy.json'),
        artifacts_legacy_pb)
    self._artifacts_legacy = artifacts_legacy_pb.inputs.artifacts

    properties_pb = pipeline_pb2.ExecutorInput()
    io_utils.parse_json_file(
        os.path.join(source_data_dir, 'exec_properties.json'), properties_pb)
    self._properties = properties_pb.inputs.parameters

  def testParseRawArtifactDict(self):
    for artifacts_dict in [self._artifacts, self._artifacts_legacy]:
      name_from_id = {}
      actual_result = kubeflow_v2_entrypoint_utils.parse_raw_artifact_dict(
          artifacts_dict, name_from_id)
      for key in self._expected_dict:
        (expected_artifact,) = self._expected_dict[key]
        (actual_artifact,) = actual_result[key]
        self.assertEqual(expected_artifact.id, actual_artifact.id)
        self.assertEqual(expected_artifact.uri, actual_artifact.uri)
        for prop in expected_artifact.artifact_type.properties:
          self.assertEqual(
              getattr(expected_artifact, prop), getattr(actual_artifact, prop))
      self.assertEqual(
          self._expected_dict[_KEY_1][0].get_string_custom_property(
              'my_property_1'),
          actual_result[_KEY_1][0].get_string_custom_property('my_property_1'))
      self.assertEqual(
          self._expected_dict[_KEY_2][0].get_string_custom_property(
              'my_property_2'),
          actual_result[_KEY_2][0].get_string_custom_property('my_property_2'))
      self.assertEqual(self._expected_dict[_KEY_3][0].span,
                       actual_result[_KEY_3][0].span)

  def testParseExecutionProperties(self):
    self.assertDictEqual(
        _EXEC_PROPERTIES,
        kubeflow_v2_entrypoint_utils.parse_execution_properties(
            self._properties))

  def testParseExecutionPropertiesMapsInputBaseUri(self):
    properties_pb = pipeline_pb2.ExecutorInput()
    properties_pb.inputs.parameters[
        'input_base_uri'].string_value = 'gs://input/base'
    self.assertDictEqual(
        {'input_base': 'gs://input/base'},
        kubeflow_v2_entrypoint_utils.parse_execution_properties(
            properties_pb.inputs.parameters))

  def testCanChangePropertiesByNameIdMapping(self):
    model_blessing = standard_artifacts.ModelBlessing()
    model_blessing.set_int_custom_property(
        constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY,
        _EXPECTED_BASELINE_MODEL_INT_ID)
    model_blessing.set_int_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY,
        _EXPECTED_CURRENT_MODEL_INT_ID)

    expected_model_blessing = standard_artifacts.ModelBlessing()
    expected_model_blessing.set_string_custom_property(
        constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY,
        _EXPECTED_BASELINE_MODEL_STRING_ID)
    expected_model_blessing.set_string_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY,
        _EXPECTED_CURRENT_MODEL_STRING_ID)
    kubeflow_v2_entrypoint_utils.refactor_model_blessing(
        model_blessing, _TEST_NAME_FROM_ID)

    self.assertDictEqual(expected_model_blessing.to_json_dict(),
                         model_blessing.to_json_dict())


if __name__ == '__main__':
  tf.test.main()
