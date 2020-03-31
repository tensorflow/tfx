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
"""Tests for tfx.scripts.entrypoint_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, Text
import tensorflow as tf

from tfx.scripts import ai_platform_entrypoint_utils
from tfx.types import standard_artifacts

_ARTIFACT_1 = standard_artifacts.StringType()
_KEY_1 = 'input_1'

_ARTIFACT_2 = standard_artifacts.ModelBlessing()
_KEY_2 = 'input_2'

_EXEC_PROPERTIES = {
    'input_config': 'input config string',
    'output_config':
        '{ \"split_config\": { \"splits\": [ { \"hash_buckets\": 2, \"name\": '
        '\"train\" }, { \"hash_buckets\": 1, \"name\": \"eval\" } ] } }',
}


class EntrypointUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(EntrypointUtilsTest, self).setUp()
    _ARTIFACT_1.type_id = 1
    _ARTIFACT_1.uri = 'gs://root/string/'
    _ARTIFACT_2.type_id = 2
    _ARTIFACT_2.uri = 'gs://root/model/'
    self._expected_dict = {
        _KEY_1: [_ARTIFACT_1],
        _KEY_2: [_ARTIFACT_2],
    }
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    with open(os.path.join(source_data_dir,
                           'artifacts.json')) as artifact_json_file:
      self._artifacts = json.load(artifact_json_file)

    with open(os.path.join(source_data_dir,
                           'exec_properties.json')) as properties_json_file:
      self._properties = json.load(properties_json_file)

  def testParseRawArtifactDict(self):
    # TODO(b/131417512): Add equal comparison to types.Artifact class so we
    # can use asserters.
    def _convert_artifact_to_str(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
      """Convert artifact to its string representation."""
      result = {}
      for k, artifacts in inputs.items():
        result[k] = [str(artifact.to_json_dict()) for artifact in artifacts]

      return result

    self.assertDictEqual(
        _convert_artifact_to_str(self._expected_dict),
        _convert_artifact_to_str(
            ai_platform_entrypoint_utils.parse_raw_artifact_dict(
                self._artifacts)))

  def testParseExecutionProperties(self):
    self.assertDictEqual(
        _EXEC_PROPERTIES,
        ai_platform_entrypoint_utils.parse_execution_properties(
            self._properties))


if __name__ == '__main__':
  tf.test.main()
