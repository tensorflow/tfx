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
"""Tests for tfx.components.example_gen.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow as tf
from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.components.example_gen import driver
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from tfx.utils import io_utils


class DriverTest(tf.test.TestCase):

  def test_prepare_input_for_processing(self):
    # Create input splits.
    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    input_base_path = os.path.join(test_dir, 'input_base')
    split1 = os.path.join(input_base_path, 'split1', 'data')
    io_utils.write_string_file(split1, 'testing')
    os.utime(split1, (0, 1))
    split2 = os.path.join(input_base_path, 'split2', 'data')
    io_utils.write_string_file(split2, 'testing2')
    os.utime(split2, (0, 3))

    # Mock metadata.
    mock_metadata = tf.test.mock.Mock()
    example_gen_driver = driver.Driver(mock_metadata)

    # Mock artifact.
    artifacts = []
    for i in [4, 3, 2, 1]:
      artifact = metadata_store_pb2.Artifact()
      artifact.id = i
      artifact.uri = input_base_path
      # Only odd ids will be matched
      if i % 2 == 1:
        artifact.custom_properties[
            'input_fingerprint'].string_value = 'split:s1,num_files:1,total_bytes:7,xor_checksum:1,sum_checksum:1\nsplit:s2,num_files:1,total_bytes:8,xor_checksum:3,sum_checksum:3'
      else:
        artifact.custom_properties[
            'input_fingerprint'].string_value = 'not_match'
      artifacts.append(artifact)

    # Create input dict.
    input_base = standard_artifacts.ExternalArtifact()
    input_base.uri = input_base_path
    input_dict = {'input_base': [input_base]}
    # Create exec proterties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(name='s1', pattern='split1/*'),
                    example_gen_pb2.Input.Split(name='s2', pattern='split2/*')
                ])),
    }

    # Cache not hit.
    mock_metadata.get_artifacts_by_uri.return_value = [artifacts[0]]
    mock_metadata.publish_artifacts.return_value = [artifacts[3]]
    updated_input_dict = example_gen_driver._prepare_input_for_processing(
        copy.deepcopy(input_dict), exec_properties)
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input_base']))
    updated_input_base = updated_input_dict['input_base'][0]
    self.assertEqual(1, updated_input_base.id)
    self.assertEqual(input_base_path, updated_input_base.uri)

    # Cache hit.
    mock_metadata.get_artifacts_by_uri.return_value = artifacts
    mock_metadata.publish_artifacts.return_value = []
    updated_input_dict = example_gen_driver._prepare_input_for_processing(
        copy.deepcopy(input_dict), exec_properties)
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input_base']))
    updated_input_base = updated_input_dict['input_base'][0]
    self.assertEqual(3, updated_input_base.id)
    self.assertEqual(input_base_path, updated_input_base.uri)


if __name__ == '__main__':
  tf.test.main()
