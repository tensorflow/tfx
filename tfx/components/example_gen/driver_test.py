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
"""Tests for tfx.components.example_gen.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.components.example_gen import driver
from tfx.proto import example_gen_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils


class DriverTest(tf.test.TestCase):

  def setUp(self):
    super(DriverTest, self).setUp()
    # Create input splits.
    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._input_base_path = os.path.join(test_dir, 'input_base')
    tf.io.gfile.makedirs(self._input_base_path)

    # Mock metadata.
    self._mock_metadata = tf.compat.v1.test.mock.Mock()
    self._example_gen_driver = driver.Driver(self._mock_metadata)

    # Create input dict.
    input_base = standard_artifacts.ExternalArtifact()
    input_base.uri = self._input_base_path
    self._input_channels = {
        'input_base': channel_utils.as_channel([input_base])
    }
    # Create exec proterties.
    self._exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='s1', pattern='span{SPAN}/split1/*'),
                    example_gen_pb2.Input.Split(
                        name='s2', pattern='span{SPAN}/split2/*')
                ]),
                preserving_proto_field_name=True),
    }

  def testResolveInputArtifacts(self):
    # Create input splits.
    split1 = os.path.join(self._input_base_path, 'split1', 'data')
    io_utils.write_string_file(split1, 'testing')
    os.utime(split1, (0, 1))
    split2 = os.path.join(self._input_base_path, 'split2', 'data')
    io_utils.write_string_file(split2, 'testing2')
    os.utime(split2, (0, 3))

    # Mock artifact.
    artifacts = []
    for i in [4, 3, 2, 1]:
      artifact = metadata_store_pb2.Artifact()
      artifact.id = i
      artifact.uri = self._input_base_path
      artifact.custom_properties['span'].string_value = '0'
      # Only odd ids will be matched
      if i % 2 == 1:
        artifact.custom_properties[
            'input_fingerprint'].string_value = 'split:s1,num_files:1,total_bytes:7,xor_checksum:1,sum_checksum:1\nsplit:s2,num_files:1,total_bytes:8,xor_checksum:3,sum_checksum:3'
      else:
        artifact.custom_properties[
            'input_fingerprint'].string_value = 'not_match'
      artifacts.append(artifact)

    # Create exec proterties.
    exec_properties = {
        'input_config':
            json_format.MessageToJson(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(name='s1', pattern='split1/*'),
                    example_gen_pb2.Input.Split(name='s2', pattern='split2/*')
                ]),
                preserving_proto_field_name=True),
    }

    # Cache not hit.
    self._mock_metadata.get_artifacts_by_uri.return_value = [artifacts[0]]
    self._mock_metadata.publish_artifacts.return_value = [artifacts[3]]
    updated_input_dict = self._example_gen_driver.resolve_input_artifacts(
        self._input_channels, exec_properties, None, None)
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input_base']))
    updated_input_base = updated_input_dict['input_base'][0]
    self.assertEqual(self._input_base_path, updated_input_base.uri)

    # Cache hit.
    self._mock_metadata.get_artifacts_by_uri.return_value = artifacts
    self._mock_metadata.publish_artifacts.return_value = []
    updated_input_dict = self._example_gen_driver.resolve_input_artifacts(
        self._input_channels, exec_properties, None, None)
    self.assertEqual(1, len(updated_input_dict))
    self.assertEqual(1, len(updated_input_dict['input_base']))
    updated_input_base = updated_input_dict['input_base'][0]
    self.assertEqual(3, updated_input_base.id)
    self.assertEqual(self._input_base_path, updated_input_base.uri)

  def testResolveInputArtifactsWithSpan(self):
    # Test align of span number.
    span1_split1 = os.path.join(self._input_base_path, 'span01', 'split1',
                                'data')
    io_utils.write_string_file(span1_split1, 'testing11')
    span1_split2 = os.path.join(self._input_base_path, 'span01', 'split2',
                                'data')
    io_utils.write_string_file(span1_split2, 'testing12')
    span2_split1 = os.path.join(self._input_base_path, 'span02', 'split1',
                                'data')
    io_utils.write_string_file(span2_split1, 'testing21')

    with self.assertRaisesRegexp(
        ValueError, 'Latest span should be the same for each split'):
      self._example_gen_driver.resolve_input_artifacts(self._input_channels,
                                                       self._exec_properties,
                                                       None, None)

    # Test if latest span is selected when span aligns for each split.
    span2_split2 = os.path.join(self._input_base_path, 'span02', 'split2',
                                'data')
    io_utils.write_string_file(span2_split2, 'testing22')

    self._mock_metadata.get_artifacts_by_uri.return_value = []
    self._mock_metadata.publish_artifacts.return_value = [
        metadata_store_pb2.Artifact()
    ]
    self._example_gen_driver.resolve_input_artifacts(self._input_channels,
                                                     self._exec_properties,
                                                     None, None)
    updated_input_config = example_gen_pb2.Input()
    json_format.Parse(self._exec_properties['input_config'],
                      updated_input_config)
    # Check if latest span is selected.
    self.assertProtoEquals(
        """
        splits {
          name: "s1"
          pattern: "span02/split1/*"
        }
        splits {
          name: "s2"
          pattern: "span02/split2/*"
        }""", updated_input_config)


if __name__ == '__main__':
  tf.test.main()
