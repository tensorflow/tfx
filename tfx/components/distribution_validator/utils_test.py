# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.distribution_validator.utils."""


import os

from absl import flags
import tensorflow as tf
from tfx.components.distribution_validator import utils
from tfx.proto import distribution_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import io_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS


class UtilsTest(tf.test.TestCase):

  def test_load_config_from_artifact(self):
    expected_config = text_format.Parse(
        """default_slice_config: {
              feature: {
                  path: {
                      step: 'company'
                  }
                  distribution_comparator: {
                    infinity_norm: {
                        threshold: 0.0
                    }
                  }
              }
            }
            """,
        distribution_validator_pb2.DistributionValidatorConfig(),
    )
    binary_proto_filepath = os.path.join(
        FLAGS.test_tmpdir, 'test', 'DVconfig.pb'
    )
    io_utils.write_bytes_file(
        binary_proto_filepath, expected_config.SerializeToString()
    )
    config_artifact = standard_artifacts.Config()
    config_artifact.uri = os.path.join(FLAGS.test_tmpdir, 'test')

    read_binary_config = utils.load_config_from_artifact(config_artifact)
    self.assertProtoEquals(read_binary_config, expected_config)
