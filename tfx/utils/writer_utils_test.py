# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.writer_utils."""

import os
from absl import flags
import tensorflow as tf
from tfx.utils import io_utils
from tfx.utils import writer_utils
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2

FLAGS = flags.FLAGS


class WriterUtilsTest(tf.test.TestCase):

  def testWriteAnomalies(self):
    anomalies = text_format.Parse(
        """anomaly_info {
               key: "feature_1"
               value {
                  description: "Some description"
                  severity: ERROR
                  short_description: "Description"
                  reason {
                    type: ENUM_TYPE_UNEXPECTED_STRING_VALUES
                    short_description: "Unexpected string values"
                    description: "Examples contain values missing"
                  }
                }
              }""",
        anomalies_pb2.Anomalies())
    binary_proto_filepath = os.path.join(FLAGS.test_tmpdir, 'SchemaDiff.pb')
    writer_utils.write_anomalies(binary_proto_filepath, anomalies)
    # Check binary proto file.
    read_binary_anomalies = anomalies_pb2.Anomalies()
    read_binary_anomalies.ParseFromString(
        io_utils.read_bytes_file(binary_proto_filepath)
    )
    self.assertProtoEquals(read_binary_anomalies, anomalies)


if __name__ == '__main__':
  tf.test.main()
