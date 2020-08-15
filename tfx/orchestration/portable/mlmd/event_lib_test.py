# Lint as: python3
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
"""Tests for tfx.orchestration.portable.mlmd.event_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import tensorflow as tf

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.portable.mlmd import event_lib


class EventLibTest(tf.test.TestCase):

  def testIsDesiredOutputEvent(self):
    legal_output_event = metadata_store_pb2.Event()
    text_format.Parse(
        """
        type: OUTPUT
        path {
          steps {
            key: 'right_key'
          }
          steps {
            index: 1
          }
        }
        """, legal_output_event)
    event_wrong_type = metadata_store_pb2.Event()
    text_format.Parse(
        """
        type: INPUT
        path {
          steps {
            key: 'right_key'
          }
          steps {
            index: 1
          }
        }
        """, event_wrong_type)
    event_no_key = metadata_store_pb2.Event()
    text_format.Parse('type: OUTPUT', event_no_key)
    self.assertTrue(
        event_lib.validate_output_event(legal_output_event, 'right_key'))
    self.assertFalse(
        event_lib.validate_output_event(event_wrong_type, 'right_key'))
    self.assertFalse(event_lib.validate_output_event(event_no_key, 'right_key'))
    self.assertTrue(event_lib.validate_output_event(event_no_key))

  def testGenerateEvent(self):
    self.assertProtoEquals(
        """
        type: INPUT
        path {
          steps {
            key: 'key'
          }
          steps {
            index: 1
          }
        }
        artifact_id: 2
        execution_id: 3
        """,
        event_lib.generate_event(
            event_type=metadata_store_pb2.Event.INPUT,
            key='key',
            index=1,
            artifact_id=2,
            execution_id=3))


if __name__ == '__main__':
  tf.test.main()
