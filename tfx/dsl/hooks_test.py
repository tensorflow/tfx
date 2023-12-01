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

"""DSL for composing execution hooks tests."""

import tensorflow as tf
from tfx.dsl import hooks
from tfx.proto.orchestration import execution_hook_pb2
from google.protobuf import json_format


class HooksTest(tf.test.TestCase):

  def test_encode_binary_component_pre_output(self):
    pre_output = hooks.BinaryComponentPreOutput(
        flags=[('flag_key', 'flag_value')], extra_flags=['extra_flag_value']
    )
    self.assertProtoEquals(
        pre_output.encode(),
        json_format.ParseDict(
            {
                'flags': [{
                    'name': 'flag_key',
                    'value': {'string_value': 'flag_value'},
                }],
                'extra_flags': [{'string_value': 'extra_flag_value'}],
            },
            execution_hook_pb2.PreExecutionOutput(),
        ),
    )

  def test_encode_bcl_component_pre_output(self):
    pre_output = hooks.BCLComponentPreOutput(
        vars={'var_key': 'var_value'},
    )
    self.assertProtoEquals(
        pre_output.encode(),
        json_format.ParseDict(
            {'vars': {'var_key': {'string_value': 'var_value'}}},
            execution_hook_pb2.PreExecutionOutput(),
        ),
    )

  def test_encode_xmanager_component_pre_output(self):
    pre_output = hooks.XManagerComponentPreOutput(
        flags=[('flag_key', 'flag_value')]
    )
    self.assertProtoEquals(
        pre_output.encode(),
        json_format.ParseDict(
            {
                'flags': [{
                    'name': 'flag_key',
                    'value': {'string_value': 'flag_value'},
                }],
            },
            execution_hook_pb2.PreExecutionOutput(),
        ),
    )


if __name__ == '__main__':
  tf.test.main()
