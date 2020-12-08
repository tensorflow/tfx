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
"""Tests for tfx.utils.proto_utils."""

import tensorflow as tf
from tfx.utils import proto_utils
from tfx.utils.testdata import foo_pb2


class ProtoUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_proto = foo_pb2.TestProto(
        string_value='hello', int_value=2, double_value=0.5)
    self.test_dict = {
        'string_value': 'hello',
        'int_value': 2,
        'double_value': 0.5
    }

  def test_gather_file_descriptors(self):
    fd_names = set()
    for fd in proto_utils.gather_file_descriptors(foo_pb2.Foo.DESCRIPTOR):
      fd_names.add(fd.name)
    self.assertEqual(
        fd_names, {
            'tfx/utils/testdata/bar.proto',
            'tfx/utils/testdata/foo.proto'
        })

  def test_proto_to_json(self):
    json_str = proto_utils.proto_to_json(self.test_proto)
    # Checks whether original field name is kept and fields are sorted.
    self.assertEqual(
        json_str.replace(' ', '').replace('\n', ''),
        '{"double_value":0.5,"int_value":2,"string_value":"hello"}')

  def test_proto_to_dict(self):
    # Checks whether original field name is kept.
    self.assertEqual(proto_utils.proto_to_dict(self.test_proto), self.test_dict)

  def test_json_to_proto(self):
    json_str = '{"obsolete_field":2,"string_value":"x"}'
    result = proto_utils.json_to_proto(json_str, foo_pb2.TestProto())
    self.assertEqual(result, foo_pb2.TestProto(string_value='x'))
    # Make sure that returned type is not message.Message
    self.assertEqual(result.string_value, 'x')

  def test_dict_to_proto(self):
    self.assertEqual(
        proto_utils.dict_to_proto(self.test_dict, foo_pb2.TestProto()),
        self.test_proto)
    dict_with_obsolete_field = {'obsolete_field': 2, 'string_value': 'x'}
    self.assertEqual(
        proto_utils.dict_to_proto(dict_with_obsolete_field,
                                  foo_pb2.TestProto()),
        foo_pb2.TestProto(string_value='x'))


if __name__ == '__main__':
  tf.test.main()
