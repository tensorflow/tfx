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
from tfx.utils import test_case_utils
from tfx.utils.testdata import foo_pb2

from google.protobuf import any_pb2
from google.protobuf import descriptor_pb2


class ProtoUtilsTest(test_case_utils.TfxTest):

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
    fd_names = {
        fd.name for fd in proto_utils.gather_file_descriptors(
            foo_pb2.Foo.DESCRIPTOR.file)
    }
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

  def test_build_descriptor(self):
    expected_file_descriptor_list = [
        """
          name: "tfx/utils/testdata/bar.proto"
          package: "tfx.utils.proto.testdata"
          message_type {
            name: "Bar"
            field {
              name: "int_field"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_INT64
            }
          }
          message_type {
            name: "Bar2"
            field {
              name: "str_field"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
          }
          syntax: "proto3"
        """, """
          name: "tfx/utils/testdata/foo.proto"
          package: "tfx.utils.proto.testdata"
          dependency: "tfx/utils/testdata/bar.proto"
          message_type {
            name: "Foo"
            field {
              name: "bar"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".tfx.utils.proto.testdata.Bar"
            }
            field {
              name: "bar2"
              number: 2
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".tfx.utils.proto.testdata.Bar2"
            }
          }
          message_type {
            name: "Foo2"
            field {
              name: "value"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_INT64
            }
          }
          message_type {
            name: "TestProto"
            field {
              name: "string_value"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
            field {
              name: "int_value"
              number: 2
              label: LABEL_OPTIONAL
              type: TYPE_INT32
            }
            field {
              name: "double_value"
              number: 3
              label: LABEL_OPTIONAL
              type: TYPE_DOUBLE
            }
          }
          syntax: "proto3"
        """
    ]
    actual_file_descriptor = descriptor_pb2.FileDescriptorSet()
    proto_utils.build_file_descriptor_set(foo_pb2.Foo, actual_file_descriptor)
    self.assertLen(actual_file_descriptor.file, 2)
    actual_file_descriptor_sorted = sorted(
        list(actual_file_descriptor.file), key=lambda fd: fd.name)
    for expected, actual in zip(expected_file_descriptor_list,
                                actual_file_descriptor_sorted):
      self.assertProtoPartiallyEquals(expected, actual)

  def test_deserialize_proto_message(self):
    expected_pb_message = """
      string_value: 'hello'
    """
    serialized_message = '{"string_value":"hello"}'
    message_type = 'tfx.utils.proto.testdata.TestProto'
    fd_set = descriptor_pb2.FileDescriptorSet()
    foo_pb2.TestProto().DESCRIPTOR.file.CopyToProto(fd_set.file.add())
    self.assertProtoPartiallyEquals(
        expected_pb_message,
        proto_utils.deserialize_proto_message(serialized_message, message_type,
                                              fd_set))

  def test_unpack_proto_any(self):
    original_proto = foo_pb2.TestProto(string_value='x')
    any_proto = any_pb2.Any()
    any_proto.Pack(original_proto)
    unpacked_proto = proto_utils.unpack_proto_any(any_proto)
    self.assertEqual(unpacked_proto.string_value, 'x')

if __name__ == '__main__':
  tf.test.main()
