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
"""Tests for tfx.utils.types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

# Standard Imports

import tensorflow as tf
from tfx.utils import types


class TypesTest(tf.test.TestCase):

  def test_tfx_type(self):
    instance = types.TfxType('MyTypeName', split='eval')

    # Test property getters.
    self.assertEqual('', instance.uri)
    self.assertEqual(0, instance.id)
    self.assertEqual(0, instance.type_id)
    self.assertEqual('MyTypeName', instance.type_name)
    self.assertEqual('', instance.state)
    self.assertEqual('eval', instance.split)
    self.assertEqual(0, instance.span)

    # Test property setters.
    instance.uri = '/tmp/uri2'
    self.assertEqual('/tmp/uri2', instance.uri)

    instance.id = 1
    self.assertEqual(1, instance.id)

    instance.type_id = 2
    self.assertEqual(2, instance.type_id)

    instance.state = types.ARTIFACT_STATE_DELETED
    self.assertEqual(types.ARTIFACT_STATE_DELETED, instance.state)

    instance.split = ''
    self.assertEqual('', instance.split)

    instance.span = 20190101
    self.assertEqual(20190101, instance.span)

    instance.set_int_custom_property('int_key', 20)
    self.assertEqual(20,
                     instance.artifact.custom_properties['int_key'].int_value)

    instance.set_string_custom_property('string_key', 'string_value')
    self.assertEqual(
        'string_value',
        instance.artifact.custom_properties['string_key'].string_value)

    self.assertEqual('MyTypeName:/tmp/uri2.1', str(instance))

    # Test json serialization.
    json_dict = instance.json_dict()
    s = json.dumps(json_dict)
    other_instance = types.TfxType.parse_from_json_dict(json.loads(s))
    self.assertEqual(instance.artifact, other_instance.artifact)
    self.assertEqual(instance.artifact_type, other_instance.artifact_type)

  def test_get_from_single_list(self):
    """Test various retrieval utilities on a single list of TfxType."""
    single_list = [types.TfxType('MyTypeName', split='eval')]
    single_list[0].uri = '/tmp/evaluri'
    self.assertEqual(single_list[0], types.get_single_instance(single_list))
    self.assertEqual('/tmp/evaluri', types.get_single_uri(single_list))
    self.assertEqual(single_list[0],
                     types._get_split_instance(single_list, 'eval'))
    self.assertEqual('/tmp/evaluri', types.get_split_uri(single_list, 'eval'))
    with self.assertRaises(ValueError):
      types._get_split_instance(single_list, 'train')
    with self.assertRaises(ValueError):
      types.get_split_uri(single_list, 'train')

  def test_get_from_split_list(self):
    """Test various retrieval utilities on a list of split TfxTypes."""
    split_list = []
    for split in ['train', 'eval']:
      instance = types.TfxType('MyTypeName', split=split)
      instance.uri = '/tmp/' + split
      split_list.append(instance)

    with self.assertRaises(ValueError):
      types.get_single_instance(split_list)

    with self.assertRaises(ValueError):
      types.get_single_uri(split_list)

    self.assertEqual(split_list[0],
                     types._get_split_instance(split_list, 'train'))
    self.assertEqual('/tmp/train', types.get_split_uri(split_list, 'train'))
    self.assertEqual(split_list[1], types._get_split_instance(
        split_list, 'eval'))
    self.assertEqual('/tmp/eval', types.get_split_uri(split_list, 'eval'))


if __name__ == '__main__':
  tf.test.main()
