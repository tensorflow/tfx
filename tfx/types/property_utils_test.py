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
"""Tests for tfx.types.property_utils."""

import tensorflow as tf

from tfx.types import property_utils

from ml_metadata.proto import metadata_store_pb2


class PropertyMapProxyTest(tf.test.TestCase):

  def testProxyWithoutSchema_getitem(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.custom_properties['intty'].int_value = 42
    artifact.custom_properties['floaty'].double_value = 3.1415
    artifact.custom_properties['stringy'].string_value = 'hello world'
    proxy = property_utils.PropertyMapProxy(artifact.custom_properties)

    self.assertEqual(proxy['intty'], 42)
    self.assertEqual(proxy['floaty'], 3.1415)
    self.assertEqual(proxy['stringy'], 'hello world')

  def testProxyWithoutSchema_get(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.custom_properties['intty'].int_value = 42
    artifact.custom_properties['floaty'].double_value = 3.1415
    artifact.custom_properties['stringy'].string_value = 'hello world'
    proxy = property_utils.PropertyMapProxy(artifact.custom_properties)

    self.assertEqual(proxy.get('intty'), 42)
    self.assertEqual(proxy.get('floaty'), 3.1415)
    self.assertEqual(proxy.get('stringy'), 'hello world')
    self.assertIsNone(proxy.get('not-exist'))
    self.assertEqual(proxy.get('not-exist', 123), 123)

  def testProxyWithoutSchema_set(self):
    artifact = metadata_store_pb2.Artifact()
    proxy = property_utils.PropertyMapProxy(artifact.custom_properties)
    proxy['intty'] = 42
    proxy['floaty'] = 3.1415
    proxy['stringy'] = 'hello world'

    self.assertEqual(artifact.custom_properties['intty'].int_value, 42)
    self.assertEqual(artifact.custom_properties['floaty'].double_value, 3.1415)
    self.assertEqual(artifact.custom_properties['stringy'].string_value,
                     'hello world')

  def testProxyWithSchema_getitem(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.properties['intty'].int_value = 42
    artifact.properties['floaty'].double_value = 3.1415
    artifact.properties['stringy'].string_value = 'hello world'
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
            'floaty': float,
            'stringy': str
        })

    self.assertEqual(proxy['intty'], 42)
    self.assertEqual(proxy['floaty'], 3.1415)
    self.assertEqual(proxy['stringy'], 'hello world')

  def testProxyWithSchema_getitem_typeNotChecked(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.properties['intty'].string_value = 'actually string'
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int  # Note that schema requires "int"
        })

    # Schema type checking is NOT done on retrieval.
    self.assertEqual(proxy['intty'], 'actually string')

  def testProxyWithSchema_getitem_defaultValue(self):
    artifact = metadata_store_pb2.Artifact()
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
            'floaty': float,
            'stringy': str
        })

    self.assertEqual(proxy['intty'], 0)
    self.assertEqual(proxy['floaty'], 0.0)
    self.assertEqual(proxy['stringy'], '')

  def testProxyWithSchema_get(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.properties['intty'].int_value = 42
    artifact.properties['floaty'].double_value = 3.1415
    artifact.properties['stringy'].string_value = 'hello world'
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
            'floaty': float,
            'stringy': str
        })

    self.assertEqual(proxy.get('intty'), 42)
    self.assertEqual(proxy.get('floaty'), 3.1415)
    self.assertEqual(proxy.get('stringy'), 'hello world')

  def testProxyWithSchema_get_defaultValue(self):
    artifact = metadata_store_pb2.Artifact()
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
            'floaty': float,
            'stringy': str
        })

    self.assertEqual(proxy.get('intty'), 0)
    self.assertEqual(proxy.get('floaty'), 0.0)
    self.assertEqual(proxy.get('stringy'), '')

  def testProxyWithSchema_get_notDefinedInSchema(self):
    artifact = metadata_store_pb2.Artifact()
    artifact.properties['not_defined_in_schema'].int_value = 123
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
            'floaty': float,
            'stringy': str
        })

    # Schema not checked if property already exists.
    self.assertEqual(proxy.get('not_defined_in_schema'), 123)

    # Schema checked if property does not exist (so that if absent we can
    # construct an empty proto default value).
    with self.assertRaises(KeyError):
      proxy.get('not_defined_in_schema_and_not_exist')

  def testProxyWithSchema_get_defaultValueCannotBeModified(self):
    artifact = metadata_store_pb2.Artifact()
    proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            'intty': int,
        })

    # Default value is a empty protobuf primitive value.
    self.assertEqual(proxy.get('intty'), 0)

    # Cannot override default value.
    with self.assertRaises(ValueError):
      proxy.get('intty', 42)


class PropertyGetterFactoryTest(tf.test.TestCase):

  def testMakePropertyGetter_getArtifactProperty(self):
    getter = property_utils.make_property_getter('foo')
    a1 = metadata_store_pb2.Artifact()
    a1.properties['foo'].int_value = 1
    a2 = metadata_store_pb2.Artifact()
    a2.properties['foo'].double_value = 1.0
    a3 = metadata_store_pb2.Artifact()
    a3.properties['foo'].string_value = 'hi'

    self.assertEqual(getter(a1), 1)
    self.assertEqual(getter(a2), 1.0)
    self.assertEqual(getter(a3), 'hi')

  def testMakePropertyGetter_getExecutionProperty(self):
    getter = property_utils.make_property_getter('foo')
    e1 = metadata_store_pb2.Execution()
    e1.properties['foo'].int_value = 1
    e2 = metadata_store_pb2.Execution()
    e2.properties['foo'].double_value = 1.0
    e3 = metadata_store_pb2.Execution()
    e3.properties['foo'].string_value = 'hi'

    self.assertEqual(getter(e1), 1)
    self.assertEqual(getter(e2), 1.0)
    self.assertEqual(getter(e3), 'hi')

  def testMakePropertyGetter_getNonExistingProperty(self):
    getter = property_utils.make_property_getter('foo')
    a1 = metadata_store_pb2.Artifact()

    with self.assertRaises(KeyError):
      getter(a1)

  def testMakeCustomPropertyGetter_getArtifactProperty(self):
    getter = property_utils.make_custom_property_getter('foo')
    a1 = metadata_store_pb2.Artifact()
    a1.custom_properties['foo'].int_value = 1
    a2 = metadata_store_pb2.Artifact()
    a2.custom_properties['foo'].double_value = 1.0
    a3 = metadata_store_pb2.Artifact()
    a3.custom_properties['foo'].string_value = 'hi'

    self.assertEqual(getter(a1), 1)
    self.assertEqual(getter(a2), 1.0)
    self.assertEqual(getter(a3), 'hi')

  def testMakeCustomPropertyGetter_getDefaultProperty(self):
    getter = property_utils.make_custom_property_getter('foo')
    getter_or_int = property_utils.make_custom_property_getter('foo', 1)
    getter_or_float = property_utils.make_custom_property_getter('foo', 1.0)
    getter_or_str = property_utils.make_custom_property_getter('foo', 'meh')
    empty_artifact = metadata_store_pb2.Artifact()

    self.assertIsNone(getter(empty_artifact))
    self.assertEqual(getter_or_int(empty_artifact), 1)
    self.assertEqual(getter_or_float(empty_artifact), 1.0)
    self.assertEqual(getter_or_str(empty_artifact), 'meh')


if __name__ == '__main__':
  tf.test.main()
