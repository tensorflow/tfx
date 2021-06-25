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
"""Tests for tfx.types.artifact."""

import json
import textwrap
from unittest import mock


import absl
import tensorflow as tf
from tfx.types import artifact
from tfx.types import value_artifact
from tfx.utils import json_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
      'int2': artifact.Property(type=artifact.PropertyType.INT),
      'float1': artifact.Property(type=artifact.PropertyType.FLOAT),
      'float2': artifact.Property(type=artifact.PropertyType.FLOAT),
      'string1': artifact.Property(type=artifact.PropertyType.STRING),
      'string2': artifact.Property(type=artifact.PropertyType.STRING),
  }

_MyArtifact2 = artifact._ArtifactType(  # pylint: disable=invalid-name
    name='MyTypeName2',
    properties={
        'int1': artifact.Property(type=artifact.PropertyType.INT),
        'int2': artifact.Property(type=artifact.PropertyType.INT),
        'float1': artifact.Property(type=artifact.PropertyType.FLOAT),
        'float2': artifact.Property(type=artifact.PropertyType.FLOAT),
        'string1': artifact.Property(type=artifact.PropertyType.STRING),
        'string2': artifact.Property(type=artifact.PropertyType.STRING),
        'jsonvalue_string':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_dict':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_int':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_float':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_list':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_null':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
        'jsonvalue_empty':
            artifact.Property(type=artifact.PropertyType.JSON_VALUE),
    })

_mlmd_artifact_type = metadata_store_pb2.ArtifactType()
json_format.Parse(
    json.dumps({
        'name': 'MyTypeName3',
        'properties': {
            'int1': 'INT',
            'int2': 'INT',
            'float1': 'DOUBLE',
            'float2': 'DOUBLE',
            'string1': 'STRING',
            'string2': 'STRING'
        }
    }), _mlmd_artifact_type)
_MyArtifact3 = artifact._ArtifactType(mlmd_artifact_type=_mlmd_artifact_type)  # pylint: disable=invalid-name


class _MyValueArtifact(value_artifact.ValueArtifact):
  TYPE_NAME = 'MyValueTypeName'

  def encode(self, value: str):
    assert isinstance(value, str), value
    return value.encode('utf-8')

  def decode(self, value: bytes):
    return value.decode('utf-8')


# Mock values for string artifact.
_STRING_VALUE = u'This is a string'
_BYTE_VALUE = b'This is a string'

# Mock paths for string artifact.
_VALID_URI = '/tmp/uri/value'
_VALID_FILE_URI = _VALID_URI

# Mock invalid paths. _BAD_URI points to a valid dir but there's no file within.
_BAD_URI = '/tmp/to/a/bad/dir'


class ArtifactTest(tf.test.TestCase):

  def testArtifact(self):
    instance = _MyArtifact()

    # Test property getters.
    self.assertEqual('', instance.uri)
    self.assertEqual(0, instance.id)
    self.assertEqual(0, instance.type_id)
    self.assertEqual('MyTypeName', instance.type_name)
    self.assertEqual('', instance.state)

    # Default property does not have span or split_names.
    with self.assertRaisesRegex(AttributeError, "has no property 'span'"):
      instance.span  # pylint: disable=pointless-statement
    with self.assertRaisesRegex(AttributeError,
                                "has no property 'split_names'"):
      instance.split_names  # pylint: disable=pointless-statement

    # Test property setters.
    instance.uri = '/tmp/uri2'
    self.assertEqual('/tmp/uri2', instance.uri)

    instance.id = 1
    self.assertEqual(1, instance.id)

    instance.type_id = 2
    self.assertEqual(2, instance.type_id)

    instance.state = artifact.ArtifactState.DELETED
    self.assertEqual(artifact.ArtifactState.DELETED, instance.state)

    # Default artifact does not have span.
    with self.assertRaisesRegex(AttributeError, "unknown property 'span'"):
      instance.span = 20190101
    # Default artifact does not have span.
    with self.assertRaisesRegex(AttributeError,
                                "unknown property 'split_names'"):
      instance.split_names = ''

    instance.set_int_custom_property('int_key', 20)
    self.assertEqual(
        20, instance.mlmd_artifact.custom_properties['int_key'].int_value)

    instance.set_string_custom_property('string_key', 'string_value')
    self.assertEqual(
        'string_value',
        instance.mlmd_artifact.custom_properties['string_key'].string_value)

    instance.set_float_custom_property('float_key', 0.5)
    self.assertEqual(
        0.5, instance.mlmd_artifact.custom_properties['float_key'].double_value)

    self.assertEqual(
        textwrap.dedent("""\
        Artifact(artifact: id: 1
        type_id: 2
        uri: "/tmp/uri2"
        custom_properties {
          key: "float_key"
          value {
            double_value: 0.5
          }
        }
        custom_properties {
          key: "int_key"
          value {
            int_value: 20
          }
        }
        custom_properties {
          key: "state"
          value {
            string_value: "deleted"
          }
        }
        custom_properties {
          key: "string_key"
          value {
            string_value: "string_value"
          }
        }
        , artifact_type: name: "MyTypeName"
        properties {
          key: "float1"
          value: DOUBLE
        }
        properties {
          key: "float2"
          value: DOUBLE
        }
        properties {
          key: "int1"
          value: INT
        }
        properties {
          key: "int2"
          value: INT
        }
        properties {
          key: "string1"
          value: STRING
        }
        properties {
          key: "string2"
          value: STRING
        }
        )"""), str(instance))

    # Test json serialization.
    json_dict = json_utils.dumps(instance)
    other_instance = json_utils.loads(json_dict)
    self.assertEqual(instance.mlmd_artifact, other_instance.mlmd_artifact)
    self.assertEqual(instance.artifact_type, other_instance.artifact_type)

  def testArtifactTypeFunctionAndProto(self):
    # Test usage of _MyArtifact2 and _MyArtifact3, which were defined using the
    # _ArtifactType function.
    types_and_names = [
        (_MyArtifact2, 'MyTypeName2'),
        (_MyArtifact3, 'MyTypeName3'),
    ]
    for type_cls, name in types_and_names:
      self.assertEqual(type_cls.TYPE_NAME, name)
      my_artifact = type_cls()
      self.assertEqual(0, my_artifact.int1)
      self.assertEqual(0, my_artifact.int2)
      my_artifact.int1 = 111
      my_artifact.int2 = 222
      self.assertEqual(0.0, my_artifact.float1)
      self.assertEqual(0.0, my_artifact.float2)
      my_artifact.float1 = 111.1
      my_artifact.float2 = 222.2
      self.assertEqual('', my_artifact.string1)
      self.assertEqual('', my_artifact.string2)
      my_artifact.string1 = '111'
      my_artifact.string2 = '222'
      self.assertEqual(my_artifact.int1, 111)
      self.assertEqual(my_artifact.int2, 222)
      self.assertEqual(my_artifact.float1, 111.1)
      self.assertEqual(my_artifact.float2, 222.2)
      self.assertEqual(my_artifact.string1, '111')
      self.assertEqual(my_artifact.string2, '222')

  def testArtifactJsonValue(self):
    # Construct artifact.
    my_artifact = _MyArtifact2()
    my_artifact.jsonvalue_string = 'aaa'
    my_artifact.jsonvalue_dict = {'k1': ['v1', 'v2', 333]}
    my_artifact.jsonvalue_int = 123
    my_artifact.jsonvalue_float = 3.14
    my_artifact.jsonvalue_list = ['a1', '2', 3, {'4': 5.0}]
    my_artifact.jsonvalue_null = None
    my_artifact.set_json_value_custom_property('customjson1', {})
    my_artifact.set_json_value_custom_property('customjson2', ['a', 'b', 3])
    my_artifact.set_json_value_custom_property('customjson3', 'xyz')
    my_artifact.set_json_value_custom_property('customjson4', 3.14)

    # Test string and proto serialization.
    self.assertEqual(
        textwrap.dedent("""\
        Artifact(artifact: properties {
          key: "jsonvalue_dict"
          value {
            struct_value {
              fields {
                key: "k1"
                value {
                  list_value {
                    values {
                      string_value: "v1"
                    }
                    values {
                      string_value: "v2"
                    }
                    values {
                      number_value: 333.0
                    }
                  }
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_float"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  number_value: 3.14
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_int"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  number_value: 123.0
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_list"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  list_value {
                    values {
                      string_value: "a1"
                    }
                    values {
                      string_value: "2"
                    }
                    values {
                      number_value: 3.0
                    }
                    values {
                      struct_value {
                        fields {
                          key: "4"
                          value {
                            number_value: 5.0
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_string"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  string_value: "aaa"
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson1"
          value {
            struct_value {
            }
          }
        }
        custom_properties {
          key: "customjson2"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  list_value {
                    values {
                      string_value: "a"
                    }
                    values {
                      string_value: "b"
                    }
                    values {
                      number_value: 3.0
                    }
                  }
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson3"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  string_value: "xyz"
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson4"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  number_value: 3.14
                }
              }
            }
          }
        }
        , artifact_type: name: "MyTypeName2"
        properties {
          key: "float1"
          value: DOUBLE
        }
        properties {
          key: "float2"
          value: DOUBLE
        }
        properties {
          key: "int1"
          value: INT
        }
        properties {
          key: "int2"
          value: INT
        }
        properties {
          key: "jsonvalue_dict"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_empty"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_float"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_int"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_list"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_null"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_string"
          value: STRUCT
        }
        properties {
          key: "string1"
          value: STRING
        }
        properties {
          key: "string2"
          value: STRING
        }
        )"""), str(my_artifact))

    copied_artifact = _MyArtifact2()
    copied_artifact.set_mlmd_artifact(my_artifact.mlmd_artifact)

    self.assertEqual(copied_artifact.jsonvalue_string, 'aaa')
    self.assertEqual(
        json.dumps(copied_artifact.jsonvalue_dict),
        '{"k1": ["v1", "v2", 333.0]}')
    self.assertEqual(copied_artifact.jsonvalue_int, 123.0)
    self.assertEqual(copied_artifact.jsonvalue_float, 3.14)
    self.assertEqual(
        json.dumps(copied_artifact.jsonvalue_list),
        '["a1", "2", 3.0, {"4": 5.0}]')
    self.assertIsNone(copied_artifact.jsonvalue_null)
    self.assertIsNone(copied_artifact.jsonvalue_empty)
    self.assertEqual(
        json.dumps(
            copied_artifact.get_json_value_custom_property('customjson1')),
        '{}')
    self.assertEqual(
        json.dumps(
            copied_artifact.get_json_value_custom_property('customjson2')),
        '["a", "b", 3.0]')
    self.assertEqual(
        copied_artifact.get_string_custom_property('customjson2'), '')
    self.assertEqual(copied_artifact.get_int_custom_property('customjson2'), 0)
    self.assertEqual(
        copied_artifact.get_float_custom_property('customjson2'), 0.0)
    self.assertEqual(
        json.dumps(copied_artifact.get_custom_property('customjson2')),
        '["a", "b", 3.0]')
    self.assertEqual(
        copied_artifact.get_json_value_custom_property('customjson3'), 'xyz')
    self.assertEqual(
        copied_artifact.get_string_custom_property('customjson3'), 'xyz')
    self.assertEqual(copied_artifact.get_custom_property('customjson3'), 'xyz')
    self.assertEqual(
        copied_artifact.get_json_value_custom_property('customjson4'), 3.14)
    self.assertEqual(
        copied_artifact.get_float_custom_property('customjson4'), 3.14)
    self.assertEqual(copied_artifact.get_int_custom_property('customjson4'), 3)
    self.assertEqual(copied_artifact.get_custom_property('customjson4'), 3.14)

    # Modify nested structure and check proto serialization reflects changes.
    copied_artifact.jsonvalue_dict['k1'].append({'4': 'x'})
    copied_artifact.jsonvalue_dict['k2'] = 'y'
    copied_artifact.jsonvalue_dict['k3'] = None
    copied_artifact.jsonvalue_int = None
    copied_artifact.jsonvalue_list.append([6, '7'])
    copied_artifact.get_json_value_custom_property('customjson1')['y'] = ['z']
    copied_artifact.get_json_value_custom_property('customjson2').append(4)

    self.assertEqual(
        textwrap.dedent("""\
        Artifact(artifact: properties {
          key: "jsonvalue_dict"
          value {
            struct_value {
              fields {
                key: "k1"
                value {
                  list_value {
                    values {
                      string_value: "v1"
                    }
                    values {
                      string_value: "v2"
                    }
                    values {
                      number_value: 333.0
                    }
                    values {
                      struct_value {
                        fields {
                          key: "4"
                          value {
                            string_value: "x"
                          }
                        }
                      }
                    }
                  }
                }
              }
              fields {
                key: "k2"
                value {
                  string_value: "y"
                }
              }
              fields {
                key: "k3"
                value {
                  null_value: NULL_VALUE
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_float"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  number_value: 3.14
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_list"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  list_value {
                    values {
                      string_value: "a1"
                    }
                    values {
                      string_value: "2"
                    }
                    values {
                      number_value: 3.0
                    }
                    values {
                      struct_value {
                        fields {
                          key: "4"
                          value {
                            number_value: 5.0
                          }
                        }
                      }
                    }
                    values {
                      list_value {
                        values {
                          number_value: 6.0
                        }
                        values {
                          string_value: "7"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        properties {
          key: "jsonvalue_string"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  string_value: "aaa"
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson1"
          value {
            struct_value {
              fields {
                key: "y"
                value {
                  list_value {
                    values {
                      string_value: "z"
                    }
                  }
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson2"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  list_value {
                    values {
                      string_value: "a"
                    }
                    values {
                      string_value: "b"
                    }
                    values {
                      number_value: 3.0
                    }
                    values {
                      number_value: 4.0
                    }
                  }
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson3"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  string_value: "xyz"
                }
              }
            }
          }
        }
        custom_properties {
          key: "customjson4"
          value {
            struct_value {
              fields {
                key: "__value__"
                value {
                  number_value: 3.14
                }
              }
            }
          }
        }
        , artifact_type: name: "MyTypeName2"
        properties {
          key: "float1"
          value: DOUBLE
        }
        properties {
          key: "float2"
          value: DOUBLE
        }
        properties {
          key: "int1"
          value: INT
        }
        properties {
          key: "int2"
          value: INT
        }
        properties {
          key: "jsonvalue_dict"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_empty"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_float"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_int"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_list"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_null"
          value: STRUCT
        }
        properties {
          key: "jsonvalue_string"
          value: STRUCT
        }
        properties {
          key: "string1"
          value: STRING
        }
        properties {
          key: "string2"
          value: STRING
        }
        )"""), str(copied_artifact))

  def testInvalidArtifact(self):
    with self.assertRaisesRegex(
        ValueError, 'The "mlmd_artifact_type" argument must be passed'):
      artifact.Artifact()

    class MyBadArtifact(artifact.Artifact):
      # No TYPE_NAME
      pass

    with self.assertRaisesRegex(
        ValueError,
        'The Artifact subclass .* must override the TYPE_NAME attribute '):
      MyBadArtifact()

    class MyNewArtifact(artifact.Artifact):
      TYPE_NAME = 'MyType'

    # Okay without additional type_name argument.
    MyNewArtifact()

    # Not okay to pass type_name on subclass.
    with self.assertRaisesRegex(
        ValueError,
        'The "mlmd_artifact_type" argument must not be passed for Artifact '
        'subclass'):
      MyNewArtifact(mlmd_artifact_type=metadata_store_pb2.ArtifactType())

  def testArtifactProperties(self):
    my_artifact = _MyArtifact()
    self.assertEqual(0, my_artifact.int1)
    self.assertEqual(0, my_artifact.int2)
    my_artifact.int1 = 111
    my_artifact.int2 = 222
    self.assertEqual('', my_artifact.string1)
    self.assertEqual('', my_artifact.string2)
    my_artifact.string1 = '111'
    my_artifact.string2 = '222'
    self.assertEqual(my_artifact.int1, 111)
    self.assertEqual(my_artifact.int2, 222)
    self.assertEqual(my_artifact.string1, '111')
    self.assertEqual(my_artifact.string2, '222')
    self.assertEqual(my_artifact.get_string_custom_property('invalid'), '')
    self.assertEqual(my_artifact.get_int_custom_property('invalid'), 0)
    self.assertIsNone(my_artifact.get_custom_property('invalid'))
    self.assertNotIn('invalid', my_artifact._artifact.custom_properties)

    with self.assertRaisesRegex(
        AttributeError, "Cannot set unknown property 'invalid' on artifact"):
      my_artifact.invalid = 1

    with self.assertRaisesRegex(
        AttributeError, "Cannot set unknown property 'invalid' on artifact"):
      my_artifact.invalid = 'x'

    with self.assertRaisesRegex(AttributeError,
                                "Artifact has no property 'invalid'"):
      my_artifact.invalid  # pylint: disable=pointless-statement

  def testStringTypeNameNotAllowed(self):
    with self.assertRaisesRegex(
        ValueError,
        'The "mlmd_artifact_type" argument must be an instance of the proto '
        'message'):
      artifact.Artifact('StringTypeName')

  @mock.patch('absl.logging.warning')
  def testDeserialize(self, *unused_mocks):
    original = _MyArtifact()
    original.uri = '/my/path'
    original.int1 = 111
    original.int2 = 222
    original.string1 = '111'
    original.string2 = '222'

    serialized = original.to_json_dict()

    rehydrated = artifact.Artifact.from_json_dict(serialized)
    absl.logging.warning.assert_not_called()
    self.assertIs(rehydrated.__class__, _MyArtifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

  @mock.patch('absl.logging.warning')
  def testDeserializeUnknownArtifactClass(self, *unused_mocks):
    original = _MyArtifact()
    original.uri = '/my/path'
    original.int1 = 111
    original.int2 = 222
    original.string1 = '111'
    original.string2 = '222'

    serialized = original.to_json_dict()
    serialized['__artifact_class_name__'] = 'MissingClassName'

    rehydrated = artifact.Artifact.from_json_dict(serialized)
    absl.logging.warning.assert_called_once()
    self.assertIs(rehydrated.__class__, artifact.Artifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

    serialized2 = rehydrated.to_json_dict()
    rehydrated = artifact.Artifact.from_json_dict(serialized2)
    self.assertIs(rehydrated.__class__, artifact.Artifact)
    self.assertEqual(rehydrated.int1, 111)
    self.assertEqual(rehydrated.int2, 222)
    self.assertEqual(rehydrated.string1, '111')
    self.assertEqual(rehydrated.string2, '222')

  def testCopyFrom(self):
    original = _MyArtifact()
    original.id = 1
    original.uri = '/my/path'
    original.int1 = 111
    original.string1 = '111'
    original.set_string_custom_property('my_custom_property', 'aaa')

    copied = _MyArtifact()
    copied.id = 2
    copied.uri = '/some/other/path'
    copied.int1 = 333
    original.set_string_custom_property('my_custom_property', 'bbb')
    copied.copy_from(original)

    # id should not be overridden.
    self.assertEqual(copied.id, 2)
    self.assertEqual(original.uri, copied.uri)
    self.assertEqual(original.int1, copied.int1)
    self.assertEqual(original.string1, copied.string1)
    self.assertEqual(
        original.get_string_custom_property('my_custom_property'),
        copied.get_string_custom_property('my_custom_property'))

  def testCopyFromDifferentArtifactType(self):
    artifact1 = _MyArtifact()
    artifact2 = _MyArtifact2()
    with self.assertRaises(AssertionError):
      artifact2.copy_from(artifact1)


if __name__ == '__main__':
  tf.test.main()
