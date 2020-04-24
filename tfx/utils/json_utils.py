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
"""Utilities to dump and load Jsonable object to/from JSONs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import importlib
import inspect
import json
from typing import Any, cast, Dict, List, Text, Type, Union

from six import with_metaclass

from google.protobuf import struct_pb2
from google.protobuf import json_format
from google.protobuf import message

# This is the special key to indicate the serialized object type.
# Depending on which, the utility knows how to deserialize it back to its
# original type.
_TFX_OBJECT_TYPE_KEY = '__tfx_object_type__'
_MODULE_KEY = '__module__'
_CLASS_KEY = '__class__'
_PROTO_VALUE_KEY = '__proto_value__'

RUNTIME_PARAMETER_PATTERN = (r'({\\*"__class__\\*": \\*"RuntimeParameter\\*", '
                             r'.*?})')


class _ObjectType(object):
  """Internal class to hold supported types."""
  # Indicates that the JSON dictionary is an instance of Jsonable type.
  # The dictionary has the states of the object and the object type info is
  # stored as __module__ and __class__ fields.
  JSONABLE = 'jsonable'
  # Indicates that the JSON dictionary is a python class.
  # The class info is stored as __module__ and __class__ fields in the
  # dictionary.
  CLASS = 'class'
  # Indicates that the JSON dictionary is an instance of a proto.Message
  # subclass. The class info of the proto python class is stored as __module__
  # and __class__ fields in the dictionary. The serialized value of the proto is
  # stored in the dictionary with key of _PROTO_VALUE_KEY.
  PROTO = 'proto'


class Jsonable(with_metaclass(abc.ABCMeta, object)):
  """Base class for serializing and deserializing objects to/from JSON.

  The default implementation assumes that the subclass can be restored by
  updating `self.__dict__` without invoking `self.__init__` function.. If the
  subclass cannot hold the assumption, it should
  override `to_json_dict` and `from_json_dict` to customize the implementation.
  """

  def to_json_dict(self) -> Dict[Text, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return self.__dict__

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    """Convert from dictionary data to an object."""
    instance = cls.__new__(cls)
    instance.__dict__ = dict_data
    return instance


JsonableValue = Union[bool, bytes, float, int, Jsonable, message.Message, Text,
                      Type]
JsonableList = List[JsonableValue]
JsonableDict = Dict[Union[bytes, Text], Union[JsonableValue, JsonableList]]
JsonableType = Union[JsonableValue, JsonableList, JsonableDict]


class _DefaultEncoder(json.JSONEncoder):
  """Default JSON Encoder which encodes Jsonable object to JSON."""

  def encode(self, obj: Any) -> Text:
    """Override encode to prevent redundant dumping."""
    if obj.__class__.__name__ == 'RuntimeParameter' and obj.ptype == Text:
      return self.default(obj)

    return super(_DefaultEncoder, self).encode(obj)

  def default(self, obj: Any) -> Any:
    # If obj is a str-typed RuntimeParameter, serialize it in place.
    if obj.__class__.__name__ == 'RuntimeParameter' and obj.ptype == Text:
      dict_data = {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.JSONABLE,
          _MODULE_KEY: obj.__class__.__module__,
          _CLASS_KEY: obj.__class__.__name__,
      }
      dict_data.update(obj.to_json_dict())
      return dumps(dict_data)

    if isinstance(obj, Jsonable):
      dict_data = {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.JSONABLE,
          _MODULE_KEY: obj.__class__.__module__,
          _CLASS_KEY: obj.__class__.__name__,
      }
      # Need to first check the existence of str-typed runtime parameter.
      data_patch = obj.to_json_dict()
      for k, v in data_patch.items():
        if v.__class__.__name__ == 'RuntimeParameter' and v.ptype == Text:
          data_patch[k] = dumps(v)
      dict_data.update(data_patch)
      return dict_data

    if inspect.isclass(obj):
      return {
          _TFX_OBJECT_TYPE_KEY: _ObjectType.CLASS,
          _MODULE_KEY: obj.__module__,
          _CLASS_KEY: obj.__name__,
      }

    if isinstance(obj, message.Message):
      return {
          _TFX_OBJECT_TYPE_KEY:
              _ObjectType.PROTO,
          _MODULE_KEY:
              obj.__class__.__module__,
          _CLASS_KEY:
              obj.__class__.__name__,
          _PROTO_VALUE_KEY:
              json_format.MessageToJson(
                  message=obj, sort_keys=True, preserving_proto_field_name=True)
      }

    return super(_DefaultEncoder, self).default(obj)


class _DefaultDecoder(json.JSONDecoder):
  """Default JSON Decoder which decodes JSON to Jsonable object."""

  def __init__(self, *args, **kwargs):
    super(_DefaultDecoder, self).__init__(
        object_hook=self._dict_to_object, *args, **kwargs)

  def _dict_to_object(self, dict_data: Dict[Text, Any]) -> Any:
    """Converts a dictionary to an object."""
    if _TFX_OBJECT_TYPE_KEY not in dict_data:
      return dict_data

    object_type = dict_data.pop(_TFX_OBJECT_TYPE_KEY)

    def _extract_class(d):
      module_name = d.pop(_MODULE_KEY)
      class_name = d.pop(_CLASS_KEY)
      return getattr(importlib.import_module(module_name), class_name)

    if object_type == _ObjectType.JSONABLE:
      jsonable_class_type = _extract_class(dict_data)
      if not issubclass(jsonable_class_type, Jsonable):
        raise ValueError('Class %s must be a subclass of Jsonable' %
                         jsonable_class_type)
      return jsonable_class_type.from_json_dict(dict_data)

    if object_type == _ObjectType.CLASS:
      return _extract_class(dict_data)

    if object_type == _ObjectType.PROTO:
      proto_class_type = _extract_class(dict_data)
      if not issubclass(proto_class_type, message.Message):
        raise ValueError('Class %s must be a subclass of proto.Message' %
                         proto_class_type)
      if _PROTO_VALUE_KEY not in dict_data:
        raise ValueError('Missing proto value in json dict')
      return json_format.Parse(dict_data[_PROTO_VALUE_KEY], proto_class_type())


def dumps(obj: Any) -> Text:
  """Dumps an object to JSON with Jsonable encoding."""
  return json.dumps(obj, cls=_DefaultEncoder, sort_keys=True)


def loads(s: Text) -> Any:
  """Loads a JSON into an object with Jsonable decoding."""
  return json.loads(s, cls=_DefaultDecoder)


# Json compatible python type definitions.
_PyJsonValue = Any  # TODO(jjong) Recursive type annotations not supported yet.
_PyJsonObject = Dict[Text, _PyJsonValue]
_PyJsonList = List[_PyJsonValue]
_PyJsonValue = Union[Text, int, float, bool, None, _PyJsonList, _PyJsonObject]


def Struct(input_dict: _PyJsonObject = None, **kwargs) -> struct_pb2.Struct:  # pylint: disable=invalid-name
  """Convenient wrapper for creating google.protobuf.Struct with dict.

  Struct is often used to configure arbitrary user-data into the component's
  execution properties within the proto buffer message. Normally user need to
  instantiate Struct message and then do parsing in two steps:

  ```python
  custom_config = google.protobuf.struct_pb2.Struct()
  json_format.ParseDict({
      'option_one': 'blah-blah'
      'option_two': 123
  }, custom_config)

  SomeComponent(custom_config=custom_config)
  ```

  This utility function helps to do it in one line:

  ```python
  SomeComponent(
      custom_config=json_utils.Struct({
          'option_one': 'blah-blah',
          'option_two': 123
      })
  )
  ```

  Or using a keyword arguments:

  ```python
  SomeComponent(
      custom_config=json_utils.Struct(
          option_one='blah-blah',
          option_two=123
      )
  )
  ```

  Args:
    input_dict: Input dict object to convert to Struct.
    **kwargs: Named arguments can be given instead of `input_dict`.

  Returns:
    google.protobuf.Struct proto message.
  """
  result = struct_pb2.Struct()
  result.update(input_dict or kwargs)
  return result


def struct_to_dict(struct: struct_pb2.Struct) -> _PyJsonObject:
  """Convert google.protobuf.Struct to python dict.

  google.protobuf.Struct stores all number as a IEEE 754 (float64) format, but
  it is mostly undesirable as a configuration object where integer is more
  frequent than floating values. This function explicitly cast integer number to
  an int type.

  Args:
    struct: google.protobuf.Struct proto message.

  Returns:
    Python dict with integer values converted to int type.
  """
  return cast(_PyJsonObject,
              _maybe_float_to_int(json_format.MessageToDict(struct)))


def _maybe_float_to_int(value: _PyJsonValue):
  if isinstance(value, float):
    return int(value) if value.is_integer() else value
  if isinstance(value, dict):
    return {k: _maybe_float_to_int(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_maybe_float_to_int(v) for v in value]
  return value
