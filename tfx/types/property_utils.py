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
"""Utiltiy for MLMD type properties and custom_properties."""

import collections
from typing import Union, Type, Text, MutableMapping, Optional, Mapping, TypeVar, Callable, Any

from ml_metadata.proto import metadata_store_pb2

_PropertyValue = Union[int, float, Text]
_PropertyType = Union[Type[int], Type[float], Type[Text]]
_T = TypeVar('_T')


# MutableMapping automatically populates other dict methods if we already have
# defined __getitem__, __setitem__, and __delitem__.
class PropertyMapProxy(collections.MutableMapping):
  """Proxy dict for `properties` and `custom_properties`.

  The proxy provides convenient usage by automatically wrapping or unwrapping
  `metadata_store_pb2.Value` while setting or getting the value. If schema is
  provided, it can also
  """

  def __init__(self, data: MutableMapping[Text, metadata_store_pb2.Value],
               schema: Optional[Mapping[Text, _PropertyType]] = None):
    self._data = data
    self._schema = schema

  @staticmethod
  def _unwrap_value(value: metadata_store_pb2.Value) -> _PropertyValue:
    value_type = value.WhichOneof('value')
    if not value_type:
      raise ValueError('property value not set')
    # List all 'if' branches to let type checker test it.
    if value_type == 'int_value':
      return value.int_value
    elif value_type == 'double_value':
      return value.double_value
    elif value_type == 'string_value':
      return value.string_value
    else:
      raise ValueError('Unknown value')

  @staticmethod
  def _wrap_value(value: _PropertyValue) -> metadata_store_pb2.Value:
    if isinstance(value, int):
      return metadata_store_pb2.Value(int_value=value)
    elif isinstance(value, float):
      return metadata_store_pb2.Value(double_value=value)
    elif isinstance(value, Text):
      return metadata_store_pb2.Value(string_value=value)
    else:
      raise TypeError('Unsupported type {}'.format(type(value)))

  def _get_type_from_schema(self, key: Text) -> _PropertyType:
    if key not in self._schema:
      raise KeyError('Key {} does not exist'.format(key))
    return self._schema[key]

  def __iter__(self):
    return iter(self._data)

  def __len__(self):
    return len(self._data)

  def get(self, key: Text, default: _T = None) -> Union[_T, _PropertyValue]:
    # Check existence using __contains__ first to prevent populating empty
    # mapping via __getitem__.
    if key not in self._data:
      if self._schema is not None:
        if default is not None:
          raise ValueError('Cannot specify default value if schema is set.')
        # Should return empty proto value. Use python type empty constructor
        # since int() == 0, float() == 0.0, str() == ''.
        return self._get_type_from_schema(key)()
      else:
        return default
    else:
      return self._unwrap_value(self._data[key])

  def __getitem__(self, key: Text) -> _PropertyValue:
    result = self.get(key)
    if result is None:
      raise KeyError('Key {} does not exist'.format(key))
    return result

  def __setitem__(self, key: Text, value: _PropertyValue):
    if self._schema is not None:
      expected_type = self._get_type_from_schema(key)
      if not isinstance(value, expected_type):
        raise TypeError('Cannot set {} to {}. Expected {} type.'
                        .format(key, value, expected_type))
    self._data[key].CopyFrom(self._wrap_value(value))

  def __delitem__(self, key: Text):
    del self._data[key]


def _make_proxy(
    data: MutableMapping[Text, metadata_store_pb2.Value]) -> PropertyMapProxy:
  if isinstance(data, PropertyMapProxy):
    return data
  return PropertyMapProxy(data)


def make_property_getter(
    property_name: Text) -> Callable[[Any], _PropertyValue]:
  def getter(obj: Any) -> _PropertyValue:
    return _make_proxy(obj.properties)[property_name]
  return getter


def make_custom_property_getter(
    property_name: Text,
    default: _T = None) -> Callable[[Any], Union[_T, _PropertyValue]]:
  def getter(obj: Any) -> Union[_T, _PropertyValue]:
    return _make_proxy(obj.custom_properties).get(property_name, default)
  return getter
