# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Property definitions for the TFX Artifact type."""

import enum

from ml_metadata.proto import metadata_store_pb2

# TODO(b/241861488): Remove safeguard once fully supported by MLMD
ENABLE_PROTO_PROPERTIES = False


class PropertyType(enum.Enum):
  """Type of an artifact property."""
  # Integer value.
  INT = 1
  # Double floating point value.
  FLOAT = 2
  # String value.
  STRING = 3
  # JSON value: a dictionary, list, string, floating point or boolean value.
  # When possible, prefer use of INT, FLOAT or STRING instead, since JSON_VALUE
  # type values may not be directly used in MLMD queries.
  # Note: when a dictionary value is used, the top-level "__value__" key is
  # reserved.
  JSON_VALUE = 4
  # TODO(b/241861488): Update comment once proto property is fully supported
  # Protocol buffer. (NOT YET SUPPORTED)
  PROTO = 5


class Property:
  """Property specified for an Artifact."""
  _ALLOWED_MLMD_TYPES = {
      PropertyType.INT: metadata_store_pb2.INT,
      PropertyType.FLOAT: metadata_store_pb2.DOUBLE,
      PropertyType.STRING: metadata_store_pb2.STRING,
      PropertyType.JSON_VALUE: metadata_store_pb2.STRUCT,
      PropertyType.PROTO: metadata_store_pb2.PROTO,
  }

  def __init__(self, type):  # pylint: disable=redefined-builtin
    if type not in Property._ALLOWED_MLMD_TYPES:
      raise ValueError('Property type must be one of %s.' %
                       list(Property._ALLOWED_MLMD_TYPES.keys()))
    # TODO(b/241861488): Remove safeguard once fully supported by MLMD.
    if (type == PropertyType.PROTO and not ENABLE_PROTO_PROPERTIES):
      raise ValueError('Proto properties are not yet supported')
    self.type = type

  def mlmd_type(self):
    return Property._ALLOWED_MLMD_TYPES[self.type]

  def __repr__(self):
    return str(self.type)
