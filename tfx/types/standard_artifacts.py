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
"""A set of standard TFX Artifact types.

Note: the artifact definitions here are expected to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decimal
import math
from typing import Text

import absl

from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from tfx.types.artifact import ValueArtifact

# Span for an artifact.
SPAN_PROPERTY = Property(type=PropertyType.INT)
# Version for an artifact.
VERSION_PROPERTY = Property(type=PropertyType.INT)
# Comma separated of splits for an artifact. Empty string means artifact
# has no split.
SPLIT_NAMES_PROPERTY = Property(type=PropertyType.STRING)
# Value for a string-typed artifact.
STRING_VALUE_PROPERTY = Property(type=PropertyType.STRING)


class Examples(Artifact):
  TYPE_NAME = 'Examples'
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'version': VERSION_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


class ExampleAnomalies(Artifact):
  TYPE_NAME = 'ExampleAnomalies'
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


class ExampleStatistics(Artifact):
  TYPE_NAME = 'ExampleStatistics'
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


# TODO(b/158334890): deprecate ExternalArtifact.
class ExternalArtifact(Artifact):
  TYPE_NAME = 'ExternalArtifact'


class InferenceResult(Artifact):
  TYPE_NAME = 'InferenceResult'


class InfraBlessing(Artifact):
  TYPE_NAME = 'InfraBlessing'


class Model(Artifact):
  TYPE_NAME = 'Model'


class ModelRun(Artifact):
  TYPE_NAME = 'ModelRun'


class ModelBlessing(Artifact):
  TYPE_NAME = 'ModelBlessing'


class ModelEvaluation(Artifact):
  TYPE_NAME = 'ModelEvaluation'


class PushedModel(Artifact):
  TYPE_NAME = 'PushedModel'


class Schema(Artifact):
  TYPE_NAME = 'Schema'


class Bytes(ValueArtifact):
  """Artifacts representing raw bytes."""
  TYPE_NAME = 'Bytes'

  def encode(self, value: bytes):
    if not isinstance(value, bytes):
      raise TypeError('Expecting bytes but got value %s of type %s' %
                      (str(value), type(value)))
    return value

  def decode(self, serialized_value: bytes):
    return serialized_value


class String(ValueArtifact):
  """String-typed artifact."""
  TYPE_NAME = 'String'

  # Note, currently we enforce unicode-encoded string.
  def encode(self, value: Text) -> bytes:
    if not isinstance(value, Text):
      raise TypeError('Expecting Text but got value %s of type %s' %
                      (str(value), type(value)))
    return value.encode('utf-8')

  def decode(self, serialized_value: bytes) -> Text:
    return serialized_value.decode('utf-8')


class Integer(ValueArtifact):
  """Integer-typed artifact."""
  TYPE_NAME = 'Integer'

  def encode(self, value: int) -> bytes:
    if not isinstance(value, int):
      raise TypeError('Expecting int but got value %s of type %s' %
                      (str(value), type(value)))
    return str(value).encode('utf-8')

  def decode(self, serialized_value: bytes) -> int:
    return int(serialized_value)


class Float(ValueArtifact):
  """Float-typed artifact."""
  TYPE_NAME = 'Float'

  _POSITIVE_INFINITY = float('Inf')
  _NEGATIVE_INFINITY = float('-Inf')

  _ENCODED_POSITIVE_INFINITY = 'Infinity'
  _ENCODED_NEGATIVE_INFINITY = '-Infinity'
  _ENCODED_NAN = 'NaN'

  def encode(self, value: float) -> bytes:
    if not isinstance(value, float):
      raise TypeError('Expecting float but got value %s of type %s' %
                      (str(value), type(value)))
    if math.isinf(value) or math.isnan(value):
      absl.logging.warning(
          '! The number "%s" may be unsupported by non-python components.' %
          value)
    str_value = str(value)
    # Special encoding for infinities and NaN to increase comatibility with
    # other languages.
    # Decoding works automatically.
    if math.isinf(value):
      if value >= 0:
        str_value = Float._ENCODED_POSITIVE_INFINITY
      else:
        str_value = Float._ENCODED_NEGATIVE_INFINITY
    if math.isnan(value):
      str_value = Float._ENCODED_NAN

    return str_value.encode('utf-8')

  def decode(self, serialized_value: bytes) -> float:
    result = float(serialized_value)

    # Check that the decoded value exactly matches the encoded string.
    # Note that float() can handle bytes, but Decimal() cannot.
    serialized_string = serialized_value.decode('utf-8')
    reserialized_string = str(result)
    is_exact = (decimal.Decimal(serialized_string) ==
                decimal.Decimal(reserialized_string))
    if not is_exact:
      absl.logging.warning(
          'The number "%s" has lost precision when converted to float "%s"' %
          (serialized_value, reserialized_string))

    return result


class TransformGraph(Artifact):
  TYPE_NAME = 'TransformGraph'


class HyperParameters(Artifact):
  TYPE_NAME = 'HyperParameters'


# WIP and subject to change.
class DataView(Artifact):
  TYPE_NAME = 'DataView'
