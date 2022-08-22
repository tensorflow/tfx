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

import decimal
import math

import absl
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from tfx.types.system_artifacts import Dataset
from tfx.types.system_artifacts import Model as SystemModel
from tfx.types.system_artifacts import Statistics
from tfx.types.value_artifact import ValueArtifact
from tfx.utils import json_utils

SPAN_PROPERTY = Property(type=PropertyType.INT)
VERSION_PROPERTY = Property(type=PropertyType.INT)
SPLIT_NAMES_PROPERTY = Property(type=PropertyType.STRING)
# (DEPRECATED. Do not use.) Value for a string-typed artifact.
STRING_VALUE_PROPERTY = Property(type=PropertyType.STRING)


class _TfxArtifact(Artifact):
  """TFX first-party component artifact definition.

  Do not construct directly, used for creating Channel, e.g.,
  ```
    Channel(type=standard_artifacts.Model)
  ```
  """

  def __init__(self, *args, **kwargs):
    """Construct TFX first-party component artifact."""
    # TODO(b/176795331): Refactor directory structure to make it clearer that
    # TFX-specific artifacts require the full "tfx" package be installed.
    #
    # Do not allow usage of TFX-specific artifact if only the core pipeline
    # SDK package is installed.
    try:
      import setuptools as _  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
      # Test import only when setuptools is available.
      try:
        # `extensions` is not included in ml_pipelines_sdk and doesn't have any
        # transitive import.
        import tfx.extensions as _  # type: ignore  # pylint: disable=g-import-not-at-top
      except ModuleNotFoundError as err:
        # The following condition detects exactly whether only the DSL package
        # is installed, and is bypassed when tests run in Bazel.
        raise RuntimeError('The "tfx" and all dependent packages need to be '
                           'installed to use this functionality.') from err
    except ModuleNotFoundError:
      pass

    super().__init__(*args, **kwargs)


class Examples(_TfxArtifact):
  """Artifact that contains the training data.

  Training data should be brought in to the TFX pipeline using components
  like ExampleGen. Data in Examples artifact is split and stored separately.
  The file and payload format must be specified as optional custom properties
  if not using default formats.
  Please see
  https://www.tensorflow.org/tfx/guide/examplegen#span_version_and_split to
  understand about span, version and splits.

  * Properties:
     - `span`: Integer to distinguish group of Examples.
     - `version`: Integer to represent updated data.
     - `split_names`: JSON string of the list of split names. For example,
        '["train", "test"]'. Empty string means artifact has no split.

  * File structure:
     - `{uri}/`
        - `Split-{split_name1}/`: Files for split
           - All direct children files are recognized as the data.
           - File format and payload format are determined by custom properties.
        - `Split-{split_name2}/`: Another split...

  * Commonly used custom properties of the Examples artifact:
    - `file_format`: a string that represents the file format. See
      tfx/components/util/tfxio_utils.py:make_tfxio for
      available values.
    - `payload_format`: int (enum) value of the data payload format.
      See tfx/proto/example_gen.proto:PayloadFormat for available formats.
  """
  TYPE_NAME = 'Examples'
  TYPE_ANNOTATION = Dataset
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'version': VERSION_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


class ExampleAnomalies(_TfxArtifact):
  TYPE_NAME = 'ExampleAnomalies'
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


class ExampleStatistics(_TfxArtifact):
  TYPE_NAME = 'ExampleStatistics'
  TYPE_ANNOTATION = Statistics
  PROPERTIES = {
      'span': SPAN_PROPERTY,
      'split_names': SPLIT_NAMES_PROPERTY,
  }


class ExamplesDiff(_TfxArtifact):
  TYPE_NAME = 'ExamplesDiff'


# TODO(b/158334890): deprecate ExternalArtifact.
class ExternalArtifact(_TfxArtifact):
  TYPE_NAME = 'ExternalArtifact'


class InferenceResult(_TfxArtifact):
  TYPE_NAME = 'InferenceResult'


class InfraBlessing(_TfxArtifact):
  TYPE_NAME = 'InfraBlessing'


class Model(_TfxArtifact):
  """Artifact that contains the actual persisted model.

  Training components stores the trained model like a saved model in this
  artifact. A `Model` artifact contains serialization of the trained model in
  one or more formats, each suitable for different usage (e.g. serving,
  evaluation), and serving environments.

  * File structure:
     - `{uri}/`
        - `Format-Serving/`: Model exported for serving.
           - `saved_model.pb`
           - Other actual model files.
        - `Format-TFMA/`: Model exported for evaluation.
           - `saved_model.pb`
           - Other actual model files.

  * Commonly used custom properties of the Model artifact:
  """
  TYPE_NAME = 'Model'
  TYPE_ANNOTATION = SystemModel


class ModelRun(_TfxArtifact):
  TYPE_NAME = 'ModelRun'


class ModelBlessing(_TfxArtifact):
  TYPE_NAME = 'ModelBlessing'


class ModelEvaluation(_TfxArtifact):
  TYPE_NAME = 'ModelEvaluation'


class PushedModel(_TfxArtifact):
  TYPE_NAME = 'PushedModel'
  TYPE_ANNOTATION = SystemModel


class Schema(_TfxArtifact):
  TYPE_NAME = 'Schema'


class TransformCache(_TfxArtifact):
  TYPE_NAME = 'TransformCache'


class JsonValue(ValueArtifact):
  """Artifacts representing a Jsonable value."""
  TYPE_NAME = 'JsonValue'

  def encode(self, value: json_utils.JsonableType) -> str:
    return json_utils.dumps(value)

  def decode(self, serialized_value: str) -> json_utils.JsonableType:
    return json_utils.loads(serialized_value)


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
  """String-typed artifact.

  String value artifacts are encoded using UTF-8.
  """
  TYPE_NAME = 'String'

  # Note, currently we enforce unicode-encoded string.
  def encode(self, value: str) -> bytes:
    if not isinstance(value, str):
      raise TypeError('Expecting Text but got value %s of type %s' %
                      (str(value), type(value)))
    return value.encode('utf-8')

  def decode(self, serialized_value: bytes) -> str:
    return serialized_value.decode('utf-8')


class Boolean(ValueArtifact):
  """Artifacts representing a boolean.

  Boolean value artifacts are encoded as "1" for True and "0" for False.
  """
  TYPE_NAME = 'Boolean'

  def encode(self, value: bool):
    if not isinstance(value, bool):
      raise TypeError('Expecting bytes but got value %s of type %s' %
                      (str(value), type(value)))
    return b'1' if value else b'0'

  def decode(self, serialized_value: bytes):
    return int(serialized_value) != 0


class Integer(ValueArtifact):
  """Integer-typed artifact.

  Integer value artifacts are encoded as a decimal string.
  """
  TYPE_NAME = 'Integer'

  def encode(self, value: int) -> bytes:
    if not isinstance(value, int):
      raise TypeError('Expecting int but got value %s of type %s' %
                      (str(value), type(value)))
    return str(value).encode('utf-8')

  def decode(self, serialized_value: bytes) -> int:
    return int(serialized_value)


class Float(ValueArtifact):
  """Float-typed artifact.

  Float value artifacts are encoded using Python str() class. However,
  Nan and Infinity are handled separately. See string constants in the
  class.
  """
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


class TransformGraph(_TfxArtifact):
  TYPE_NAME = 'TransformGraph'


class HyperParameters(_TfxArtifact):
  TYPE_NAME = 'HyperParameters'


class TunerResults(_TfxArtifact):
  TYPE_NAME = 'TunerResults'


# WIP and subject to change.
class DataView(_TfxArtifact):
  TYPE_NAME = 'DataView'
