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
"""TFX artifact type definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import builtins
import copy
import enum
import importlib
import json
from typing import Any, Dict, Optional, Text, Type

import absl
import tensorflow as tf

from tfx.utils import json_utils
from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class ArtifactState(object):
  """Enumeration of possible Artifact states."""

  # Indicates that there is a pending execution producing the artifact.
  PENDING = 'pending'
  # Indicates that the artifact ready to be consumed.
  PUBLISHED = 'published'
  # Indicates that the no data at the artifact uri, though the artifact is not
  # marked as deleted.
  MISSING = 'missing'
  # Indicates that the artifact should be garbage collected.
  MARKED_FOR_DELETION = 'MARKED_FOR_DELETION'
  # Indicates that the artifact has been garbage collected.
  DELETED = 'deleted'


# Default split of examples data.
DEFAULT_EXAMPLE_SPLITS = ['train', 'eval']

# Prefix for custom properties to prevent name collision.
# TODO(b/152444458): Revisit this part after we have a better aligned type
# system.
CUSTOM_PROPERTIES_PREFIX = 'custom:'


class PropertyType(enum.Enum):
  INT = 1
  FLOAT = 2
  STRING = 3


class Property(object):
  """Property specified for an Artifact."""
  _ALLOWED_MLMD_TYPES = {
      PropertyType.INT: metadata_store_pb2.INT,
      PropertyType.FLOAT: metadata_store_pb2.DOUBLE,
      PropertyType.STRING: metadata_store_pb2.STRING,
  }

  def __init__(self, type):  # pylint: disable=redefined-builtin
    if type not in Property._ALLOWED_MLMD_TYPES:
      raise ValueError('Property type must be one of %s.' %
                       list(Property._ALLOWED_MLMD_TYPES.keys()))
    self.type = type

  def mlmd_type(self):
    return Property._ALLOWED_MLMD_TYPES[self.type]


class Artifact(json_utils.Jsonable):
  """TFX artifact used for orchestration.

  This is used for type-checking and inter-component communication. Currently,
  it wraps a tuple of (ml_metadata.proto.Artifact,
  ml_metadata.proto.ArtifactType) with additional property accessors for
  internal state.

  A user may create a subclass of Artifact and override the TYPE_NAME property
  with the type for this artifact subclass. Users of the subclass may then omit
  the "type_name" field when construction the object.

  A user may specify artifact type-specific properties for an Artifact subclass
  by overriding the PROPERTIES dictionary, as detailed below.

  Note: the behavior of this class is experimental, without backwards
  compatibility guarantees, and may change in upcoming releases.
  """

  # String artifact type name used to identify the type in ML Metadata
  # database. Must be overridden by subclass.
  #
  # Example usage:
  #
  # TYPE_NAME = 'MyTypeName'
  TYPE_NAME = None

  # Optional dictionary of property name strings as keys and `Property`
  # objects as values, used to specify the artifact type's properties.
  # Subsequently, this artifact property may be accessed as Python attributes
  # of the artifact object.
  #
  # Example usage:
  #
  # PROPERTIES = {
  #   'span': Property(type=PropertyType.INT),
  #   # Comma separated of splits for an artifact. Empty string means artifact
  #   # has no split.
  #   'split_names': Property(type=PropertyType.STRING),
  # }
  #
  # Subsequently, these properties can be stored and accessed as
  # `myartifact.span` and `myartifact.split_name`, respectively.
  PROPERTIES = None

  # Initialization flag to support setattr / getattr behavior.
  _initialized = False

  def __init__(
      self,
      mlmd_artifact_type: Optional[metadata_store_pb2.ArtifactType] = None):
    """Construct an instance of Artifact.

    Used by TFX internal implementation: create an empty Artifact with
    type_name and optional split info specified. The remaining info will be
    filled in during compiling and running time. The Artifact should be
    transparent to end users and should not be initiated directly by pipeline
    users.

    Args:
      mlmd_artifact_type: Proto message defining the underlying ArtifactType.
        Optional and intended for internal use.
    """
    if self.__class__ == Artifact:
      if not mlmd_artifact_type:
        raise ValueError(
            'The "mlmd_artifact_type" argument must be passed to specify a '
            'type for this Artifact.')
      if not isinstance(mlmd_artifact_type, metadata_store_pb2.ArtifactType):
        raise ValueError(
            'The "mlmd_artifact_type" argument must be an instance of the '
            'proto message ml_metadata.proto.metadata_store_pb2.ArtifactType.')
    else:
      if mlmd_artifact_type:
        raise ValueError(
            'The "mlmd_artifact_type" argument must not be passed for '
            'Artifact subclass %s.' % self.__class__)
      mlmd_artifact_type = self._get_artifact_type()

    # MLMD artifact type proto object.
    self._artifact_type = mlmd_artifact_type
    # Underlying MLMD artifact proto object.
    self._artifact = metadata_store_pb2.Artifact()
    # Initialization flag to prevent recursive getattr / setattr errors.
    self._initialized = True

  @classmethod
  def _get_artifact_type(cls):
    if not getattr(cls, '_MLMD_ARTIFACT_TYPE', None):
      type_name = cls.TYPE_NAME
      if not (type_name and isinstance(type_name, (str, Text))):
        raise ValueError(
            ('The Artifact subclass %s must override the TYPE_NAME attribute '
             'with a string type name identifier (got %r instead).') %
            (cls, type_name))
      artifact_type = metadata_store_pb2.ArtifactType()
      artifact_type.name = type_name
      if cls.PROPERTIES:
        # Perform validation on PROPERTIES dictionary.
        if not isinstance(cls.PROPERTIES, dict):
          raise ValueError(
              'Artifact subclass %s.PROPERTIES is not a dictionary.' % cls)
        for key, value in cls.PROPERTIES.items():
          if not (isinstance(key,
                             (Text, bytes)) and isinstance(value, Property)):
            raise ValueError(
                ('Artifact subclass %s.PROPERTIES dictionary must have keys of '
                 'type string and values of type artifact.Property.') % cls)

        # Populate ML Metadata artifact properties dictionary.
        for key, value in cls.PROPERTIES.items():
          artifact_type.properties[key] = value.mlmd_type()
      cls._MLMD_ARTIFACT_TYPE = artifact_type
    return copy.deepcopy(cls._MLMD_ARTIFACT_TYPE)

  def __getattr__(self, name: Text) -> Any:
    """Custom __getattr__ to allow access to artifact properties."""
    if name == '_artifact_type':
      # Prevent infinite recursion when used with copy.deepcopy().
      raise AttributeError()
    if name not in self._artifact_type.properties:
      raise AttributeError('Artifact has no property %r.' % name)
    property_mlmd_type = self._artifact_type.properties[name]
    if property_mlmd_type == metadata_store_pb2.STRING:
      if name not in self._artifact.properties:
        # Avoid populating empty property protobuf with the [] operator.
        return ''
      return self._artifact.properties[name].string_value
    elif property_mlmd_type == metadata_store_pb2.INT:
      if name not in self._artifact.properties:
        # Avoid populating empty property protobuf with the [] operator.
        return 0
      return self._artifact.properties[name].int_value
    elif property_mlmd_type == metadata_store_pb2.DOUBLE:
      if name not in self._artifact.properties:
        # Avoid populating empty property protobuf with the [] operator.
        return 0.0
      return self._artifact.properties[name].double_value
    else:
      raise Exception('Unknown MLMD type %r for property %r.' %
                      (property_mlmd_type, name))

  def __setattr__(self, name: Text, value: Any):
    """Custom __setattr__ to allow access to artifact properties."""
    if not self._initialized:
      object.__setattr__(self, name, value)
      return
    if name not in self._artifact_type.properties:
      if (name in self.__dict__ or
          any(name in c.__dict__ for c in self.__class__.mro())):
        # Use any provided getter / setter if available.
        object.__setattr__(self, name, value)
        return
      # In the case where we do not handle this via an explicit getter /
      # setter, we assume that the user implied an artifact attribute store,
      # and we raise an exception since such an attribute was not explicitly
      # defined in the Artifact PROPERTIES dictionary.
      raise AttributeError('Cannot set unknown property %r on artifact %r.' %
                           (name, self))
    property_mlmd_type = self._artifact_type.properties[name]
    if property_mlmd_type == metadata_store_pb2.STRING:
      if not isinstance(value, (Text, bytes)):
        raise Exception(
            'Expected string value for property %r; got %r instead.' %
            (name, value))
      self._artifact.properties[name].string_value = value
    elif property_mlmd_type == metadata_store_pb2.INT:
      if not isinstance(value, int):
        raise Exception(
            'Expected integer value for property %r; got %r instead.' %
            (name, value))
      self._artifact.properties[name].int_value = value
    elif property_mlmd_type == metadata_store_pb2.DOUBLE:
      if not isinstance(value, float):
        raise Exception(
            'Expected integer value for property %r; got %r instead.' %
            (name, value))
      self._artifact.properties[name].double_value = value
    else:
      raise Exception('Unknown MLMD type %r for property %r.' %
                      (property_mlmd_type, name))

  def set_mlmd_artifact(self, artifact: metadata_store_pb2.Artifact):
    """Replace the MLMD artifact object on this artifact."""
    if not isinstance(artifact, metadata_store_pb2.Artifact):
      raise ValueError(
          ('Expected instance of metadata_store_pb2.Artifact, got %s '
           'instead.') % (artifact,))
    self._artifact = artifact

  def set_mlmd_artifact_type(self,
                             artifact_type: metadata_store_pb2.ArtifactType):
    """Set entire ArtifactType in this object."""
    if not isinstance(artifact_type, metadata_store_pb2.ArtifactType):
      raise ValueError(
          ('Expected instance of metadata_store_pb2.ArtifactType, got %s '
           'instead.') % (artifact_type,))
    self._artifact_type = artifact_type
    self._artifact.type_id = artifact_type.id

  def __repr__(self):
    return 'Artifact(artifact: {}, artifact_type: {})'.format(
        str(self._artifact), str(self._artifact_type))

  def to_json_dict(self) -> Dict[Text, Any]:
    return {
        'artifact':
            json.loads(
                json_format.MessageToJson(
                    message=self._artifact, preserving_proto_field_name=True)),
        'artifact_type':
            json.loads(
                json_format.MessageToJson(
                    message=self._artifact_type,
                    preserving_proto_field_name=True)),
        '__artifact_class_module__':
            self.__class__.__module__,
        '__artifact_class_name__':
            self.__class__.__name__,
    }

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    module_name = dict_data['__artifact_class_module__']
    class_name = dict_data['__artifact_class_name__']
    artifact = metadata_store_pb2.Artifact()
    artifact_type = metadata_store_pb2.ArtifactType()
    json_format.Parse(json.dumps(dict_data['artifact']), artifact)
    json_format.Parse(json.dumps(dict_data['artifact_type']), artifact_type)

    # First, try to resolve the specific class used for the artifact; if this
    # is not possible, use a generic artifact.Artifact object.
    result = None
    try:
      artifact_cls = getattr(importlib.import_module(module_name), class_name)
      # If the artifact type is the base Artifact class, do not construct the
      # object here since that constructor requires the mlmd_artifact_type
      # argument.
      if artifact_cls != Artifact:
        result = artifact_cls()
    except (AttributeError, ImportError, ValueError):
      absl.logging.warning((
          'Could not load artifact class %s.%s; using fallback deserialization '
          'for the relevant artifact. Please make sure that any artifact '
          'classes can be imported within your container or environment.') %
                           (module_name, class_name))
    if not result:
      result = Artifact(mlmd_artifact_type=artifact_type)
    result.set_mlmd_artifact_type(artifact_type)
    result.set_mlmd_artifact(artifact)
    return result

  # Read-only properties.
  @property
  def type(self):
    return self.__class__

  @property
  def type_name(self):
    return self._artifact_type.name

  @property
  def artifact_type(self):
    return self._artifact_type

  @property
  def mlmd_artifact(self):
    return self._artifact

  # Settable properties for all artifact types.
  @property
  def uri(self) -> Text:
    """Artifact URI."""
    return self._artifact.uri

  @uri.setter
  def uri(self, uri: Text):
    """Setter for artifact URI."""
    self._artifact.uri = uri

  @property
  def id(self) -> int:
    """Id of underlying artifact."""
    return self._artifact.id

  @id.setter
  def id(self, artifact_id: int):
    """Set id of underlying artifact."""
    self._artifact.id = artifact_id

  @property
  def type_id(self) -> int:
    """Id of underlying artifact type."""
    return self._artifact.type_id

  @type_id.setter
  def type_id(self, type_id: int):
    """Set id of underlying artifact type."""
    self._artifact.type_id = type_id

  # System-managed properties for all artifact types. Will be deprecated soon
  # in favor of a unified getter / setter interface and MLMD context.
  #
  # TODO(b/135056715): Rely on MLMD context for pipeline grouping for
  # artifacts once it's ready.
  #
  # The following system properties are used:
  #   - name: The name of the artifact, used to differentiate same type of
  #       artifact produced by the same component (in a subsequent change, this
  #       information will move to the associated ML Metadata Event object).
  #   - state: The state of an artifact; can be one of PENDING, PUBLISHED,
  #       MISSING, DELETING, DELETED (in a subsequent change, this information
  #       will move to a top-level ML Metadata Artifact attribute).
  #   - pipeline_name: The name of the pipeline that produces the artifact (in
  #       a subsequent change, this information will move to an associated ML
  #       Metadata Context attribute).
  #   - producer_component: The name of the component that produces the
  #       artifact (in a subsequent change, this information will move to the
  #       associated ML Metadata Event object).
  def _get_system_property(self, key: Text) -> Text:
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Legacy artifact types which have explicitly defined system properties.
      return self._artifact.properties[key].string_value
    return self._artifact.custom_properties[key].string_value

  def _set_system_property(self, key: Text, value: Text):
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Clear non-custom property in legacy artifact types.
      del self._artifact.properties[key]
    self._artifact.custom_properties[key].string_value = value

  @property
  def name(self) -> Text:
    """Name of the underlying artifact."""
    return self._get_system_property('name')

  @name.setter
  def name(self, name: Text):
    """Set name of the underlying artifact."""
    self._set_system_property('name', name)

  @property
  def state(self) -> Text:
    """State of the underlying artifact."""
    return self._get_system_property('state')

  @state.setter
  def state(self, state: Text):
    """Set state of the underlying artifact."""
    self._set_system_property('state', state)

  @property
  def pipeline_name(self) -> Text:
    """Name of the pipeline that produce the artifact."""
    return self._get_system_property('pipeline_name')

  @pipeline_name.setter
  def pipeline_name(self, pipeline_name: Text):
    """Set name of the pipeline that produce the artifact."""
    self._set_system_property('pipeline_name', pipeline_name)

  @property
  def producer_component(self) -> Text:
    """Producer component of the artifact."""
    return self._get_system_property('producer_component')

  @producer_component.setter
  def producer_component(self, producer_component: Text):
    """Set producer component of the artifact."""
    self._set_system_property('producer_component', producer_component)

  # Custom property accessors.
  def set_string_custom_property(self, key: Text, value: Text):
    """Set a custom property of string type."""
    self._artifact.custom_properties[key].string_value = value

  def set_int_custom_property(self, key: Text, value: int):
    """Set a custom property of int type."""
    self._artifact.custom_properties[key].int_value = builtins.int(value)

  def get_string_custom_property(self, key: Text) -> Text:
    """Get a custom property of string type."""
    return self._artifact.custom_properties[key].string_value

  def get_int_custom_property(self, key: Text) -> int:
    """Get a custom property of int type."""
    return self._artifact.custom_properties[key].int_value

  def copy_from(self, other: 'Artifact'):
    """Set uri, properties and custom properties from a given Artifact."""
    assert self.type is other.type, (
        'Unable to set properties from an artifact of different type: {} vs {}'
        .format(self.type_name, other.type_name))
    self.uri = other.uri

    self._artifact.properties.clear()
    self._artifact.properties.MergeFrom(other._artifact.properties)  # pylint: disable=protected-access
    self._artifact.custom_properties.clear()
    self._artifact.custom_properties.MergeFrom(
        other._artifact.custom_properties)  # pylint: disable=protected-access


class ValueArtifact(Artifact):
  """Artifacts of small scalar-values that can be easily loaded into memory."""

  def __init__(self, *args, **kwargs):
    self._has_value = False
    self._modified = False
    self._value = None
    super(ValueArtifact, self).__init__(*args, **kwargs)

  def read(self):
    if not self._has_value:
      file_path = self.uri
      # Assert there is a file exists.
      if not tf.io.gfile.exists(file_path):
        raise RuntimeError(
            'Given path does not exist or is not a valid file: %s' % file_path)

      serialized_value = tf.io.gfile.GFile(file_path, 'rb').read()
      self._has_value = True
      self._value = self.decode(serialized_value)
    return self._value

  def write(self, value):
    serialized_value = self.encode(value)
    tf.io.gfile.GFile(self.uri, 'wb').write(serialized_value)

  @property
  def value(self):
    if not self._has_value:
      raise ValueError('The artifact value has not yet been read from storage.')
    return self._value

  @value.setter
  def value(self, value):
    self._modified = True
    self._value = value
    self.write(value)

  # Note: behavior of decode() method should not be changed to provide
  # backward/forward compatibility.
  @abc.abstractmethod
  def decode(self, serialized_value) -> bytes:
    """Method decoding the file content. Implemented by subclasses."""
    pass

  # Note: behavior of encode() method should not be changed to provide
  # backward/forward compatibility.
  @abc.abstractmethod
  def encode(self, value) -> Any:
    """Method encoding the file content. Implemented by subclasses."""
    pass


def _ArtifactType(  # pylint: disable=invalid-name
    name: Optional[str] = None,  # pylint: disable=g-ambiguous-str-annotation
    properties: Optional[Dict[Text, Property]] = None,
    mlmd_artifact_type: Optional[metadata_store_pb2.ArtifactType] = None
) -> Type[Artifact]:
  """Experimental interface: internal use only.

  Construct an artifact type.

  Equivalent to subclassing Artifact and providing relevant properties. The user
  must either provide (1) a type "name" and "properties" or (2) a MLMD
  metadata_store_pb2.ArtifactType protobuf message as the "mlmd_artifact_type"
  parameter.

  Args:
    name: Name of the artifact type in MLMD. Must be provided unless a protobuf
      message is provided in the "mlmd_artifact_type" parameter.
    properties: Dictionary of properties mapping property name keys to
      `Parameter` object instances. Must be provided unless a protobuf message
      is provided in the "mlmd_artifact_type" parameter.
    mlmd_artifact_type: A ML Metadata metadata_store_pb2.ArtifactType protobuf
      message corresponding to the type being created.

  Returns:
    An Artifact class corresponding to the specified type.
  """
  if mlmd_artifact_type:
    if name or properties:
      raise ValueError(
          'The "name" and "properties" fields should not be passed when the '
          '"mlmd_artifact_type" parameter is set, in _ArtifactType call.')
    if not mlmd_artifact_type.name:
      raise ValueError('Artifact type proto must have "name" field set.')
    properties = {}
    for name, property_type in mlmd_artifact_type.properties.items():
      if property_type == metadata_store_pb2.PropertyType.INT:
        properties[name] = Property(PropertyType.INT)
      elif property_type == metadata_store_pb2.PropertyType.DOUBLE:
        properties[name] = Property(PropertyType.FLOAT)
      elif property_type == metadata_store_pb2.PropertyType.STRING:
        properties[name] = Property(PropertyType.STRING)
      else:
        raise ValueError('Unsupported MLMD property type: %s.' % property_type)
    return type(
        str(mlmd_artifact_type.name), (Artifact,), {
            'TYPE_NAME': mlmd_artifact_type.name,
            'PROPERTIES': properties,
        })
  else:
    if not name:
      raise ValueError(
          '"name" parameter must be passed to _ArtifactType when a '
          'metadata_store_pb2.ArtifactType object is not passed for the '
          '"mlmd_artifact_type" parameter.')
    return type(name, (Artifact,), {
        'TYPE_NAME': name,
        'PROPERTIES': properties
    })
