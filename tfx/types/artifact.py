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

import builtins
import copy
import importlib
import json
from typing import Any, Dict, List, Optional, Type, Union

from absl import logging
from tfx.types import artifact_property
from tfx.types.system_artifacts import SystemArtifact
from tfx.utils import doc_controls
from tfx.utils import json_utils
from tfx.utils import proto_utils

from google.protobuf import struct_pb2
from google.protobuf import json_format
from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

Property = artifact_property.Property
PropertyType = artifact_property.PropertyType


class ArtifactState:
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


JsonValueType = Union[Dict, List, int, float, type(None), str]
_JSON_SINGLE_VALUE_KEY = '__value__'


def _decode_struct_value(
    struct_value: Optional[struct_pb2.Struct]) -> JsonValueType:
  if struct_value is None:
    return None
  result = json_format.MessageToDict(struct_value)
  if _JSON_SINGLE_VALUE_KEY in result:
    return result[_JSON_SINGLE_VALUE_KEY]
  else:
    return result


def _encode_struct_value(value: JsonValueType) -> Optional[struct_pb2.Struct]:
  if value is None:
    return None
  if not isinstance(value, dict):
    value = {
        _JSON_SINGLE_VALUE_KEY: value,
    }
  result = struct_pb2.Struct()
  json_format.ParseDict(value, result)
  return result


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

  # The system artifact class used to annotate the artifact type. It is a
  # subclass of SystemArtifact.
  # These subclasses (system artifact classses) are defined in
  # third_party/py/tfx/types/system_artifacts.py.
  #
  # Example usage:
  #
  # TYPE_ANNOTATION = system_artifacts.Dataset
  TYPE_ANNOTATION: Type[SystemArtifact] = None

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
    # When list/dict JSON or proto value properties are read, it is possible
    # they will be modified without knowledge of this class. Therefore,
    # deserialized values need to be cached here and reserialized into the
    # metadata proto when requested.
    self._cached_modifiable_properties = {}
    self._cached_modifiable_custom_properties = {}
    # Initialization flag to prevent recursive getattr / setattr errors.
    self._initialized = True

  @classmethod
  def _get_artifact_type(cls):
    existing_artifact_type = getattr(cls, '_MLMD_ARTIFACT_TYPE', None)
    if (not existing_artifact_type) or (cls.TYPE_NAME !=
                                        existing_artifact_type.name):
      type_name = cls.TYPE_NAME
      if not (type_name and isinstance(type_name, str)):
        raise ValueError(
            ('The Artifact subclass %s must override the TYPE_NAME attribute '
             'with a string type name identifier (got %r instead).') %
            (cls, type_name))
      artifact_type = metadata_store_pb2.ArtifactType()
      artifact_type.name = type_name

      # Populate ML Metadata artifact properties dictionary.
      if cls.PROPERTIES:
        # Perform validation on PROPERTIES dictionary.
        if not isinstance(cls.PROPERTIES, dict):
          raise ValueError(
              'Artifact subclass %s.PROPERTIES is not a dictionary.' % cls)
        for key, value in cls.PROPERTIES.items():
          if not (isinstance(key,
                             (str, bytes)) and isinstance(value, Property)):
            raise ValueError(
                ('Artifact subclass %s.PROPERTIES dictionary must have keys of '
                 'type string and values of type artifact.Property.') % cls)

        for key, value in cls.PROPERTIES.items():
          artifact_type.properties[key] = value.mlmd_type()

      # Populate ML Metadata artifact type field: `base_type`.
      type_annotation_cls = cls.TYPE_ANNOTATION
      if type_annotation_cls:
        if not issubclass(type_annotation_cls, SystemArtifact):
          raise ValueError(
              'TYPE_ANNOTATION %s is not a subclass of SystemArtifact.' %
              type_annotation_cls)
        if type_annotation_cls.MLMD_SYSTEM_BASE_TYPE:
          artifact_type.base_type = type_annotation_cls.MLMD_SYSTEM_BASE_TYPE

      cls._MLMD_ARTIFACT_TYPE = artifact_type
    return copy.deepcopy(cls._MLMD_ARTIFACT_TYPE)

  def __getattr__(self, name: str) -> Any:
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
    elif property_mlmd_type == metadata_store_pb2.STRUCT:
      if name not in self._artifact.properties:
        # Avoid populating empty property protobuf with the [] operator.
        return None
      if name in self._cached_modifiable_properties:
        return self._cached_modifiable_properties[name]
      value = _decode_struct_value(self._artifact.properties[name].struct_value)
      # We must cache the decoded lists or dictionaries returned here so that
      # if their recursive contents are modified, the Metadata proto message
      # can be updated to reflect this.
      if isinstance(value, (dict, list)):
        self._cached_modifiable_properties[name] = value
      return value
    elif property_mlmd_type == metadata_store_pb2.PROTO:
      if name not in self._artifact.properties:
        # Avoid populating empty property protobuf with the [] operator.
        return None
      if name in self._cached_modifiable_properties:
        return self._cached_modifiable_properties[name]
      value = proto_utils.unpack_proto_any(
          self._artifact.properties[name].proto_value)
      # We must cache the protobuf message here so that if its contents are
      # modified, the Metadata proto message can be updated to reflect this.
      self._cached_modifiable_properties[name] = value
      return value
    else:
      raise Exception('Unknown MLMD type %r for property %r.' %
                      (property_mlmd_type, name))

  def __setattr__(self, name: str, value: Any):
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
      if not isinstance(value, (str, bytes)):
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
            'Expected float value for property %r; got %r instead.' %
            (name, value))
      self._artifact.properties[name].double_value = value
    elif property_mlmd_type == metadata_store_pb2.STRUCT:
      if not isinstance(value, (dict, list, str, float, int, type(None))):
        raise Exception(
            ('Expected JSON value (dict, list, string, float, int or None) '
             'for property %r; got %r instead.') % (name, value))
      encoded_value = _encode_struct_value(value)
      if encoded_value is None:
        self._artifact.properties[name].struct_value.Clear()
      else:
        self._artifact.properties[name].struct_value.CopyFrom(encoded_value)
      self._cached_modifiable_properties[name] = value
    elif property_mlmd_type == metadata_store_pb2.PROTO:
      if not isinstance(value, (message.Message, type(None))):
        raise Exception(
            'Expected protobuf message value or None for property %r; got %r '
            'instead.' % (name, value))
      if value is None:
        self._artifact.properties[name].proto_value.Clear()
      else:
        self._artifact.properties[name].proto_value.Pack(value)
      self._cached_modifiable_properties[name] = value
    else:
      raise Exception('Unknown MLMD type %r for property %r.' %
                      (property_mlmd_type, name))

  @doc_controls.do_not_doc_inheritable
  def set_mlmd_artifact(self, artifact: metadata_store_pb2.Artifact):
    """Replace the MLMD artifact object on this artifact."""
    if not isinstance(artifact, metadata_store_pb2.Artifact):
      raise ValueError(
          ('Expected instance of metadata_store_pb2.Artifact, got %s '
           'instead.') % (artifact,))
    self._artifact = artifact
    self._cached_modifiable_properties = {}
    self._cached_modifiable_custom_properties = {}

  @doc_controls.do_not_doc_inheritable
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
        str(self.mlmd_artifact), str(self._artifact_type))

  @doc_controls.do_not_doc_inheritable
  def to_json_dict(self) -> Dict[str, Any]:
    return {
        'artifact':
            json.loads(
                json_format.MessageToJson(
                    message=self.mlmd_artifact,
                    preserving_proto_field_name=True)),
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
  @doc_controls.do_not_doc_inheritable
  def from_json_dict(cls, dict_data: Dict[str, Any]) -> Any:
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
      logging.warning((
          'Could not load artifact class %s.%s; using fallback deserialization '
          'for the relevant artifact. Please make sure that any artifact '
          'classes can be imported within your container or environment.'),
                      module_name, class_name)
    if not result:
      result = Artifact(mlmd_artifact_type=artifact_type)
    result.set_mlmd_artifact_type(artifact_type)
    result.set_mlmd_artifact(artifact)
    return result

  # Read-only properties.
  @property
  @doc_controls.do_not_doc_in_subclasses
  def type(self):
    """Type of the artifact."""
    return self.__class__

  @property
  @doc_controls.do_not_doc_in_subclasses
  def type_name(self):
    """Type name of the underlying mlmd artifact."""
    return self._artifact_type.name

  @property
  @doc_controls.do_not_doc_in_subclasses
  def artifact_type(self):
    """Type of the underlying mlmd artifact."""
    return self._artifact_type

  @property
  @doc_controls.do_not_doc_in_subclasses
  def mlmd_artifact(self):
    """Underlying mlmd artifact."""
    # Update the Metadata proto message to reflect the contents of any
    # possibly-modified JSON value properties, which may be dicts or lists
    # modifiable by the user.
    for cache_map, target_proto_properties in [
        (self._cached_modifiable_properties, self._artifact.properties),
        (self._cached_modifiable_custom_properties,
         self._artifact.custom_properties)
    ]:
      for key, cached_value in cache_map.items():
        if cached_value is None:
          if key in target_proto_properties:
            del target_proto_properties[key]
        elif isinstance(cached_value, message.Message):
          target_proto_properties[key].proto_value.Pack(cached_value)
        else:
          struct_value = _encode_struct_value(cached_value)
          target_proto_properties[key].struct_value.CopyFrom(struct_value)
    return self._artifact

  # Settable properties for all artifact types.
  @property
  @doc_controls.do_not_doc_in_subclasses
  def uri(self) -> str:
    """Artifact URI."""
    return self._artifact.uri

  @uri.setter
  def uri(self, uri: str):
    """Setter for artifact URI."""
    self._artifact.uri = uri

  @property
  @doc_controls.do_not_doc_in_subclasses
  def id(self) -> int:
    """Id of the underlying mlmd artifact."""
    return self._artifact.id

  @id.setter
  def id(self, artifact_id: int):
    """Set id of underlying artifact."""
    self._artifact.id = artifact_id

  @property
  @doc_controls.do_not_doc_in_subclasses
  def type_id(self) -> int:
    """Type id of the underlying mlmd artifact."""
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
  def _get_system_property(self, key: str) -> str:
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Legacy artifact types which have explicitly defined system properties.
      return self._artifact.properties[key].string_value
    return self._artifact.custom_properties[key].string_value

  def _set_system_property(self, key: str, value: str):
    if (key in self._artifact_type.properties and
        key in self._artifact.properties):
      # Clear non-custom property in legacy artifact types.
      del self._artifact.properties[key]
    self._artifact.custom_properties[key].string_value = value

  @property
  @doc_controls.do_not_doc_inheritable
  def name(self) -> str:
    """Name of the underlying mlmd artifact."""
    return self._get_system_property('name')

  @name.setter
  def name(self, name: str):
    """Set name of the underlying artifact."""
    self._set_system_property('name', name)
    self._artifact.name = name

  @property
  @doc_controls.do_not_doc_in_subclasses
  def state(self) -> str:
    """State of the underlying mlmd artifact."""
    return self._get_system_property('state')

  @state.setter
  def state(self, state: str):
    """Set state of the underlying artifact."""
    self._set_system_property('state', state)

  @property
  @doc_controls.do_not_doc_in_subclasses
  def pipeline_name(self) -> str:
    """Name of the pipeline that produce the artifact."""
    return self._get_system_property('pipeline_name')

  @pipeline_name.setter
  def pipeline_name(self, pipeline_name: str):
    """Set name of the pipeline that produce the artifact."""
    self._set_system_property('pipeline_name', pipeline_name)

  @property
  @doc_controls.do_not_doc_inheritable
  def producer_component(self) -> str:
    """Producer component of the artifact."""
    return self._get_system_property('producer_component')

  @producer_component.setter
  def producer_component(self, producer_component: str):
    """Set producer component of the artifact."""
    self._set_system_property('producer_component', producer_component)

  @property
  @doc_controls.do_not_doc_in_subclasses
  def is_external(self) -> bool:
    """Returns true if the artifact is external."""
    return self.get_int_custom_property('is_external') == 1

  @is_external.setter
  def is_external(self, is_external: bool):
    """Sets if the artifact is external."""
    self.set_int_custom_property('is_external', is_external)

  # Custom property accessors.
  @doc_controls.do_not_doc_in_subclasses
  def set_string_custom_property(self, key: str, value: str):
    """Set a custom property of string type."""
    self._artifact.custom_properties[key].string_value = value

  @doc_controls.do_not_doc_in_subclasses
  def set_int_custom_property(self, key: str, value: int):
    """Set a custom property of int type."""
    self._artifact.custom_properties[key].int_value = builtins.int(value)

  @doc_controls.do_not_doc_in_subclasses
  def set_float_custom_property(self, key: str, value: float):
    """Sets a custom property of float type."""
    self._artifact.custom_properties[key].double_value = builtins.float(value)

  @doc_controls.do_not_doc_inheritable
  def set_json_value_custom_property(self, key: str, value: JsonValueType):
    """Sets a custom property of JSON type."""
    self._cached_modifiable_custom_properties[key] = value

  @doc_controls.do_not_doc_inheritable
  def set_proto_custom_property(self, key: str, value: message.Message):
    """Sets a custom property of proto type."""
    # TODO(b/241861488): Remove safeguard once fully supported by MLMD.
    if not artifact_property.ENABLE_PROTO_PROPERTIES:
      raise ValueError('Proto properties are not yet supported')
    self._cached_modifiable_custom_properties[key] = value

  @doc_controls.do_not_doc_in_subclasses
  def has_custom_property(self, key: str) -> bool:
    return key in self._artifact.custom_properties

  @doc_controls.do_not_doc_in_subclasses
  def get_string_custom_property(self, key: str) -> str:
    """Get a custom property of string type."""
    if key not in self._artifact.custom_properties:
      return ''
    json_value = self.get_json_value_custom_property(key)
    if isinstance(json_value, str):
      return json_value
    return self._artifact.custom_properties[key].string_value

  @doc_controls.do_not_doc_in_subclasses
  def get_int_custom_property(self, key: str) -> int:
    """Get a custom property of int type."""
    if key not in self._artifact.custom_properties:
      return 0
    json_value = self.get_json_value_custom_property(key)
    if isinstance(json_value, float):
      return int(json_value)
    return self._artifact.custom_properties[key].int_value

  # TODO(b/179215351): Standardize type name into one of float and double.
  @doc_controls.do_not_doc_in_subclasses
  def get_float_custom_property(self, key: str) -> float:
    """Gets a custom property of float type."""
    if key not in self._artifact.custom_properties:
      return 0.0
    json_value = self.get_json_value_custom_property(key)
    if isinstance(json_value, float):
      return json_value
    return self._artifact.custom_properties[key].double_value

  @doc_controls.do_not_doc_in_subclasses
  def get_custom_property(
      self, key: str) -> Optional[Union[int, float, str, JsonValueType]]:
    """Gets a custom property with key. Return None if not found."""
    if key not in self._artifact.custom_properties:
      return None

    json_value = self.get_json_value_custom_property(key)
    if json_value:
      return json_value

    mlmd_value = self._artifact.custom_properties[key]
    if mlmd_value.HasField('int_value'):
      return mlmd_value.int_value
    elif mlmd_value.HasField('double_value'):
      return mlmd_value.double_value
    elif mlmd_value.HasField('string_value'):
      return mlmd_value.string_value
    return None

  @doc_controls.do_not_doc_inheritable
  def get_json_value_custom_property(self, key: str) -> JsonValueType:
    """Get a custom property of JSON type."""
    if key in self._cached_modifiable_custom_properties:
      return self._cached_modifiable_custom_properties[key]
    if (key not in self._artifact.custom_properties or
        not self._artifact.custom_properties[key].HasField('struct_value')):
      return None
    value = _decode_struct_value(
        self._artifact.custom_properties[key].struct_value)
    # We must cache the decoded lists or dictionaries returned here so that
    # if their recursive contents are modified, the Metadata proto message
    # can be updated to reflect this.
    if isinstance(value, (dict, list)):
      self._cached_modifiable_custom_properties[key] = value
    return value

  @doc_controls.do_not_doc_inheritable
  def get_proto_custom_property(self, key: str) -> Optional[message.Message]:
    """Get a custom property of proto type."""
    if not artifact_property.ENABLE_PROTO_PROPERTIES:
      raise ValueError('Proto properties are not yet supported')
    if key in self._cached_modifiable_custom_properties:
      return self._cached_modifiable_custom_properties[key]
    if (key not in self._artifact.custom_properties or
        not self._artifact.custom_properties[key].HasField('proto_value')):
      return None
    value = proto_utils.unpack_proto_any(
        self._artifact.custom_properties[key].proto_value)
    # We must cache the protobuf message here so that if its contents are
    # modified, the Metadata proto message can be updated to reflect this.
    if isinstance(value, message.Message):
      self._cached_modifiable_custom_properties[key] = value
    return value

  @doc_controls.do_not_doc_inheritable
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
    self._cached_modifiable_properties = copy.deepcopy(
        other._cached_modifiable_properties)  # pylint: disable=protected-access
    self._cached_modifiable_custom_properties = copy.deepcopy(
        other._cached_modifiable_custom_properties)  # pylint: disable=protected-access


def _ArtifactType(  # pylint: disable=invalid-name
    name: Optional[str] = None,
    annotation: Optional[Type[SystemArtifact]] = None,
    properties: Optional[Dict[str, Property]] = None,
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
    annotation: Annotation of the artifact type. It can be any of the system
      artifact classes from third_party/py/tfx/types/system_artifacts.py.
    properties: Dictionary of properties mapping property name keys to
      `Parameter` object instances. Must be provided unless a protobuf message
      is provided in the "mlmd_artifact_type" parameter.
    mlmd_artifact_type: A ML Metadata metadata_store_pb2.ArtifactType protobuf
      message corresponding to the type being created.

  Returns:
    An Artifact class corresponding to the specified type.
  """
  if mlmd_artifact_type:
    if name or annotation or properties:
      raise ValueError(
          'The "name", "annotation" and "properties" fields should not be '
          'passed when the "mlmd_artifact_type" parameter is set, in '
          '_ArtifactType call.')
    if not mlmd_artifact_type.name:
      raise ValueError('Artifact type proto must have "name" field set.')
    properties = {}
    for name, property_type in mlmd_artifact_type.properties.items():
      if property_type == metadata_store_pb2.PropertyType.INT:
        properties[name] = Property(PropertyType.INT)
      elif property_type == metadata_store_pb2.PropertyType.DOUBLE:
        properties[name] = Property(PropertyType.FLOAT)
      elif property_type == metadata_store_pb2.PropertyType.PROTO:
        properties[name] = Property(PropertyType.PROTO)
      elif property_type == metadata_store_pb2.PropertyType.STRING:
        properties[name] = Property(PropertyType.STRING)
      else:
        raise ValueError('Unsupported MLMD property type: %s.' % property_type)
    annotation = None
    if mlmd_artifact_type.base_type != metadata_store_pb2.ArtifactType.UNSET:
      extensions = (
          metadata_store_pb2.ArtifactType.SystemDefinedBaseType.DESCRIPTOR.
          values_by_number[mlmd_artifact_type.base_type].GetOptions().Extensions
      )
      mlmd_base_type_name = extensions[
          metadata_store_pb2.system_type_extension].type_name
      annotation = type(mlmd_base_type_name, (SystemArtifact,), {
          'MLMD_SYSTEM_BASE_TYPE': mlmd_artifact_type.base_type,
      })

    return type(
        str(mlmd_artifact_type.name), (Artifact,), {
            'TYPE_NAME': mlmd_artifact_type.name,
            'TYPE_ANNOTATION': annotation,
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
        'TYPE_ANNOTATION': annotation,
        'PROPERTIES': properties
    })
