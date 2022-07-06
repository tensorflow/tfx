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

import abc
from typing import Any, Type, Optional

from tfx.dsl.io import fileio
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from tfx.types.system_artifacts import SystemArtifact
from tfx.utils import doc_controls
from ml_metadata.proto import metadata_store_pb2

_IS_NULL_KEY = '__is_null__'


class ValueArtifact(Artifact):
  """Artifacts of small scalar-values that can be easily loaded into memory.

  Value artifacts are stored to a file located at the `uri` of the artifact.
  This is different from other kinds of artifact types that has a directory at
  the `uri`. The payload of the file will be determined by each value artifact
  types which is a subclass of this class.

  The content of a value artifact can be read or written using `.value`
  property.
  """

  def __init__(self, *args, **kwargs):
    """Initializes ValueArtifact."""
    self._has_value = False
    self._modified = False
    self._value = None
    super().__init__(*args, **kwargs)

  @doc_controls.do_not_doc_inheritable
  def read(self):
    if not self._has_value:
      file_path = self.uri
      # Assert there is a file exists.
      if not fileio.exists(file_path):
        raise RuntimeError(
            'Given path does not exist or is not a valid file: %s' % file_path)

      self._has_value = True
      if not self.get_int_custom_property(_IS_NULL_KEY):
        serialized_value = fileio.open(file_path, 'rb').read()
        self._value = self.decode(serialized_value)
    return self._value

  @doc_controls.do_not_doc_inheritable
  def write(self, value):
    if value is None:
      self.set_int_custom_property(_IS_NULL_KEY, 1)
      serialized_value = b''
    else:
      self.set_int_custom_property(_IS_NULL_KEY, 0)
      serialized_value = self.encode(value)
    with fileio.open(self.uri, 'wb') as f:
      f.write(serialized_value)

  @property
  def value(self):
    """Value stored in the artifact."""
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
  @doc_controls.do_not_doc_inheritable
  @abc.abstractmethod
  def decode(self, serialized_value) -> bytes:
    """Method decoding the file content. Implemented by subclasses."""
    pass

  # Note: behavior of encode() method should not be changed to provide
  # backward/forward compatibility.
  @doc_controls.do_not_doc_inheritable
  @abc.abstractmethod
  def encode(self, value) -> Any:
    """Method encoding the file content. Implemented by subclasses."""
    pass

  @classmethod
  def annotate_as(cls, type_annotation: Optional[Type[SystemArtifact]] = None):
    """Annotate the value artifact type with a system artifact class.

    Example usage:

    from tfx.types.system_artifacts import Model
    ...
    tfx.Binary(
      name=component_name,
      mpm_or_target=...,
      flags=...,
      outputs={
          'experiment_id': standard_artifacts.String.annotate_as(Model)
      })

    Args:
      type_annotation: the system artifact class used to annotate the value
        artifact type. It is a subclass of SystemArtifact. The subclasses are
        defined in third_party/py/tfx/types/system_artifacts.py.

    Returns:
      A subclass of the method caller class (e.g., standard_artifacts.String,
      standard_artifacts.Float) with TYPE_ANNOTATION attribute set to be
      `type_annotation`; returns the original class if`type_annotation` is None.
    """
    if not type_annotation:
      return cls
    if not issubclass(type_annotation, SystemArtifact):
      raise ValueError(
          'type_annotation %s is not a subclass of SystemArtifact.' %
          type_annotation)
    type_annotation_str = str(type_annotation.__name__)
    return type(
        str(cls.__name__) + '_' + type_annotation_str,
        (cls,),
        {
            'TYPE_NAME': str(cls.TYPE_NAME) + '_' + type_annotation_str,
            'TYPE_ANNOTATION': type_annotation,
            '__module__': cls.__module__,
        },
    )


def _ValueArtifactType(  # pylint: disable=invalid-name
    mlmd_artifact_type: metadata_store_pb2.ArtifactType,
    base: Type[ValueArtifact],
) -> Type[ValueArtifact]:
  """Experimental interface: internal use only.

  Construct a value artifact type. Equivalent to subclassing ValueArtifact and
  providing relevant properties.

  Args:
    mlmd_artifact_type: A ML Metadata metadata_store_pb2.ArtifactType protobuf
      message corresponding to the type being created.
    base: base class of the created value artifact type. It is a subclass of
      ValueArtifact, for example, Integer, String.

  Returns:
    A ValueArtifact subclass corresponding to the specified type and base.
  """

  if not mlmd_artifact_type.name:
    raise ValueError('ValueArtifact type proto must have "name" field set.')
  if not (base and issubclass(base, ValueArtifact)):
    raise ValueError(
        'Input argumment "base" must be a subclass of ValueArtifact; got : %s.'
        % base)
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
  annotation = None
  if mlmd_artifact_type.base_type != metadata_store_pb2.ArtifactType.UNSET:
    extensions = (
        metadata_store_pb2.ArtifactType.SystemDefinedBaseType.DESCRIPTOR
        .values_by_number[mlmd_artifact_type.base_type].GetOptions().Extensions)
    mlmd_base_type_name = extensions[
        metadata_store_pb2.system_type_extension].type_name
    annotation = type(mlmd_base_type_name, (SystemArtifact,), {
        'MLMD_SYSTEM_BASE_TYPE': mlmd_artifact_type.base_type,
    })

  return type(
      str(mlmd_artifact_type.name), (base,), {
          'TYPE_NAME': mlmd_artifact_type.name,
          'TYPE_ANNOTATION': annotation,
          'PROPERTIES': properties,
      })
