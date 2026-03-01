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
"""TFX Artifact utilities."""

import itertools
import json
import os
from typing import Dict, List, Optional, Type, Union

from absl import logging
from tfx.dsl.io import fileio
from tfx.types import artifact as artifact_lib
from tfx.types import standard_artifact_utils
from tfx.types import value_artifact as value_artifact_lib

from ml_metadata.proto import metadata_store_pb2


Artifact = artifact_lib.Artifact
_ArtifactType = artifact_lib._ArtifactType  # pylint: disable=protected-access
ValueArtifact = value_artifact_lib.ValueArtifact
_ValueArtifactType = value_artifact_lib._ValueArtifactType  # pylint: disable=protected-access


# Reexport standard_artifact_utils symbols to artifact_utils for backward
# compatibility.
ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY = (
    standard_artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY
)
# pylint: disable=protected-access
_ARTIFACT_VERSION_FOR_SPLIT_UPDATE = (
    standard_artifact_utils._ARTIFACT_VERSION_FOR_SPLIT_UPDATE
)
_ARTIFACT_VERSION_FOR_MODEL_UPDATE = (
    standard_artifact_utils._ARTIFACT_VERSION_FOR_MODEL_UPDATE
)
_ARTIFACT_VERSION_FOR_STATS_UPDATE = (
    standard_artifact_utils._ARTIFACT_VERSION_FOR_STATS_UPDATE
)
_ARTIFACT_VERSION_FOR_ANOMALIES_UPDATE = (
    standard_artifact_utils._ARTIFACT_VERSION_FOR_ANOMALIES_UPDATE
)
# pylint: enable=protected-access
is_artifact_version_older_than = (
    standard_artifact_utils.is_artifact_version_older_than
)
get_split_uris = standard_artifact_utils.get_split_uris
get_split_uri = standard_artifact_utils.get_split_uri
encode_split_names = standard_artifact_utils.encode_split_names
decode_split_names = standard_artifact_utils.decode_split_names


# TODO(ruoyu): Deprecate this function since it is no longer needed.
def parse_artifact_dict(json_str: str) -> Dict[str, List[Artifact]]:
  """Parse a dict from key to list of Artifact from its json format."""
  tfx_artifacts = {}
  for k, j in json.loads(json_str).items():
    tfx_artifacts[k] = [Artifact.from_json_dict(v) for v in j]
  return tfx_artifacts


# TODO(ruoyu): Deprecate this function since it is no longer needed.
def jsonify_artifact_dict(artifact_dict: Dict[str, List[Artifact]]) -> str:
  """Serialize a dict from key to list of Artifact into json format."""
  d = {}
  for k, j in artifact_dict.items():
    d[k] = [v.to_json_dict() for v in j]
  return json.dumps(d)


def get_single_instance(artifact_list: List[Artifact]) -> Artifact:
  """Get a single instance of Artifact from a list of length one.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.

  Returns:
    The single Artifact object in artifact_list.

  Raises:
    ValueError: If length of artifact_list is not one.
  """
  if len(artifact_list) != 1:
    raise ValueError(
        f'expected list length of one but got {len(artifact_list)}')
  return artifact_list[0]


def get_single_uri(artifact_list: List[Artifact]) -> str:
  """Get the uri of Artifact from a list of length one.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.

  Returns:
    The uri of the single Artifact object in artifact_list.

  Raises:
    ValueError: If length of artifact_list is not one.
  """
  return get_single_instance(artifact_list).uri


def replicate_artifacts(source: Artifact, count: int) -> List[Artifact]:
  """Replicate given artifact and return a list with `count` artifacts."""
  result = []
  artifact_cls = source.type
  for i in range(count):
    new_instance = artifact_cls()
    new_instance.copy_from(source)
    # New uris should be sub directories of the original uri. See
    # https://github.com/tensorflow/tfx/blob/1a1a53e17626d636f403b6dd16f8635e80755682/tfx/orchestration/portable/execution_publish_utils.py#L35
    new_instance.uri = os.path.join(source.uri, str(i))
    result.append(new_instance)
  return result


def _get_subclasses(cls: Type[Artifact]) -> List[Type[Artifact]]:
  """Internal method. Get transitive subclasses of an Artifact subclass."""
  all_subclasses = []
  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(_get_subclasses(subclass))
  return all_subclasses


def get_artifact_type_class(
    artifact_type: metadata_store_pb2.ArtifactType) -> Type[Artifact]:
  """Get the artifact type class corresponding to an MLMD type proto."""

  # Make sure this module path containing the standard Artifact subclass
  # definitions is imported. Modules containing custom artifact subclasses that
  # need to be deserialized should be imported by the entrypoint of the
  # application or container.

  # Enumerate the Artifact type ontology, separated into auto-generated and
  # natively-defined classes.
  artifact_classes = _get_subclasses(Artifact)
  native_artifact_classes = []
  generated_artifact_classes = []
  value_artifact_classes = []
  for cls in artifact_classes:
    if not cls.TYPE_NAME:
      # Skip abstract classes.
      continue
    if getattr(cls, '_AUTOGENERATED', False):
      generated_artifact_classes.append(cls)
    else:
      native_artifact_classes.append(cls)
    if issubclass(cls, ValueArtifact):
      value_artifact_classes.append(cls)

  # Try to find an existing class for the artifact type, if it exists. Prefer
  # to use a native artifact class.
  for cls in itertools.chain(native_artifact_classes,
                             generated_artifact_classes):
    candidate_type = cls._get_artifact_type()  # pylint: disable=protected-access
    # We need to compare `.name` and `.properties` (and not the entire proto
    # directly), because the proto `.id` field will be populated when the type
    # is read from MLMD.
    if (artifact_type.name == candidate_type.name and
        artifact_type.properties == candidate_type.properties):
      return cls

  # Generate a class for the artifact type on the fly.
  logging.warning(
      'Could not find matching artifact class for type %r (proto: %r); '
      'generating an ephemeral artifact class on-the-fly. If this is not '
      'intended, please make sure that the artifact class for this type can '
      'be imported within your container or environment where a component '
      'is executed to consume this type.', artifact_type.name,
      str(artifact_type))

  for cls in value_artifact_classes:
    if not cls.TYPE_NAME:
      continue
    if artifact_type.name.startswith(cls.TYPE_NAME):
      new_artifact_class = _ValueArtifactType(
          mlmd_artifact_type=artifact_type, base=cls)
      setattr(new_artifact_class, '_AUTOGENERATED', True)
      return new_artifact_class

  new_artifact_class = _ArtifactType(mlmd_artifact_type=artifact_type)
  setattr(new_artifact_class, '_AUTOGENERATED', True)
  return new_artifact_class


def deserialize_artifact(
    artifact_type: metadata_store_pb2.ArtifactType,
    artifact: Optional[metadata_store_pb2.Artifact] = None) -> Artifact:
  """Reconstructs an Artifact object from MLMD proto descriptors.

  Internal method, no backwards compatibility guarantees.

  Args:
    artifact_type: A metadata_store_pb2.ArtifactType proto object describing the
      type of the artifact.
    artifact: A metadata_store_pb2.Artifact proto object describing the contents
      of the artifact. If not provided, an Artifact of the desired type with
      empty contents is created.

  Returns:
    Artifact subclass object for the given MLMD proto descriptors.
  """
  if artifact is None:
    artifact = metadata_store_pb2.Artifact()
  return deserialize_artifacts(artifact_type, [artifact])[0]


def deserialize_artifacts(
    artifact_type: metadata_store_pb2.ArtifactType,
    artifacts: List[metadata_store_pb2.Artifact]) -> List[Artifact]:
  """Reconstructs Artifact objects from MLMD proto descriptors.

  Internal method, no backwards compatibility guarantees.

  Args:
    artifact_type: A metadata_store_pb2.ArtifactType proto object describing the
      type of the artifact.
    artifacts: List of metadata_store_pb2.Artifact proto describing the contents
      of the artifact.

  Returns:
    Artifact subclass object for the given MLMD proto descriptors.
  """
  # Validate inputs.
  if not isinstance(artifact_type, metadata_store_pb2.ArtifactType):
    raise ValueError(
        'Expected metadata_store_pb2.ArtifactType for artifact_type, got '
        f'{artifact_type} instead')
  for artifact in artifacts:
    if not isinstance(artifact, metadata_store_pb2.Artifact):
      raise ValueError(
          f'Expected metadata_store_pb2.Artifact for artifact, got {artifact} '
          'instead')

  # Get the artifact's class and construct the Artifact object.
  artifact_cls = get_artifact_type_class(artifact_type)
  result = []
  for artifact in artifacts:
    item = artifact_cls()
    item.artifact_type.CopyFrom(artifact_type)
    item.set_mlmd_artifact(artifact)
    result.append(item)
  return result


def verify_artifacts(
    artifacts: Union[Dict[str, List[Artifact]], List[Artifact],
                     Artifact]) -> None:
  """Check that all artifacts have uri and exist at that uri.

  Args:
      artifacts: artifacts dict (key -> types.Artifact), single artifact list,
        or artifact instance.

  Raises:
    TypeError: if the input is an invalid type.
    RuntimeError: if artifact is not valid.
  """
  if isinstance(artifacts, Artifact):
    artifact_list = [artifacts]
  elif isinstance(artifacts, list):
    artifact_list = artifacts
  elif isinstance(artifacts, dict):
    artifact_list = list(itertools.chain(*artifacts.values()))
  else:
    raise TypeError

  for artifact_instance in artifact_list:
    if not artifact_instance.uri:
      raise RuntimeError(f'Artifact {artifact_instance} does not have uri')
    if not fileio.exists(artifact_instance.uri):
      raise RuntimeError(f'Artifact uri {artifact_instance.uri} is missing')
