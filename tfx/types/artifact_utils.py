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
"""TFX Artifact utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import re

from typing import Dict, List, Optional, Text

import absl

from ml_metadata.proto import metadata_store_pb2
from tfx.types.artifact import Artifact


# TODO(ruoyu): Deprecate this function since it is no longer needed.
def parse_artifact_dict(json_str: Text) -> Dict[Text, List[Artifact]]:
  """Parse a dict from key to list of Artifact from its json format."""
  tfx_artifacts = {}
  for k, l in json.loads(json_str).items():
    tfx_artifacts[k] = [Artifact.from_json_dict(v) for v in l]
  return tfx_artifacts


# TODO(ruoyu): Deprecate this function since it is no longer needed.
def jsonify_artifact_dict(artifact_dict: Dict[Text, List[Artifact]]) -> Text:
  """Serialize a dict from key to list of Artifact into json format."""
  d = {}
  for k, l in artifact_dict.items():
    d[k] = [v.to_json_dict() for v in l]
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
    raise ValueError('expected list length of one but got {}'.format(
        len(artifact_list)))
  return artifact_list[0]


def get_single_uri(artifact_list: List[Artifact]) -> Text:
  """Get the uri of Artifact from a list of length one.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.

  Returns:
    The uri of the single Artifact object in artifact_list.

  Raises:
    ValueError: If length of artifact_list is not one.
  """
  return get_single_instance(artifact_list).uri


def get_split_uri(artifact_list: List[Artifact], split: Text) -> Text:
  """Get the uri of Artifact with matching split from given list.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.
    split: Name of split.

  Returns:
    The uri of Artifact object in artifact_list with matching split.

  Raises:
    ValueError: If number with matching split in artifact_list is not one.
  """
  matching_artifacts = []
  for artifact in artifact_list:
    split_names = decode_split_names(artifact.split_names)
    if split in split_names:
      matching_artifacts.append(artifact)
  if len(matching_artifacts) != 1:
    raise ValueError(
        ('Expected exactly one artifact with split %r, but found matching '
         'artifacts %s.') % (split, matching_artifacts))
  return os.path.join(matching_artifacts[0].uri, split)


def encode_split_names(splits: List[Text]) -> Text:
  """Get the encoded representation of a list of split names."""
  rewritten_splits = []
  for split in splits:
    # TODO(b/146759051): Remove workaround for RuntimeParameter object once
    # this bug is clarified.
    if split.__class__.__name__ == 'RuntimeParameter':
      absl.logging.warning(
          'RuntimeParameter provided for split name: this functionality may '
          'not be supported in the future.')
      split = str(split)
      # Intentionally ignore split format check to pass through the template for
      # now. This behavior is very fragile and should be fixed (see
      # b/146759051).
    elif not re.match('^([A-Za-z0-9][A-Za-z0-9_-]*)?$', split):
      # TODO(ccy): Disallow empty split names once the importer removes split as
      # a property for all artifacts.
      raise ValueError(
          ('Split names are expected to be alphanumeric (allowing dashes and '
           'underscores, provided they are not the first character); got %r '
           'instead.') % (split,))
    rewritten_splits.append(split)
  return json.dumps(rewritten_splits)


def decode_split_names(split_names: Text) -> List[Text]:
  """Decode an encoded list of split names."""
  if not split_names:
    return []
  return json.loads(split_names)


def deserialize_artifact(
    artifact_type: metadata_store_pb2.ArtifactType,
    artifact: Optional[metadata_store_pb2.Artifact] = None) -> Artifact:
  """Reconstruct Artifact object from MLMD proto descriptors.

  Internal method, no backwards compatibility guarantees.

  Args:
    artifact_type: A metadata_store_pb2.ArtifactType proto object describing the
      type of the artifact.
    artifact: A metadata_store_pb2.Artifact proto object describing the contents
      of the artifact.  If not provided, an Artifact of the desired type with
      empty contents is created.

  Returns:
    Artifact subclass object for the given MLMD proto descriptors.
  """
  # Validate inputs.
  if not isinstance(artifact_type, metadata_store_pb2.ArtifactType):
    raise ValueError(
        ('Expected metadata_store_pb2.ArtifactType for artifact_type, got %s '
         'instead') % (artifact_type,))
  if artifact and not isinstance(artifact, metadata_store_pb2.Artifact):
    raise ValueError(
        ('Expected metadata_store_pb2.Artifact for artifact, got %s '
         'instead') % (artifact,))

  # Make sure this module path containing the standard Artifact subclass
  # definitions is imported. Modules containing custom artifact subclasses that
  # need to be deserialized should be imported by the entrypoint of the
  # application or container.
  from tfx.types import standard_artifacts  # pylint: disable=g-import-not-at-top,unused-variable

  # Attempt to find the appropriate Artifact subclass for reconstructing this
  # object.
  artifact_cls = None

  # Search the whole Artifact type ontology for a matching class.
  def find_subclasses(cls):
    result = []
    for subclass in cls.__subclasses__():
      result.append(subclass)
      result.extend(find_subclasses(subclass))

    return result

  for cls in find_subclasses(Artifact):
    if cls.TYPE_NAME == artifact_type.name:
      artifact_cls = cls

  # Construct the Artifact object, using a concrete Artifact subclass when
  # possible.
  if artifact_cls:
    result = artifact_cls()
    result.set_mlmd_artifact_type(artifact_type)
  else:
    absl.logging.warning(
        ('Could not load artifact class for type %r; using fallback '
         'deserialization for the relevant artifact. If this is not intended, '
         'please make sure that the artifact class for this type can be '
         'imported within your container or environment where a component is '
         'executed to consume this type.') % (artifact_type.name))
    result = Artifact(mlmd_artifact_type=artifact_type)
  if artifact:
    result.set_mlmd_artifact(artifact)
  return result
