# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Utilities for standard artifacts that does not depend on standard_artifacts.

This module was introduced to break the cyclic dependency of standard_artifacts
-> artifact_utils -> standard_artifacts.
"""

import json
import os
import re
from typing import List

from absl import logging
from packaging import version
from tfx.types import artifact as artifact_lib

from ml_metadata.proto import metadata_store_pb2


_Artifact = artifact_lib.Artifact


ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY = 'tfx_version'

# TODO(b/182526033): deprecate old artifact payload format.
# Version that "Split-{split_name}" is introduced.
_ARTIFACT_VERSION_FOR_SPLIT_UPDATE = '0.29.0.dev'
# Version that "Format-TFMA/Format-Serving" is introduced.
_ARTIFACT_VERSION_FOR_MODEL_UPDATE = '0.29.0.dev'
# Version that "FeatureStats.pb" is introduced.
_ARTIFACT_VERSION_FOR_STATS_UPDATE = '0.29.0.dev'
# Version that "SchemaDiff.pb" is introduced.
_ARTIFACT_VERSION_FOR_ANOMALIES_UPDATE = '0.29.0.dev'


def is_artifact_version_older_than(
    artifact: _Artifact, artifact_version: str
) -> bool:
  """Check if artifact belongs to old version."""
  if artifact.mlmd_artifact.state == metadata_store_pb2.Artifact.UNKNOWN:
    # Newly generated artifact should use the latest artifact payload format.
    return False

  # For artifact that resolved from MLMD.
  if not artifact.has_custom_property(ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY):
    # Artifact without version.
    return True

  # Artifact with old version
  return bool(
      version.parse(
          artifact.get_string_custom_property(
              ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY
          )
      )
      < version.parse(artifact_version)
  )


def get_split_uris(artifact_list: List[_Artifact], split: str) -> List[str]:
  """Get the uris of Artifacts with matching split from given list.

  Args:
    artifact_list: A list of Artifact objects.
    split: Name of split.

  Returns:
    A list of uris of Artifact object in artifact_list with matching split.

  Raises:
    ValueError: If number of artifacts matching the split is not equal to
      number of input artifacts.
  """
  result = []
  for artifact in artifact_list:
    split_names = decode_split_names(artifact.split_names)
    if split in split_names:
      # TODO(b/182526033): deprecate old split format.
      if is_artifact_version_older_than(
          artifact, _ARTIFACT_VERSION_FOR_SPLIT_UPDATE
      ):
        result.append(os.path.join(artifact.uri, split))
      else:
        result.append(os.path.join(artifact.uri, f'Split-{split}'))
  if len(result) != len(artifact_list):
    raise ValueError(
        f'Split does not exist over all example artifacts: {split}'
    )
  return result


def get_split_uri(artifact_list: List[_Artifact], split: str) -> str:
  """Get the uri of Artifact with matching split from given list.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.
    split: Name of split.

  Returns:
    The uri of Artifact object in artifact_list with matching split.

  Raises:
    ValueError: If number with matching split in artifact_list is not one.
  """
  artifact_split_uris = get_split_uris(artifact_list, split)
  if len(artifact_split_uris) != 1:
    raise ValueError(
        f'Expected exactly one artifact with split {repr(split)}, but found '
        f'matching artifacts {artifact_split_uris}.'
    )
  return artifact_split_uris[0]


def encode_split_names(splits: List[str]) -> str:
  """Get the encoded representation of a list of split names."""
  rewritten_splits = []
  for split in splits:
    # TODO(b/146759051): Remove workaround for RuntimeParameter object once
    # this bug is clarified.
    if split.__class__.__name__ == 'RuntimeParameter':
      logging.warning(
          'RuntimeParameter provided for split name: this functionality may '
          'not be supported in the future.'
      )
      split = str(split)
      # Intentionally ignore split format check to pass through the template for
      # now. This behavior is very fragile and should be fixed (see
      # b/146759051).
    elif not re.match('^([A-Za-z0-9][A-Za-z0-9_-]*)?$', split):
      # TODO(ccy): Disallow empty split names once the importer removes split as
      # a property for all artifacts.
      raise ValueError(
          'Split names are expected to be alphanumeric (allowing dashes and '
          'underscores, provided they are not the first character); got '
          f'{repr(split)} instead.'
      )
    rewritten_splits.append(split)
  return json.dumps(rewritten_splits)


def decode_split_names(split_names: str) -> List[str]:
  """Decode an encoded list of split names."""
  if not split_names:
    return []
  return json.loads(split_names)
