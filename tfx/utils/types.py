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
"""Definition of TFX Artifact type.

Deprecated: please see the new location of this module at `tfx.types.artifact`
and `tfx.types.artifact_utils`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Text

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact

TfxType = deprecation.deprecated(  # pylint: disable=invalid-name
    None,
    'tfx.utils.types.TfxType has been renamed to tfx.types.Artifact as of '
    'TFX 0.14.0.')(
        Artifact)

TfxArtifact = deprecation.deprecated(  # pylint: disable=invalid-name
    None,
    'tfx.utils.types.TfxArtifact has been renamed to tfx.types.Artifact as of '
    'TFX 0.14.0.')(
        Artifact)


@deprecation.deprecated(
    None, 'tfx.utils.types.parse_tfx_type_dict has been renamed to '
    'tfx.types.artifact_utils.parse_artifact_dict as of TFX 0.14.0.')
def parse_tfx_type_dict(json_str: Text) -> Dict[Text, List[Artifact]]:
  return artifact_utils.parse_artifact_dict(json_str)


@deprecation.deprecated(
    None, 'tfx.utils.types.jsonify_tfx_type_dict has been renamed to '
    'tfx.types.artifact_utils.jsonify_artifact_dict as of TFX 0.14.0.')
def jsonify_tfx_type_dict(artifact_dict: Dict[Text, List[Artifact]]) -> Text:
  return artifact_utils.jsonify_artifact_dict(artifact_dict)


@deprecation.deprecated(
    None, 'tfx.utils.types.get_single_instance has been renamed to '
    'tfx.types.artifact_utils.get_single_instance as of TFX 0.14.0.')
def get_single_instance(artifact_list: List[Artifact]) -> Artifact:
  """Get a single instance of TfxArtifact from a list of length one."""
  if len(artifact_list) != 1:
    raise ValueError('expected list length of one but got {}'.format(
        len(artifact_list)))
  return artifact_utils.get_single_instance(artifact_list)


@deprecation.deprecated(
    None, 'tfx.utils.types.get_single_uri has been renamed to '
    'tfx.types.artifact_utils.get_single_uri as of TFX 0.14.0.')
def get_single_uri(artifact_list: List[Artifact]) -> Text:
  """Get the uri of TfxArtifact from a list of length one."""
  return artifact_utils.get_single_uri(artifact_list)


@deprecation.deprecated(
    None, 'tfx.utils.types.get_split_uri has been renamed to '
    'tfx.types.artifact_utils.get_split_uri as of TFX 0.14.0.')
def get_split_uri(artifact_list: List[Artifact], split: Text) -> Text:
  """Get the uri of the Artifact with matching split from given list."""
  return artifact_utils.get_split_uri(artifact_list, split)
