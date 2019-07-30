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

from typing import Dict, List, Text

from tfx.types.artifact import Artifact


def parse_artifact_dict(json_str: Text) -> Dict[Text, List[Artifact]]:
  """Parse a dict from key to list of Artifact from its json format."""
  tfx_artifacts = {}
  for k, l in json.loads(json_str).items():
    tfx_artifacts[k] = [Artifact.parse_from_json_dict(v) for v in l]
  return tfx_artifacts


def jsonify_artifact_dict(artifact_dict: Dict[Text, List[Artifact]]) -> Text:
  """Serialize a dict from key to list of Artifact into json format."""
  d = {}
  for k, l in artifact_dict.items():
    d[k] = [v.json_dict() for v in l]
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


def _get_split_instance(artifact_list: List[Artifact], split: Text) -> Artifact:
  """Get an instance of Artifact with matching split from given list.

  Args:
    artifact_list: A list of Artifact objects whose length must be one.
    split: Name of split.

  Returns:
    The single Artifact object in artifact_list with matching split.

  Raises:
    ValueError: If number with matching split in artifact_list is not one.
  """
  matched = [x for x in artifact_list if x.split == split]
  if len(matched) != 1:
    raise ValueError('{} elements matches split {}'.format(len(matched), split))
  return matched[0]


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
  return _get_split_instance(artifact_list, split).uri
