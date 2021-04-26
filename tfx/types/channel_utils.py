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
"""TFX Channel utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Iterable, List, Text

from tfx.types.artifact import Artifact
from tfx.types.channel import Channel


def as_channel(artifacts: Iterable[Artifact]) -> Channel:
  """Converts artifact collection of the same artifact type into a Channel.

  Args:
    artifacts: An iterable of Artifact.

  Returns:
    A static Channel containing the source artifact collection.

  Raises:
    ValueError when source is not a non-empty iterable of Artifact.
  """
  try:
    first_element = next(iter(artifacts))
    if isinstance(first_element, Artifact):
      return Channel(type=first_element.type).set_artifacts(artifacts)
    else:
      raise ValueError('Invalid artifact iterable: {}'.format(artifacts))
  except StopIteration:
    raise ValueError('Cannot convert empty artifact iterable into Channel')


def unwrap_channel_dict(
    channel_dict: Dict[Text, Channel]) -> Dict[Text, List[Artifact]]:
  """Unwrap dict of channels to dict of lists of Artifact.

  Args:
    channel_dict: a dict of Text -> Channel

  Returns:
    a dict of Text -> List[Artifact]
  """
  return dict((k, list(v.get())) for k, v in channel_dict.items())
