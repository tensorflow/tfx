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
"""TFX Channel definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Iterable
from typing import Text
from typing import Union
from tfx.utils import types


class Channel(object):
  """Tfx Channel.

  TFX Channel is an abstract concept that connects data producers and data
  consumers. It contains restriction of the artifact type that should be fed
  into or read from it.

  Attributes:
    type_name: A string representing the artifact type the Channel takes.
  """

  # TODO(b/124763842): Consider replace type_name with ArtifactType.
  # TODO(b/125348988): Add support for real Channel in addition to static ones.
  def __init__(self,
               type_name,
               static_artifact_collection = None):
    """Initialization of Channel.

    Args:
      type_name: Name of the type that should be fed into or read from the
        Channel.
      static_artifact_collection: (Optional) A collection of artifacts as the
        values that can be read from the Channel. This is used to construct a
        static Channel.
    """

    self.type_name = type_name
    self._static_artifact_collection = static_artifact_collection or []
    self._validate_type()

  def __str__(self):
    return 'Channel<{}: {}>'.format(self.type_name,
                                    self._static_artifact_collection)

  def __repr__(self):
    return self.__str__()

  def _validate_type(self):
    for artifact in self._static_artifact_collection:
      if artifact.type_name != self.type_name:
        raise ValueError(
            'Static artifact collection with different artifact type than {}'
            .format(self.type_name))

  def get(self):
    """Returns all artifacts that can be get from this Channel.

    Returns:
      An artifact collection.
    """
    # TODO(b/125037186): We should support dynamic query against a Channel
    #  instead of a static Artifact collection.
    return self._static_artifact_collection

  def type_check(self, expected_type_name):
    """Checks whether a Channel has the expected type name.

    Args:
      expected_type_name: Expected type_name to check against.

    Raises:
      TypeError if the type_name of given Channel is different from expected.
    """
    if self.type_name != expected_type_name:
      raise TypeError('Expects {} but found {}'.format(expected_type_name,
                                                       str(self.type_name)))


def as_channel(source):
  """Converts artifact collection of the same artifact type into a Channel.

  Args:
    source: Either a Channel or an iterable of TfxType.

  Returns:
    A static Channel containing the source artifact collection.

  Raises:
    ValueError when source is not a non-empty iterable of TfxType.
  """

  if isinstance(source, Channel):
    return source
  elif isinstance(source, collections.Iterable):
    try:
      first_element = next(iter(source))
      if isinstance(first_element, types.TfxType):
        return Channel(
            type_name=first_element.type_name,
            static_artifact_collection=source)
      else:
        raise ValueError('Invalid source to be a channel: {}'.format(source))
    except StopIteration:
      raise ValueError('Cannot convert empty artifact collection into Channel')
  else:
    raise ValueError('Invalid source to be a channel: {}'.format(source))
