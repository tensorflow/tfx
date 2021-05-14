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
from typing import Any

from tfx.dsl.io import fileio
from tfx.types.artifact import Artifact
from tfx.utils import doc_controls


class ValueArtifact(Artifact):
  """Artifacts of small scalar-values that can be easily loaded into memory."""

  def __init__(self, *args, **kwargs):
    """Initializes ValueArtifact."""
    self._has_value = False
    self._modified = False
    self._value = None
    super(ValueArtifact, self).__init__(*args, **kwargs)

  @doc_controls.do_not_doc_inheritable
  def read(self):
    if not self._has_value:
      file_path = self.uri
      # Assert there is a file exists.
      if not fileio.exists(file_path):
        raise RuntimeError(
            'Given path does not exist or is not a valid file: %s' % file_path)

      serialized_value = fileio.open(file_path, 'rb').read()
      self._has_value = True
      self._value = self.decode(serialized_value)
    return self._value

  @doc_controls.do_not_doc_inheritable
  def write(self, value):
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
