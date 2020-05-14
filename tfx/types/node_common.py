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
"""ComponentSpec for defining inputs/outputs/properties of TFX components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Text

from tfx.types.channel import Channel
from tfx.utils import json_utils


class _PropertyDictWrapper(json_utils.Jsonable):
  """Helper class to wrap inputs/outputs from TFX nodes.

  Currently, this class is read-only (setting properties is not implemented).

  Internal class: no backwards compatibility guarantees. Will be deprecated
  after 0.15 release.
  """

  def __init__(self,
               data: Dict[Text, Channel],
               compat_aliases: Optional[Dict[Text, Text]] = None):
    self._data = data
    self._compat_aliases = compat_aliases or {}

  def __getitem__(self, key):
    if key in self._compat_aliases:
      key = self._compat_aliases[key]
    return self._data[key]

  def __getattr__(self, key):
    if key in self._compat_aliases:
      key = self._compat_aliases[key]
    try:
      return self._data[key]
    except KeyError:
      raise AttributeError

  def __repr__(self):
    return repr(self._data)

  def get_all(self) -> Dict[Text, Channel]:
    return self._data

  def keys(self):
    return self._data.keys()

  def values(self):
    return self._data.values()

  def items(self):
    return self._data.items()
