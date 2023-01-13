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
"""Lightweight wrapper for building MLMD filter query."""

import collections
from typing import Union

import ml_metadata as mlmd


def list_options(v: Union[str, 'And', 'Or']) -> mlmd.ListOptions:
  """Converts query value to ListOptions."""
  return mlmd.ListOptions(filter_query=str(v))


class _ClauseList(collections.UserList):
  """List of clauses container."""
  separator = None
  list_options = list_options

  def __str__(self) -> str:
    if not self:
      return ''
    elif len(self) == 1:
      return self[0]
    else:
      return f' {self.separator} '.join(f'({clause})' for clause in self)


class And(_ClauseList):
  separator = 'AND'


class Or(_ClauseList):
  separator = 'OR'
