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
from typing import Any, List, Union

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
      return str(self[0])
    else:
      return f' {self.separator} '.join(f'({clause})' for clause in self)


class And(_ClauseList):
  separator = 'AND'


class Or(_ClauseList):
  separator = 'OR'


def to_sql_string(value: Union[bool, int, float, str, List[Any]]) -> str:
  """Converts python value to appropriate GoogleSQL string."""
  if isinstance(value, list):
    inner = ', '.join(to_sql_string(v) for v in value)
    return f'({inner})'
  if isinstance(value, bool):
    return 'TRUE' if value else 'FALSE'
  if isinstance(value, (int, float)):
    return str(value)
  if isinstance(value, str):
    return f'"{value}"'
  raise NotImplementedError(f'Cannot convert {value} to SQL string.')
