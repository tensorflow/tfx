# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Partition utility module."""

import collections
from typing import Callable, Mapping, Sequence, Tuple, TypeVar

_T = TypeVar('_T')


class Partition(collections.UserDict, Mapping[str, int]):
  """An immutable partition with multiple dimension (str -> int).

  Partition is used together with _join as a key to join values.

  - If partition dimensions does not overlap at all, then joined result is a
  cartesian product.
  - If partition dimensions are exactly the same, then the joined result is a
  zip().
  - If partition dimensions partially overlaps, then the overlapping dimensions
  are zipped, and distinct dimensions are cartesian producted.
  """

  @property
  def dimensions(self) -> Tuple[str, ...]:
    return tuple(sorted(self.keys()))

  def partial(self, dimensions: Sequence[str]) -> Sequence[int]:
    return tuple(self[d] for d in dimensions)

  def __or__(self, other) -> 'Partition':
    common_dims = list(set(self) & set(other))
    other_partial = tuple(other[d] for d in common_dims)
    if self.partial(common_dims) != other_partial:
      raise ValueError(
          f'Cannot merge {self} and {other} on {common_dims}')
    return Partition({**self, **other})


NO_PARTITION = Partition({})


def _group_by(
    values: Sequence[Tuple[Partition, _T]],
    dimensions: Sequence[str],
) -> Mapping[Sequence[int], Sequence[Tuple[Partition, _T]]]:
  result = collections.defaultdict(list)
  for partition, value in values:
    result[partition.partial(dimensions)].append((partition, value))
  return result


def join(
    lhs: Sequence[Tuple[Partition, _T]],
    rhs: Sequence[Tuple[Partition, _T]],
    merge_fn: Callable[[_T, _T], _T],
) -> Sequence[Tuple[Partition, _T]]:
  """Join values by the partition key.

  Example:
    join(
        [
            ({x=1, y=1}, 'xy-11'),
            ({x=2, y=2}, 'xy-22'),
        ],
        [
            ({x=1, z=1}, 'xz-11'),
            ({x=1, z=2}, 'xz-12'),
            ({x=2, z=1}, 'xz-21'),
            ({x=2, z=2}, 'xz-22'),
        ],
        merge_fn=lambda left, right: f'{left}_{right}'
    ) == [
        ({x=1, y=1, z=1}, 'xy-11_xz-11'),
        ({x=1, y=1, z=2}, 'xy-11_xz-12'),
        ({x=2, y=2, z=1}, 'xy-22_xz-21'),
        ({x=2, y=2, z=2}, 'xy-22_xz-22'),
    ]

  Args:
    lhs: LHS values.
    rhs: RHS values.
    merge_fn: A merge function that is called for each joined pair of values.

  Returns:
    A inner-joined value with merged Partition and values.
  """
  if not lhs or not rhs:
    return []

  common_dims = sorted(set(lhs[0][0].dimensions) & set(rhs[0][0].dimensions))
  lhs_by_part = _group_by(lhs, common_dims)
  rhs_by_part = _group_by(rhs, common_dims)

  result = []
  for part, sub_lhs in lhs_by_part.items():
    if part not in rhs_by_part:
      continue
    sub_rhs = rhs_by_part[part]
    for left_partition, left_value in sub_lhs:
      for right_partition, right_value in sub_rhs:
        merged_partition = left_partition | right_partition
        merged_value = merge_fn(left_value, right_value)
        result.append((merged_partition, merged_value))
  return result
