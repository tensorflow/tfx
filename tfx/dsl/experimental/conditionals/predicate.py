# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Predicate for Conditional channels."""

import enum

from typing import Any, Dict, Optional, Union
from tfx.utils import json_utils

# To resolve circular dependency caused by type annotations.
ph = Any  # tfx/dsl/placeholder/placeholder.py imports this module.


class CompareOp(enum.Enum):
  EQUAL = '__eq__'
  NOT_EQUAL = '__ne__'
  LESS_THAN = '__lt__'
  LESS_THAN_OR_EQUAL = '__le__'
  GREATER_THAN = '__gt__'
  GREATER_THAN_OR_EQUAL = '__ge__'


class LogicalOp(enum.Enum):
  NOT = 'not'
  AND = 'and'
  OR = 'or'


PrimitiveValueTypes = Union[int, float, str]
ValueLikeTypes = Union[PrimitiveValueTypes, 'ph.ArtifactPlaceholder']


def _encode_value_like(x: ValueLikeTypes) -> PrimitiveValueTypes:
  if isinstance(x, str):
    return x
  if hasattr(x, 'encode'):
    return x.encode()
  return x


class _Comparison:
  """Represents a comparison between two placeholders."""

  # TODO(b/190409099): Support RuntimeParameter.
  def __init__(self, compare_op: CompareOp, left: ValueLikeTypes,
               right: ValueLikeTypes):
    self.compare_op = compare_op
    self.left = left
    self.right = right

  # TODO(b/190408540): Make this a proto, then use proto_to_json.
  def to_json_dict(self):
    return {
        'cmp_op': str(self.compare_op),
        'left': _encode_value_like(self.left),
        'right': _encode_value_like(self.right),
    }


class _LogicalExpression:
  """Represents a boolean logical expression."""

  def __init__(self,
               logical_op: LogicalOp,
               left: Union[_Comparison, '_LogicalExpression'],
               right: Optional[Union[_Comparison,
                                     '_LogicalExpression']] = None):
    if logical_op == LogicalOp.NOT:
      if right is not None:
        raise ValueError(
            f'right must be None if logical_op is NOT. right is {right}')
    else:
      if right is None:
        raise ValueError(
            'right must be not be None if logical_op is AND or OR.')

    self.logical_op = logical_op
    self.left = left
    self.right = right

  # TODO(b/190408540): Make this a proto, then use proto_to_json.
  def to_json_dict(self):
    return {
        'logical_op': str(self.logical_op),
        'left': self.left.to_json_dict(),
        'right': self.right.to_json_dict() if self.right else None,
    }


class Predicate(json_utils.Jsonable):
  """Experimental Predicate object.

  Note that we don't overwrite the default implementation of `from_json_dict`,
  because it is not used, so it doesn't matter.
  """

  def __init__(self, value: Union[_Comparison, _LogicalExpression]):
    self.value = value

  @classmethod
  def from_comparison(cls, cmp_op: CompareOp, left: ValueLikeTypes,
                      right: ValueLikeTypes) -> 'Predicate':
    return Predicate(_Comparison(cmp_op, left, right))

  def to_json_dict(self) -> Dict[str, Any]:
    return self.value.to_json_dict()


def logical_not(predicate: Predicate) -> Predicate:
  """Applies a NOT operation to the Predicate."""
  return Predicate(_LogicalExpression(LogicalOp.NOT, predicate.value))


def logical_and(left_predicate: Predicate,
                right_predicate: Predicate) -> Predicate:
  """Applies an AND operation."""
  return Predicate(
      _LogicalExpression(LogicalOp.AND, left_predicate.value,
                         right_predicate.value))


def logical_or(left_predicate: Predicate,
               right_predicate: Predicate) -> Predicate:
  """Applies an OR operation."""
  return Predicate(
      _LogicalExpression(LogicalOp.OR, left_predicate.value,
                         right_predicate.value))
