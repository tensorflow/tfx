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
"""JSON compatible type checking functions.

Internal use only. No backwards compatibility guarantees.
"""

from typing import Any, Type, Union

_JSON_COMPATIBLE_PRIMITIVES = frozenset(
    [int, float, str, bool, type(None), Any])


def is_json_compatible(
    typehint: Type,  # pylint: disable=g-bare-generic
) -> bool:
  """Check if a type hint represents a JSON-compatible type.

  Currently, 'JSON-compatible' can be the following two cases:
    1. A type conforms with T, where T is defined
    as `X = Union['T', int, float, str, bool, NoneType]`,
    `T = Union[List['X'], Dict[str, 'X']]`, It can be Optional. ForwardRef is
    not allowed.
    2. Use `Any` to indicate any type conforms with case 1. Note Any is only
    allowed with Dict or List. A standalone `Any` is invalid.

  Args:
    typehint: The typehint to check.
  Returns:
    True if typehint is a JSON-compatible type.
  """
  def check(typehint: Any, not_primitive: bool = True) -> bool:
    origin = getattr(typehint, '__origin__', typehint)
    args = getattr(typehint, '__args__', None)
    if origin is dict or origin is list or origin is Union:

    # Starting from Python 3.9 Dict won't have default args (~KT, ~VT)
    # and List won't have default args (~T).
      if not args:
        return False
      elif origin is dict and args[0] is not str:
        return False
      elif origin is dict and args[0] is str:
        return check(typehint=args[1], not_primitive=False)
      # Handle top level optional.
      elif origin is Union and not_primitive:
        return all([
            arg is type(None) or
            check(typehint=arg, not_primitive=True) for arg in args
        ])
      else:
        return all([check(typehint=arg, not_primitive=False) for arg in args])
    else:
      return not not_primitive and origin in _JSON_COMPATIBLE_PRIMITIVES
  return check(typehint, not_primitive=True)


def check_strict_json_compat(
    in_type: Any, expect_type: Type) -> bool:  # pylint: disable=g-bare-generic
  """Check if in_type conforms with expect_type.

  Args:
    in_type: Input type hint. It can be any JSON-compatible type. It can also be
    an instance.
    expect_type: Expected type hint. It can be any JSON-compatible type.

  Returns:
    True if in_type is valid w.r.t. expect_type.
  """
  check_instance = False
  if getattr(in_type, '__module__', None) not in {'typing', 'builtins'}:
    check_instance = True

  def _check(in_type: Any, expect_type: Type) -> bool:  # pylint: disable=g-bare-generic
    """Check if in_type conforms with expect_type."""
    if in_type is Any:
      return expect_type is Any
    elif expect_type is Any:
      return True

    in_obj = None
    if check_instance:
      in_obj, in_type = in_type, type(in_type)

    in_args = getattr(in_type, '__args__', ())
    in_origin = getattr(in_type, '__origin__', in_type)
    expect_args = getattr(expect_type, '__args__', ())
    expect_origin = getattr(expect_type, '__origin__', expect_type)

    if in_origin is Union:
      return all(_check(arg, expect_type) for arg in in_args)
    if expect_origin is Union:
      if check_instance:
        return any(_check(in_obj, arg) for arg in expect_args)
      else:
        return any(_check(in_type, arg) for arg in expect_args)

    if in_origin != expect_origin:
      return False
    elif in_origin in (
        dict, list
    ) and expect_args and expect_args[0].__class__.__name__ == 'TypeVar':
      return True
    elif check_instance:
      if isinstance(in_obj, list):
        return not expect_args or all(
            [_check(o, expect_args[0]) for o in in_obj])
      elif isinstance(in_obj, dict):
        return not expect_args or (
            all(_check(k, expect_args[0]) for k in in_obj.keys()) and
            all(_check(v, expect_args[1]) for v in in_obj.values()))
      else:
        return True
    # For List -> List[X] and Dict -> Dict[X, Y].
    elif len(in_args) < len(expect_args):
      return False
    # For Python 3.7, where Dict and List have args KT, KV, T. Return True
    # whenever the expect type is Dict or List.
    else:
      return all(_check(*arg) for arg in zip(in_args, expect_args))

  return _check(in_type, expect_type)
