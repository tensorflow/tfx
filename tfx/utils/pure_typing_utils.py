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
"""Utility for typing without other TFX dependencies (thus pure)."""

import collections.abc
import inspect
import sys
import types
import typing
from typing import Any, Iterable, Literal, Mapping, Type, TypeVar, TypedDict

from typing_extensions import (  # pylint: disable=g-multiple-import
    Annotated,  # New in python 3.9
    NotRequired,  # New in python 3.11  # pytype: disable=not-supported-yet
    Required,  # New in python 3.11  # pytype: disable=not-supported-yet
    TypeGuard,  # New in python 3.10
)

from google.protobuf.internal import enum_type_wrapper

_T = TypeVar('_T')
_TTypedDict = TypeVar('_TTypedDict', bound=TypedDict)

if sys.version_info >= (3, 10):
  _UNION_ORIGINS = (types.UnionType, typing.Union)
else:
  _UNION_ORIGINS = (typing.Union,)


# This is intentionally not public as it aims for the annotations resolver only
# for `is_compatible`.
def _get_annotations(value: Any) -> Mapping[str, Any]:
  # In python 3.10+ we have `inspect.get_annotations` which is very similar to
  # `typing.get_type_hints`, but does not perform stringized type evaluation and
  # ignores annotation inheritance by default. (Check each code docstring for
  # subtle behavior diff.) The intended behavior for is_compatible is more close
  # to `typing.get_type_hints`, so we use this.
  return typing.get_type_hints(value)


def _get_typed_dict_required_keys(tp: TypedDict) -> Iterable[str]:
  if hasattr(tp, '__required_keys__'):
    return tp.__required_keys__
  if tp.__total__:
    return _get_annotations(tp).keys()
  else:
    return []


def _is_typed_dict_compatible(
    value: Any, tp: _TTypedDict
) -> TypeGuard[_TTypedDict]:
  """Checks if the value is compatible with the given TypedDict."""
  # pytype: disable=attribute-error
  return (
      isinstance(value, dict)
      and all(k in value for k in _get_typed_dict_required_keys(tp))
      and all(k in _get_annotations(tp) for k in value)
      and all(
          is_compatible(v, _get_annotations(tp)[k]) for k, v in value.items()
      )
  )
  # pytype: enable=attribute-error


def is_compatible(value: Any, tp: Type[_T]) -> TypeGuard[_T]:
  """Whether the value is compatible with the type.

  Similar to builtin.isinstance(), but accepts more advanced subscripted type
  hints.

  Args:
    value: The value under test.
    tp: The type to check acceptability.

  Returns:
    Whether the `value` is compatible with the type `tp`.
  """
  maybe_origin = typing.get_origin(tp)
  maybe_args = typing.get_args(tp)
  if tp is Any:
    return True
  if tp in (None, type(None)):
    return value is None
  if isinstance(tp, enum_type_wrapper.EnumTypeWrapper):
    return value in tp.values()
  if inspect.isclass(tp):
    if not maybe_args:
      if issubclass(tp, dict) and hasattr(tp, '__annotations__'):
        return _is_typed_dict_compatible(value, tp)
      return isinstance(value, tp)
  if maybe_origin is not None:
    # Union[T, U] or T | U
    if maybe_origin in _UNION_ORIGINS:
      assert maybe_args, f'{tp} should be subscripted.'
      return any(is_compatible(value, arg) for arg in maybe_args)
    # Type[T]
    elif maybe_origin is type:
      if not maybe_args:
        return inspect.isclass(value)
      assert len(maybe_args) == 1
      subtype = maybe_args[0]
      if subtype is Any:
        return inspect.isclass(value)
      elif typing.get_origin(subtype) is typing.Union:
        # Convert Type[Union[x, y, ...]] to Union[Type[x], Type[y], ...].
        subtypes = [typing.Type[a] for a in typing.get_args(subtype)]
        return any(is_compatible(value, t) for t in subtypes)
      elif inspect.isclass(subtype):
        return inspect.isclass(value) and issubclass(value, subtype)
    # List[T], Set[T], FrozenSet[T], Iterable[T], Sequence[T], MutableSeuence[T]
    elif maybe_origin in (
        list,
        set,
        frozenset,
        collections.abc.Iterable,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Collection,
        collections.abc.Container,
    ):
      if not isinstance(value, maybe_origin):
        return False
      if not maybe_args:
        return True
      assert len(maybe_args) == 1
      if maybe_args[0] is str and isinstance(value, str):
        # `str` is technically Iterable[str], etc., but it's mostly not intended
        # for type checking and fail to catch a bug. Therefore we don't regard
        # str as Iterable[str], etc. This is also consistent with pytype
        # behavior:
        # https://github.com/google/pytype/blob/main/docs/faq.md#why-doesnt-str-match-against-string-iterables
        return False
      return all(is_compatible(v, maybe_args[0]) for v in value)
    # Tuple[T]
    elif maybe_origin is tuple:
      if not isinstance(value, tuple):
        return False
      if not maybe_args:
        return True
      if len(maybe_args) == 2 and maybe_args[-1] is Ellipsis:
        return all(is_compatible(v, maybe_args[0]) for v in value)
      return len(maybe_args) == len(value) and all(
          is_compatible(v, arg) for v, arg in zip(value, maybe_args)
      )
    # Dict[K, V], Mapping[K, V], MutableMapping[K, V]
    elif maybe_origin in (
        dict,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
    ):
      if not isinstance(value, maybe_origin):
        return False
      if not maybe_args:  # Unsubscripted Dict.
        return True
      assert len(maybe_args) == 2
      kt, vt = maybe_args
      return all(
          is_compatible(k, kt) and is_compatible(v, vt)
          for k, v in value.items()
      )
    # Literal[T]
    elif maybe_origin is Literal:
      assert maybe_args
      return value in maybe_args
    elif maybe_origin is Annotated:
      assert maybe_args
      return is_compatible(value, maybe_args[0])
    # Required[T] and NotRequired[T]
    elif maybe_origin in (Required, NotRequired):
      assert len(maybe_args) == 1
      return is_compatible(value, maybe_args[0])
    else:
      raise NotImplementedError(
          f'Type {tp} with unsupported origin type {maybe_origin}.'
      )
  raise NotImplementedError(f'Unsupported type {tp}.')


def maybe_unwrap_optional(type_hint: Any) -> tuple[bool, Any]:
  origin = typing.get_origin(type_hint)
  args = typing.get_args(type_hint)
  if origin and origin in _UNION_ORIGINS and args and type(None) in args:
    args_except_none = [t for t in args if t != type(None)]
    if len(args_except_none) == 1:
      return True, args_except_none[0]
  return False, type_hint
