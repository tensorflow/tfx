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
"""Utility for frequently used types and its typecheck."""

import collections.abc
import inspect
import typing
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, MutableSequence, Sequence, Type, TypeVar, TypedDict, get_args, get_origin

import tfx.types
from typing_extensions import (  # pylint: disable=g-multiple-import
    Annotated,  # New in python 3.9
    NotRequired,  # New in python 3.11  # pytype: disable=not-supported-yet
    Required,  # New in python 3.11  # pytype: disable=not-supported-yet
    TypeGuard,  # New in python 3.10
)

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_VT_co = TypeVar('_VT_co', covariant=True)  # pylint: disable=invalid-name # pytype: disable=not-supported-yet

# Note: Only immutable multimap can have covariant value types, because for
# zoo: MutableMapping[str, Animal], zoo['cat'].append(Dog()) is invalid.
MultiMap = Mapping[_KT, Sequence[_VT_co]]
MutableMultiMap = MutableMapping[_KT, MutableSequence[_VT]]

# Note: We don't use TypeVar for Artifact (e.g.
# TypeVar('Artifact', bound=tfx.types.Artifact)) because different key contains
# different Artifact subtypes (e.g. "examples" has Examples, "model" has Model).
# This makes, for example, artifact_dict['examples'].append(Examples()) invalid,
# but this is the best type effort we can make.
ArtifactMultiMap = MultiMap[str, tfx.types.Artifact]
ArtifactMutableMultiMap = MutableMultiMap[str, tfx.types.Artifact]
# Commonly used legacy artifact dict concrete type. Always prefer to use
# ArtifactMultiMap or ArtifactMutableMultiMap.
ArtifactMultiDict = Dict[str, List[tfx.types.Artifact]]

_T = TypeVar('_T')
_TTypedDict = TypeVar('_TTypedDict', bound=TypedDict)


def _get_typed_dict_required_keys(tp: TypedDict) -> Iterable[str]:
  if hasattr(tp, '__required_keys__'):
    return tp.__required_keys__
  if tp.__total__:
    return tp.__annotations__.keys()
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
      and all(k in tp.__annotations__ for k in value)
      and all(is_compatible(v, tp.__annotations__[k]) for k, v in value.items())
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
  maybe_origin = get_origin(tp)
  maybe_args = get_args(tp)
  if inspect.isclass(tp):
    if not maybe_args:
      if issubclass(tp, dict) and hasattr(tp, '__annotations__'):
        return _is_typed_dict_compatible(value, tp)
      return isinstance(value, tp)
  if tp is Any:
    return True
  if tp in (None, type(None)):
    return value is None
  if maybe_origin is not None:
    # Union[T]
    if maybe_origin is typing.Union:
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
      elif get_origin(subtype) is typing.Union:
        # Convert Type[Union[x, y, ...]] to Union[Type[x], Type[y], ...].
        subtypes = [typing.Type[a] for a in get_args(subtype)]
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
    ):
      if not isinstance(value, maybe_origin):
        return False
      if not maybe_args:
        return True
      assert len(maybe_args) == 1
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
          is_compatible(v, arg) for v, arg in zip(value, maybe_args))
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
          for k, v in value.items())
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
          f'Type {tp} with unsupported origin type {maybe_origin}.')
  raise NotImplementedError(f'Unsupported type {tp}.')


_TArtifact = TypeVar('_TArtifact', bound=tfx.types.Artifact)


def is_homogeneous_artifact_list(value: Any) -> TypeGuard[Sequence[_TArtifact]]:
  """Checks value is Sequence[T] where T is subclass of Artifact."""
  return (
      is_compatible(value, Sequence[tfx.types.Artifact]) and
      all(isinstance(v, type(value[0])) for v in value[1:]))


def is_artifact_list(value: Any) -> TypeGuard[Sequence[tfx.types.Artifact]]:
  return is_compatible(value, Sequence[tfx.types.Artifact])


def is_artifact_multimap(
    value: Any,
) -> TypeGuard[Mapping[str, Sequence[tfx.types.Artifact]]]:
  """Checks value is Mapping[str, Sequence[Artifact]] type."""
  return is_compatible(value, ArtifactMultiMap) or is_compatible(
      value, ArtifactMultiDict
  )


def is_list_of_artifact_multimap(
    value,
) -> TypeGuard[Sequence[Mapping[str, Sequence[tfx.types.Artifact]]]]:
  """Checks value is Sequence[Mapping[str, Sequence[Artifact]]] type."""
  return is_compatible(value, Sequence[ArtifactMultiMap])
