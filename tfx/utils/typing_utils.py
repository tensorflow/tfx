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

from typing import Any, Dict, List, Mapping, MutableMapping, MutableSequence, Sequence, TypeVar

import tfx.types
from tfx.utils import pure_typing_utils
from typing_extensions import (  # pylint: disable=g-multiple-import
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

# Keep for backward compatibility.
is_compatible = pure_typing_utils.is_compatible

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
