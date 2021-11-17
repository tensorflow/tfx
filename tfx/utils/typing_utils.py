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

import collections
from typing import TypeVar, Mapping, MutableMapping, Sequence, MutableSequence, Any, Dict, List

import tfx.types

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


def is_homogeneous_artifact_list(value: Any) -> bool:
  """Checks value is Sequence[T] where T is subclass of Artifact."""
  return (
      isinstance(value, collections.abc.Sequence) and
      (not value or
       (issubclass(type(value[0]), tfx.types.Artifact) and
        all(isinstance(v, type(value[0])) for v in value[1:]))))


def is_artifact_multimap(value: Any) -> bool:
  """Checks value is Mapping[str, Sequence[Artifact]] type."""
  if not isinstance(value, collections.abc.Mapping):
    return False
  for key, list_artifacts in value.items():
    if (not isinstance(key, str) or
        not isinstance(list_artifacts, collections.abc.Sequence) or
        not all(isinstance(v, tfx.types.Artifact) for v in list_artifacts)):
      return False
  return True


def is_list_of_artifact_multimap(value):
  """Checks value is Sequence[Mapping[str, Sequence[Artifact]]] type."""
  return (isinstance(value, collections.abc.Sequence) and
          all(is_artifact_multimap(v) for v in value))
