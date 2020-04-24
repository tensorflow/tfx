# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX Python function component type annotations.

Experimental. No backwards compatibility guarantees.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from typing import Text, Type, Union
from six import with_metaclass

from tfx.types import artifact


class _ArtifactGenericMeta(type):
  """Metaclass for _ArtifactGeneric, to enable class indexing."""

  def __getitem__(cls: Type['_ArtifactGeneric'],
                  params: Type[artifact.Artifact]):
    """Metaclass method allowing indexing class (`_ArtifactGeneric[T]`)."""
    return cls._generic_getitem(params)  # pytype: disable=attribute-error


class _ArtifactGeneric(with_metaclass(_ArtifactGenericMeta, object)):
  """A generic that takes a Type[tfx.types.Artifact] as its single argument."""

  def __init__(  # pylint: disable=invalid-name
      self,
      artifact_type: Type[artifact.Artifact],
      _init_via_getitem=False):
    if not _init_via_getitem:
      class_name = self.__class__.__name__
      raise ValueError(
          ('%s should be instantiated via the syntax `%s[T]`, where T is a '
           'subclass of tfx.types.Artifact.') % (class_name, class_name))
    self.type = artifact_type

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_ArtifactGeneric[T]` for a given type T."""
    # Check that the given parameter is a concrete (i.e. non-abstract) subclass
    # of `tfx.types.Artifact`.
    if (inspect.isclass(params) and issubclass(params, artifact.Artifact) and
        params.TYPE_NAME):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          ('Generic type `%s[T]` expects the single parameter T to be a '
           'concrete subclass of `tfx.types.Artifact` (got %r instead).') %
          (class_name, params))

  def __repr__(self):
    return '%s[%s]' % (self.__class__.__name__, self.type)


class _PrimitiveTypeGenericMeta(type):
  """Metaclass for _PrimitiveTypeGeneric, to enable primitive type indexing."""

  def __getitem__(cls: Type[Union[int, float, Text, bytes]],
                  params: Type[artifact.Artifact]):
    """Metaclass method allowing indexing class (`_PrimitiveTypeGeneric[T]`)."""
    return cls._generic_getitem(params)  # pytype: disable=attribute-error


class _PrimitiveTypeGeneric(with_metaclass(_PrimitiveTypeGenericMeta, object)):
  """A generic that takes a primitive type as its single argument."""

  def __init__(  # pylint: disable=invalid-name
      self,
      artifact_type: Type[Union[int, float, Text, bytes]],
      _init_via_getitem=False):
    if not _init_via_getitem:
      class_name = self.__class__.__name__
      raise ValueError(
          ('%s should be instantiated via the syntax `%s[T]`, where T is '
           '`int`, `float`, `str` or `bytes`.') % (class_name, class_name))
    self.type = artifact_type

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_PrimitiveTypeGeneric[T]` for a given type T."""
    # Check that the given parameter is a primitive type.
    if inspect.isclass(params) and params in (int, float, Text, bytes):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          ('Generic type `%s[T]` expects the single parameter T to be '
           '`int`, `float`, `str` or `bytes` (got %r instead).') %
          (class_name, params))

  def __repr__(self):
    return '%s[%s]' % (self.__class__.__name__, self.type)

# Typehint annotations for component authoring.


class InputArtifact(_ArtifactGeneric):
  """Input artifact object type annotation."""
  pass


class OutputArtifact(_ArtifactGeneric):
  """Output artifact object type annotation."""
  pass


class Parameter(_PrimitiveTypeGeneric):
  """Component parameter type annotation."""
  pass


# TODO(ccy): potentially make this compatible `typing.TypedDict` in
# Python 3.8, to allow for component return value type checking.
class OutputDict(object):
  """Decorator declaring component executor function outputs."""

  def __init__(self, **kwargs):
    self.kwargs = kwargs
