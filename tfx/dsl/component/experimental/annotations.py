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

import inspect
from typing import Any, Dict, List, Type, Union, get_args, get_origin

from tfx.dsl.component.experimental import json_compat
from tfx.types import artifact
from tfx.utils import deprecation_utils

try:
  import apache_beam as beam  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top

  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


def _check_valid_input_artifact_params(params):
  """Check if the annotation params is an Artifact or a List[Artifact]."""
  # If the typehint is List[Artifact], unwrap it.
  if params is list or get_origin(params) is list:
    generic_arg = get_args(params)
    if generic_arg is not None and len(generic_arg) == 1:
      params = generic_arg[0]
    else:
      return False
  if (
      inspect.isclass(params)
      and issubclass(params, artifact.Artifact)
      and hasattr(params, 'TYPE_NAME')
  ):
    return True
  else:
    return False


class _ArtifactGenericMeta(type):
  """Metaclass for _ArtifactGeneric, to enable class indexing."""

  def __getitem__(
      cls: Type['_ArtifactGeneric'],
      params: Union[Type[artifact.Artifact], Type[List[artifact.Artifact]]],
  ):
    """Metaclass method allowing indexing class (`_ArtifactGeneric[T]`)."""
    return cls._generic_getitem(params)  # pytype: disable=attribute-error


class _ArtifactGeneric(metaclass=_ArtifactGenericMeta):
  """A generic that takes a Type[tfx.types.Artifact] as its single argument."""

  def __init__(  # pylint: disable=invalid-name
      self,
      artifact_type: Union[
          Type[artifact.Artifact], Type[List[artifact.Artifact]]
      ],
      _init_via_getitem=False,
  ):
    if not _init_via_getitem:
      class_name = self.__class__.__name__
      raise ValueError(
          '%s should be instantiated via the syntax `%s[T]`, where T is a '
          'subclass of tfx.types.Artifact.' % (class_name, class_name)
      )
    self.type = artifact_type

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_ArtifactGeneric[T]` for a given type T."""
    # Check that the given parameter is a concrete (i.e. non-abstract) subclass
    # of `tfx.types.Artifact`.
    if (
        inspect.isclass(params)
        and issubclass(params, artifact.Artifact)
        and params.TYPE_NAME
    ):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          (
              'Generic type `%s[T]` expects the single parameter T to be a '
              'concrete subclass of `tfx.types.Artifact` (got %r instead).'
          )
          % (class_name, params)
      )

  def __repr__(self):
    return '%s[%s]' % (self.__class__.__name__, self.type)


class _PrimitiveTypeGenericMeta(type):
  """Metaclass for _PrimitiveTypeGeneric, to enable primitive type indexing."""

  def __getitem__(
      cls: Type['_PrimitiveTypeGeneric'],
      params: Type[Union[int, float, str, bool, List[Any], Dict[Any, Any]]],
  ):
    """Metaclass method allowing indexing class (`_PrimitiveTypeGeneric[T]`)."""
    return cls._generic_getitem(params)  # pytype: disable=attribute-error


class _PrimitiveTypeGeneric(metaclass=_PrimitiveTypeGenericMeta):
  """A generic that takes a primitive type as its single argument."""

  def __init__(  # pylint: disable=invalid-name
      self,
      artifact_type: Type[Union[int, float, str, bool]],
      _init_via_getitem=False,
  ):
    if not _init_via_getitem:
      class_name = self.__class__.__name__
      raise ValueError(
          (
              '%s should be instantiated via the syntax `%s[T]`, where T is '
              '`int`, `float`, `str`, or `bool`.'
          )
          % (class_name, class_name)
      )
    self._type = artifact_type

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_PrimitiveTypeGeneric[T]` for a given type T."""
    # Check that the given parameter is a primitive type.
    if (
        inspect.isclass(params)
        and params in (int, float, str, bool)
        or json_compat.is_json_compatible(params)
    ):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          (
              'Generic type `%s[T]` expects the single parameter T to be '
              '`int`, `float`, `str`, `bool` or JSON-compatible types '
              '(Dict[str, T], List[T]) (got %r instead).'
          )
          % (class_name, params)
      )

  def __repr__(self):
    return '%s[%s]' % (self.__class__.__name__, self._type)

  @property
  def type(self):
    return self._type


class _PipelineTypeGenericMeta(type):
  """Metaclass for _PipelineTypeGeneric."""

  def __getitem__(
      cls: Type['_PipelineTypeGeneric'], params: Type[_BeamPipeline]
  ):
    """Metaclass method allowing indexing class (`_PipelineTypeGeneric[T]`)."""
    return cls._generic_getitem(params)  # pytype: disable=attribute-error


class _PipelineTypeGeneric(metaclass=_PipelineTypeGenericMeta):
  """A generic that takes a beam.Pipeline as its single argument."""

  def __init__(  # pylint: disable=invalid-name
      self, artifact_type: Type[_BeamPipeline], _init_via_getitem=False
  ):
    if not _init_via_getitem:
      class_name = self.__class__.__name__
      raise ValueError(
          (
              '%s should be instantiated via the syntax `%s[T]`, where T is '
              '`beam.Pipeline`.'
          )
          % (class_name, class_name)
      )
    self._type = artifact_type

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_PrimitiveTypeGeneric[T]` for a given type T."""
    # Check that the given parameter is a primitive type.
    if inspect.isclass(params) and params in (_BeamPipeline,):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          (
              'Generic type `%s[T]` expects the single parameter T to be '
              '`beam.Pipeline`, got %r instead.'
          )
          % (class_name, params)
      )

  def __repr__(self):
    return '%s[%s]' % (self.__class__.__name__, self._type)

  @property
  def type(self):
    return self._type


# Typehint annotations for component authoring.


class InputArtifact(_ArtifactGeneric):
  """Input artifact object type annotation."""

  @classmethod
  def _generic_getitem(cls, params):
    """Return the result of `_ArtifactGeneric[T]` for a given type T."""
    # Check that the given parameter is a concrete (i.e. non-abstract) subclass
    # of `tfx.types.Artifact`, or a List of `tfx.types.Artifact`.
    if _check_valid_input_artifact_params(params):
      return cls(params, _init_via_getitem=True)
    else:
      class_name = cls.__name__
      raise ValueError(
          (
              'Generic type `%s[T]` expects the single parameter T to be a '
              'concrete subclass of `tfx.types.Artifact` or a List of '
              '`tfx.types.Artifact` (got %r instead).'
          )
          % (class_name, params)
      )


class OutputArtifact(_ArtifactGeneric):
  """Output artifact object type annotation."""


class Parameter(_PrimitiveTypeGeneric):
  """Component parameter type annotation."""


class BeamComponentParameter(_PipelineTypeGeneric):
  """Component parameter type annotation."""


class OutputDict:
  """Decorator declaring component executor function outputs.

  Now @component can understand TypedDict return type annotation as well, so
  please use a TypedDict instead of using an OutputDict.
  """

  @deprecation_utils.deprecated('2023-08-25', 'Please use TypedDict instead.')
  def __init__(self, **kwargs):
    self.kwargs = kwargs
