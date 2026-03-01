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
"""Dependency injection providers for node execution."""

from collections.abc import Container, MutableSequence, Sequence
import inspect
from typing import Any, Callable, TypeVar, get_args, get_origin, Optional

from tfx.dsl.component.experimental import json_compat
from tfx.orchestration.portable import data_types
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx.utils import pure_typing_utils
from tfx.utils.di import errors
from tfx.utils.di import providers

_TfxArtifact = artifact.Artifact
_AT = TypeVar('_AT', bound=_TfxArtifact)

_PRIMITIVE_TO_ARTIFACT = {
    int: standard_artifacts.Integer,
    float: standard_artifacts.Float,
    str: standard_artifacts.String,
    bytes: standard_artifacts.Bytes,
    bool: standard_artifacts.Boolean,
}


def _type_check(value: Any, type_hint: Any) -> bool:
  if type_hint is None:
    return True
  try:
    return pure_typing_utils.is_compatible(value, type_hint)
  except NotImplementedError:
    return True


def _is_valid_artifact_type(artifact_type: Any) -> bool:
  return (
      inspect.isclass(artifact_type)
      and issubclass(artifact_type, _TfxArtifact)
      and artifact_type.TYPE_NAME
  )


def _try_infer(
    type_hint: Any,
) -> Optional[type[standard_artifacts.ValueArtifact]]:
  if type_hint in _PRIMITIVE_TO_ARTIFACT:
    return _PRIMITIVE_TO_ARTIFACT[type_hint]
  elif json_compat.is_json_compatible(type_hint):
    return standard_artifacts.JsonValue
  return None


def _deserialize_artifact(
    target_type: type[_AT], artifacts: list[_TfxArtifact]
) -> list[_AT]:
  """Transforms list[Artifact] into desired tfx artifact class.

  This is different from artifact_utils.deserialize_artifacts which depends on
  the globally imported classes to search for the tfx artifact class. The target
  artifact class is explicitly passed, thus it is guaranteed that the artifact
  class is already imported.

  Args:
    target_type: TFX artifact type of the result.
    artifacts: Already deserialized artifacts given from ExecutionInfo.

  Returns:
    Correctly deserialized artifact list.
  """
  result = []
  for a in artifacts:
    if a.type_name != target_type.TYPE_NAME:
      raise errors.InvalidTypeHintError(
          f'type_hint uses {target_type.TYPE_NAME} but the resolved artifacts'
          f' have type_name = {a.type_name}'
      )
    if type(a) is target_type:  # pylint: disable=unidiomatic-typecheck
      result.append(a)
    else:
      new_artifact = target_type()
      new_artifact.set_mlmd_artifact_type(a.artifact_type)
      new_artifact.set_mlmd_artifact(a.mlmd_artifact)
      result.append(new_artifact)
  return result


def _transform_artifacts(
    artifacts: list[_TfxArtifact], type_hint: Any, is_input: bool = False
) -> Any:
  """Transforms raw list[Artifact] to target type_hint with type checking."""
  if type_hint is None:
    return artifacts

  origin = get_origin(type_hint)
  args = get_args(type_hint)
  if origin and args:
    # List[T]
    if (
        origin in (list, Sequence, MutableSequence)
        and len(args) == 1
        and _is_valid_artifact_type(args[0])
    ):
      return _deserialize_artifact(args[0], artifacts)

  # Optional[T]
  is_opt, unwrapped_type = pure_typing_utils.maybe_unwrap_optional(type_hint)
  if is_opt and _is_valid_artifact_type(unwrapped_type):
    artifact_type = unwrapped_type
    artifacts = _deserialize_artifact(artifact_type, artifacts)
    if not artifacts:
      return None
    elif len(artifacts) == 1:
      return artifacts[0]
    else:
      raise errors.InvalidTypeHintError(
          f'type_hint = {type_hint} but got {len(artifacts)} artifacts. Please'
          f' use list[{artifact_type.__name__}] annotation instead.'
      )

  # Just T
  if _is_valid_artifact_type(unwrapped_type):
    artifact_type = unwrapped_type
    artifacts = _deserialize_artifact(artifact_type, artifacts)
    if len(artifacts) == 1:
      return artifacts[0]
    else:
      raise errors.InvalidTypeHintError(
          f'type_hint = {type_hint} but got {len(artifacts)} artifacts. Please'
          f' use list[{artifact_type.__name__}] or'
          f' Optional[{artifact_type.__name__}] annotation instead.'
      )

  # Primitive or jsonable type_hint for a value artifact
  if (
      is_input
      and (artifact_type := _try_infer(unwrapped_type))
      is not None
  ):
    artifacts = _deserialize_artifact(artifact_type, artifacts)
    if is_opt and not artifacts:
      return None
    if len(artifacts) == 1:
      artifacts[0].read()
      return artifacts[0].value
    else:
      raise errors.InvalidTypeHintError(
          f'type_hint = {type_hint} but got {len(artifacts)} artifacts.'
          ' Please use a single value artifact for primitive types.'
      )

  raise errors.InvalidTypeHintError(f'Unsupported annotation: {type_hint}')


class FlatExecutionInfoProvider(providers.Provider):
  """Provider for flattened input/output/exec_property."""

  def __init__(self, names: Container[str], /, strict: bool = True):
    self._names = names
    self._strict = strict

  def match(self, name: str, type_hint: Any) -> bool:
    return name in self._names

  def make_factory(self, name: str, type_hint: Any) -> Callable[..., Any]:
    def provide(exec_info: data_types.ExecutionInfo):
      if name in exec_info.input_dict:
        debug_str = f'input[{name}]'
        result = _transform_artifacts(
            exec_info.input_dict[name], type_hint, is_input=True
        )
      elif name in exec_info.output_dict:
        debug_str = f'output[{name}]'
        result = _transform_artifacts(exec_info.output_dict[name], type_hint)
      elif name in exec_info.exec_properties:
        debug_str = f'exec_property[{name}]'
        result = exec_info.exec_properties[name]
      else:
        assert name in self._names, f'name = {name} not in {self._names}.'
        raise errors.FalseMatchError()

      if self._strict and not _type_check(result, type_hint):
        raise errors.InvalidTypeHintError(
            f'Given type_hint = {type_hint} but {debug_str} = {result} is not'
            ' compatible.'
        )
      return result

    return provide
