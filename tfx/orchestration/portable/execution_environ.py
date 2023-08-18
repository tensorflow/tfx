# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Environment for component execution."""

from collections.abc import MutableSequence, Sequence
import contextlib
import inspect
from typing import Any, List, Type, TypeVar, get_args, get_origin, Optional, Union

from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact as artifact_lib
from tfx.types import artifact_utils
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


_TAny = TypeVar('_TAny')
_TArtifact = TypeVar('_TArtifact', bound=artifact_lib.Artifact)


class Environ(contextlib.ExitStack):
  """Tflex component execution environment."""

  def __init__(
      self,
      *,
      execution_info: data_types.ExecutionInfo,
      executor_output: Optional[execution_result_pb2.ExecutorOutput] = None,
  ):
    super().__init__()
    self._execution_info = execution_info
    self._executor_output = executor_output

  def _get_updated_output_artifacts(
      self, key: str
  ) -> Optional[List[metadata_store_pb2.Artifact]]:
    if (
        self._executor_output is None
        or key not in self._executor_output.output_artifacts
    ):
      return None
    return list(self._executor_output.output_artifacts[key].artifacts)

  def _get_updated_exec_properties(self, key) -> Optional[Any]:
    if self._executor_output is None:
      return None
    return self._executor_output.execution_properties.get(key, default=None)

  def strict_get(self, name: str, type_hint: Type[_TAny]) -> _TAny:
    """Get environment value with name and type hint."""

    def assert_type_hint(expected):
      if type_hint != expected:
        raise TypeError(f'Expected {type_hint} for {name} but got {expected}.')

    def try_deserialize_artifact(
        artifact: Union[metadata_store_pb2.Artifact, _TArtifact],
        artifact_type: Type[_TArtifact],
    ) -> _TArtifact:
      if isinstance(artifact, metadata_store_pb2.Artifact):
        return artifact_utils.deserialize_artifact(
            artifact_type.artifact_type,
            artifact,
        )
      return artifact

    def get_artifact_composite(
        artifact_list: Sequence[
            Union[artifact_lib.Artifact, metadata_store_pb2.Artifact]
        ],
        *,
        is_output: bool,
    ):
      debug_target = (
          f'output_dict[{name}]' if is_output else f'input_dict[{name}]'
      )
      if inspect.isclass(type_hint):
        if issubclass(type_hint, artifact_lib.Artifact):
          if len(artifact_list) != 1:
            raise TypeError(
                f'Expected 1 artifact for {debug_target} but got'
                f' {len(artifact_list)}.'
            )
          result = artifact_list[0]
          if isinstance(result, metadata_store_pb2.Artifact):
            result = artifact_utils.deserialize_artifact(
                type_hint.artifact_type, result
            )
          if not isinstance(result, type_hint):
            raise TypeError(
                f'Expected {type_hint} for {debug_target} but got'
                f' {result.__class__.__name__}.'
            )
          return result
        else:
          raise TypeError(
              f'Expected {type_hint} for {debug_target} but got'
              f' {type_hint.__name__}.'
          )
        # TODO(jjong): Add PreOutputArtifact and AsyncOutputArtifact support.
      if origin := get_origin(type_hint):
        if origin in (list, Sequence, MutableSequence):
          if args := get_args(type_hint):
            artifact_type = args[0]
            if inspect.isclass(artifact_type) and issubclass(
                artifact_type, artifact_lib.Artifact
            ):
              artifact_list = [
                  try_deserialize_artifact(a, artifact_type)
                  for a in artifact_list
              ]
              if any(not isinstance(a, artifact_type) for a in artifact_list):
                raise TypeError(
                    f'Expected {type_hint} for {debug_target} but got'
                    f' {artifact_list}'
                )
              return artifact_list
      raise TypeError(
          f'Invalid type hint {type_hint} for {debug_target}. Must be one of'
          ' `YourArtifactType`, `list[YourArtifactType]`,'
      )

    if name in self._execution_info.input_dict:
      return get_artifact_composite(
          self._execution_info.input_dict[name], is_output=False
      )
    if artifact_list := (
        self._get_updated_output_artifacts(name)
        or self._execution_info.output_dict.get(name)
    ):
      return get_artifact_composite(
          artifact_list, is_output=True
      )
    if result := (
        self._get_updated_exec_properties(name)
        or self._execution_info.exec_properties.get(name)
    ):
      if not typing_utils.is_compatible(result, type_hint):
        raise TypeError(
            f'Expected {type_hint} for exec_properties[{name}] but got'
            f' {result}.'
        )
      return result
    if name == 'execution_id':
      assert_type_hint(int)
      return self._execution_info.execution_id
    if name == 'stateful_working_dir':
      assert_type_hint(str)
      return self._execution_info.stateful_working_dir
    if name == 'tmp_dir':
      assert_type_hint(str)
      return self._execution_info.tmp_dir
    if name == 'pipeline_id':
      assert_type_hint(str)
      if self._execution_info.pipeline_info is None:
        raise RuntimeError('There is no pipeline_info to get pipeline_id')
      return self._execution_info.pipeline_info.id
    if name == 'pipeline_run_id':
      assert_type_hint(str)
      return self._execution_info.pipeline_run_id

    valid_names: set[str] = {
        *self._execution_info.input_dict,
        *self._execution_info.output_dict,
        *self._execution_info.exec_properties,
        'execution_id',
        'stateful_working_dir',
        'tmp_dir',
        'pipeline_id',
        'pipeline_run_id',
    }
    if self._executor_output is not None:
      valid_names.update({
          *self._executor_output.output_artifacts,
          *self._executor_output.execution_properties,
      })
    raise AttributeError(
        f'Unknown attribute {name}. Valid names: {valid_names}'
    )
