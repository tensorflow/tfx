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
"""Portable library for output artifacts resolution including caching decision."""

import collections
import copy
import datetime
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from absl import logging
from tfx import types
from tfx import version
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import node_proto_view
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.types.value_artifact import ValueArtifact
from tfx.utils import proto_utils

from ml_metadata.proto import metadata_store_pb2

_SYSTEM = '.system'
_EXECUTOR_EXECUTION = 'executor_execution'
_DRIVER_EXECUTION = 'driver_execution'
_STATEFUL_WORKING_DIR = 'stateful_working_dir'
_DRIVER_OUTPUT_FILE = 'driver_output.pb'
_EXECUTOR_OUTPUT_FILE = 'executor_output.pb'
_VALUE_ARTIFACT_FILE_NAME = 'value'
# The fixed special value to indicate that the binary will set the output URI
# value during its execution.
RESOLVED_AT_RUNTIME = '{resolved_at_runtime}'


def make_output_dirs(
    output_dict: Mapping[str, Sequence[types.Artifact]]) -> None:
  """Make dirs for output artifacts' URI."""
  for _, artifact_list in output_dict.items():
    for artifact in artifact_list:
      # Omit lifecycle management for external artifacts.
      if artifact.is_external:
        continue
      if isinstance(artifact, ValueArtifact):
        # If this is a ValueArtifact, create the file if it does not exist.
        if not fileio.exists(artifact.uri):
          artifact_dir = os.path.dirname(artifact.uri)
          fileio.makedirs(artifact_dir)
          with fileio.open(artifact.uri, 'w') as f:
            # Because fileio.open won't create an empty file, we write an
            # empty string to it to force the creation.
            f.write('')
      else:
        # Otherwise create a dir.
        fileio.makedirs(artifact.uri)


def remove_output_dirs(
    output_dict: Mapping[str, Sequence[types.Artifact]]) -> None:
  """Remove dirs of output artifacts' URI."""
  for _, artifact_list in output_dict.items():
    for artifact in artifact_list:
      # Omit lifecycle management for external artifacts.
      if artifact.is_external:
        continue
      if fileio.isdir(artifact.uri):
        fileio.rmtree(artifact.uri)
      else:
        fileio.remove(artifact.uri)


def clear_output_dirs(
    output_dict: Mapping[str, Sequence[types.Artifact]]) -> None:
  """Clear dirs of output artifacts' URI."""
  for _, artifact_list in output_dict.items():
    for artifact in artifact_list:
      # Omit lifecycle management for external artifacts.
      if artifact.is_external:
        continue
      if fileio.isdir(artifact.uri) and fileio.listdir(artifact.uri):
        fileio.rmtree(artifact.uri)
        fileio.mkdir(artifact.uri)


def remove_stateful_working_dir(stateful_working_dir: str) -> None:
  """Remove stateful_working_dir."""
  # Clean up stateful working dir
  # Note that:
  # stateful_working_dir = os.path.join(
  #    self._node_dir,
  #    _SYSTEM,
  #    _STATEFUL_WORKING_DIR, <-- we want to clean from this level down.
  #    dir_suffix)
  stateful_working_dir = os.path.abspath(
      os.path.join(stateful_working_dir, os.pardir))
  try:
    fileio.rmtree(stateful_working_dir)
  except fileio.NotFoundError:
    logging.warning(
        'stateful_working_dir %s is not found, not going to delete it.',
        stateful_working_dir)


def _attach_artifact_properties(spec: pipeline_pb2.OutputSpec.ArtifactSpec,
                                artifact: types.Artifact):
  """Attaches properties of an artifact using ArtifactSpec."""
  for key, value in spec.additional_properties.items():
    if not value.HasField('field_value'):
      raise RuntimeError('Property value is not a field_value for %s' % key)
    if value.field_value.HasField('proto_value'):
      # Proto properties need to be unpacked from the google.protobuf.Any
      # message to its concrete message before setting the artifact property
      property_value = proto_utils.unpack_proto_any(
          value.field_value.proto_value)
    else:
      property_value = data_types_utils.get_metadata_value(value.field_value)
    setattr(artifact, key, property_value)

  for key, value in spec.additional_custom_properties.items():
    if not value.HasField('field_value'):
      raise RuntimeError('Property value is not a field_value for %s' % key)
    value_type = value.field_value.WhichOneof('value')
    if value_type == 'int_value':
      artifact.set_int_custom_property(key, value.field_value.int_value)
    elif value_type == 'string_value':
      artifact.set_string_custom_property(key, value.field_value.string_value)
    elif value_type == 'double_value':
      artifact.set_float_custom_property(key, value.field_value.double_value)
    elif value_type == 'proto_value':
      proto_value = proto_utils.unpack_proto_any(value.field_value.proto_value)
      artifact.set_proto_custom_property(key, proto_value)
    else:
      raise RuntimeError(f'Unexpected value_type: {value_type}')


class OutputsResolver:
  """This class has methods to handle launcher output related logic."""

  def __init__(self,
               pipeline_node: Union[pipeline_pb2.PipelineNode,
                                    node_proto_view.NodeProtoView],
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               execution_mode: 'pipeline_pb2.Pipeline.ExecutionMode' = (
                   pipeline_pb2.Pipeline.SYNC)):
    self._pipeline_node = pipeline_node
    self._pipeline_info = pipeline_info
    self._pipeline_root = (
        pipeline_runtime_spec.pipeline_root.field_value.string_value)
    self._pipeline_run_id = (
        pipeline_runtime_spec.pipeline_run_id.field_value.string_value)
    self._execution_mode = execution_mode
    self._node_dir = os.path.join(self._pipeline_root,
                                  pipeline_node.node_info.id)

  def generate_output_artifacts(
      self, execution_id: int) -> Dict[str, List[types.Artifact]]:
    """Generates output artifacts given execution_id."""
    return generate_output_artifacts(
        execution_id=execution_id,
        outputs=self._pipeline_node.outputs.outputs,
        node_dir=self._node_dir,
        pipeline_root=self._pipeline_root)

  def get_executor_output_uri(self, execution_id: int) -> str:
    """Generates executor output uri given execution_id."""
    execution_dir = os.path.join(self._node_dir, _SYSTEM, _EXECUTOR_EXECUTION,
                                 str(execution_id))
    fileio.makedirs(execution_dir)
    return os.path.join(execution_dir, _EXECUTOR_OUTPUT_FILE)

  def get_driver_output_uri(self) -> str:
    driver_output_dir = os.path.join(
        self._node_dir, _SYSTEM, _DRIVER_EXECUTION,
        str(int(datetime.datetime.now().timestamp() * 1000000)))
    fileio.makedirs(driver_output_dir)
    return os.path.join(driver_output_dir, _DRIVER_OUTPUT_FILE)

  def get_stateful_working_directory(self,
                                     execution_id: Optional[int] = None) -> str:
    """Generates stateful working directory given (optional) execution id.

    Args:
      execution_id: An optional execution id which will be used as part of the
        stateful working dir path if provided. The stateful working dir path
        will be <node_dir>/.system/stateful_working_dir/<execution_id>. If
        execution_id is not provided, for backward compatibility purposes,
        <pipeline_run_id> is used instead of <execution_id> but an error is
        raised if the execution_mode is not SYNC (since ASYNC pipelines have
        no pipeline_run_id).

    Returns:
      Path to stateful working directory.

    Raises:
      ValueError: If execution_id is not provided and execution_mode of the
        pipeline is not SYNC.
    """
    return get_stateful_working_directory(self._node_dir, self._execution_mode,
                                          self._pipeline_run_id, execution_id)

  def make_tmp_dir(self, execution_id: int) -> str:
    """Generates a temporary directory."""
    return make_tmp_dir(self._node_dir, execution_id)


def _generate_output_artifact(
    output_spec: pipeline_pb2.OutputSpec) -> types.Artifact:
  """Generates each output artifact given output_spec."""
  artifact = artifact_utils.deserialize_artifact(
      output_spec.artifact_spec.type)
  _attach_artifact_properties(output_spec.artifact_spec, artifact)

  return artifact


def _validate_external_uri(external_uri: str,
                           pipeline_root: Optional[str]) -> None:
  """Validates a user-defined external artifact URI."""
  if external_uri == RESOLVED_AT_RUNTIME:
    return

  if pipeline_root and pipeline_root in external_uri:
    raise ValueError('External artifact URI %s is not allowed within the '
                     'pipeline base directory.' % external_uri)


def generate_output_artifacts(
    execution_id: int,
    outputs: Mapping[str, pipeline_pb2.OutputSpec],
    node_dir: str,
    pipeline_root: Optional[str] = None) -> Dict[str, List[types.Artifact]]:
  """Generates output artifacts.

  Args:
    execution_id: The id of the execution.
    outputs: Mapping from artifact key to its OutputSpec value in pipeline IR.
    node_dir: The root directory of the node.
    pipeline_root: Path to root directory of the pipeline.

  Returns:
    Mapping from artifact key to the list of TFX artifacts.

  Raises:
    ValueError: If any external artifact uri is inside the pipeline_root.
  """

  output_artifacts = collections.defaultdict(list)
  for key, output_spec in outputs.items():
    artifact = _generate_output_artifact(output_spec)
    if output_spec.artifact_spec.external_artifact_uris:
      for external_uri in output_spec.artifact_spec.external_artifact_uris:
        _validate_external_uri(external_uri, pipeline_root)
        external_artifact = copy.deepcopy(artifact)
        external_artifact.uri = external_uri
        external_artifact.is_external = True

        logging.debug('Creating external output artifact uri %s',
                      external_artifact.uri)
        output_artifacts[key].append(external_artifact)

    else:
      artifact.uri = os.path.join(node_dir, key, str(execution_id))
      if isinstance(artifact, ValueArtifact):
        artifact.uri = os.path.join(artifact.uri, _VALUE_ARTIFACT_FILE_NAME)

      logging.debug('Creating output artifact uri %s', artifact.uri)
      output_artifacts[key].append(artifact)

  return output_artifacts


def get_stateful_working_directory(node_dir: str,
                                   execution_mode: pipeline_pb2.Pipeline
                                   .ExecutionMode = pipeline_pb2.Pipeline.SYNC,
                                   pipeline_run_id: str = '',
                                   execution_id: Optional[int] = None) -> str:
  """Generates stateful working directory.

  Args:
    node_dir: The root directory of the node.
    execution_mode: Execution mode of the pipeline.
    pipeline_run_id: Optional pipeline_run_id, only available if execution mode
      is SYNC.
    execution_id: An optional execution id which will be used as part of the
      stateful working dir path if provided. The stateful working dir path will
      be <node_dir>/.system/stateful_working_dir/<execution_id>. If execution_id
      is not provided, for backward compatibility purposes, <pipeline_run_id> is
      used instead of <execution_id> but an error is raised if the
      execution_mode is not SYNC (since ASYNC pipelines have no
      pipeline_run_id).

  Returns:
    Path to stateful working directory.

  Raises:
    ValueError: If execution_id is not provided and execution_mode of the
      pipeline is not SYNC.
  """
  if (execution_id is None and execution_mode != pipeline_pb2.Pipeline.SYNC):
    raise ValueError(
        'Cannot create stateful working dir if execution id is `None` and '
        'the execution mode of the pipeline is not `SYNC`.')

  if execution_id is None:
    dir_suffix = pipeline_run_id
  else:
    dir_suffix = str(execution_id)

  # TODO(b/150979622): We should introduce an id that is not changed across
  # retries of the same component run to provide better isolation between
  # "retry" and "new execution". When it is available, introduce it into
  # stateful working directory.
  # NOTE: If this directory structure is changed, please update
  # the remove_stateful_working_dir function in this file accordingly.
  stateful_working_dir = os.path.join(node_dir, _SYSTEM, _STATEFUL_WORKING_DIR,
                                      dir_suffix)
  try:
    fileio.makedirs(stateful_working_dir)
  except Exception:  # pylint: disable=broad-except
    logging.exception('Failed to make stateful working dir: %s',
                      stateful_working_dir)
    raise
  return stateful_working_dir


def make_tmp_dir(node_dir: str, execution_id: int) -> str:
  """Generates a temporary directory."""
  result = os.path.join(node_dir, _SYSTEM, _EXECUTOR_EXECUTION,
                        str(execution_id), '.temp', '')
  fileio.makedirs(result)
  return result


def tag_output_artifacts_with_version(
    output_artifacts: Optional[Mapping[str, Sequence[types.Artifact]]] = None):
  """Tag output artifacts with the current TFX version."""
  if not output_artifacts:
    return
  for unused_key, artifact_list in output_artifacts.items():
    for artifact in artifact_list:
      if not artifact.has_custom_property(
          artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY):
        artifact.set_string_custom_property(
            artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY,
            version.__version__)


def populate_output_artifact(
    executor_output: execution_result_pb2.ExecutorOutput,
    output_dict: Mapping[str, Sequence[types.Artifact]]):
  """Populate output_dict to executor_output."""
  for key, artifact_list in output_dict.items():
    artifacts = execution_result_pb2.ExecutorOutput.ArtifactList()
    for artifact in artifact_list:
      artifacts.artifacts.append(artifact.mlmd_artifact)
    executor_output.output_artifacts[key].CopyFrom(artifacts)


def populate_exec_properties(
    executor_output: execution_result_pb2.ExecutorOutput,
    exec_properties: Mapping[str, Any]):
  """Populate exec_properties to executor_output."""
  for key, value in exec_properties.items():
    v = metadata_store_pb2.Value()
    if isinstance(value, str):
      v.string_value = value
    elif isinstance(value, int):
      v.int_value = value
    elif isinstance(value, float):
      v.double_value = value
    else:
      logging.info(
          'Value type %s of key %s in exec_properties is not '
          'supported, going to drop it', type(value), key)
      continue
    executor_output.execution_properties[key].CopyFrom(v)
