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
"""Portable library for output artifacts resolution including caching decision.
"""

import collections
import copy
import datetime
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import uuid

from absl import logging
from tfx import types
from tfx import version
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import node_proto_view
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact as tfx_artifact
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
# LINT.IfChange
RESOLVED_AT_RUNTIME = '{resolved_at_runtime}'
# LINT.ThenChange(<Internal source code>)
_ORCHESTRATOR_GENERATED_BCL_DIR = 'orchestrator_generated_bcl'
_STATEFUL_WORKING_DIR_INDEX = '__stateful_working_dir_index__'


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


def remove_stateful_working_dir(stateful_working_dir: str) -> None:
  """Remove stateful_working_dir."""
  try:
    fileio.rmtree(stateful_working_dir)
    logging.info('Deleted stateful_working_dir %s', stateful_working_dir)
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


def get_node_dir(
    pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec, node_id: str
) -> str:
  """Gets node dir for the given pipeline node."""
  return os.path.join(
      pipeline_runtime_spec.pipeline_root.field_value.string_value, node_id
  )


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
    self._node_dir = get_node_dir(
        pipeline_runtime_spec, pipeline_node.node_info.id
    )

  def generate_output_artifacts(
      self, execution_id: int) -> Dict[str, List[types.Artifact]]:
    """Generates output artifacts given execution_id."""
    return generate_output_artifacts(
        execution_id=execution_id,
        outputs=self._pipeline_node.outputs.outputs,
        node_dir=self._node_dir)

  def get_executor_output_uri(self, execution_id: int) -> str:
    """Generates executor output uri given execution_id."""
    return get_executor_output_uri(self._node_dir, execution_id)

  def get_driver_output_uri(self) -> str:
    driver_output_dir = os.path.join(
        self._node_dir, _SYSTEM, _DRIVER_EXECUTION,
        str(int(datetime.datetime.now().timestamp() * 1000000)))
    fileio.makedirs(driver_output_dir)
    return os.path.join(driver_output_dir, _DRIVER_OUTPUT_FILE)

  def get_stateful_working_directory(
      self,
      execution: metadata_store_pb2.Execution,
  ) -> str:
    """Generates stateful working directory.

    Args:
      execution: execution containing stateful_working_dir_index.

    Returns:
      Path to stateful working directory.
    """
    return get_stateful_working_directory(
        self._node_dir,
        execution=execution,
    )

  def make_tmp_dir(self, execution_id: int) -> str:
    """Generates a temporary directory."""
    return make_tmp_dir(self._node_dir, execution_id)


def _generate_output_artifact(
    output_spec: pipeline_pb2.OutputSpec) -> types.Artifact:
  """Generates each output artifact given output_spec."""
  artifact = artifact_utils.deserialize_artifact(output_spec.artifact_spec.type)
  _attach_artifact_properties(output_spec.artifact_spec, artifact)

  if output_spec.artifact_spec.is_async:
    # Mark the artifact state as REFERENCE to distinguish it from PUBLISHED
    # (LIVE in MLMD) intermediate artifacts emitted during a component's
    # execution. At the end  of the component's execution, its state will remain
    # REFERENCE instead of changing to PUBLISHED.
    artifact.state = tfx_artifact.ArtifactState.REFERENCE

  return artifact


def generate_output_artifacts(
    execution_id: int,
    outputs: Mapping[str, pipeline_pb2.OutputSpec],
    node_dir: str) -> Dict[str, List[types.Artifact]]:
  """Generates output artifacts.

  Args:
    execution_id: The id of the execution.
    outputs: Mapping from artifact key to its OutputSpec value in pipeline IR.
    node_dir: The root directory of the node.

  Returns:
    Mapping from artifact key to the list of TFX artifacts.
  """

  output_artifacts = collections.defaultdict(list)
  for key, output_spec in outputs.items():
    artifact = _generate_output_artifact(output_spec)
    if output_spec.artifact_spec.external_artifact_uris:
      for external_uri in output_spec.artifact_spec.external_artifact_uris:
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


def get_executor_output_dir(execution_info: data_types.ExecutionInfo) -> str:
  """Generates executor output directory for a given execution info."""
  return os.path.dirname(execution_info.execution_output_uri)


def get_executor_output_uri(node_dir, execution_id: int) -> str:
  """Generates executor output uri for a given execution_id."""
  execution_dir = os.path.join(node_dir, _SYSTEM, _EXECUTOR_EXECUTION,
                               str(execution_id))
  fileio.makedirs(execution_dir)
  return os.path.join(execution_dir, _EXECUTOR_OUTPUT_FILE)


def get_stateful_working_dir_index(
    execution: Optional[metadata_store_pb2.Execution] = None,
) -> str:
  """Gets stateful working directory index.

  Returned the UUID stored in the execution. If the execution is not provided or
  UUID is not found in the execution, a new UUID will be returned.

  Args:
    execution: execution that stores the stateful_working_dir_index.

  Returns:
    an index for stateful working dir.
  """
  index = None
  if (
      execution is not None
      and _STATEFUL_WORKING_DIR_INDEX in execution.custom_properties
  ):
    index = data_types_utils.get_metadata_value(
        execution.custom_properties[_STATEFUL_WORKING_DIR_INDEX])
  return str(index) if index is not None else str(uuid.uuid4())


def get_stateful_working_directory(
    node_dir: str,
    execution: metadata_store_pb2.Execution,
) -> str:
  """Generates stateful working directory.

  The generated stateful working directory will have the following pattern:
  {node_id}/.system/stateful_working_dir/{stateful_working_dir_index}. The
  stateful_working_dir_index is an UUID stored as a custom property in the
  execution. If no UUID was found in the execution, a new UUID will be
  generated and used as the directory suffix.

  Args:
    node_dir: root directory of the node.
    execution: execution containing stateful_working_dir_index.

  Returns:
    Path to stateful working directory.
  """
  # NOTE: If this directory structure is changed, please update
  # the remove_stateful_working_dir function in this file accordingly.
  # Create stateful working dir for the execution.
  stateful_working_dir_index = get_stateful_working_dir_index(execution)
  stateful_working_dir = os.path.join(
      node_dir, _SYSTEM, _STATEFUL_WORKING_DIR, stateful_working_dir_index
  )
  if not fileio.exists(stateful_working_dir):
    try:
      fileio.makedirs(stateful_working_dir)
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          'Failed to make stateful working dir: %s',
          stateful_working_dir,
      )
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


def get_orchestrator_generated_bcl_dir(
    pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec, node_id: str
) -> str:
  """Generates a root directory to hold orchestrator generated BCLs for the given node.

  Args:
    pipeline_runtime_spec: pipeline runtime specifications.
    node_id: unique id of the node within the pipeline.

  Returns:
    Path to orchestrator generated bcl root dir, which has the format
    `<node_dir>/.system/orchestrator_generated_bcl`
  """
  node_dir = get_node_dir(pipeline_runtime_spec, node_id)
  orchestrator_generated_bcl_dir = os.path.join(
      node_dir, _SYSTEM, _ORCHESTRATOR_GENERATED_BCL_DIR
  )
  if not fileio.exists(orchestrator_generated_bcl_dir):
    try:
      fileio.makedirs(orchestrator_generated_bcl_dir)
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          'Failed to make orchestrator generated bcl dir: %s',
          orchestrator_generated_bcl_dir,
      )
      raise
  return orchestrator_generated_bcl_dir
