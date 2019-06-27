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
"""TFX ml metadata library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import hashlib
import os
import types
import tensorflow as tf
from typing import Any, Dict, List, Optional, Set, Text, Type
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.orchestration import data_types
from tfx.utils.types import ARTIFACT_STATE_PUBLISHED
from tfx.utils.types import TfxArtifact

# Maximum number of executions we look at for previous result.
MAX_EXECUTIONS_FOR_CACHE = 100


def filed_based_metadata_connection_config(
    metadata_db_uri: Text) -> metadata_store_pb2.ConnectionConfig:
  """Convenient function to create file based metadata connection config.

  Args:
    metadata_db_uri: uri to metadata db.

  Returns:
    A metadata_store_pb2.ConnectionConfig based on given metadata db uri.
  """
  tf.gfile.MakeDirs(os.path.dirname(metadata_db_uri))
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.filename_uri = metadata_db_uri
  connection_config.sqlite.connection_mode = \
    metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
  return connection_config


# TODO(ruoyu): Figure out the story mutable UDFs. We should not reuse previous
# run when having different UDFs.
class Metadata(object):
  """Helper class to handle metadata I/O."""

  def __init__(self,
               connection_config: metadata_store_pb2.ConnectionConfig) -> None:
    self._connection_config = connection_config
    self._store = None

  def __enter__(self) -> 'Metadata':
    # TODO(ruoyu): Establishing a connection pool instead of newing
    # a connection every time. Until then, check self._store before usage
    # in every method.
    self._store = metadata_store.MetadataStore(self._connection_config)
    return self

  def __exit__(self, exc_type: Optional[Type[Exception]],
               exc_value: Optional[Exception],
               exc_tb: Optional[types.TracebackType]) -> None:
    self._store = None

  @property
  def store(self) -> metadata_store.MetadataStore:
    """Returns underlying MetadataStore.

    Raises:
      RuntimeError: if this instance is not in enter state.
    """
    if self._store is None:
      raise RuntimeError('Metadata object is not in enter state')
    return self._store

  def _prepare_artifact_type(self,
                             artifact_type: metadata_store_pb2.ArtifactType
                            ) -> metadata_store_pb2.ArtifactType:
    if artifact_type.id:
      return artifact_type
    type_id = self._store.put_artifact_type(artifact_type)
    artifact_type.id = type_id
    return artifact_type

  def update_artifact_state(self, artifact: metadata_store_pb2.Artifact,
                            new_state: Text) -> None:
    if not artifact.id:
      raise ValueError('Artifact id missing for {}'.format(artifact))
    artifact.properties['state'].string_value = new_state
    self._store.put_artifacts([artifact])

  # This should be atomic. However this depends on ML metadata transaction
  # support.
  def check_artifact_state(self, artifact: metadata_store_pb2.Artifact,
                           expected_states: Set[Text]) -> None:
    if not artifact.id:
      raise ValueError('Artifact id missing for {}'.format(artifact))
    [artifact_in_metadata] = self._store.get_artifacts_by_id([artifact.id])
    current_artifact_state = artifact_in_metadata.properties[
        'state'].string_value
    if current_artifact_state not in expected_states:
      raise RuntimeError(
          'Artifact state for {} is {}, but one of {} expected'.format(
              artifact_in_metadata, current_artifact_state, expected_states))

  # TODO(ruoyu): Make this transaction-based once b/123573724 is fixed.
  def publish_artifacts(self, raw_artifact_list: List[TfxArtifact]
                       ) -> List[metadata_store_pb2.Artifact]:
    """Publish a list of artifacts if any is not already published."""
    artifact_list = []
    for raw_artifact in raw_artifact_list:
      artifact_type = self._prepare_artifact_type(raw_artifact.artifact_type)
      raw_artifact.set_artifact_type(artifact_type)
      if not raw_artifact.artifact.id:
        raw_artifact.state = ARTIFACT_STATE_PUBLISHED
        [artifact_id] = self._store.put_artifacts([raw_artifact.artifact])
        raw_artifact.id = artifact_id
      artifact_list.append(raw_artifact.artifact)
    return artifact_list

  def get_all_artifacts(self) -> List[metadata_store_pb2.Artifact]:
    try:
      return self._store.get_artifacts()
    except tf.errors.NotFoundError:
      return []

  def _prepare_event(self, execution_id: int, artifact_id: int, key: Text,
                     index: int, event_type: Any) -> metadata_store_pb2.Event:
    """Commits a single event to the repository."""
    event = metadata_store_pb2.Event()
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    step = event.path.steps.add()
    step.key = key
    step = event.path.steps.add()
    step.index = index
    event.type = event_type
    return event

  def _prepare_execution_type(self, type_name: Text,
                              exec_properties: Dict[Text, Any]) -> int:
    """Get a execution type. Use existing type if available."""
    try:
      execution_type = self._store.get_execution_type(type_name)
      if execution_type is None:
        raise RuntimeError('Execution type is None for {}.'.format(type_name))
      return execution_type.id
    except tf.errors.NotFoundError:
      execution_type = metadata_store_pb2.ExecutionType(name=type_name)
      execution_type.properties['state'] = metadata_store_pb2.STRING
      for k in exec_properties.keys():
        execution_type.properties[k] = metadata_store_pb2.STRING
      # TODO(ruoyu): Find a better place / solution to the checksum logic.
      if 'module_file' in exec_properties:
        execution_type.properties['checksum_md5'] = metadata_store_pb2.STRING
      execution_type.properties['pipeline_name'] = metadata_store_pb2.STRING
      execution_type.properties['pipeline_root'] = metadata_store_pb2.STRING
      execution_type.properties['run_id'] = metadata_store_pb2.STRING
      execution_type.properties['component_id'] = metadata_store_pb2.STRING

      return self._store.put_execution_type(execution_type)

  # TODO(ruoyu): Make pipeline_info and component_info required once migration
  # to go/tfx-oss-artifact-passing finishes.
  def _prepare_execution(
      self,
      type_name: Text,
      state: Text,
      exec_properties: Dict[Text, Any],
      pipeline_info: Optional[data_types.PipelineInfo] = None,
      component_info: Optional[data_types.ComponentInfo] = None
  ) -> metadata_store_pb2.Execution:
    """Create a new execution with given type and state."""
    execution = metadata_store_pb2.Execution()
    execution.type_id = self._prepare_execution_type(type_name, exec_properties)
    execution.properties['state'].string_value = tf.compat.as_text(state)
    for k, v in exec_properties.items():
      # We always convert execution properties to unicode.
      execution.properties[k].string_value = tf.compat.as_text(
          tf.compat.as_str_any(v))
    # We also need to checksum UDF file to identify different binary being
    # used. Do we have a better way to checksum a file than hashlib.md5?
    # TODO(ruoyu): Find a better place / solution to the checksum logic.
    # TODO(ruoyu): SHA instead of MD5.
    if 'module_file' in exec_properties and exec_properties[
        'module_file'] and tf.gfile.Exists(exec_properties['module_file']):
      contents = file_io.read_file_to_string(exec_properties['module_file'])
      execution.properties['checksum_md5'].string_value = tf.compat.as_text(
          tf.compat.as_str_any(
              hashlib.md5(tf.compat.as_bytes(contents)).hexdigest()))
    if pipeline_info:
      execution.properties[
          'pipeline_name'].string_value = pipeline_info.pipeline_name
      execution.properties[
          'pipeline_root'].string_value = pipeline_info.pipeline_root
      if pipeline_info.run_id:
        execution.properties['run_id'].string_value = pipeline_info.run_id
    if component_info:
      execution.properties[
          'component_id'].string_value = component_info.component_id
    return execution

  def _update_execution_state(self, execution: metadata_store_pb2.Execution,
                              new_state: Text) -> None:
    execution.properties['state'].string_value = tf.compat.as_text(new_state)
    self._store.put_executions([execution])

  # TODO(ruoyu): Deprecate this in favor of register_execution() once migration
  # to go/tfx-oss-artifact-passing finishes.
  def prepare_execution(
      self,
      type_name: Text,
      exec_properties: Dict[Text, Any],
      pipeline_info: Optional[data_types.PipelineInfo] = None,
      component_info: Optional[data_types.ComponentInfo] = None) -> int:
    """Create a new execution in metadata.

    This will be superseded by register_execution().

    Args:
      type_name: the type name of the execution.
      exec_properties: the execution properties of the execution
      pipeline_info: optional pipeline info of the execution
      component_info: optional component info of the execution

    Returns:
      execution id of the new execution
    """
    execution = self._prepare_execution(type_name, 'new', exec_properties,
                                        pipeline_info, component_info)
    [execution_id] = self._store.put_executions([execution])
    return execution_id

  def register_execution(self, exec_properties: Dict[Text, Any],
                         pipeline_info: data_types.PipelineInfo,
                         component_info: data_types.ComponentInfo) -> int:
    """Create a new execution in metadata.

    Args:
      exec_properties: the execution properties of the execution
      pipeline_info: optional pipeline info of the execution
      component_info: optional component info of the execution

    Returns:
      execution id of the new execution
    """
    return self.prepare_execution(component_info.component_type,
                                  exec_properties, pipeline_info,
                                  component_info)

  def publish_execution(
      self,
      execution_id: int,
      input_dict: Dict[Text, List[TfxArtifact]],
      output_dict: Dict[Text, List[TfxArtifact]],
      state: Optional[Text] = 'complete',
  ) -> Dict[Text, List[TfxArtifact]]:
    """Publish an execution with input and output artifacts info.

    Args:
      execution_id: id of execution to be published.
      input_dict: inputs artifacts used by the execution with id ready.
      output_dict: output artifacts produced by the execution without id.
      state: optional state of the execution, default to be 'complete'.

    Returns:
      Updated outputs with artifact ids.

    Raises:
      ValueError: If any output artifact already has id set.
    """
    [execution] = self._store.get_executions_by_id([execution_id])
    self._update_execution_state(execution, state)

    tf.logging.info(
        'Publishing execution {}, with inputs {} and outputs {}'.format(
            execution, input_dict, output_dict))
    events = []
    if input_dict:
      for key, input_list in input_dict.items():
        for index, single_input in enumerate(input_list):
          if not single_input.artifact.id:
            raise ValueError(
                'input artifact {} has missing id'.format(single_input))
          events.append(
              self._prepare_event(
                  execution_id=execution_id,
                  artifact_id=single_input.artifact.id,
                  key=key,
                  index=index,
                  event_type=metadata_store_pb2.Event.INPUT))
    if output_dict:
      for key, output_list in output_dict.items():
        for index, single_output in enumerate(output_list):
          if single_output.artifact.id:
            raise ValueError(
                'output artifact {} already has an id'.format(single_output))
          [published_artifact] = self.publish_artifacts([single_output])  # pylint: disable=unbalanced-tuple-unpacking
          single_output.set_artifact(published_artifact)
          events.append(
              self._prepare_event(
                  execution_id=execution_id,
                  artifact_id=published_artifact.id,
                  key=key,
                  index=index,
                  event_type=metadata_store_pb2.Event.OUTPUT))
    if events:
      self._store.put_events(events)
    tf.logging.info(
        'Published execution with final outputs {}'.format(output_dict))
    return output_dict

  def _get_cached_execution_id(self, input_dict: Dict[Text, List[TfxArtifact]],
                               candidate_execution_ids: List[int]
                              ) -> Optional[int]:
    """Gets common execution ids that are related to all the artifacts in input.

    Args:
      input_dict: input used by a component run.
      candidate_execution_ids: a list of id of candidate execution.

    Returns:
      a qualified execution id or None.

    """
    input_ids = set()
    for input_list in input_dict.values():
      for single_input in input_list:
        input_ids.add(single_input.artifact.id)

    for execution_id in candidate_execution_ids:
      events = self._store.get_events_by_execution_ids([execution_id])
      execution_input_ids = set([
          event.artifact_id for event in events if event.type in [
              metadata_store_pb2.Event.INPUT,
              metadata_store_pb2.Event.DECLARED_INPUT
          ]
      ])
      if input_ids == execution_input_ids:
        tf.logging.info(
            'Found matching execution with all input artifacts: {}'.format(
                execution_id))
        return execution_id
      else:
        tf.logging.debug('Execution %d does not match desired input artifacts',
                         execution_id)
    tf.logging.info('No execution matching type id and input artifacts found')
    return None

  def _is_eligible_previous_execution(
      self, currrent_execution: metadata_store_pb2.Execution,
      target_execution: metadata_store_pb2.Execution) -> bool:
    currrent_execution.properties['run_id'].string_value = ''
    target_execution.properties['run_id'].string_value = ''
    currrent_execution.id = target_execution.id
    return currrent_execution == target_execution

  def previous_execution(self, input_artifacts: Dict[Text, List[TfxArtifact]],
                         exec_properties: Dict[Text, Any],
                         pipeline_info: data_types.PipelineInfo,
                         component_info: data_types.ComponentInfo
                        ) -> Optional[int]:
    """Gets eligible previous execution that takes the same inputs.

    An eligible execution should take the same inputs, execution properties and
    with the same pipeline and component properties.

    Args:
      input_artifacts: inputs used by the run.
      exec_properties: execution properties used by the run.
      pipeline_info: info of the current pipeline run.
      component_info: info of the current component.

    Returns:
      Execution id of previous run that takes the input dict. None if not found.
    """
    return self.previous_run(
        component_info.component_type,
        input_artifacts,
        exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info)

  # TODO(ruoyu): Deprecate this in favor of previous_execution() once migration
  def previous_run(self,
                   type_name: Text,
                   input_dict: Dict[Text, List[TfxArtifact]],
                   exec_properties: Dict[Text, Any],
                   pipeline_info: Optional[data_types.PipelineInfo] = None,
                   component_info: Optional[data_types.ComponentInfo] = None
                  ) -> Optional[int]:
    """Gets previous run of same type that takes current set of input.

    Args:
      type_name: the type of run.
      input_dict: inputs used by the run.
      exec_properties: execution properties used by the run.
      pipeline_info: optional pipeline info.
      component_info: optional component info.

    Returns:
      Execution id of previous run that takes the input dict. None if not found.
    """

    tf.logging.info(
        'Checking previous run for execution_type_name {} and input_dict {}'
        .format(type_name, input_dict))

    # Ids of candidate executions which share the same execution property as
    # current.
    candidate_execution_ids = []
    expected_previous_execution = self._prepare_execution(
        type_name,
        'complete',
        exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info)
    for execution in self._store.get_executions_by_type(type_name):
      if self._is_eligible_previous_execution(
          copy.deepcopy(expected_previous_execution), copy.deepcopy(execution)):
        candidate_execution_ids.append(execution.id)
    candidate_execution_ids.sort(reverse=True)
    candidate_execution_ids = candidate_execution_ids[0:min(
        len(candidate_execution_ids), MAX_EXECUTIONS_FOR_CACHE)]

    return self._get_cached_execution_id(input_dict, candidate_execution_ids)

  # TODO(b/136031301): This should be merged with previous_run.
  def fetch_previous_result_artifacts(
      self, output_dict: Dict[Text, List[TfxArtifact]],
      execution_id: int) -> Dict[Text, List[TfxArtifact]]:
    """Fetches output with artifact ids produced by a previous run.

    Args:
      output_dict: a dict from name to a list of output TfxArtifact objects.
      execution_id: the id of the execution that produced the outputs.

    Returns:
      Original output_dict with artifact id inserted.

    Raises:
      RuntimeError: path change without clean metadata.
    """

    name_to_index_to_artifacts = collections.defaultdict(dict)
    for event in self._store.get_events_by_execution_ids([execution_id]):
      if event.type == metadata_store_pb2.Event.OUTPUT:
        [artifact] = self._store.get_artifacts_by_id([event.artifact_id])
        output_key = event.path.steps[0].key
        output_index = event.path.steps[1].index
        name_to_index_to_artifacts[output_key][output_index] = artifact
    for output_name, output_list in output_dict.items():
      if output_name not in name_to_index_to_artifacts:
        raise RuntimeError('Unmatched output name from previous execution.')
      index_to_artifacts = name_to_index_to_artifacts[output_name]
      if len(output_list) != len(index_to_artifacts):
        raise RuntimeError(
            'Output name expected {} items but {} retrieved'.format(
                len(output_list), len(index_to_artifacts)))
      for index, output in enumerate(output_list):
        output.set_artifact(index_to_artifacts[index])
    return dict(output_dict)

  def search_artifacts(self, artifact_name: Text, pipeline_name: Text,
                       run_id: Text,
                       producer_component_id: Text) -> List[TfxArtifact]:
    """Search artifacts that matches given info.

    Args:
      artifact_name: the name of the artifact that set by producer component.
        The name is logged both in artifacts and the events when the execution
        being published.
      pipeline_name: the name of the pipeline that produces the artifact
      run_id: the run id of the pipeline run that produces the artifact
      producer_component_id: the id of the component that produces the artifact

    Returns:
      A list of TfxArtifacts that matches the given info

    Raises:
      RuntimeError: when no matching execution is found given producer info.
    """
    producer_execution = None
    matching_artifact_ids = set()
    for execution in self._store.get_executions():
      if (execution.properties['pipeline_name'].string_value == pipeline_name
          and execution.properties['run_id'].string_value == run_id and
          execution.properties['component_id'].string_value ==
          producer_component_id):
        producer_execution = execution
    if not producer_execution:
      raise RuntimeError('Cannot find matching execution with pipeline name {},'
                         'run id {} and component id {}'.format(
                             pipeline_name, run_id, producer_component_id))
    for event in self._store.get_events_by_execution_ids(
        [producer_execution.id]):
      if (event.type == metadata_store_pb2.Event.OUTPUT and
          event.path.steps[0].key == artifact_name):
        matching_artifact_ids.add(event.artifact_id)

    result_artifacts = []
    for a in self._store.get_artifacts_by_id(list(matching_artifact_ids)):
      tfx_artifact = TfxArtifact(a.properties['type_name'].string_value)
      tfx_artifact.artifact = a
      result_artifacts.append(tfx_artifact)
    return result_artifacts
