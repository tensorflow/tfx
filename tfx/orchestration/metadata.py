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
import hashlib
import logging
import types
import tensorflow as tf
from typing import Any, Dict, List, Optional, Set, Text, Type
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.utils.types import ARTIFACT_STATE_PUBLISHED
from tfx.utils.types import TfxType


# Maximum number of executions we look at for previous result.
MAX_EXECUTIONS_FOR_CACHE = 100


# TODO(ruoyu): Figure out the story mutable UDFs. We should not reuse previous
# run when having different UDFs.
class Metadata(object):
  """Helper class to handle metadata I/O."""

  def __init__(self,
               connection_config,
               logger):
    self._connection_config = connection_config
    self._store = None
    self._logger = logger  # For future use; no logging done yet in class.

  def __enter__(self):
    # TODO(ruoyu): Establishing a connection pool instead of newing
    # a connection every time. Until then, check self._store before usage
    # in every method.
    self._store = metadata_store.MetadataStore(self._connection_config)
    return self

  def __exit__(self, exc_type,
               exc_value,
               exc_tb):
    self._store = None

  @property
  def store(self):
    """Returns underlying MetadataStore.

    Raises:
      RuntimeError: if this instance is not in enter state.
    """
    if self._store is None:
      raise RuntimeError('Metadata object is not in enter state')
    return self._store

  def _prepare_artifact_type(self,
                             artifact_type
                            ):
    if artifact_type.id:
      return artifact_type
    type_id = self._store.put_artifact_type(artifact_type)
    artifact_type.id = type_id
    return artifact_type

  def update_artifact_state(self, artifact,
                            new_state):
    if not artifact.id:
      raise ValueError('Artifact id missing for {}'.format(artifact))
    artifact.properties['state'].string_value = new_state
    self._store.put_artifacts([artifact])

  # This should be atomic. However this depends on ML metadata transaction
  # support.
  def check_artifact_state(self, artifact,
                           expected_states):
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
  def publish_artifacts(self, raw_artifact_list
                       ):
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

  def get_all_artifacts(self):
    try:
      return self._store.get_artifacts()
    except tf.errors.NotFoundError:
      return []

  def _prepare_event(self, execution_id, artifact_id, key,
                     index, is_input):
    """Commits a single event to the repository."""
    event = metadata_store_pb2.Event()
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    step = event.path.steps.add()
    step.key = key
    step = event.path.steps.add()
    step.index = index
    if is_input:
      event.type = metadata_store_pb2.Event.DECLARED_INPUT
    else:
      event.type = metadata_store_pb2.Event.DECLARED_OUTPUT
    return event

  def _prepare_input_event(self, execution_id, artifact_id, key,
                           index):
    return self._prepare_event(execution_id, artifact_id, key, index, True)

  def _prepare_output_event(self, execution_id, artifact_id,
                            key, index):
    return self._prepare_event(execution_id, artifact_id, key, index, False)

  def _prepare_execution_type(self, type_name,
                              exec_properties):
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

      return self._store.put_execution_type(execution_type)

  def _prepare_execution(
      self, type_name, state,
      exec_properties):
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
    return execution

  def _update_execution_state(self, execution,
                              new_state):
    execution.properties['state'].string_value = tf.compat.as_text(new_state)
    self._store.put_executions([execution])

  def prepare_execution(self, type_name, exec_properties):
    execution = self._prepare_execution(type_name, 'new', exec_properties)
    [execution_id] = self._store.put_executions([execution])
    return execution_id

  def publish_execution(
      self, execution_id, input_dict,
      output_dict):
    """Publish an execution with input and output artifacts info.

    Args:
      execution_id: id of execution to be published.
      input_dict: inputs artifacts used by the execution with id ready.
      output_dict: output artifacts produced by the execution without id.

    Returns:
      Updated outputs with artifact ids.

    Raises:
      ValueError: If any output artifact already has id set.
    """
    [execution] = self._store.get_executions_by_id([execution_id])
    self._update_execution_state(execution, 'complete')

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
              self._prepare_input_event(execution_id, single_input.artifact.id,
                                        key, index))
    if output_dict:
      for key, output_list in output_dict.items():
        for index, single_output in enumerate(output_list):
          if single_output.artifact.id:
            raise ValueError(
                'output artifact {} already has an id'.format(single_output))
          [published_artifact] = self.publish_artifacts([single_output])  # pylint: disable=unbalanced-tuple-unpacking
          single_output.set_artifact(published_artifact)
          events.append(
              self._prepare_output_event(execution_id, published_artifact.id,
                                         key, index))
    if events:
      self._store.put_events(events)
    tf.logging.info(
        'Published execution with final outputs {}'.format(output_dict))
    return output_dict

  def _get_cached_execution_id(self, input_dict,
                               candidate_execution_ids):
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

  def previous_run(self, type_name, input_dict,
                   exec_properties):
    """Gets previous run of same type that takes current set of input.

    Args:
      type_name: the type of run.
      input_dict: inputs used by the run.
      exec_properties: execution properties used by the run.

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
        type_name, 'complete', exec_properties)
    for execution in self._store.get_executions_by_type(type_name):
      expected_previous_execution.id = execution.id
      if execution == expected_previous_execution:
        candidate_execution_ids.append(execution.id)
    candidate_execution_ids.sort(reverse=True)
    candidate_execution_ids = candidate_execution_ids[0:min(
        len(candidate_execution_ids), MAX_EXECUTIONS_FOR_CACHE)]

    return self._get_cached_execution_id(input_dict, candidate_execution_ids)

  # TODO(ruoyu): This should be merged with previous_run, otherwise we cannot
  # handle the case if output dict structure is changed.
  def fetch_previous_result_artifacts(
      self, output_dict,
      execution_id):
    """Fetches output with artifact ids produced by a previous run.

    Args:
      output_dict: a dict from name to a list of output TfxType objects.
      execution_id: the id of the execution that produced the outputs.

    Returns:
      Original output_dict with artifact id inserted.

    Raises:
      RuntimeError: path change without clean metadata.
    """

    name_to_index_to_artifacts = collections.defaultdict(dict)
    for event in self._store.get_events_by_execution_ids([execution_id]):
      if event.type == metadata_store_pb2.Event.DECLARED_OUTPUT:
        [artifact] = self._store.get_artifacts_by_id([event.artifact_id])
        output_key = event.path.steps[0].key
        output_index = event.path.steps[1].index
        [artifact] = self._store.get_artifacts_by_id([event.artifact_id])
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
