# Lint as: python3
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
"""Recording pipeline from MLMD metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tensorflow as tf
from typing import Iterable, List, Mapping, Optional, Text, Tuple

from tfx.orchestration import metadata
from tfx.utils import io_utils

def _get_paths(metadata_connection: metadata.Metadata,
               execution_ids: List[int],
               output_dir: Text) -> Iterable[Tuple[Text, Text]]:
  """Yields tuple with source and destination artifact uris.

  The destination artifact uris are located in the output_dir. The source
  artifact uris are retrieved using execution ids. If the artifact name is
  missing, the default name would be set to {component_id}_name.

  Args:
    metadata_connection: Instance of metadata.Metadata for I/O to MLMD.
    execution_ids: List of execution ids of a pipeline run.
    output_dir: Directory path where the pipeline outputs should be recorded.

  Yields:
    Iterable over tuples of source uri and destination uri.
  """
  events = metadata_connection.store.get_events_by_execution_ids(
      execution_ids)
  output_events = [
      x for x in events
      if x.type == metadata_store_pb2.Event.OUTPUT
  ]
  unique_artifact_ids = list({x.artifact_id for x in output_events})

  for artifact in metadata_connection.store.get_artifacts_by_id(
      unique_artifact_ids):
    src_uri = artifact.uri
    artifact_properties = artifact.custom_properties
    component_id = artifact_properties['producer_component'].string_value
    output_key = artifact_properties['name'].string_value
    if not component_id or not output_key:
      raise ValueError("component_id and output_key cannot be None.")
    dest_uri = os.path.join(output_dir, component_id, output_key)
    yield (src_uri, dest_uri)

def _get_execution_dict(metadata_connection: metadata.Metadata
                       ) -> Mapping[Text, List[metadata_store_pb2.Execution]]:
  """Returns a dictionary holding list of executions for all run_id in MLMD.

  Args:
    metadata_connection: Instance of metadata.Metadata for I/O to MLMD.

  Returns:
    A dictionary that holds list of executions for a run_id.
  """
  execution_dict = collections.defaultdict(list)
  for execution in metadata_connection.store.get_executions():
    execution_run_id = execution.properties['run_id'].string_value
    execution_dict[execution_run_id].append(execution)
  return execution_dict

def _get_latest_executions(metadata_connection: metadata.Metadata,
                           pipeline_name: Text
                           ) -> List[metadata_store_pb2.Execution]:
  """Fetches executions associated with the latest context.

  Args:
    metadata_connection: Instance of metadata.Metadata for I/O to MLMD.
    pipeline_name: Name of the pipeline to rerieve the latest executions for.

  Returns:
    List of executions for the latest run of a pipeline with the given
    pipeline_name.
  """
  pipeline_run_contexts = [
      c for c in metadata_connection.store.get_contexts_by_type(
          metadata._CONTEXT_TYPE_PIPELINE_RUN)  # pylint: disable=protected-access
      if c.properties['pipeline_name'].string_value == pipeline_name
  ]
  latest_context = max(pipeline_run_contexts,
                       key=lambda c: c.last_update_time_since_epoch)
  return metadata_connection.store.get_executions_by_context(latest_context.id)

def record_pipeline(output_dir: Text,
                    metadata_db_uri: Optional[Text],
                    host: Optional[Text],
                    port: Optional[int],
                    pipeline_name: Optional[Text],
                    run_id: Optional[Text]) -> None:
  """Record pipeline run with run_id to output_dir.

  For the beam pipeline, metadata_db_uri is required. For KFP pipeline,
  host and port should be specified. If run_id is not specified, then
  pipeline_name ought to be specified in order to fetch the latest execution
  for the specified pipeline.

  Args:
    output_dir: Directory path where the pipeline outputs should be recorded.
    metadata_db_uri: Uri to metadata db.
    host: Hostname of the metadata grpc server
    port: Port number of the metadata grpc server.
    pipeline_name: Pipeline name, which is required if run_id isn't specified.
    run_id: Pipeline execution run_id.

  Raises:
    ValueError: In cases of invalid arguments:
      - metadata_db_uri is None or host and/or port is None.
      - run_id is None and pipeline_name is None.
    FileNotFoundError: if the source artifact uri does not already exist.
  """
  if host is not None and port is not None:
    metadata_config = metadata_store_pb2.MetadataStoreClientConfig(
        host=host, port=port)
  elif metadata_db_uri is not None:
    metadata_config = metadata.sqlite_metadata_connection_config(
        metadata_db_uri)
  else:
    raise ValueError('For KFP, host and port are required. '
                     'For beam pipeline, metadata_db_uri is required.')

  with metadata.Metadata(metadata_config) as metadata_connection:
    if run_id is None:
      if pipeline_name is None:
        raise ValueError('If the run_id is not specified,'
                         ' pipeline_name should be specified')
      # fetch executions of the most recently updated execution context.
      executions = _get_latest_executions(metadata_connection,
                                          pipeline_name)
    else:
      execution_dict = _get_execution_dict(metadata_connection)
      if run_id in execution_dict:
        executions = execution_dict[run_id]
      else:
        raise ValueError(
            "run_id {} is not recorded in the MLMD metadata".format(run_id))
    execution_ids = [e.id for e in executions]
    for src_uri, dest_uri in _get_paths(metadata_connection,
                                        execution_ids,
                                        output_dir):
      if not tf.io.gfile.exists(src_uri):
        raise FileNotFoundError("{} does not exist".format(src_uri))
      io_utils.copy_dir(src_uri, dest_uri)
    logging.info("Pipeline Recorded at %s", output_dir)
