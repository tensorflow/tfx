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

from collections import defaultdict
import os

import absl
from ml_metadata.proto import metadata_store_pb2
from typing import Dict, List, Text
import tensorflow as tf

from tfx.orchestration import metadata
from tfx.utils import io_utils

def get_paths(metadata_connection: metadata.Metadata,
              executions: Dict[Text, List[metadata_store_pb2.Execution]],
              output_dir: Text):
  """Returns a zipped list of source artifact uris and destination uris.
  Destination uris are stored in output_dir.
  Args:
    metadata_connection: A class for metadata I/O to metadata db.
    executions:
    output_dir: Uri to metadata db.

  Returns:
    Zip of src_uris and dest_uris.
  """
  events = [
      x for x in metadata_connection.store.get_events_by_execution_ids(
          [e.id for e in executions])
      if x.type == metadata_store_pb2.Event.OUTPUT
  ]
  unique_artifact_ids = list({x.artifact_id for x in events})

  src_uris = []
  dest_uris = []
  for artifact in \
        metadata_connection.store.get_artifacts_by_id(unique_artifact_ids):
    src_uris.append(artifact.uri)
    component_id = \
        artifact.custom_properties['producer_component'].string_value
    name = artifact.custom_properties['name'].string_value
    dest_uris.append(os.path.join(output_dir, component_id, name))
  return zip(src_uris, dest_uris)

def get_execution_dict(metadata_connection: metadata.Metadata
                      ) -> Dict[Text, List[metadata_store_pb2.Execution]]:
  """Returns dictionary mapping holding executions for run_id.

  Args:
    metadata_connection: A class for metadata I/O to metadata db.

  Returns:
    A dictionary that holds executions for pipeline run_id
  """
  execution_dict = defaultdict(list)
  for execution in metadata_connection.store.get_executions():
    execution_run_id = execution.properties['run_id'].string_value
    execution_dict[execution_run_id].append(execution)
  return execution_dict

def get_latest_executions(metadata_connection: metadata.Metadata,
                          pipeline_name: Text
                          ) -> List[metadata_store_pb2.Execution]:
  """Fetches executions associated with the latest context.
  Args:
    metadata_connection: A class for metadata I/O to metadata db.

  Returns:
    A dictionary that holds executions for pipeline run_id
  """
  latest_context_id = None
  latest_updated_time = float('-inf')

  for pipeline_run_context in \
      metadata_connection.store.get_contexts_by_type(
          metadata._CONTEXT_TYPE_PIPELINE_RUN):  # pylint: disable=protected-access
    if pipeline_name != \
          pipeline_run_context.properties['pipeline_name'].string_value:
      continue
    if latest_updated_time < pipeline_run_context.last_update_time_since_epoch:
      latest_updated_time = pipeline_run_context.last_update_time_since_epoch
      latest_context_id = pipeline_run_context.id
  return metadata_connection.store.get_executions_by_context(latest_context_id)

def record_pipeline(output_dir: Text,
                    metadata_db_uri: Text,
                    host: Text,
                    port: int,
                    pipeline_name: Text,
                    run_id: Text) -> None:
  """Record pipeline run with run_id to output_dir. For the beam pipeline,
  metadata_db_uri is required. For KFP, host and port should be specified.

  Args:
    output_dir: Directory to record pipeline outputs to.
    metadata_db_uri: Uri to metadata db.
    host: The name or network address of the instance of MySQL to connect to.
    port: The port MySQL is using to listen for connections.
    run_id: Pipeline execution run_id.

  Raises:
    ValueError if metadata_db_uri is None or host and/or port is None.
  """
  if host is not None and port is not None:
    metadata_config = metadata_store_pb2.MetadataStoreClientConfig()
    metadata_config.host = host
    metadata_config.port = port
  elif metadata_db_uri is not None:
    metadata_config = metadata.sqlite_metadata_connection_config(
        metadata_db_uri)
  else:
    raise ValueError("For KFP, host and port are required. "\
                     "For beam pipeline, metadata_db_uri is required.")

  with metadata.Metadata(metadata_config) as metadata_connection:
    if run_id is None:
      # fetch executions of the most recently updated execution context
      executions = get_latest_executions(metadata_connection,
                                         pipeline_name)
    else:
      execution_dict = get_execution_dict(metadata_connection)
      if run_id in execution_dict:
        executions = execution_dict[run_id]
      else:
        raise ValueError(
            "run_id {} is not recorded in the MLMD metadata".format(run_id))
    for src_uri, dest_uri in \
          get_paths(metadata_connection, executions, output_dir):
      if not tf.io.gfile.exists(src_uri):
        raise FileNotFoundError("{} does not exist".format(src_uri))
      os.makedirs(dest_uri, exist_ok=True)
      io_utils.copy_dir(src_uri, dest_uri)
    absl.logging.info("Pipeline Recorded at %s", output_dir)
