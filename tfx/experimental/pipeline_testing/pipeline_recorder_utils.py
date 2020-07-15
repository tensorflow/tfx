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
import subprocess

import absl
from ml_metadata.proto import metadata_store_pb2
from typing import Dict, List, Text, Union

from tfx.orchestration import metadata
from tfx.utils import io_utils

def get_paths(metadata_connection: metadata.Metadata,
              execution_dict: Dict[Text, List[metadata_store_pb2.Execution]],
              output_dir: Text,
              run_id: Text):
  """Returns a zip of source artifact uris and destination uris in output_dir.

  Args:
    metadata_connection: A class for metadata I/O to metadata db.
    execution_dict: A dictionary that holds executions for pipeline run_id.
    output_dir: Uri to metadata db.
    run_id: Pipeline execution run_id.

  Returns:
    Zip of src_uris and dest_uris.
  """
  events = [
      x for x in metadata_connection.store.get_events_by_execution_ids(
          [e.id for e in execution_dict[run_id]])
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

def record_pipeline(output_dir: Text, metadata_db_uri: Text, run_id: Text
                    ) -> None:
  """Record pipeline run with run_id to output_dir.

  Args:
    output_dir: Directory to record pipeline outputs to.
    metadata_db_uri: Uri to metadata db.
    run_id: Pipeline execution run_id.
  """
  metadata_config = metadata.sqlite_metadata_connection_config(metadata_db_uri)
  with metadata.Metadata(metadata_config) as metadata_connection:
    execution_dict = get_execution_dict(metadata_connection)
    if run_id is None:
      run_id = max(execution_dict.keys()) # fetch the latest run_id
    elif run_id not in execution_dict:
      raise ValueError(
          "run_id {} is not recorded in the MLMD metadata".format(run_id))
    for src_uri, dest_uri in \
          get_paths(metadata_connection, execution_dict, output_dir, run_id):
      if not os.path.exists(src_uri):
        raise FileNotFoundError("{} does not exist".format(src_uri))
      os.makedirs(dest_uri, exist_ok=True)
      io_utils.copy_dir(src_uri, dest_uri)
    absl.logging.info("Pipeline Recorded at %s", output_dir)

def record_kfp_pipeline(
    connection_config: metadata_store_pb2.MetadataStoreClientConfig,
    run_id: Text, output_dir: Text) -> None:
  """Record KFP pipeline run with run_id to output_dir.

  Args:
    connection_config: an instance used to configure connection to
      a ML Metadata connection.
    run_id: Pipeline execution run_id.
    output_dir: Directory to record pipeline outputs to.
  """
  with metadata.Metadata(connection_config) as metadata_connection:
    execution_dict = get_execution_dict(metadata_connection)
    if run_id not in execution_dict:
      raise ValueError(
          "run_id {} is not recorded in the MLMD metadata".format(
              FLAGS.run_id))
    for src_uri, dest_uri in \
          get_paths(metadata_connection, execution_dict, output_dir, run_id):
      src_uri = src_uri + '/*'
      os.makedirs(dest_uri, exist_ok=True)
      subprocess.run(['gsutil', 'cp', '-r', src_uri, dest_uri], check=True)
      io_utils.copy_dir(src_uri, dest_uri)
    absl.logging.info("Pipeline Recorded at %s", output_dir)
