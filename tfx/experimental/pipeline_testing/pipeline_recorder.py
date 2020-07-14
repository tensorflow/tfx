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

from absl import logging
from absl import app
from absl import flags
from ml_metadata.proto import metadata_store_pb2

from tfx.orchestration import metadata
from tfx.utils import io_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Path to record the pipeline outputs.')
flags.DEFINE_string('metadata_db_uri', None, 'Path to metadata db.')
flags.DEFINE_string('run_id', None, 'Pipeline Run Id (default=latest run_id).')

flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('metadata_db_uri')

def get_paths(metadata_connection, execution_dict, output_dir, run_id):
  events = [
      x for x in metadata_connection.store.get_events_by_execution_ids(
          [e.id for e in execution_dict[run_id]])
      if x.type == metadata_store_pb2.Event.OUTPUT
  ]
  unique_artifact_ids = list({x.artifact_id for x in events})

  src_uris = []
  dest_uris = []
  for artifact in metadata_connection.store.get_artifacts_by_id(unique_artifact_ids):
    src_uris.append(artifact.uri)
    component_id = \
        artifact.custom_properties['producer_component'].string_value
    name = artifact.custom_properties['name'].string_value
    dest_uris.append(os.path.join(output_dir, component_id, name))
  return zip(src_uris, dest_uris)

def get_execution_dict(metadata_connection):
  execution_dict = defaultdict(list)
  for execution in metadata_connection.store.get_executions():
    execution_run_id = execution.properties['run_id'].string_value
    execution_dict[execution_run_id].append(execution)
  return execution_dict

def record_pipeline(output_dir, metadata_db_uri, run_id):
  metadata_config = metadata.sqlite_metadata_connection_config(metadata_db_uri)
  with metadata.Metadata(metadata_config) as metadata_connection:
    execution_dict = get_execution_dict(metadata_connection)
    if run_id is None:
      run_id = max(execution_dict.keys()) # fetch the latest run_id
    elif run_id not in execution_dict:
      raise ValueError(
          "run_id {} is not recorded in the MLMD metadata".format(run_id))
    for src_uri, dest_uri in get_paths(metadata_connection, execution_dict, output_dir, run_id):
      if not os.path.exists(src_uri):
        raise FileNotFoundError("{} does not exist".format(src_uri))
      os.makedirs(dest_uri, exist_ok=True)
      io_utils.copy_dir(src_uri, dest_uri)
    logging.info("Pipeline Recorded at %s", output_dir)

def main(unused_argv):
  record_pipeline(FLAGS.output_dir, FLAGS.metadata_db_uri, FLAGS.run_id)

if __name__ == '__main__':
  app.run(main)
