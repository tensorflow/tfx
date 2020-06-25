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
"""Recording pipeline from MLMD metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from distutils.dir_util import copy_tree
from collections import defaultdict

from tfx.orchestration import metadata
from ml_metadata.proto import metadata_store_pb2

def main(record_dir, pipeline_path, run_id):
  metadata_dir = os.path.join(os.environ['HOME'],
                              'tfx/tfx/experimental/pipeline_testing/',
                              'metadata.db')

  metadata_config = metadata.sqlite_metadata_connection_config(metadata_dir)
  with metadata.Metadata(metadata_config) as m:
    if not run_id:
      execution_dict = defaultdict(list)# TODO: better naming
      for execution in m.store.get_executions():
        execution_run_id = execution.properties['run_id'].string_value
        execution_dict[execution_run_id].append(execution)
      run_id = max(execution_dict.keys()) # fetch the latest run_id

    events = [
        x for x in m.store.get_events_by_execution_ids(
            [e.id for e in execution_dict[run_id]])
        if x.type == metadata_store_pb2.Event.OUTPUT
    ]
    unique_artifact_ids = list({x.artifact_id for x in events})

    for artifact in m.store.get_artifacts_by_id(unique_artifact_ids):
      src_path = artifact.uri
      dest_path = src_path.replace(pipeline_path, "")
      dest_path = dest_path[:dest_path.rfind('/')]
      dest_path = os.path.join(record_dir, dest_path)

      os.makedirs(dest_path, exist_ok=True)
      copy_tree(src_path, dest_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--pipeline_path',
      type=str,
      default=os.path.join(os.environ['HOME'],
                           "tfx/pipelines/chicago_taxi_beam/"),
      help='Path to pipeline')
  parser.add_argument(
      '--record_path',
      type=str,
      default=os.path.join(os.environ['HOME'],
                           'tfx/tfx/experimental/pipeline_testing/',
                           'testdata'),
      help='Path to record')
  parser.add_argument(
      '--run_id',
      type=str,
      default=None,
      help='Pipeline Run Id')

  args = parser.parse_args()
  main(args.record_path, args.pipeline_path, args.run_id)
