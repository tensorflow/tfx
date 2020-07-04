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

import absl
from absl import app
from absl import flags
from collections import defaultdict
from distutils.dir_util import copy_tree
from ml_metadata.proto import metadata_store_pb2
import os
from tfx.orchestration import metadata

FLAGS = flags.FLAGS

_pipeline_name = 'chicago_taxi_beam'
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')

default_record_dir = os.path.join('examples/chicago_taxi_pipeline/testdata')
default_metadata_dir = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                    'metadata.db')

flags.DEFINE_string('record_dir', default_record_dir, 'Path to record')
flags.DEFINE_string('metadata_dir', default_metadata_dir, 'Path to metadata')
flags.DEFINE_string('run_id', None, 'Pipeline Run Id')

def main(unused_argv):
  run_id = FLAGS.run_id
  metadata_dir = FLAGS.metadata_dir
  metadata_config = metadata.sqlite_metadata_connection_config(metadata_dir)
  with metadata.Metadata(metadata_config) as m:
    execution_dict = defaultdict(list)
    for execution in m.store.get_executions():
      execution_run_id = execution.properties['run_id'].string_value
      execution_dict[execution_run_id].append(execution)
    if run_id is None:
      run_id = max(execution_dict.keys()) # fetch the latest run_id

    events = [
        x for x in m.store.get_events_by_execution_ids(
            [e.id for e in execution_dict[run_id]])
        if x.type == metadata_store_pb2.Event.OUTPUT
    ]
    unique_artifact_ids = list({x.artifact_id for x in events})

    for artifact in m.store.get_artifacts_by_id(unique_artifact_ids):
      src_dir = artifact.uri
      component_id = \
          artifact.custom_properties['producer_component'].string_value
      name = artifact.custom_properties['name'].string_value
      dest_dir = os.path.join(FLAGS.record_dir, component_id, name)
      os.makedirs(dest_dir, exist_ok=True)
      copy_tree(src_dir, dest_dir)
    absl.logging.info("Pipeline Recorded at %s", FLAGS.record_dir)

if __name__ == '__main__':
  app.run(main)
