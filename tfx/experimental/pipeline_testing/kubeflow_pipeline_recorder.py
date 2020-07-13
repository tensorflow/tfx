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
"""Recording pipeline using Kubeflow gRPC metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os
import subprocess

from absl import app
from absl import flags
from ml_metadata.proto import metadata_store_pb2

from tfx.orchestration import metadata


FLAGS = flags.FLAGS

flags.DEFINE_string('record_dir', None, 'Path to record')
flags.DEFINE_integer('local_port', None, 'Local port number')
flags.DEFINE_string('run_id', None, 'KfP Pipeline Run Id')

flags.mark_flag_as_required('record_dir')
flags.mark_flag_as_required('local_port')
flags.mark_flag_as_required('run_id')

def main(unused_argv):
  connection_config = metadata_store_pb2.MetadataStoreClientConfig()
  connection_config.host = 'localhost'
  connection_config.port = FLAGS.local_port
  with metadata.Metadata(connection_config) as m:
    execution_dict = defaultdict(list)
    for execution in m.store.get_executions():
      execution_run_id = execution.properties['run_id'].string_value
      execution_dict[execution_run_id].append(execution)
    if FLAGS.run_id not in execution_dict:
      raise ValueError(
          "run_id {} is not recorded in the MLMD metadata".format(FLAGS.run_id))
    events = [
        x for x in m.store.get_events_by_execution_ids(
            [e.id for e in execution_dict[FLAGS.run_id]])
        if x.type == metadata_store_pb2.Event.OUTPUT
    ]
    unique_artifact_ids = list({x.artifact_id for x in events})

    for artifact in m.store.get_artifacts_by_id(unique_artifact_ids):
      src_dir = artifact.uri + '/*'
      print(src_dir)
      component_id = \
          artifact.custom_properties['producer_component'].string_value
      name = artifact.custom_properties['name'].string_value
      dest_dir = os.path.join(FLAGS.record_dir, component_id, name)
      os.makedirs(dest_dir, exist_ok=True)
      subprocess.run(['gsutil', 'cp', '-r', src_dir, dest_dir], check=True)

if __name__ == '__main__':
  app.run(main)
